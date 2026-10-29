"""Jointly train a direct spectrum decoder with a low-weight distribution loss."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import LossConfig, ModelConfig
from dataset import (
    BlockShuffleBatchSampler,
    MultiChargeHDFDataset,
    prebatched_collate,
    validate_target_values,
)
from decision_model import DecisionConfig, DistributionToSpectrumDecoder
from losses import loss_components, make_theoretical_mask
from metrics import MetricAccumulator
from model import MultiChargeHurdleMVN


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(os.path.dirname(HERE), "data")
DEFAULT_STAGE1 = os.path.join(
    os.path.dirname(HERE), "outputs", "nvidia", "stage1", "stage1_best.pt"
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict_distribution(
    model: MultiChargeHurdleMVN, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        batch["left_ids"],
        batch["right_ids"],
        batch["cleavage_mask"],
        batch["frag_id"],
        batch["instrument_id"],
        batch["e_bucket"],
        batch["precursor_charge"],
        batch["n_term_form_id"],
    )


def metric_view(
    distribution: dict[str, torch.Tensor], intensity: torch.Tensor
) -> dict[str, torch.Tensor]:
    output = dict(distribution)
    output["expected_intensity"] = intensity
    return output


def run_epoch(
    stage1: MultiChargeHurdleMVN,
    decoder: DistributionToSpectrumDecoder,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: LossConfig,
    distribution_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    collect_metrics: bool = False,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict | None]:
    training = optimizer is not None
    stage1.train(training)
    decoder.train(training)
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    direct_metrics = MetricAccumulator() if collect_metrics else None
    base_metrics = MetricAccumulator() if collect_metrics else None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        batch_size = int(batch["y_intensity"].shape[0])
        theoretical = make_theoretical_mask(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                distribution = predict_distribution(stage1, batch)
                output = decoder(distribution, theoretical)
            with torch.autocast(device_type=device.type, enabled=False):
                components = loss_components(
                    metric_view(distribution, output["predicted_intensity"]),
                    batch,
                    loss_cfg,
                    phase="stage2",
                    stage2_distribution_weight=distribution_weight,
                )

        if training:
            assert optimizer is not None and scaler is not None
            if not torch.isfinite(components["total"]):
                raise FloatingPointError(
                    f"non-finite direct-decoder loss at batch {batch_index}"
                )
            scaler.scale(components["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(stage1.parameters()) + list(decoder.parameters()), 5.0
            )
            scaler.step(optimizer)
            scaler.update()

        for key, value in components.items():
            totals[key] += float(value.detach()) * batch_size
        samples += batch_size

        if direct_metrics is not None and base_metrics is not None:
            direct_metrics.update(
                metric_view(distribution, output["predicted_intensity"]), batch
            )
            base_metrics.update(
                metric_view(distribution, output["base_intensity"]), batch
            )

    losses = {key: value / max(1, samples) for key, value in totals.items()}
    metrics = None
    if direct_metrics is not None and base_metrics is not None:
        metrics = {
            "direct": direct_metrics.compute(),
            "stage1_base": base_metrics.compute(),
        }
    return losses, metrics


def make_optimizer(
    stage1: MultiChargeHurdleMVN,
    decoder: DistributionToSpectrumDecoder,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    stage1_heads: list[torch.nn.Parameter] = []
    stage1_backbone: list[torch.nn.Parameter] = []
    for name, parameter in stage1.named_parameters():
        destination = (
            stage1_heads
            if name.startswith("intensity_") or name.startswith("occurrence_")
            else stage1_backbone
        )
        destination.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": stage1_backbone,
                "lr": args.stage1_backbone_learning_rate,
                "name": "stage1_backbone",
            },
            {
                "params": stage1_heads,
                "lr": args.stage1_head_learning_rate,
                "name": "stage1_heads",
            },
            {
                "params": decoder.parameters(),
                "lr": args.decoder_learning_rate,
                "name": "decoder",
            },
        ],
        weight_decay=args.weight_decay,
    )


def make_loaders(args: argparse.Namespace):
    train_dataset = MultiChargeHDFDataset(
        args.train_hdf5, args.max_train_samples, prebatch=True
    )
    val_dataset = MultiChargeHDFDataset(
        args.val_hdf5, args.max_val_samples, prebatch=True
    )
    train_sampler = BlockShuffleBatchSampler(
        len(train_dataset),
        args.batch_size,
        block_size=args.shuffle_block_size,
        seed=args.seed,
    )
    worker_options = {
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        worker_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=prebatched_collate,
        **worker_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=prebatched_collate,
        **worker_options,
    )
    return train_dataset, val_dataset, train_sampler, train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly train a direct decoder from hurdle-MVN parameters"
    )
    parser.add_argument(
        "--train-hdf5", default=os.path.join(DEFAULT_DATA, "train_multi_charge.hdf5")
    )
    parser.add_argument(
        "--val-hdf5", default=os.path.join(DEFAULT_DATA, "val_multi_charge.hdf5")
    )
    parser.add_argument("--stage1-checkpoint", default=DEFAULT_STAGE1)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "log"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--shuffle-block-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--decoder-learning-rate", type=float, default=5e-4)
    parser.add_argument("--stage1-backbone-learning-rate", type=float, default=5e-5)
    parser.add_argument("--stage1-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage2-distribution-weight", type=float, default=0.01)
    parser.add_argument("--w-sa", type=float, default=1.0)
    parser.add_argument("--w-mvn", type=float, default=0.1)
    parser.add_argument("--w-bce", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--occurrence-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and torch.version.cuda is None:
        raise RuntimeError("The NVIDIA entry point requires a CUDA-enabled PyTorch build")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.stage2_distribution_weight < 0.0:
        raise ValueError("--stage2-distribution-weight must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but it is unavailable")

    device = torch.device(args.device)
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location=device, weights_only=False
    )
    stage1_cfg = ModelConfig(**stage1_checkpoint["model_config"])
    stage1 = MultiChargeHurdleMVN(stage1_cfg).to(device)
    stage1.load_state_dict(stage1_checkpoint["model"])
    stage1.requires_grad_(True)

    loss_cfg = LossConfig(
        w_sa=args.w_sa,
        w_mvn=args.w_mvn,
        w_bce=args.w_bce,
        jitter=args.jitter,
        occurrence_threshold=args.occurrence_threshold,
    )

    decision_cfg = DecisionConfig(
        covariance_rank=stage1_cfg.covariance_rank,
        hidden_size=args.hidden_size,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
    )
    decoder = DistributionToSpectrumDecoder(decision_cfg).to(device)
    optimizer = make_optimizer(stage1, decoder, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    train_dataset, val_dataset, train_sampler, train_loader, val_loader = make_loaders(args)

    stage1_parameters = sum(parameter.numel() for parameter in stage1.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())
    run_config = {
        "args": vars(args),
        "stage1_model_config": asdict(stage1_cfg),
        "loss_config": asdict(loss_cfg),
        "decision_config": asdict(decision_cfg),
        "stage1_parameters_trainable": stage1_parameters,
        "decoder_parameters_trainable": decoder_parameters,
        "stage1_source_epoch": stage1_checkpoint.get("epoch"),
        "target_diagnostics": validate_target_values(args.train_hdf5),
        "decision_rule": "max_normalize(softplus(decoder(pi, mu, covariance)))",
        "training_objective": (
            "w_sa * SA(direct_prediction, Y) + lambda_dist * "
            "(w_mvn * MVN_NLL + w_bce * BCE)"
        ),
        "decoder_inputs": list(decoder.feature_names),
        "covariance_parameterization": decoder.covariance_parameterization,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as handle:
        json.dump(run_config, handle, indent=2)

    log_path = os.path.join(args.out_dir, "train.log")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    checkpoint_path = os.path.join(args.out_dir, "best.pt")
    header = (
        f"train={len(train_dataset)} val={len(val_dataset)} device={device} "
        f"stage1_trainable={stage1_parameters} decoder_trainable={decoder_parameters} "
        f"batch={args.batch_size} lambda_dist={args.stage2_distribution_weight}"
    )
    print(header, flush=True)
    with open(log_path, "w") as handle:
        handle.write(header + "\n")
    with open(metrics_path, "w"):
        pass

    best_sa = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        started = time.time()
        train_losses, _ = run_epoch(
            stage1,
            decoder,
            train_loader,
            device,
            loss_cfg,
            args.stage2_distribution_weight,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_losses, val_metrics = run_epoch(
                stage1,
                decoder,
                val_loader,
                device,
                loss_cfg,
                args.stage2_distribution_weight,
                use_amp=args.amp,
                collect_metrics=True,
                max_batches=args.max_val_batches,
            )
        assert val_metrics is not None
        direct = val_metrics["direct"]
        base = val_metrics["stage1_base"]
        improved = val_losses["sa"] < best_sa - args.min_delta
        if improved:
            best_sa = val_losses["sa"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "stage1_model": stage1.state_dict(),
                    "decoder": decoder.state_dict(),
                    "stage1_model_config": asdict(stage1_cfg),
                    "decision_config": asdict(decision_cfg),
                    "stage1_checkpoint": args.stage1_checkpoint,
                    "epoch": epoch,
                    "val_sa_loss": best_sa,
                    "stage2_distribution_weight": args.stage2_distribution_weight,
                    "loss_config": asdict(loss_cfg),
                    "metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "learning_rates": {
                group["name"]: group["lr"] for group in optimizer.param_groups
            },
            "elapsed_seconds": time.time() - started,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "improved": improved,
        }
        with open(metrics_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        resolved_gain = (
            direct["charge_resolved_cosine"]
            - base["charge_resolved_cosine"]
        )
        summed_gain = (
            direct["charge_aggregated_cosine"]
            - base["charge_aggregated_cosine"]
        )
        line = (
            f"epoch={epoch:03d} train_total={train_losses['total']:.5f} "
            f"val_total={val_losses['total']:.5f} val_sa={val_losses['sa']:.5f} "
            f"dist={val_losses['distribution']:.5f} "
            f"mvn={val_losses['mvn_nll']:.5f} bce={val_losses['bce']:.5f} "
            f"resolved_cos={direct['charge_resolved_cosine']:.6f} "
            f"base_resolved_cos={base['charge_resolved_cosine']:.6f} "
            f"gain={resolved_gain:+.6f} "
            f"summed_cos={direct['charge_aggregated_cosine']:.6f} "
            f"base_summed_cos={base['charge_aggregated_cosine']:.6f} "
            f"gain={summed_gain:+.6f} best_sa={best_sa:.5f} "
            f"time={row['elapsed_seconds']:.1f}s"
        )
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")
        scheduler.step()

        if args.patience > 0 and stale_epochs >= args.patience:
            stop_line = (
                f"early_stop={epoch} best_epoch={best_epoch} best_sa={best_sa:.5f}"
            )
            print(stop_line, flush=True)
            with open(log_path, "a") as handle:
                handle.write(stop_line + "\n")
            break

    print(
        f"best_epoch={best_epoch} best_sa={best_sa:.5f} checkpoint={checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
