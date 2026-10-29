"""Two-stage training for the charge-resolved hurdle-MVN spectrum model."""
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
from losses import loss_components
from metrics import MetricAccumulator
from model import MultiChargeHurdleMVN, load_backbone_checkpoint


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(os.path.dirname(HERE), "data")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict(model: MultiChargeHurdleMVN, batch: dict[str, torch.Tensor]):
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


def run_epoch(
    model: MultiChargeHurdleMVN,
    loader: DataLoader,
    loss_cfg: LossConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    collect_metrics: bool = False,
    max_batches: int | None = None,
    phase: str = "stage2",
    stage2_distribution_weight: float = 0.1,
) -> tuple[dict[str, float], dict | None]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    metrics = MetricAccumulator(loss_cfg.occurrence_threshold) if collect_metrics else None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        batch_size = int(batch["y_intensity"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                output = predict(model, batch)
            # Cholesky and likelihood calculations need float32 stability.
            with torch.autocast(device_type=device.type, enabled=False):
                components = loss_components(
                    output,
                    batch,
                    loss_cfg,
                    phase=phase,
                    stage2_distribution_weight=stage2_distribution_weight,
                )

        if training:
            assert optimizer is not None and scaler is not None
            if not torch.isfinite(components["total"]):
                raise FloatingPointError(
                    f"non-finite {phase} loss at batch {batch_index}: "
                    f"{float(components['total'].detach())}"
                )
            scaler.scale(components["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        for key, value in components.items():
            totals[key] += float(value.detach()) * batch_size
        samples += batch_size
        if metrics is not None:
            metrics.update(output, batch)

    losses = {key: value / max(1, samples) for key, value in totals.items()}
    return losses, metrics.compute() if metrics is not None else None


def make_optimizer(
    model: MultiChargeHurdleMVN,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        destination = (
            head
            if name.startswith("intensity_") or name.startswith("occurrence_")
            else backbone
        )
        destination.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ],
        weight_decay=weight_decay,
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


def save_checkpoint(
    path: str,
    model: MultiChargeHurdleMVN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    phase: str,
    phase_epoch: int,
    global_epoch: int,
    selection_name: str,
    selection_value: float,
    stage2_distribution_weight: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": asdict(model_cfg),
            "loss_config": asdict(loss_cfg),
            "phase": phase,
            "phase_epoch": phase_epoch,
            "epoch": global_epoch,
            "selection_metric": selection_name,
            "val_loss": selection_value,
            "stage2_distribution_weight": stage2_distribution_weight,
            "probability_model": {
                "family": "hurdle_low_rank_mvn",
                "expected_intensity": "sigmoid(occurrence_logit) * conditional_mean",
            },
        },
        path,
    )


def train_phase(
    *,
    phase: str,
    epochs: int,
    patience: int,
    global_epoch: int,
    model: MultiChargeHurdleMVN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: BlockShuffleBatchSampler,
    loss_cfg: LossConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: str,
    log_path: str,
    metrics_path: str,
) -> tuple[int, dict]:
    selection_name = "distribution" if phase == "stage1" else "sa"
    distribution_weight = 0.0 if phase == "stage1" else args.stage2_distribution_weight
    best_value = float("inf")
    best_phase_epoch = 0
    stale_epochs = 0

    for phase_epoch in range(1, epochs + 1):
        global_epoch += 1
        train_sampler.set_epoch(global_epoch)
        started = time.time()
        train_losses, _ = run_epoch(
            model,
            train_loader,
            loss_cfg,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
            max_batches=args.max_train_batches,
            phase=phase,
            stage2_distribution_weight=distribution_weight,
        )
        with torch.no_grad():
            val_losses, val_metrics = run_epoch(
                model,
                val_loader,
                loss_cfg,
                device,
                use_amp=args.amp,
                collect_metrics=True,
                max_batches=args.max_val_batches,
                phase=phase,
                stage2_distribution_weight=distribution_weight,
            )
        assert val_metrics is not None
        selection_value = val_losses[selection_name]
        improved = selection_value < best_value - args.min_delta
        if improved:
            best_value = selection_value
            best_phase_epoch = phase_epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                model_cfg,
                loss_cfg,
                phase,
                phase_epoch,
                global_epoch,
                selection_name,
                best_value,
                args.stage2_distribution_weight,
            )
        else:
            stale_epochs += 1

        row = {
            "phase": phase,
            "phase_epoch": phase_epoch,
            "global_epoch": global_epoch,
            "elapsed_seconds": time.time() - started,
            "selection_metric": selection_name,
            "selection_value": selection_value,
            "improved": improved,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
        }
        with open(metrics_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        line = (
            f"phase={phase} epoch={phase_epoch:03d}/{epochs:03d} "
            f"train={train_losses['total']:.5f} val={val_losses['total']:.5f} "
            f"dist={val_losses['distribution']:.5f} sa={val_losses['sa']:.5f} "
            f"mvn={val_losses['mvn_nll']:.5f} bce={val_losses['bce']:.5f} "
            f"resolved_pcc={val_metrics['charge_resolved_pcc']:.4f} "
            f"summed_pcc={val_metrics['charge_aggregated_pcc']:.4f} "
            f"occ_f1={val_metrics['occurrence']['f1']:.4f} "
            f"best_{selection_name}={best_value:.5f} "
            f"time={row['elapsed_seconds']:.1f}s"
        )
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")

        if patience > 0 and stale_epochs >= patience:
            stop_line = (
                f"phase={phase} early_stop={phase_epoch} "
                f"best_epoch={best_phase_epoch} best_{selection_name}={best_value:.5f}"
            )
            print(stop_line, flush=True)
            with open(log_path, "a") as handle:
                handle.write(stop_line + "\n")
            break

    return global_epoch, {
        "phase": phase,
        "best_epoch": best_phase_epoch,
        "best_metric": selection_name,
        "best_value": best_value,
        "checkpoint": checkpoint_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage training for charge-resolved hurdle-MVN prediction"
    )
    parser.add_argument(
        "--train-hdf5", default=os.path.join(DEFAULT_DATA, "train_multi_charge.hdf5")
    )
    parser.add_argument(
        "--val-hdf5", default=os.path.join(DEFAULT_DATA, "val_multi_charge.hdf5")
    )
    parser.add_argument("--out-dir", default=os.path.join(HERE, "log"))
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage1-patience", type=int, default=5)
    parser.add_argument("--stage2-patience", type=int, default=10)
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
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--stage1-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-head-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-backbone-lr", type=float, default=5e-5)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--w-sa", type=float, default=1.0)
    parser.add_argument("--w-mvn", type=float, default=0.1)
    parser.add_argument("--w-bce", type=float, default=0.5)
    parser.add_argument("--stage2-distribution-weight", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--occurrence-threshold", type=float, default=0.5)
    parser.add_argument("--covariance-rank", type=int, default=4)
    parser.add_argument("--mean-min", type=float, default=1e-5)
    parser.add_argument("--mean-max", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=1e-3)
    parser.add_argument("--scale-max", type=float, default=0.5)
    parser.add_argument("--factor-max", type=float, default=0.1)
    parser.add_argument("--initial-mean", type=float, default=0.1)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--occurrence-prior", type=float, default=0.02)
    parser.add_argument("--aa-embed-dim", type=int, default=64)
    parser.add_argument("--cond-embed-dim", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and torch.version.cuda is None:
        raise RuntimeError("The NVIDIA entry point requires a CUDA-enabled PyTorch build")
    if args.init_checkpoint and args.stage1_checkpoint:
        raise ValueError("--init-checkpoint and --stage1-checkpoint are mutually exclusive")
    if args.stage1_epochs < 1 and not args.stage1_checkpoint:
        raise ValueError("stage1 requires --stage1-epochs >= 1 or --stage1-checkpoint")
    if args.stage2_epochs < 0:
        raise ValueError("--stage2-epochs must be non-negative")
    if args.stage2_distribution_weight < 0.0:
        raise ValueError("--stage2-distribution-weight must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but it is unavailable")

    device = torch.device(args.device)
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    model_cfg = ModelConfig(
        aa_embed_dim=args.aa_embed_dim,
        cond_embed_dim=args.cond_embed_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        covariance_rank=args.covariance_rank,
        mean_min=args.mean_min,
        mean_max=args.mean_max,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        factor_max=args.factor_max,
        initial_mean=args.initial_mean,
        initial_scale=args.initial_scale,
        occurrence_prior=args.occurrence_prior,
    )
    loss_cfg = LossConfig(
        w_sa=args.w_sa,
        w_mvn=args.w_mvn,
        w_bce=args.w_bce,
        jitter=args.jitter,
        occurrence_threshold=args.occurrence_threshold,
    )
    train_dataset, val_dataset, train_sampler, train_loader, val_loader = make_loaders(args)
    model = MultiChargeHurdleMVN(model_cfg).to(device)

    transfer_report = None
    if args.init_checkpoint:
        transfer_report = load_backbone_checkpoint(model, args.init_checkpoint)
    if args.stage1_checkpoint:
        stage1_source = torch.load(
            args.stage1_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(stage1_source["model"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log_path = os.path.join(args.out_dir, "train.log")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    stage1_path = os.path.join(args.out_dir, "stage1_best.pt")
    final_path = os.path.join(args.out_dir, "best.pt")
    run_config_path = os.path.join(args.out_dir, "run_config.json")
    run_config = {
        "args": vars(args),
        "model_config": asdict(model_cfg),
        "loss_config": asdict(loss_cfg),
        "parameter_count": parameter_count,
        "backbone": model.backbone_name,
        "target_diagnostics": validate_target_values(args.train_hdf5),
        "training_framework": {
            "stage1": "w_mvn * MVN_NLL(nonzero Y) + w_bce * BCE(Z)",
            "stage1_selection": "validation distribution loss",
            "stage2": "w_sa * SA(sigmoid(logit) * mu, Y) + lambda_dist * stage1_loss",
            "stage2_selection": "validation SA loss",
            "lambda_dist": args.stage2_distribution_weight,
            "recommended_setting": "lambda_dist=0.1 (SA-dominant fine-tuning)",
        },
        "transfer_report": transfer_report,
    }
    with open(run_config_path, "w") as handle:
        json.dump(run_config, handle, indent=2)

    header = (
        f"train={len(train_dataset)} val={len(val_dataset)} device={device} "
        f"parameters={parameter_count} batch={args.batch_size} "
        f"stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs} "
        f"stage2_distribution_weight={args.stage2_distribution_weight}"
    )
    print(header, flush=True)
    with open(log_path, "w") as handle:
        handle.write(header + "\n")
    with open(metrics_path, "w"):
        pass

    global_epoch = 0
    phase_summaries = []
    if args.stage1_checkpoint:
        phase_summaries.append(
            {
                "phase": "stage1",
                "checkpoint": args.stage1_checkpoint,
                "source": "provided",
            }
        )
    else:
        optimizer = make_optimizer(
            model,
            args.stage1_backbone_lr,
            args.stage1_head_lr,
            args.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=args.amp and device.type == "cuda"
        )
        global_epoch, summary = train_phase(
            phase="stage1",
            epochs=args.stage1_epochs,
            patience=args.stage1_patience,
            global_epoch=global_epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            loss_cfg=loss_cfg,
            model_cfg=model_cfg,
            device=device,
            args=args,
            checkpoint_path=stage1_path,
            log_path=log_path,
            metrics_path=metrics_path,
        )
        phase_summaries.append(summary)
        stage1_source = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(stage1_source["model"])

    if args.stage2_epochs == 0:
        run_config["phase_summaries"] = phase_summaries
        with open(run_config_path, "w") as handle:
            json.dump(run_config, handle, indent=2)
        selected_stage1 = (
            args.stage1_checkpoint if args.stage1_checkpoint else stage1_path
        )
        print(f"stage1_checkpoint={selected_stage1} stage2=skipped", flush=True)
        return

    transition = "stage1 best checkpoint restored; Stage-2 optimizer reset"
    print(transition, flush=True)
    with open(log_path, "a") as handle:
        handle.write(transition + "\n")

    optimizer = make_optimizer(
        model,
        args.stage2_backbone_lr,
        args.stage2_head_lr,
        args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    global_epoch, summary = train_phase(
        phase="stage2",
        epochs=args.stage2_epochs,
        patience=args.stage2_patience,
        global_epoch=global_epoch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        loss_cfg=loss_cfg,
        model_cfg=model_cfg,
        device=device,
        args=args,
        checkpoint_path=final_path,
        log_path=log_path,
        metrics_path=metrics_path,
    )
    phase_summaries.append(summary)
    run_config["phase_summaries"] = phase_summaries
    with open(run_config_path, "w") as handle:
        json.dump(run_config, handle, indent=2)
    print(
        f"stage1_checkpoint={stage1_path if not args.stage1_checkpoint else args.stage1_checkpoint} "
        f"final_checkpoint={final_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
