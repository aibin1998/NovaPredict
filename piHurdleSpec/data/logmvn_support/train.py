"""Train a hurdle low-rank Log-MVN CNN-BiLSTM on the controlled TDRepo split."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
from .compat import (
    TrainConfig,
    collate_batch,
    masked_bce_with_logits,
    masked_ce,
    spectral_angle_similarity,
)
from .metrics import (
    batch_break_counts,
    batch_charge_pcc,
    batch_intensity_pcc,
    summarize_break_metrics,
    summarize_pcc,
)
from .dataset import BaselineRAMDataset
from .losses import masked_low_rank_log_mvn_nll, probability_summary
from .model import (
    LogMVNCNNBiLSTMFragmentPredictor,
    load_deterministic_checkpoint,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, batch):
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


def loss_components(output, batch, cfg):
    observed = batch["cleavage_mask"].unsqueeze(-1) * batch["ion_mask"]
    theoretical = batch["cleavage_mask"].unsqueeze(-1).expand_as(batch["ion_mask"])
    log_mvn_nll = masked_low_rank_log_mvn_nll(
        output["log_location"],
        output["log_scale"],
        output["low_rank_factor"],
        batch["y_intensity"],
        observed,
        target_eps=cfg.log_mvn_target_eps,
        jitter=cfg.log_mvn_jitter,
    )
    shape = spectral_angle_similarity(
        output["expected_intensity"], batch["y_intensity"], theoretical
    )
    charge = masked_ce(
        output["charge_logits"], batch["y_charge"], observed, ignore_index=0
    )
    break_loss = masked_bce_with_logits(
        output["break_logits"], batch["ion_mask"], theoretical
    )
    total = (
        cfg.w_intensity * shape
        + cfg.w_log_mvn * log_mvn_nll
        + cfg.w_charge * charge
        + cfg.w_break * break_loss
    )
    return {
        "total": total,
        "shape": shape,
        "log_mvn_nll": log_mvn_nll,
        "charge": charge,
        "break": break_loss,
    }


def run_epoch(model, loader, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    for batch_index, batch in enumerate(loader):
        batch = {
            key: value.to(cfg.device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.set_grad_enabled(training):
            output = predict(model, batch)
            components = loss_components(output, batch, cfg)
        if training:
            optimizer.zero_grad(set_to_none=True)
            components["total"].backward()
            if (
                cfg.check_encoder_gradient
                and model.encoder.weight_ih_l0.requires_grad
                and batch_index < 2
            ):
                encoder_grad = model.encoder.weight_ih_l0.grad
                if (
                    encoder_grad is None
                    or not bool(torch.isfinite(encoder_grad).all())
                    or not bool(encoder_grad.abs().max() > 0)
                ):
                    raise RuntimeError("encoder gradient vanished; check the LSTM backend")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for key, value in components.items():
            totals[key] += float(value.detach().item())
    denominator = max(1, len(loader))
    return {key: value / denominator for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    conditional = {key: [] for key in ("all", "all_id", "obs", "obs_id")}
    expected = {key: [] for key in ("all", "all_id", "obs", "obs_id")}
    charge, charge_ids = [], []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    probability_totals = defaultdict(float)
    batches = 0

    for batch in loader:
        batch = {
            key: value.to(cfg.device, non_blocking=True)
            for key, value in batch.items()
        }
        output = predict(model, batch)
        conditional_row = batch_intensity_pcc(
            output["conditional_mean"],
            batch["y_intensity"],
            batch["cleavage_mask"],
            batch["ion_mask"],
            batch["prsm_id"],
        )
        expected_row = batch_intensity_pcc(
            output["expected_intensity"],
            batch["y_intensity"],
            batch["cleavage_mask"],
            batch["ion_mask"],
            batch["prsm_id"],
        )
        for destination, source in (("all", "all_valid"), ("obs", "observed_only")):
            conditional[destination] += conditional_row[source]
            conditional[destination + "_id"] += conditional_row[source + "_prsm"]
            expected[destination] += expected_row[source]
            expected[destination + "_id"] += expected_row[source + "_prsm"]

        charge_row = batch_charge_pcc(
            output["charge_logits"],
            batch["y_charge"],
            batch["cleavage_mask"],
            batch["ion_mask"],
            batch["prsm_id"],
        )
        charge += charge_row["observed_only"]
        charge_ids += charge_row["observed_only_prsm"]
        break_row = batch_break_counts(
            output["break_logits"],
            batch["cleavage_mask"],
            batch["ion_mask"],
            cfg.break_prob_threshold,
        )
        for key in counts:
            counts[key] += break_row[key]

        observed_mask = batch["cleavage_mask"].unsqueeze(-1) * batch["ion_mask"]
        theoretical_mask = batch["cleavage_mask"].unsqueeze(-1).expand_as(
            batch["ion_mask"]
        )
        probability = probability_summary(
            output,
            batch["y_intensity"],
            observed_mask,
            theoretical_mask,
            target_eps=cfg.log_mvn_target_eps,
        )
        for key, value in probability.items():
            probability_totals[key] += float(value.item())
        batches += 1

    return {
        "conditional_theoretical": summarize_pcc(
            conditional["all"], conditional["all_id"]
        ),
        "conditional_annotated": summarize_pcc(
            conditional["obs"], conditional["obs_id"]
        ),
        "expected_theoretical": summarize_pcc(expected["all"], expected["all_id"]),
        "expected_annotated": summarize_pcc(expected["obs"], expected["obs_id"]),
        "charge_annotated": summarize_pcc(charge, charge_ids),
        "break": summarize_break_metrics(
            **counts, threshold=cfg.break_prob_threshold
        ),
        "probability": {
            key: value / max(1, batches)
            for key, value in probability_totals.items()
        },
    }


def set_head_only_training(model, enabled: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("intensity_") if enabled else True


def make_optimizer(model, cfg, backbone_lr: float, head_lr: float):
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("intensity_"):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hdf5", required=True)
    parser.add_argument("--val-hdf5", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detail-interval", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "log"))
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--w-log-mvn", type=float, default=0.1)
    parser.add_argument("--covariance-rank", type=int, default=8)
    parser.add_argument("--location-min", type=float, default=float(np.log(1e-4)))
    parser.add_argument("--location-max", type=float, default=0.0)
    parser.add_argument("--scale-min", type=float, default=0.05)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--factor-max", type=float, default=0.25)
    parser.add_argument("--target-eps", type=float, default=1e-4)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--check-encoder-gradient", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({args.device}) but CUDA is unavailable")

    cfg = TrainConfig()
    cfg.device = args.device
    cfg.intensity_log1p = False
    cfg.intensity_per_prsm_norm = True
    cfg.eval_break_vec_pcc = True
    cfg.w_log_mvn = args.w_log_mvn
    cfg.log_mvn_target_eps = args.target_eps
    cfg.log_mvn_jitter = args.jitter
    cfg.check_encoder_gradient = args.check_encoder_gradient
    if args.batch_size:
        cfg.batch_size = args.batch_size

    seed_all(cfg.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds = BaselineRAMDataset(args.train_hdf5, args.max_train_samples)
    val_ds = BaselineRAMDataset(args.val_hdf5, args.max_val_samples)
    train_loader = DataLoader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=cfg.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=cfg.device.startswith("cuda"),
    )

    model = LogMVNCNNBiLSTMFragmentPredictor(
        cfg,
        covariance_rank=args.covariance_rank,
        location_min=args.location_min,
        location_max=args.location_max,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        factor_max=args.factor_max,
    ).to(cfg.device)
    transfer_report = None
    if args.init_checkpoint:
        transfer_report = load_deterministic_checkpoint(model, args.init_checkpoint)
        print(f"warm-start: {json.dumps(transfer_report)}", flush=True)

    optimizer = make_optimizer(model, cfg, args.backbone_lr, args.head_lr)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "args": vars(args),
        "cfg": cfg.__dict__,
        "parameter_count": parameter_count,
        "backbone": model.backbone_name,
        "effective_receptive_field": model.effective_receptive_field,
        "probability_model": {
            "family": "hurdle_low_rank_log_mvn",
            "log_intensity_distribution": "Normal(location, D + U U^T)",
            "covariance_rank": args.covariance_rank,
            "location_range": [args.location_min, args.location_max],
            "scale_range": [args.scale_min, args.scale_max],
            "factor_max_abs": args.factor_max,
            "expected_intensity": (
                "sigmoid(break_logit) * "
                "exp(location + 0.5 * diag(D + U U^T))"
            ),
        },
        "transfer_report": transfer_report,
        "conditions": [
            "fragmentation",
            "instrument",
            "collision_energy",
            "precursor_charge",
            "n_terminal_form",
        ],
        "excluded_metadata": [
            "faims_cv",
            "precursor_intensity",
            "retention_time",
        ],
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)

    log_path = os.path.join(args.out_dir, "train.log")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    checkpoint_path = os.path.join(args.out_dir, "best.pt")
    header = (
        f"train={len(train_ds)} val={len(val_ds)} device={cfg.device} "
        f"parameters={parameter_count} backbone={model.backbone_name} "
        f"log_mvn_weight={cfg.w_log_mvn} rank={args.covariance_rank} "
        f"warmup={args.head_warmup_epochs}"
    )
    print(header, flush=True)
    with open(log_path, "w") as handle:
        handle.write(header + "\n")
    with open(metrics_path, "w"):
        pass

    best = float("inf")
    warmup_enabled = bool(args.init_checkpoint and args.head_warmup_epochs > 0)
    set_head_only_training(model, warmup_enabled)

    for epoch in range(1, args.epochs + 1):
        if warmup_enabled and epoch == args.head_warmup_epochs + 1:
            set_head_only_training(model, False)
            warmup_enabled = False
            message = f"epoch {epoch:03d} unfreeze=all"
            print(message, flush=True)
            with open(log_path, "a") as handle:
                handle.write(message + "\n")

        start = time.time()
        train_losses = run_epoch(model, train_loader, cfg, optimizer)
        val_losses = run_epoch(model, val_loader, cfg)
        line = (
            f"epoch {epoch:03d} train_total={train_losses['total']:.5f} "
            f"val_total={val_losses['total']:.5f} "
            f"val_shape={val_losses['shape']:.5f} "
            f"val_log_mvn_nll={val_losses['log_mvn_nll']:.5f} "
            f"val_break={val_losses['break']:.5f} "
            f"time={time.time() - start:.1f}s"
        )
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")

        if val_losses["total"] < best:
            best = val_losses["total"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": best,
                    "parameter_count": parameter_count,
                    "probability_model": config["probability_model"],
                },
                checkpoint_path,
            )

        if epoch % args.detail_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, cfg)
            row = {
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                **metrics,
            }
            with open(metrics_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            detail = (
                f"epoch {epoch:03d} "
                f"conditional_theo={metrics['conditional_theoretical']['pcc_mean']:.4f} "
                f"conditional_ann={metrics['conditional_annotated']['pcc_mean']:.4f} "
                f"expected_theo={metrics['expected_theoretical']['pcc_mean']:.4f} "
                f"expected_ann={metrics['expected_annotated']['pcc_mean']:.4f} "
                f"break_f1={metrics['break']['f1']:.4f} "
                f"log_mvn_nll={metrics['probability']['log_mvn_nll']:.4f} "
                f"log_scale={metrics['probability']['mean_log_scale']:.3f} "
                f"factor_rms={metrics['probability']['factor_rms']:.4f}"
            )
            print(detail, flush=True)
            with open(log_path, "a") as handle:
                handle.write(detail + "\n")

    line = f"best_val={best:.5f} checkpoint={checkpoint_path}"
    print(line, flush=True)
    with open(log_path, "a") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()
