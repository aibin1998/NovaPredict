"""Train the controlled TD-repo CNN-BiLSTM architecture ablation."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, RELEASE_ROOT)
from data.logmvn_support.compat import (  # noqa: E402
    TrainConfig, collate_batch, masked_bce_with_logits, masked_ce,
    spectral_angle_similarity,
)
from data.logmvn_support.metrics import (  # noqa: E402
    batch_break_counts, batch_break_vec_pcc, batch_charge_pcc,
    batch_intensity_pcc, summarize_break_metrics, summarize_pcc,
)

from dataset import BaselineRAMDataset  # noqa: E402
from model import CNNBiLSTMFragmentPredictor  # noqa: E402


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def predict(model, batch):
    return model(batch["left_ids"], batch["right_ids"], batch["cleavage_mask"],
                 batch["frag_id"], batch["instrument_id"], batch["e_bucket"],
                 batch["precursor_charge"], batch["n_term_form_id"])


def run_epoch(model, loader, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(cfg.device, non_blocking=True) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            pred_i, pred_c, pred_b = predict(model, batch)
            observed = batch["cleavage_mask"].unsqueeze(-1) * batch["ion_mask"]
            theoretical = batch["cleavage_mask"].unsqueeze(-1).expand_as(batch["ion_mask"])
            loss = (cfg.w_intensity * spectral_angle_similarity(pred_i, batch["y_intensity"], observed)
                    + cfg.w_charge * masked_ce(pred_c, batch["y_charge"], observed, ignore_index=0)
                    + cfg.w_break * masked_bce_with_logits(pred_b, batch["ion_mask"], theoretical))
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if cfg.device.startswith("cuda") and torch.version.hip is not None and batch_index < 2:
                encoder_grad = model.encoder.weight_ih_l0.grad
                if (encoder_grad is None or not bool(torch.isfinite(encoder_grad).all())
                        or not bool(encoder_grad.abs().max() > 0)):
                    raise RuntimeError("ROCm encoder gradient vanished; use the native LSTM backend")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    vals = {key: [] for key in ("all", "all_id", "obs", "obs_id", "gate_all", "gate_all_id", "gate_obs", "gate_obs_id")}
    charge, charge_ids = [], []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for batch in loader:
        batch = {key: value.to(cfg.device, non_blocking=True) for key, value in batch.items()}
        pred_i, pred_c, pred_b = predict(model, batch)
        raw = batch_intensity_pcc(pred_i, batch["y_intensity"], batch["cleavage_mask"], batch["ion_mask"], batch["prsm_id"])
        gate = batch_break_vec_pcc(pred_i, pred_b, batch["y_intensity"], batch["cleavage_mask"], batch["ion_mask"], batch["prsm_id"])
        ch = batch_charge_pcc(pred_c, batch["y_charge"], batch["cleavage_mask"], batch["ion_mask"], batch["prsm_id"])
        for dst, src in (("all", "all_valid"), ("obs", "observed_only")):
            vals[dst] += raw[src]; vals[dst + "_id"] += raw[src + "_prsm"]
            vals["gate_" + dst] += gate[src]; vals["gate_" + dst + "_id"] += gate[src + "_prsm"]
        charge += ch["observed_only"]; charge_ids += ch["observed_only_prsm"]
        row = batch_break_counts(pred_b, batch["cleavage_mask"], batch["ion_mask"], cfg.break_prob_threshold)
        for key in counts: counts[key] += row[key]
    return {
        "raw_theoretical": summarize_pcc(vals["all"], vals["all_id"]),
        "raw_annotated": summarize_pcc(vals["obs"], vals["obs_id"]),
        "gate_theoretical": summarize_pcc(vals["gate_all"], vals["gate_all_id"]),
        "gate_annotated": summarize_pcc(vals["gate_obs"], vals["gate_obs_id"]),
        "charge_annotated": summarize_pcc(charge, charge_ids),
        "break": summarize_break_metrics(**counts, threshold=cfg.break_prob_threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hdf5", required=True); parser.add_argument("--val-hdf5", required=True)
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--detail-interval", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "log"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and torch.version.hip is None:
        raise RuntimeError("The AMD entry point requires a ROCm-enabled PyTorch build")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"ROCm device requested ({args.device}) but it is unavailable")
    # ROCm's MIOpen LSTM can yield zero encoder gradients on this path.
    torch.backends.cudnn.enabled = False
    cfg = TrainConfig(); cfg.device = args.device; cfg.intensity_log1p = False; cfg.eval_break_vec_pcc = True
    if args.batch_size: cfg.batch_size = args.batch_size
    seed_all(cfg.seed); os.makedirs(args.out_dir, exist_ok=True)
    train_ds = BaselineRAMDataset(args.train_hdf5); val_ds = BaselineRAMDataset(args.val_hdf5)
    if args.max_train_samples: train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples: val_ds = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch, pin_memory=cfg.device.startswith("cuda"))
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch, pin_memory=cfg.device.startswith("cuda"))
    model = CNNBiLSTMFragmentPredictor(cfg, encoder_layers=args.encoder_layers).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    backbone_name = getattr(model, "backbone_name", "residual_cnn_bilstm")
    receptive_field = getattr(model, "effective_receptive_field", None)
    log_path = os.path.join(args.out_dir, "train.log"); metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    checkpoint_path = os.path.join(args.out_dir, "best.pt"); best = float("inf")
    config = {"args": vars(args), "cfg": cfg.__dict__, "parameter_count": parameter_count,
              "backbone": backbone_name, "effective_receptive_field": receptive_field,
              "architecture": {"encoder_layers": args.encoder_layers, "decoder": None},
              "conditions": ["fragmentation", "instrument", "collision_energy", "precursor_charge", "n_terminal_form"],
              "excluded_metadata": ["faims_cv", "precursor_intensity", "retention_time"]}
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as handle: json.dump(config, handle, indent=2)
    header = (f"train={len(train_ds)} val={len(val_ds)} device={cfg.device} "
              f"parameters={parameter_count} backbone={backbone_name} receptive_field={receptive_field}")
    print(header, flush=True)
    with open(log_path, "a") as handle: handle.write(header + "\n")
    for epoch in range(1, args.epochs + 1):
        start = time.time(); train_loss = run_epoch(model, train_loader, cfg, optimizer); val_loss = run_epoch(model, val_loader, cfg)
        line = f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} time={time.time()-start:.1f}s"
        print(line, flush=True)
        with open(log_path, "a") as handle: handle.write(line + "\n")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "val_loss": val_loss, "parameter_count": parameter_count}, checkpoint_path)
        if epoch % args.detail_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, cfg)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            with open(metrics_path, "a") as handle: handle.write(json.dumps(row) + "\n")
            detail = (f"epoch {epoch:03d} raw_theo={metrics['raw_theoretical']['pcc_mean']:.4f} "
                      f"raw_ann={metrics['raw_annotated']['pcc_mean']:.4f} "
                      f"gate_theo={metrics['gate_theoretical']['pcc_mean']:.4f} "
                      f"gate_ann={metrics['gate_annotated']['pcc_mean']:.4f} "
                      f"break_f1={metrics['break']['f1']:.4f}")
            print(detail, flush=True)
            with open(log_path, "a") as handle: handle.write(detail + "\n")
    line = f"best_val={best:.5f} checkpoint={checkpoint_path}"
    print(line, flush=True)
    with open(log_path, "a") as handle: handle.write(line + "\n")


if __name__ == "__main__":
    main()
