"""Evaluate a trained multi-charge hurdle-MVN checkpoint."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from config import LossConfig, ModelConfig
from dataset import MultiChargeHDFDataset, prebatched_collate
from model import MultiChargeHurdleMVN
from train import DEFAULT_DATA, run_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--hdf5", default=f"{DEFAULT_DATA}/val_multi_charge.hdf5"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.device.startswith("cuda") and torch.version.cuda is None:
        raise RuntimeError("The NVIDIA entry point requires a CUDA-enabled PyTorch build")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**checkpoint["model_config"])
    loss_cfg = LossConfig(**checkpoint["loss_config"])
    model = MultiChargeHurdleMVN(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset = MultiChargeHDFDataset(args.hdf5, args.max_samples, prebatch=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=prebatched_collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    with torch.no_grad():
        losses, metrics = run_epoch(
            model,
            loader,
            loss_cfg,
            device,
            use_amp=args.amp,
            collect_metrics=True,
            max_batches=args.max_batches,
            phase=checkpoint.get("phase", "stage2"),
            stage2_distribution_weight=checkpoint.get(
                "stage2_distribution_weight", 0.1
            ),
        )
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "hdf5": args.hdf5,
        "spectra": len(dataset),
        "losses": losses,
        "metrics": metrics,
        "stage2_distribution_weight": checkpoint.get(
            "stage2_distribution_weight", 0.1
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
