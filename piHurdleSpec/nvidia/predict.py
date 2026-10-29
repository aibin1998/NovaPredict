"""Stream Stage-2 intensity predictions from a charge-resolved HDF5 file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from dataset import MultiChargeHDFDataset, prebatched_collate
from decision_model import DecisionConfig, DistributionToSpectrumDecoder
from losses import make_theoretical_mask
from metrics import MetricAccumulator
from model import MultiChargeHurdleMVN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-2 best.pt")
    parser.add_argument("--hdf5", type=Path, required=True, help="Input multi-charge HDF5")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction HDF5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--metrics-json", type=Path, help="Also evaluate against input labels")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and worker count nonnegative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.output.resolve() in {args.hdf5.resolve(), args.checkpoint.resolve()}:
        parser.error("output must differ from the input and checkpoint")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists: {args.output}; pass --overwrite to replace it")
    if args.metrics_json:
        if args.metrics_json.resolve() in {
            args.hdf5.resolve(), args.checkpoint.resolve(), args.output.resolve()
        }:
            parser.error("metrics JSON must differ from the input, checkpoint and output")
        if args.metrics_json.exists() and not args.overwrite:
            parser.error(f"metrics JSON exists: {args.metrics_json}; pass --overwrite to replace it")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and torch.version.cuda is None:
        raise RuntimeError("The NVIDIA entry point requires a CUDA-enabled PyTorch build")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_keys = {"stage1_model", "decoder", "stage1_model_config", "decision_config"}
    missing = expected_keys - set(checkpoint)
    if missing:
        raise KeyError(f"not a Stage-2 decision checkpoint; missing {sorted(missing)}")

    stage1 = MultiChargeHurdleMVN(ModelConfig(**checkpoint["stage1_model_config"])).to(device)
    decoder = DistributionToSpectrumDecoder(DecisionConfig(**checkpoint["decision_config"])).to(device)
    stage1.load_state_dict(checkpoint["stage1_model"])
    decoder.load_state_dict(checkpoint["decoder"])
    stage1.eval()
    decoder.eval()

    dataset = MultiChargeHDFDataset(str(args.hdf5), args.max_samples, prebatch=True)
    if not dataset:
        raise ValueError("input HDF5 contains no spectra")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=prebatched_collate, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = MetricAccumulator() if args.metrics_json else None

    with h5py.File(args.output, "w") as output, torch.inference_mode():
        chunk_rows = min(args.batch_size, len(dataset))
        identifiers = output.create_dataset(
            "prsm_id", (len(dataset),), dtype="i8", chunks=(chunk_rows,), compression="lzf"
        )
        intensities = output.create_dataset(
            "predicted_intensity", (len(dataset), 199, 2, 30), dtype="f4",
            chunks=(chunk_rows, 199, 2, 30), compression="lzf",
        )
        output.attrs["layout"] = "[spectrum, cleavage, ion_type, fragment_charge]"
        output.attrs["checkpoint"] = str(args.checkpoint.resolve())
        offset = 0
        for batch in loader:
            count = int(batch["prsm_id"].shape[0])
            identifiers[offset:offset + count] = batch["prsm_id"].numpy()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                distribution = stage1(
                    batch["left_ids"], batch["right_ids"], batch["cleavage_mask"],
                    batch["frag_id"], batch["instrument_id"], batch["e_bucket"],
                    batch["precursor_charge"], batch["n_term_form_id"],
                )
                prediction = decoder(distribution, make_theoretical_mask(batch))[
                    "predicted_intensity"
                ]
            intensities[offset:offset + count] = prediction.float().cpu().numpy()
            if metrics is not None:
                metric_output = dict(distribution)
                metric_output["expected_intensity"] = prediction
                metrics.update(metric_output, batch)
            offset += count

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "input_hdf5": str(args.hdf5.resolve()),
        "output_hdf5": str(args.output.resolve()),
        "spectra": offset,
        "metrics": metrics.compute() if metrics is not None else None,
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
