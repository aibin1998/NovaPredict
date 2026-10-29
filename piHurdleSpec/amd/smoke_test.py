"""Run one Stage-1 and one Stage-2 CPU step against the real HDF5 file."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from config import LossConfig, ModelConfig
from dataset import MultiChargeHDFDataset, validate_target_values
from losses import loss_components, make_theoretical_mask
from metrics import MetricAccumulator
from model import MultiChargeHurdleMVN
from train import DEFAULT_DATA, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5", default=f"{DEFAULT_DATA}/train_multi_charge.hdf5"
    )
    args = parser.parse_args()

    dataset = MultiChargeHDFDataset(args.hdf5, max_samples=4, cache_bytes=32 * 1024 * 1024)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    cfg = ModelConfig(
        aa_embed_dim=16,
        cond_embed_dim=4,
        lstm_hidden=16,
        lstm_layers=1,
        head_hidden=16,
        dropout=0.0,
        covariance_rank=2,
    )
    model = MultiChargeHurdleMVN(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    phase_losses = {}
    for phase in ("stage1", "stage2"):
        output = predict(model, batch)
        losses = loss_components(
            output,
            batch,
            LossConfig(),
            phase=phase,
            stage2_distribution_weight=0.1,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        phase_losses[phase] = {
            key: float(value.detach()) for key, value in losses.items()
        }

    expected_shape = (2, 199, 2, 30)
    assert tuple(output["conditional_mean"].shape) == expected_shape
    assert tuple(output["occurrence_logits"].shape) == expected_shape
    assert tuple(output["low_rank_factor"].shape) == expected_shape + (2,)
    assert all(torch.isfinite(value) for value in losses.values())
    assert torch.all((output["conditional_mean"] > 0) & (output["conditional_mean"] <= 1))
    assert torch.all(batch["ion_mask"] <= make_theoretical_mask(batch))

    metrics = MetricAccumulator()
    metrics.update(output, batch)
    result = {
        "status": "ok",
        "hdf5": args.hdf5,
        "target_diagnostics": validate_target_values(args.hdf5, sample_count=64),
        "output_shapes": {key: list(value.shape) for key, value in output.items()},
        "phase_losses": phase_losses,
        "metrics": metrics.compute(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
