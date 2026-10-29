#!/usr/bin/env python
"""Validate schema and invariants of a charge-resolved TDRepo HDF5 file."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def validate(path: Path, chunk_size: int = 512) -> dict:
    required = {
        "prsm_id", "left_ids", "right_ids", "cleavage_mask",
        "charge_valid_mask", "y_intensity", "ion_mask", "isotope_shift",
        "precursor_charge", "spectrum_intensity_max_raw",
        "spectrum_intensity_sum_raw", "annotated_ion_count",
    }
    totals = Counter()
    with h5py.File(path, "r") as handle:
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise KeyError(f"missing datasets: {missing}")
        count = len(handle["prsm_id"])
        if handle["y_intensity"].shape != (count, 199, 2, 30):
            raise ValueError(f"unexpected target shape: {handle['y_intensity'].shape}")
        for name in required:
            if handle[name].shape[0] != count:
                raise ValueError(f"row mismatch for {name}: {handle[name].shape[0]} != {count}")

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            intensity = handle["y_intensity"][start:stop]
            ion_mask = handle["ion_mask"][start:stop].astype(bool)
            cleavage = handle["cleavage_mask"][start:stop].astype(bool)
            charge_valid = handle["charge_valid_mask"][start:stop].astype(bool)
            precursor = handle["precursor_charge"][start:stop]
            shifts = handle["isotope_shift"][start:stop]
            observed_count = ion_mask.sum(axis=(1, 2, 3))

            if not np.array_equal(observed_count, handle["annotated_ion_count"][start:stop]):
                raise ValueError(f"annotated count mismatch in rows {start}:{stop}")
            if not np.array_equal(intensity > 0, ion_mask):
                raise ValueError(f"intensity/mask mismatch in rows {start}:{stop}")
            if not np.allclose(intensity.max(axis=(1, 2, 3)), 1.0, atol=1e-6):
                raise ValueError(f"max normalization failure in rows {start}:{stop}")
            theoretical = cleavage[:, :, None, None] & charge_valid[:, None, :, :]
            if np.any(ion_mask & ~theoretical):
                raise ValueError(f"label outside theoretical mask in rows {start}:{stop}")
            expected_charge_mask = (
                np.arange(1, 31)[None, None, :] <= precursor[:, None, None]
            )
            expected_charge_mask = np.repeat(expected_charge_mask, 2, axis=1)
            if not np.array_equal(charge_valid, expected_charge_mask):
                raise ValueError(f"charge mask mismatch in rows {start}:{stop}")
            if np.any(~np.isin(shifts[ion_mask], [-1, 0, 1])):
                raise ValueError(f"invalid isotope shift in rows {start}:{stop}")

            totals["spectra"] += stop - start
            totals["observed_ions"] += int(ion_mask.sum())
            for shift in (-1, 0, 1):
                totals[f"isotope_shift_{shift:+d}"] += int((shifts[ion_mask] == shift).sum())

        result = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "shape": list(handle["y_intensity"].shape),
            "dtype": str(handle["y_intensity"].dtype),
            "statistics": dict(totals),
            "source_build_statistics": json.loads(
                handle.attrs.get("build_statistics_json", "{}")
            ),
            "status": "PASS",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    results = [validate(path, args.chunk_size) for path in args.paths]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

