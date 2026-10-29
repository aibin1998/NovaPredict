"""Build the legacy charge-aggregated B/Y HDF5 used by LogMVN pretraining.

The target at each (cleavage, B/Y) slot uses the highest-intensity matched
charge-state row and its charge label. Intensities are then divided by the
maximum target in each spectrum. Train/validation input msalign files must
already be separated by proteoform; this converter never makes a split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.logmvn_support.compat import (  # noqa: E402
    encode_cleavage_sequence,
    encode_energy_to_bucket,
    encode_fragmentation,
    encode_instrument,
    encode_n_term_form,
    encode_precursor_charge,
    normalize_proteoform_sequence,
)


FIELDS = {
    "prsm_id": ((), np.int64),
    "left_ids": ((199,), np.int16),
    "right_ids": ((199,), np.int16),
    "cleavage_mask": ((199,), np.float32),
    "frag_id": ((), np.int16),
    "instrument_id": ((), np.int16),
    "e_bucket": ((), np.int16),
    "precursor_charge": ((), np.int16),
    "n_term_form_id": ((), np.int16),
    "y_intensity": ((199, 2), np.float32),
    "y_charge": ((199, 2), np.int16),
    "ion_mask": ((199, 2), np.float32),
}


def iter_blocks(path: Path, max_spectra: int | None = None):
    """Yield metadata and matched-peak rows from annotated msalign."""
    metadata: dict[str, str] = {}
    peaks: list[list[str]] = []
    inside = False
    seen = 0
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text == "BEGIN IONS":
                metadata, peaks, inside = {}, [], True
            elif text == "END IONS":
                if inside:
                    yield metadata, peaks
                    seen += 1
                    if max_spectra is not None and seen >= max_spectra:
                        return
                inside = False
            elif inside and "=" in text and text and (text[0].isalpha() or text[0] == "_"):
                key, _, value = text.partition("=")
                metadata[key] = value
            elif inside and text and (text[0].isdigit() or text[0] == "-"):
                peaks.append(text.split())


def convert_block(metadata: dict[str, str], peaks: list[list[str]]):
    """Match the original `max_intensity_row` target construction."""
    sequence = normalize_proteoform_sequence(
        metadata.get("PROTEOFORM") or metadata.get("DATABASE_SEQUENCE", "")
    )
    length = len(sequence)
    if not 3 <= length <= 200:
        return None
    try:
        precursor_charge = int(metadata.get("PRECURSOR_CHARGE", "0").split(":")[0])
    except ValueError:
        return None

    # The original pandas groupby selects the first row when tied for the
    # maximum intensity within one ion-type/cleavage group.
    selected: dict[tuple[int, int], tuple[float, int]] = {}
    observed: set[tuple[int, int]] = set()
    for columns in peaks:
        if len(columns) < 10:
            continue
        try:
            value = float(columns[1])
            charge = int(float(columns[2]))
            ion_type = columns[4].strip().upper()[:1]
            ion_number = int(float(columns[5]))
        except (ValueError, OverflowError):
            continue
        cleavage = ion_number if ion_type == "B" else length - ion_number if ion_type == "Y" else None
        if cleavage is None or not 1 <= cleavage < length:
            continue
        index = (cleavage - 1, 0 if ion_type == "B" else 1)
        observed.add(index)
        if np.isfinite(value) and (index not in selected or value > selected[index][0]):
            selected[index] = (value, charge)
    if not observed:
        return None

    intensity = np.zeros((199, 2), dtype=np.float32)
    charge_target = np.zeros((199, 2), dtype=np.int16)
    ion_mask = np.zeros((199, 2), dtype=np.float32)
    for index in observed:
        ion_mask[index] = 1.0
    for index, (value, charge) in selected.items():
        intensity[index] = value
        charge_target[index] = max(0, min(20, charge))
    maximum = float(intensity.max())
    if maximum > 0:
        intensity /= maximum

    left, right, cleavage_mask, _ = encode_cleavage_sequence(sequence)
    return {
        "left_ids": left,
        "right_ids": right,
        "cleavage_mask": cleavage_mask,
        "frag_id": encode_fragmentation(metadata.get("ACTIVATION", "HCD")),
        "instrument_id": encode_instrument(metadata.get("INSTRUMENT", "UNKNOWN")),
        "e_bucket": encode_energy_to_bucket(metadata.get("COLLISION_ENERGY")),
        "precursor_charge": encode_precursor_charge(precursor_charge),
        "n_term_form_id": encode_n_term_form("UNKNOWN"),
        "y_intensity": intensity,
        "y_charge": charge_target,
        "ion_mask": ion_mask,
    }


def append_rows(handle: h5py.File, rows: list[dict], first_id: int) -> None:
    count = len(rows)
    if count == 0:
        return
    for name, (shape, dtype) in FIELDS.items():
        dataset = handle[name]
        dataset.resize(first_id + count, axis=0)
        if name == "prsm_id":
            dataset[first_id:first_id + count] = np.arange(first_id, first_id + count)
        else:
            dataset[first_id:first_id + count] = np.asarray(
                [row[name] for row in rows], dtype=dtype
            ).reshape((count,) + shape)


def build(input_path: Path, output_path: Path, max_spectra: int | None, chunk_size: int) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = seen = 0
    rows: list[dict] = []
    with h5py.File(output_path, "w") as handle:
        handle.attrs["source_msalign"] = str(input_path.resolve())
        handle.attrs["normalization"] = "linear"
        handle.attrs["target_layout"] = "[spectrum, cleavage, B/Y]"
        handle.attrs["charge_aggregation"] = "max_intensity_row"
        handle.attrs["proteoform_split"] = "input files must already be split by proteoform"
        for name, (shape, dtype) in FIELDS.items():
            handle.create_dataset(
                name, shape=(0,) + shape, maxshape=(None,) + shape,
                chunks=(chunk_size,) + shape, dtype=dtype,
            )
        for metadata, peaks in iter_blocks(input_path, max_spectra):
            seen += 1
            row = convert_block(metadata, peaks)
            if row is None:
                continue
            rows.append(row)
            if len(rows) >= chunk_size:
                append_rows(handle, rows, accepted)
                accepted += len(rows)
                rows.clear()
        if rows:
            append_rows(handle, rows, accepted)
            accepted += len(rows)
    return accepted, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="pre-split annotated msalign")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.max_spectra is not None and args.max_spectra < 1:
        parser.error("--max-spectra must be positive")
    accepted, seen = build(args.input, args.output, args.max_spectra, args.chunk_size)
    print(f"Wrote {accepted:,} spectra to {args.output} ({seen - accepted:,} skipped)")


if __name__ == "__main__":
    main()
