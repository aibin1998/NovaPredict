#!/usr/bin/env python
"""Build charge-resolved NovaPredict/TD-Pred targets from annotated msalign.

The target layout is [cleavage, ion_type, charge] = [199, 2, 30].  Ion type
0 is N-terminal (b/c) and ion type 1 is C-terminal (y/z).  C-terminal ion y_i
is placed at cleavage L-i so complementary b/y ions share a cleavage index.

TopRepo isotope-offset labels (for example b24+1 and b31-1) are mapped to the
same canonical (ion_type, cleavage, charge) variable.  The offset is retained
in a separate isotope_shift tensor for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


MAX_LEN = 200
MAX_CLEAVAGES = MAX_LEN - 1
MAX_FRAGMENT_CHARGE = 30
NUM_ION_TYPES = 2
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_VOCAB)}

FRAG_TO_ID = {
    "UNKNOWN": 0, "CID": 1, "HCD": 2, "ETD": 3, "ECD": 4,
    "ETHCD": 5, "UVPD": 6,
}
INSTRUMENT_TO_ID = {
    "UNKNOWN": 0, "QE": 1, "LUMOS": 2, "VELOS": 3, "ELITE": 4,
    "FUSION": 5, "ORBITRAP": 6,
}

TENSOR_FIELDS = {
    "left_ids": ((MAX_CLEAVAGES,), np.int16),
    "right_ids": ((MAX_CLEAVAGES,), np.int16),
    "cleavage_mask": ((MAX_CLEAVAGES,), np.uint8),
    "charge_valid_mask": ((NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), np.uint8),
    "y_intensity": (
        (MAX_CLEAVAGES, NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), np.float32
    ),
    "ion_mask": (
        (MAX_CLEAVAGES, NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), np.uint8
    ),
    "isotope_shift": (
        (MAX_CLEAVAGES, NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), np.int8
    ),
}
SCALAR_FIELDS = {
    "prsm_id": np.int64,
    "source_spectrum_index": np.int64,
    "spectrum_key_hash": np.uint64,
    "ms2_id": np.int64,
    "ms2_scan": np.int64,
    "frag_id": np.int16,
    "instrument_id": np.int16,
    "e_bucket": np.int16,
    "precursor_charge": np.int16,
    "n_term_form_id": np.int16,
    "spectrum_intensity_max_raw": np.float32,
    "spectrum_intensity_sum_raw": np.float64,
    "annotated_ion_count": np.int16,
}


def normalize_sequence(value: str) -> str:
    sequence = str(value or "").strip()
    parts = sequence.split(".")
    if len(parts) >= 3:
        sequence = ".".join(parts[1:-1])
    sequence = re.sub(r"\[[^\]]*\]", "", sequence)
    return "".join(char for char in sequence.upper() if char.isalpha())


def parse_primary_int(value: str | None, default: int = 0) -> int:
    try:
        return int(round(float(str(value).split(":")[0])))
    except (TypeError, ValueError):
        return default


def parse_primary_float(value: str | None, default: float = math.nan) -> float:
    try:
        result = float(str(value).split(":")[0])
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def encode_fragmentation(value: str | None) -> int:
    return FRAG_TO_ID.get(str(value or "UNKNOWN").strip().upper(), 0)


def encode_instrument(value: str | None) -> int:
    instrument = str(value or "UNKNOWN").strip().upper()
    if "LUMOS" in instrument:
        return INSTRUMENT_TO_ID["LUMOS"]
    if "ECLIPSE" in instrument or "FUSION" in instrument:
        return INSTRUMENT_TO_ID["FUSION"]
    if "VELOS" in instrument:
        return INSTRUMENT_TO_ID["VELOS"]
    if "ELITE" in instrument:
        return INSTRUMENT_TO_ID["ELITE"]
    if "Q EXACTIVE" in instrument or instrument == "QE":
        return INSTRUMENT_TO_ID["QE"]
    if "ORBITRAP" in instrument:
        return INSTRUMENT_TO_ID["ORBITRAP"]
    return INSTRUMENT_TO_ID.get(instrument, 0)


def encode_energy_bucket(value: str | None) -> int:
    energy = parse_primary_float(value)
    if not np.isfinite(energy):
        return 0
    return int(min(9, max(0, math.floor(energy / 10.0)))) + 1


def encode_sequence(sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    left = np.zeros(MAX_CLEAVAGES, dtype=np.int16)
    right = np.zeros(MAX_CLEAVAGES, dtype=np.int16)
    mask = np.zeros(MAX_CLEAVAGES, dtype=np.uint8)
    count = len(sequence) - 1
    for index in range(count):
        left[index] = AA_TO_ID.get(sequence[index], 0)
        right[index] = AA_TO_ID.get(sequence[index + 1], 0)
    mask[:count] = 1
    unknown = sum(char not in AA_TO_ID for char in sequence)
    return left, right, mask, unknown


def iter_msalign(path: Path, max_spectra: int | None = None):
    metadata: dict[str, str] = {}
    peaks: list[list[str]] = []
    inside = False
    source_index = 0
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text == "BEGIN IONS":
                metadata, peaks, inside = {}, [], True
            elif text == "END IONS":
                if inside:
                    yield source_index, metadata, peaks
                    source_index += 1
                    if max_spectra is not None and source_index >= max_spectra:
                        return
                inside = False
            elif inside and "=" in text and (text[0].isalpha() or text[0] == "_"):
                key, _, value = text.partition("=")
                metadata[key] = value
            elif inside and text and (text[0].isdigit() or text[0] == "-"):
                peaks.append(text.split())


def spectrum_hash(metadata: dict[str, str]) -> np.uint64:
    key = "|".join(
        [
            metadata.get("DATASET_ID", ""),
            metadata.get("MSALIGN_FILE_NAME", ""),
            metadata.get("MS2_ID", ""),
            metadata.get("MS2_SCAN", ""),
        ]
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(digest, dtype="<u8")[0]


def process_spectrum(
    source_index: int,
    metadata: dict[str, str],
    peaks: list[list[str]],
    duplicate_policy: str,
    evalue_cutoff: float,
    coverage_cutoff: float,
) -> tuple[dict | None, Counter]:
    stats = Counter(source_spectra=1)
    sequence = normalize_sequence(
        metadata.get("DATABASE_SEQUENCE") or metadata.get("PROTEOFORM", "")
    )
    length = len(sequence)
    if length < 2 or length > MAX_LEN:
        stats["skipped_sequence_length"] += 1
        return None, stats

    evalue = parse_primary_float(metadata.get("E_VALUE"), default=math.inf)
    if evalue > evalue_cutoff:
        stats["skipped_evalue"] += 1
        return None, stats

    precursor_charge = parse_primary_int(metadata.get("PRECURSOR_CHARGE"))
    if precursor_charge < 1 or precursor_charge > MAX_FRAGMENT_CHARGE:
        stats["skipped_precursor_charge"] += 1
        return None, stats

    intensity = np.zeros(
        (MAX_CLEAVAGES, NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), dtype=np.float32
    )
    ion_mask = np.zeros_like(intensity, dtype=np.uint8)
    isotope_shift = np.zeros_like(intensity, dtype=np.int8)

    for columns in peaks:
        if len(columns) != 10:
            stats["unannotated_peak_rows"] += 1
            continue
        annotation = columns[4].strip()
        if not annotation:
            stats["unannotated_peak_rows"] += 1
            continue
        series = annotation[0].upper()
        if series in {"B", "C"}:
            ion_type = 0
        elif series in {"Y", "Z"}:
            ion_type = 1
        else:
            stats["dropped_unsupported_ion_type"] += 1
            continue
        try:
            value = float(columns[1])
            charge = int(float(columns[2]))
            ion_number = int(float(columns[5]))
            shift = int(float(columns[7]))
        except ValueError:
            stats["dropped_parse_error"] += 1
            continue
        if not np.isfinite(value) or value < 0:
            stats["dropped_invalid_intensity"] += 1
            continue
        if charge < 1 or charge > MAX_FRAGMENT_CHARGE:
            stats["dropped_fragment_charge_out_of_range"] += 1
            continue
        if charge > precursor_charge:
            stats["dropped_fragment_charge_above_precursor"] += 1
            continue
        if ion_number < 1 or ion_number >= length:
            stats["dropped_ion_position"] += 1
            continue

        cleavage = ion_number - 1 if ion_type == 0 else length - ion_number - 1
        index = (cleavage, ion_type, charge - 1)
        if ion_mask[index]:
            stats["duplicate_target_rows"] += 1
            if duplicate_policy == "error":
                raise ValueError(
                    f"duplicate target at source spectrum {source_index}: "
                    f"series={series} ion={ion_number} charge={charge}"
                )
            if duplicate_policy == "max" and value <= intensity[index]:
                continue
            if duplicate_policy == "sum":
                intensity[index] += value
                continue
        intensity[index] = value
        ion_mask[index] = 1
        isotope_shift[index] = shift
        stats[f"isotope_shift_{shift:+d}"] += 1

    annotated_count = int(ion_mask.sum())
    coverage = annotated_count / float(2 * (length - 1))
    if coverage < coverage_cutoff:
        stats["skipped_coverage"] += 1
        return None, stats

    raw_max = float(intensity.max())
    raw_sum = float(intensity.sum(dtype=np.float64))
    if raw_max <= 0:
        stats["skipped_no_annotated_intensity"] += 1
        return None, stats
    intensity /= raw_max

    left, right, cleavage_mask, unknown = encode_sequence(sequence)
    stats["unknown_amino_acid_count"] += unknown
    charge_valid_mask = np.zeros(
        (NUM_ION_TYPES, MAX_FRAGMENT_CHARGE), dtype=np.uint8
    )
    charge_valid_mask[:, :precursor_charge] = 1

    record = {
        "left_ids": left,
        "right_ids": right,
        "cleavage_mask": cleavage_mask,
        "charge_valid_mask": charge_valid_mask,
        "y_intensity": intensity,
        "ion_mask": ion_mask,
        "isotope_shift": isotope_shift,
        "source_spectrum_index": source_index,
        "spectrum_key_hash": spectrum_hash(metadata),
        "ms2_id": parse_primary_int(metadata.get("MS2_ID"), default=-1),
        "ms2_scan": parse_primary_int(metadata.get("MS2_SCAN"), default=-1),
        "frag_id": encode_fragmentation(metadata.get("ACTIVATION")),
        "instrument_id": encode_instrument(metadata.get("INSTRUMENT")),
        "e_bucket": encode_energy_bucket(metadata.get("COLLISION_ENERGY")),
        "precursor_charge": precursor_charge,
        "n_term_form_id": 0,
        "spectrum_intensity_max_raw": raw_max,
        "spectrum_intensity_sum_raw": raw_sum,
        "annotated_ion_count": annotated_count,
    }
    stats["accepted_spectra"] += 1
    stats["accepted_ion_rows"] += annotated_count
    return record, stats


def create_datasets(handle: h5py.File, compression: str, batch_size: int):
    datasets = {}
    for name, (tail, dtype) in TENSOR_FIELDS.items():
        first_chunk = min(batch_size, 32) if name in {
            "y_intensity", "ion_mask", "isotope_shift"
        } else min(batch_size, 1024)
        datasets[name] = handle.create_dataset(
            name,
            shape=(0, *tail),
            maxshape=(None, *tail),
            chunks=(first_chunk, *tail),
            dtype=dtype,
            compression=compression,
            shuffle=True,
        )
    for name, dtype in SCALAR_FIELDS.items():
        datasets[name] = handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1024, batch_size),),
            dtype=dtype,
            compression=compression,
            shuffle=True,
        )
    return datasets


def append_records(datasets: dict, records: list[dict], start: int) -> int:
    stop = start + len(records)
    for dataset in datasets.values():
        dataset.resize(stop, axis=0)
    for name in TENSOR_FIELDS:
        datasets[name][start:stop] = np.stack([record[name] for record in records])
    for name, dtype in SCALAR_FIELDS.items():
        if name == "prsm_id":
            values = np.arange(start, stop, dtype=dtype)
        else:
            values = np.asarray([record[name] for record in records], dtype=dtype)
        datasets[name][start:stop] = values
    return stop


def build(args) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    stats = Counter()
    buffer: list[dict] = []
    written = 0
    with h5py.File(args.output, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": "novapredict_multi_charge_v1",
                "source_msalign": str(args.input.resolve()),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "max_sequence_length": MAX_LEN,
                "max_fragment_charge": MAX_FRAGMENT_CHARGE,
                "ion_type_mapping": "0=N-terminal b/c;1=C-terminal y/z",
                "target_layout": "[spectrum, cleavage, ion_type, charge_1_to_30]",
                "flatten_order": "cleavage-major; within cleavage B charges 1..30 then Y charges 1..30",
                "c_terminal_alignment": "y_i/z_i -> cleavage_index=L-i-1",
                "intensity_normalization": "per-spectrum max over all charge-resolved ions",
                "raw_intensity_recovery": "y_intensity * spectrum_intensity_max_raw",
                "isotope_policy": "-1/0/+1 offsets merged into canonical target; offset retained in isotope_shift",
                "duplicate_policy": args.duplicate_policy,
                "evalue_cutoff": args.evalue_cutoff,
                "coverage_cutoff": args.coverage_cutoff,
                "aa_vocab": "PAD=0;" + ",".join(
                    f"{aa}={index}" for aa, index in AA_TO_ID.items()
                ),
            }
        )
        datasets = create_datasets(handle, args.compression, args.batch_size)
        for source_index, metadata, peaks in iter_msalign(args.input, args.max_spectra):
            record, row_stats = process_spectrum(
                source_index,
                metadata,
                peaks,
                args.duplicate_policy,
                args.evalue_cutoff,
                args.coverage_cutoff,
            )
            stats.update(row_stats)
            if record is not None:
                buffer.append(record)
            if len(buffer) >= args.batch_size:
                written = append_records(datasets, buffer, written)
                buffer.clear()
            if stats["source_spectra"] % args.progress_interval == 0:
                print(
                    f"source={stats['source_spectra']:,} accepted={written + len(buffer):,} "
                    f"ions={stats['accepted_ion_rows']:,}",
                    flush=True,
                )
        if buffer:
            written = append_records(datasets, buffer, written)
        handle.attrs["build_statistics_json"] = json.dumps(dict(stats), sort_keys=True)
        handle.flush()

    result = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "statistics": dict(stats),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    parser.add_argument("--duplicate-policy", choices=["error", "max", "sum"], default="error")
    parser.add_argument("--evalue-cutoff", type=float, default=1e-4)
    parser.add_argument("--coverage-cutoff", type=float, default=0.2)
    parser.add_argument("--progress-interval", type=int, default=20_000)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()

