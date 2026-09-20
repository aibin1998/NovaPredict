"""Lazy HDF5 input pipeline for charge-resolved TDRepo spectra."""
from __future__ import annotations

import math
import os
from collections.abc import Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate


REQUIRED_FIELDS = (
    "prsm_id",
    "left_ids",
    "right_ids",
    "cleavage_mask",
    "frag_id",
    "instrument_id",
    "e_bucket",
    "precursor_charge",
    "n_term_form_id",
    "charge_valid_mask",
    "y_intensity",
    "ion_mask",
)


class MultiChargeHDFDataset(Dataset):
    """Open one HDF5 handle per DataLoader worker and read samples on demand."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        cache_bytes: int = 256 * 1024 * 1024,
        prebatch: bool = False,
    ) -> None:
        self.path = os.path.abspath(path)
        self.cache_bytes = int(cache_bytes)
        self.prebatch = bool(prebatch)
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            missing = [key for key in REQUIRED_FIELDS if key not in handle]
            if missing:
                raise KeyError(f"{self.path} missing HDF5 fields: {missing}")
            total = int(handle["prsm_id"].shape[0])
            target_shape = tuple(handle["y_intensity"].shape[1:])
            if target_shape != (199, 2, 30):
                raise ValueError(
                    f"expected y_intensity shape [N,199,2,30], got {handle['y_intensity'].shape}"
                )
        self.length = total if max_samples is None else min(total, int(max_samples))

    def __len__(self) -> int:
        return self.length

    def _hdf(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(
                self.path,
                "r",
                rdcc_nbytes=self.cache_bytes,
                rdcc_nslots=1_000_003,
            )
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        hdf = self._hdf()
        data = {key: hdf[key][index] for key in REQUIRED_FIELDS}
        return self._tensor_row(data)

    @staticmethod
    def _tensor_row(data: dict) -> dict[str, torch.Tensor]:
        return {
            "prsm_id": torch.tensor(int(data["prsm_id"]), dtype=torch.long),
            "left_ids": torch.as_tensor(data["left_ids"], dtype=torch.long),
            "right_ids": torch.as_tensor(data["right_ids"], dtype=torch.long),
            "cleavage_mask": torch.as_tensor(
                data["cleavage_mask"], dtype=torch.float32
            ),
            "frag_id": torch.tensor(int(data["frag_id"]), dtype=torch.long),
            "instrument_id": torch.tensor(
                int(data["instrument_id"]), dtype=torch.long
            ),
            "e_bucket": torch.tensor(int(data["e_bucket"]), dtype=torch.long),
            "precursor_charge": torch.tensor(
                int(data["precursor_charge"]), dtype=torch.long
            ),
            "n_term_form_id": torch.tensor(
                int(data["n_term_form_id"]), dtype=torch.long
            ),
            "charge_valid_mask": torch.as_tensor(
                data["charge_valid_mask"], dtype=torch.float32
            ),
            "y_intensity": torch.as_tensor(
                data["y_intensity"], dtype=torch.float32
            ),
            "ion_mask": torch.as_tensor(data["ion_mask"], dtype=torch.float32),
        }

    @staticmethod
    def _tensor_batch(data: dict) -> dict[str, torch.Tensor]:
        return {
            "prsm_id": torch.as_tensor(data["prsm_id"], dtype=torch.long),
            "left_ids": torch.as_tensor(data["left_ids"], dtype=torch.long),
            "right_ids": torch.as_tensor(data["right_ids"], dtype=torch.long),
            "cleavage_mask": torch.as_tensor(data["cleavage_mask"], dtype=torch.float32),
            "frag_id": torch.as_tensor(data["frag_id"], dtype=torch.long),
            "instrument_id": torch.as_tensor(data["instrument_id"], dtype=torch.long),
            "e_bucket": torch.as_tensor(data["e_bucket"], dtype=torch.long),
            "precursor_charge": torch.as_tensor(
                data["precursor_charge"], dtype=torch.long
            ),
            "n_term_form_id": torch.as_tensor(
                data["n_term_form_id"], dtype=torch.long
            ),
            "charge_valid_mask": torch.as_tensor(
                data["charge_valid_mask"], dtype=torch.float32
            ),
            "y_intensity": torch.as_tensor(data["y_intensity"], dtype=torch.float32),
            "ion_mask": torch.as_tensor(data["ion_mask"], dtype=torch.float32),
        }

    def __getitems__(
        self, indices: list[int]
    ) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        """Read dense shuffled batches as one contiguous HDF5 slice.

        With block_size=batch_size, the sampler supplies every row in one HDF
        interval but in random order. A single slice avoids thousands of HDF5
        calls and repeated chunk decompression. Sparse index lists retain the
        row-wise path to avoid reading a much larger interval.
        """

        if not indices:
            return []
        index_array = np.asarray(indices, dtype=np.int64)
        start = int(index_array.min())
        stop = int(index_array.max()) + 1
        if stop - start > int(1.25 * len(index_array)):
            return [self[int(index)] for index in index_array]

        hdf = self._hdf()
        block = {key: hdf[key][start:stop] for key in REQUIRED_FIELDS}
        if self.prebatch:
            if stop - start != len(index_array):
                positions = index_array - start
                block = {key: values[positions] for key, values in block.items()}
            return self._tensor_batch(block)
        return [
            self._tensor_row(
                {key: values[int(index) - start] for key, values in block.items()}
            )
            for index in index_array
        ]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        self.close()


def prebatched_collate(data):
    """Pass an HDF batch through, with a fallback for sparse row-wise reads."""

    return data if isinstance(data, dict) else default_collate(data)


class BlockShuffleBatchSampler(Sampler[list[int]]):
    """Shuffle HDF-friendly blocks, then shuffle samples inside each block."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        block_size: int = 1024,
        seed: int = 1337,
        drop_last: bool = False,
    ) -> None:
        if dataset_size < 0 or batch_size < 1 or block_size < batch_size:
            raise ValueError("require dataset_size >= 0 and block_size >= batch_size >= 1")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        starts = list(range(0, self.dataset_size, self.block_size))
        if starts:
            order = torch.randperm(len(starts), generator=generator).tolist()
            starts = [starts[i] for i in order]
        for start in starts:
            stop = min(start + self.block_size, self.dataset_size)
            local = torch.randperm(stop - start, generator=generator).tolist()
            indices = [start + offset for offset in local]
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)


def validate_target_values(path: str, sample_count: int = 4096, seed: int = 1337) -> dict:
    """Quickly verify the normalization and binary masks used by the model."""

    with h5py.File(path, "r") as handle:
        size = int(handle["y_intensity"].shape[0])
        count = min(size, sample_count)
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, max(1, size - count + 1)))
        intensity = handle["y_intensity"][start : start + count]
        ion_mask = handle["ion_mask"][start : start + count]
    positive = intensity[ion_mask.astype(bool)]
    return {
        "sampled_spectra": int(count),
        "positive_values": int(positive.size),
        "minimum": float(positive.min()) if positive.size else None,
        "maximum": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
        "mask_value_mismatches": int(np.count_nonzero((intensity > 0) != (ion_mask > 0))),
    }
