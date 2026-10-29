"""Baseline-only TD-repo HDF5 dataset for the CNN-BiLSTM experiment."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = (
    "prsm_id", "left_ids", "right_ids", "cleavage_mask", "frag_id", "instrument_id",
    "e_bucket", "precursor_charge", "n_term_form_id", "y_intensity", "y_charge", "ion_mask",
)


class BaselineRAMDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, "r") as handle:
            missing = [key for key in FIELDS if key not in handle]
            if missing:
                raise KeyError(f"{path} missing HDF5 fields: {missing}")
            self.d = {key: handle[key][:] for key in FIELDS}

    def __len__(self):
        return len(self.d["prsm_id"])

    def __getitem__(self, i):
        d = self.d
        return {
            "prsm_id": torch.tensor(int(d["prsm_id"][i]), dtype=torch.long),
            "left_ids": torch.from_numpy(d["left_ids"][i].astype(np.int64)),
            "right_ids": torch.from_numpy(d["right_ids"][i].astype(np.int64)),
            "cleavage_mask": torch.from_numpy(d["cleavage_mask"][i].astype(np.float32)),
            "frag_id": torch.tensor(int(d["frag_id"][i]), dtype=torch.long),
            "instrument_id": torch.tensor(int(d["instrument_id"][i]), dtype=torch.long),
            "e_bucket": torch.tensor(int(d["e_bucket"][i]), dtype=torch.long),
            "precursor_charge": torch.tensor(int(d["precursor_charge"][i]), dtype=torch.long),
            "n_term_form_id": torch.tensor(int(d["n_term_form_id"][i]), dtype=torch.long),
            "y_intensity": torch.from_numpy(d["y_intensity"][i].astype(np.float32)),
            "y_charge": torch.from_numpy(d["y_charge"][i].astype(np.int64)),
            "ion_mask": torch.from_numpy(d["ion_mask"][i].astype(np.float32)),
        }
