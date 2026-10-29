"""Small, portable subset of the original TDRepo CNN–BiLSTM helpers.

The encodings and loss definitions match the modules in
``NovaPredict_Human_Tissue`` that the original pretraining code imported.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_VOCAB)}
PAD_ID = 0
FRAG_TO_ID = {
    "CID": 1, "HCD": 2, "ETD": 3, "ECD": 4, "EThcD": 5,
    "UVPD": 6, "UNKNOWN": 0,
}
INSTRUMENT_TO_ID = {
    "UNKNOWN": 0, "QE": 1, "LUMOS": 2, "VELOS": 3,
    "ELITE": 4, "FUSION": 5, "ORBITRAP": 6,
}
N_TERM_FORM_TO_ID = {
    "UNKNOWN": 0, "NONE": 1, "NME": 2,
    "M_ACETYLATION": 3, "NME_ACETYLATION": 4,
}
ION_TO_ID = {"B": 0, "Y": 1}
NUM_ION_TYPES = 2


@dataclass
class TrainConfig:
    max_len: int = 200
    aa_embed_dim: int = 64
    cond_embed_dim: int = 16
    max_precursor_charge: int = 30
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.2
    head_hidden: int = 128
    charge_num_classes: int = 21
    group_label_strategy: str = "max_intensity_row"
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1337
    device: str = "cpu"
    w_intensity: float = 1.0
    w_charge: float = 0.2
    w_break: float = 0.5
    break_prob_threshold: float = 0.5
    intensity_log1p: bool = False
    intensity_per_prsm_norm: bool = True
    eval_break_vec_pcc: bool = True
    w_log_mvn: float = 0.1
    log_mvn_target_eps: float = 1e-4
    log_mvn_jitter: float = 1e-5


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def spectral_angle_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is not None:
        mask = mask.to(dtype=prediction.dtype)
        prediction = prediction * mask
        target = target * mask
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    prediction = prediction / (prediction.norm(dim=1, keepdim=True) + eps)
    target = target / (target.norm(dim=1, keepdim=True) + eps)
    cosine = F.cosine_similarity(prediction, target, dim=1)
    cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
    return (2.0 * torch.arccos(cosine) / torch.pi).mean()


def masked_ce(logits, target, mask, ignore_index: int = 0):
    classes = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, classes), target.reshape(-1),
        reduction="none", ignore_index=ignore_index,
    )
    flat_mask = mask.reshape(-1)
    return (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)


def masked_bce_with_logits(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def normalize_proteoform_sequence(value: str) -> str:
    sequence = str(value).strip()
    parts = sequence.split(".")
    if len(parts) >= 3:
        sequence = ".".join(parts[1:-1])
    sequence = re.sub(r"\[[^\]]*\]", "", sequence)
    return "".join(char for char in sequence.upper() if char.isalpha())


def encode_cleavage_sequence(sequence: str, max_len: int = 200):
    import numpy as np

    sequence = normalize_proteoform_sequence(sequence)
    left = np.zeros(max_len - 1, dtype=np.int16)
    right = np.zeros(max_len - 1, dtype=np.int16)
    mask = np.zeros(max_len - 1, dtype=np.float32)
    for index in range(min(len(sequence), max_len) - 1):
        left[index] = AA_TO_ID.get(sequence[index], PAD_ID)
        right[index] = AA_TO_ID.get(sequence[index + 1], PAD_ID)
        mask[index] = 1.0
    return left, right, mask, len(sequence)


def encode_fragmentation(value: str | None) -> int:
    return FRAG_TO_ID.get(str(value).strip(), FRAG_TO_ID["UNKNOWN"])


def encode_instrument(value: str | None) -> int:
    if value is None:
        return INSTRUMENT_TO_ID["UNKNOWN"]
    instrument = str(value).strip().upper()
    if "LUMOS" in instrument:
        return INSTRUMENT_TO_ID["LUMOS"]
    if "ECLIPSE" in instrument or "FUSION" in instrument:
        return INSTRUMENT_TO_ID["FUSION"]
    for text in ("VELOS", "ELITE", "ORBITRAP"):
        if text in instrument:
            return INSTRUMENT_TO_ID[text]
    if "Q EXACTIVE" in instrument or instrument == "QE":
        return INSTRUMENT_TO_ID["QE"]
    return INSTRUMENT_TO_ID.get(instrument, INSTRUMENT_TO_ID["UNKNOWN"])


def encode_energy_to_bucket(value: str | float | None) -> int:
    try:
        energy = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(energy):
        return 0
    return int(min(9, max(0, math.floor(energy / 10.0)))) + 1


def encode_precursor_charge(value: str | float | None, max_charge: int = 30) -> int:
    try:
        charge = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(charge):
        return 0
    return max(0, min(max_charge, int(round(charge))))


def encode_n_term_form(value: str | None) -> int:
    if value is None:
        return 0
    normalized = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    return N_TERM_FORM_TO_ID.get(normalized, 0)
