"""Residual CNN-BiLSTM with a conditional low-rank log-normal head."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .compat import (
    FRAG_TO_ID,
    INSTRUMENT_TO_ID,
    N_TERM_FORM_TO_ID,
    NUM_ION_TYPES,
    PAD_ID,
)


class LogMVNCNNBiLSTMFragmentPredictor(nn.Module):
    """Predict a hurdle probability and a joint low-rank Gaussian law for log intensities."""

    backbone_name = "residual_cnn_bilstm_rf7_hurdle_logmvn"
    effective_receptive_field = 7

    def __init__(
        self,
        cfg,
        aa_vocab_size: int = 21,
        covariance_rank: int = 8,
        location_min: float = math.log(1e-4),
        location_max: float = 0.0,
        scale_min: float = 0.05,
        scale_max: float = 1.5,
        factor_max: float = 0.25,
        initial_mean: float = 0.16,
        initial_scale: float = 0.5,
    ):
        super().__init__()
        if covariance_rank < 1:
            raise ValueError("covariance_rank must be positive")
        if not location_min < math.log(initial_mean) < location_max:
            raise ValueError("initial_mean must lie inside the configured log-location range")
        if not 0.0 < scale_min < initial_scale < scale_max:
            raise ValueError("require 0 < scale_min < initial_scale < scale_max")

        self.cfg = cfg
        self.covariance_rank = int(covariance_rank)
        self.location_min = float(location_min)
        self.location_max = float(location_max)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.factor_max = float(factor_max)
        d = cfg.cond_embed_dim

        self.aa_emb = nn.Embedding(aa_vocab_size, cfg.aa_embed_dim, padding_idx=PAD_ID)
        self.frag_emb = nn.Embedding(max(FRAG_TO_ID.values()) + 1, d)
        self.instrument_emb = nn.Embedding(max(INSTRUMENT_TO_ID.values()) + 1, d)
        self.energy_emb = nn.Embedding(11, d)
        self.precursor_charge_emb = nn.Embedding(cfg.max_precursor_charge + 1, d)
        self.n_term_form_emb = nn.Embedding(max(N_TERM_FORM_TO_ID.values()) + 1, d)

        self.input_dim = 2 * cfg.aa_embed_dim + 5 * d
        self.cnn_input_norm = nn.LayerNorm(self.input_dim)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        enc_dim = 2 * cfg.lstm_hidden
        self.intensity_shared = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.intensity_location_head = nn.Linear(cfg.head_hidden, NUM_ION_TYPES)
        self.intensity_scale_head = nn.Linear(cfg.head_hidden, NUM_ION_TYPES)
        self.intensity_factor_head = nn.Linear(
            cfg.head_hidden, NUM_ION_TYPES * self.covariance_rank
        )
        self.charge_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, NUM_ION_TYPES * cfg.charge_num_classes),
        )
        self.break_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, NUM_ION_TYPES),
        )

        nn.init.zeros_(self.intensity_location_head.weight)
        initial_location_fraction = (
            (math.log(initial_mean) - self.location_min)
            / (self.location_max - self.location_min)
        )
        nn.init.constant_(
            self.intensity_location_head.bias,
            math.log(initial_location_fraction / (1.0 - initial_location_fraction)),
        )
        nn.init.zeros_(self.intensity_scale_head.weight)
        initial_scale_fraction = (
            (initial_scale - self.scale_min) / (self.scale_max - self.scale_min)
        )
        nn.init.constant_(
            self.intensity_scale_head.bias,
            math.log(initial_scale_fraction / (1.0 - initial_scale_fraction)),
        )
        # U must not start exactly at zero: d(UU^T)/dU is zero at U=0.
        nn.init.normal_(self.intensity_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.intensity_factor_head.bias)

    def forward(
        self,
        left_ids,
        right_ids,
        cleavage_mask,
        frag_id,
        instrument_id,
        e_bucket,
        precursor_charge,
        n_term_form_id,
    ) -> Dict[str, torch.Tensor]:
        batch, steps = left_ids.shape
        expand = lambda value: value.unsqueeze(1).expand(batch, steps, -1)
        x = torch.cat(
            [
                self.aa_emb(left_ids),
                self.aa_emb(right_ids),
                expand(self.frag_emb(frag_id)),
                expand(self.instrument_emb(instrument_id)),
                expand(self.energy_emb(e_bucket)),
                expand(self.precursor_charge_emb(precursor_charge)),
                expand(self.n_term_form_emb(n_term_form_id)),
            ],
            dim=-1,
        )

        position_mask = cleavage_mask.unsqueeze(-1).to(dtype=x.dtype)
        cnn_input = self.cnn_input_norm(x) * position_mask
        local = self.local_cnn(cnn_input.transpose(1, 2)).transpose(1, 2) * position_mask
        x = (x + local) * position_mask

        lengths = cleavage_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=steps
        )

        intensity_features = self.intensity_shared(hidden)
        location_fraction = torch.sigmoid(
            self.intensity_location_head(intensity_features)
        )
        log_location = self.location_min + (
            self.location_max - self.location_min
        ) * location_fraction
        scale_fraction = torch.sigmoid(self.intensity_scale_head(intensity_features))
        log_scale = self.scale_min + (
            self.scale_max - self.scale_min
        ) * scale_fraction
        low_rank_factor = self.factor_max * torch.tanh(
            self.intensity_factor_head(intensity_features).view(
                batch, steps, NUM_ION_TYPES, self.covariance_rank
            )
        )

        marginal_log_variance = log_scale.square() + low_rank_factor.square().sum(dim=-1)
        conditional_median = torch.exp(log_location)
        conditional_mean = torch.exp(log_location + 0.5 * marginal_log_variance)
        break_logits = self.break_head(hidden)
        expected_intensity = torch.sigmoid(break_logits) * conditional_mean
        charge_logits = self.charge_head(hidden).view(
            batch, steps, NUM_ION_TYPES, self.cfg.charge_num_classes
        )

        return {
            "log_location": log_location,
            "log_scale": log_scale,
            "low_rank_factor": low_rank_factor,
            "marginal_log_variance": marginal_log_variance,
            "conditional_median": conditional_median,
            "conditional_mean": conditional_mean,
            "expected_intensity": expected_intensity,
            "charge_logits": charge_logits,
            "break_logits": break_logits,
        }


def load_deterministic_checkpoint(model: nn.Module, checkpoint_path: str) -> Dict[str, object]:
    """Warm-start the shared encoder and auxiliary heads from the deterministic model."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    transferred = {}
    skipped = []

    for key, value in source.items():
        mapped_key = key
        if key.startswith("intensity_head.0."):
            mapped_key = key.replace("intensity_head.0.", "intensity_shared.0.")
        elif key.startswith("intensity_head."):
            skipped.append(key)
            continue
        if mapped_key in target and target[mapped_key].shape == value.shape:
            transferred[mapped_key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(transferred, strict=False)
    return {
        "source_epoch": checkpoint.get("epoch"),
        "source_val_loss": checkpoint.get("val_loss"),
        "transferred_count": len(transferred),
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
