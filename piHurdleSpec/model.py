"""Residual CNN-BiLSTM with a charge-resolved hurdle low-rank MVN head."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from config import ModelConfig


PAD_ID = 0
FRAGMENTATION_VOCAB_SIZE = 7
INSTRUMENT_VOCAB_SIZE = 7
ENERGY_VOCAB_SIZE = 11
N_TERM_FORM_VOCAB_SIZE = 5


class MultiChargeHurdleMVN(nn.Module):
    """Predict occurrence and a joint Gaussian law for nonzero intensities."""

    backbone_name = "residual_cnn_bilstm_rf7_multicharge_hurdle_mvn_two_stage"
    effective_receptive_field = 7

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.covariance_rank < 1:
            raise ValueError("covariance_rank must be positive")
        if not cfg.mean_min < cfg.initial_mean < cfg.mean_max:
            raise ValueError("initial_mean must lie strictly inside [mean_min, mean_max]")
        if not cfg.scale_min < cfg.initial_scale < cfg.scale_max:
            raise ValueError("initial_scale must lie strictly inside [scale_min, scale_max]")
        if not 0.0 < cfg.occurrence_prior < 1.0:
            raise ValueError("occurrence_prior must lie in (0, 1)")
        self.cfg = cfg

        d = cfg.cond_embed_dim
        self.aa_emb = nn.Embedding(cfg.aa_vocab_size, cfg.aa_embed_dim, padding_idx=PAD_ID)
        self.frag_emb = nn.Embedding(FRAGMENTATION_VOCAB_SIZE, d)
        self.instrument_emb = nn.Embedding(INSTRUMENT_VOCAB_SIZE, d)
        self.energy_emb = nn.Embedding(ENERGY_VOCAB_SIZE, d)
        self.precursor_charge_emb = nn.Embedding(cfg.max_precursor_charge + 1, d)
        self.n_term_form_emb = nn.Embedding(N_TERM_FORM_VOCAB_SIZE, d)

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

        encoded_dim = 2 * cfg.lstm_hidden
        self.intensity_shared = nn.Sequential(
            nn.Linear(encoded_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        outputs = cfg.outputs_per_cleavage
        self.intensity_mean_head = nn.Linear(cfg.head_hidden, outputs)
        self.intensity_scale_head = nn.Linear(cfg.head_hidden, outputs)
        self.intensity_factor_head = nn.Linear(
            cfg.head_hidden, outputs * cfg.covariance_rank
        )
        self.occurrence_head = nn.Sequential(
            nn.Linear(encoded_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, outputs),
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        cfg = self.cfg
        nn.init.zeros_(self.intensity_mean_head.weight)
        mean_fraction = (cfg.initial_mean - cfg.mean_min) / (cfg.mean_max - cfg.mean_min)
        nn.init.constant_(
            self.intensity_mean_head.bias,
            math.log(mean_fraction / (1.0 - mean_fraction)),
        )

        nn.init.zeros_(self.intensity_scale_head.weight)
        scale_fraction = (cfg.initial_scale - cfg.scale_min) / (
            cfg.scale_max - cfg.scale_min
        )
        nn.init.constant_(
            self.intensity_scale_head.bias,
            math.log(scale_fraction / (1.0 - scale_fraction)),
        )

        # U cannot start exactly at zero because d(UU^T)/dU is then zero.
        nn.init.normal_(self.intensity_factor_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.intensity_factor_head.bias)
        nn.init.constant_(
            self.occurrence_head[-1].bias,
            math.log(cfg.occurrence_prior / (1.0 - cfg.occurrence_prior)),
        )

    def forward(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
        cleavage_mask: torch.Tensor,
        frag_id: torch.Tensor,
        instrument_id: torch.Tensor,
        e_bucket: torch.Tensor,
        precursor_charge: torch.Tensor,
        n_term_form_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, steps = left_ids.shape

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1).expand(batch, steps, -1)

        features = torch.cat(
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
        position_mask = cleavage_mask.unsqueeze(-1).to(features.dtype)
        cnn_input = self.cnn_input_norm(features) * position_mask
        local = self.local_cnn(cnn_input.transpose(1, 2)).transpose(1, 2)
        features = (features + local * position_mask) * position_mask

        lengths = cleavage_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        packed, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=steps
        )

        cfg = self.cfg
        intensity_features = self.intensity_shared(hidden)
        mean_fraction = torch.sigmoid(self.intensity_mean_head(intensity_features))
        conditional_mean = cfg.mean_min + (cfg.mean_max - cfg.mean_min) * mean_fraction

        scale_fraction = torch.sigmoid(self.intensity_scale_head(intensity_features))
        residual_scale = cfg.scale_min + (
            cfg.scale_max - cfg.scale_min
        ) * scale_fraction
        low_rank_factor = cfg.factor_max * torch.tanh(
            self.intensity_factor_head(intensity_features).view(
                batch, steps, cfg.outputs_per_cleavage, cfg.covariance_rank
            )
        )
        occurrence_logits = self.occurrence_head(hidden)

        target_shape = (batch, steps, cfg.num_ion_types, cfg.max_fragment_charge)
        conditional_mean = conditional_mean.view(target_shape)
        residual_scale = residual_scale.view(target_shape)
        low_rank_factor = low_rank_factor.view(
            *target_shape, cfg.covariance_rank
        )
        occurrence_logits = occurrence_logits.view(target_shape)
        occurrence_probability = torch.sigmoid(occurrence_logits)
        marginal_variance = residual_scale.square() + low_rank_factor.square().sum(-1)

        return {
            "conditional_mean": conditional_mean,
            "residual_scale": residual_scale,
            "low_rank_factor": low_rank_factor,
            "marginal_variance": marginal_variance,
            "occurrence_logits": occurrence_logits,
            "occurrence_probability": occurrence_probability,
            "expected_intensity": occurrence_probability * conditional_mean,
        }


def load_backbone_checkpoint(model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Transfer compatible encoder parameters from an existing NovaPredict model."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in source.items():
        mapped = key
        if key.startswith("intensity_head.0."):
            mapped = key.replace("intensity_head.0.", "intensity_shared.0.")
        elif key.startswith("break_head.0."):
            mapped = key.replace("break_head.0.", "occurrence_head.0.")
        elif key.startswith("intensity_head.") or key.startswith("break_head."):
            skipped.append(key)
            continue
        if mapped in target and target[mapped].shape == value.shape:
            transferred[mapped] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(transferred, strict=False)
    return {
        "source_epoch": checkpoint.get("epoch"),
        "source_val_loss": checkpoint.get("val_loss"),
        "transferred_count": len(transferred),
        "skipped_count": len(skipped),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
