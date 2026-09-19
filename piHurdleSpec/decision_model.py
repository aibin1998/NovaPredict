"""Direct point-spectrum decoder for hurdle-MVN distribution parameters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecisionConfig:
    num_ion_types: int = 2
    max_fragment_charge: int = 30
    covariance_rank: int = 4
    hidden_size: int = 192
    residual_blocks: int = 3
    dropout: float = 0.1
    epsilon: float = 1e-8

    @property
    def outputs_per_cleavage(self) -> int:
        return self.num_ion_types * self.max_fragment_charge

    @property
    def flattened_feature_size(self) -> int:
        # pi, mu, diagonal scale, and the rank columns of U, where
        # Sigma = diag(scale^2) + U U^T.
        return self.outputs_per_cleavage * (3 + self.covariance_rank)


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        residual = self.net(self.norm(hidden).transpose(1, 2)).transpose(1, 2)
        return (hidden + residual) * position_mask


class DistributionToSpectrumDecoder(nn.Module):
    """Directly map pi, mu, and covariance to a nonnegative point spectrum."""

    feature_names = (
        "fragmentation_probability",
        "conditional_mean",
        "conditional_covariance",
    )
    covariance_parameterization = "diag(residual_scale^2) + low_rank_factor @ low_rank_factor.T"

    def __init__(self, cfg: DecisionConfig):
        super().__init__()
        if cfg.covariance_rank < 1:
            raise ValueError("covariance_rank must be positive")
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.flattened_feature_size)
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.flattened_feature_size, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.global_context = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(cfg.hidden_size, cfg.dropout)
                for _ in range(cfg.residual_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.outputs_per_cleavage),
        )

    def distribution_features(
        self,
        distribution: dict[str, torch.Tensor],
        theoretical_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probability = distribution["occurrence_probability"].float()
        mean = distribution["conditional_mean"].float()
        residual_scale = distribution["residual_scale"].float()
        low_rank_factor = distribution["low_rank_factor"].float()
        if probability.shape != mean.shape or probability.shape != residual_scale.shape:
            raise ValueError("pi, mu, and residual_scale must have identical shapes")
        if low_rank_factor.shape[:-1] != probability.shape:
            raise ValueError("low_rank_factor must match pi except for its rank axis")
        if low_rank_factor.shape[-1] != self.cfg.covariance_rank:
            raise ValueError(
                "low_rank_factor rank does not match DecisionConfig.covariance_rank"
            )

        mask = theoretical_mask.float()
        probability = probability * mask
        mean = mean * mask
        residual_scale = residual_scale * mask
        low_rank_factor = low_rank_factor * mask.unsqueeze(-1)
        batch_size, steps = probability.shape[:2]
        features = torch.cat(
            [
                probability.reshape(batch_size, steps, -1),
                mean.reshape(batch_size, steps, -1),
                residual_scale.reshape(batch_size, steps, -1),
                low_rank_factor.reshape(batch_size, steps, -1),
            ],
            dim=-1,
        )
        expected = probability * mean
        return features, expected

    @staticmethod
    def _max_normalize(
        intensity: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        maximum = intensity.flatten(1).amax(dim=1).clamp_min(epsilon)
        return intensity / maximum[:, None, None, None]

    def forward(
        self,
        distribution: dict[str, torch.Tensor],
        theoretical_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flattened, expected = self.distribution_features(
            distribution, theoretical_mask
        )
        position_mask = theoretical_mask.any(dim=(-1, -2), keepdim=False)
        position_mask = position_mask.unsqueeze(-1).to(flattened.dtype)

        hidden = self.input_projection(self.input_norm(flattened)) * position_mask
        pooled = hidden.sum(dim=1) / position_mask.sum(dim=1).clamp_min(1.0)
        hidden = (hidden + self.global_context(pooled).unsqueeze(1)) * position_mask
        for block in self.blocks:
            hidden = block(hidden, position_mask)

        intensity_logits = self.output_head(hidden).view_as(expected)
        raw_intensity = F.softplus(intensity_logits) * theoretical_mask.float()
        predicted = self._max_normalize(raw_intensity, self.cfg.epsilon)
        base = self._max_normalize(expected, self.cfg.epsilon)
        return {
            "predicted_intensity": predicted,
            "base_intensity": base,
            "intensity_logits": intensity_logits,
        }
