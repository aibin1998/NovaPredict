"""Configuration for the multi-charge hurdle-MVN model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    max_len: int = 200
    max_fragment_charge: int = 30
    max_precursor_charge: int = 30
    aa_vocab_size: int = 21
    aa_embed_dim: int = 64
    cond_embed_dim: int = 16
    lstm_hidden: int = 256
    lstm_layers: int = 2
    head_hidden: int = 128
    dropout: float = 0.2
    covariance_rank: int = 4
    mean_min: float = 1e-5
    mean_max: float = 1.0
    scale_min: float = 1e-3
    scale_max: float = 0.5
    factor_max: float = 0.1
    initial_mean: float = 0.1
    initial_scale: float = 0.1
    occurrence_prior: float = 0.02

    @property
    def num_cleavages(self) -> int:
        return self.max_len - 1

    @property
    def num_ion_types(self) -> int:
        return 2

    @property
    def outputs_per_cleavage(self) -> int:
        return self.num_ion_types * self.max_fragment_charge


@dataclass
class LossConfig:
    w_sa: float = 1.0
    w_mvn: float = 0.1
    w_bce: float = 0.5
    jitter: float = 1e-5
    occurrence_threshold: float = 0.5

