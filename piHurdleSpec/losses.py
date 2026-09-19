"""Losses for two-stage charge-resolved hurdle-MVN training."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from config import LossConfig


def make_theoretical_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask valid cleavage, ion-type, and charge combinations."""

    return (
        batch["cleavage_mask"][:, :, None, None]
        * batch["charge_valid_mask"][:, None, :, :]
    )


def masked_low_rank_mvn_nll(
    mean: torch.Tensor,
    residual_scale: torch.Tensor,
    factor: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Exact masked MVN NLL for Y with covariance D + U U^T.

    Only nonzero observed ions enter the conditional likelihood. The matrix
    determinant lemma and Woodbury identity reduce the factorization to an
    R x R matrix for each spectrum, where R is the covariance rank.
    """

    batch_size = target.shape[0]
    rank = factor.shape[-1]
    mean = mean.float().reshape(batch_size, -1)
    residual_scale = residual_scale.float().reshape(batch_size, -1)
    factor = factor.float().reshape(batch_size, -1, rank)
    target = target.float().reshape(batch_size, -1)
    mask = observed_mask.float().reshape(batch_size, -1)

    residual = (target - mean) * mask
    diagonal_variance = residual_scale.square().clamp_min(jitter)
    inverse_diagonal = mask / diagonal_variance
    masked_factor = factor * mask.unsqueeze(-1)
    weighted_factor = masked_factor * inverse_diagonal.unsqueeze(-1)

    eye = torch.eye(rank, device=target.device, dtype=target.dtype).unsqueeze(0)
    small_matrix = eye + torch.einsum(
        "bnr,bns->brs", masked_factor, weighted_factor
    )
    chol = torch.linalg.cholesky(small_matrix + jitter * eye)

    projected = torch.einsum(
        "bnr,bn->br", masked_factor, inverse_diagonal * residual
    )
    solved = torch.cholesky_solve(projected.unsqueeze(-1), chol).squeeze(-1)
    diagonal_quadratic = (residual.square() * inverse_diagonal).sum(dim=1)
    correction = (projected * solved).sum(dim=1)
    quadratic = (diagonal_quadratic - correction).clamp_min(0.0)

    log_determinant = (mask * torch.log(diagonal_variance)).sum(dim=1)
    log_determinant = log_determinant + 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)
    ).sum(dim=1)
    observed_count = mask.sum(dim=1)
    nll = 0.5 * (
        observed_count * math.log(2.0 * math.pi) + log_determinant + quadratic
    )
    return nll.sum() / observed_count.sum().clamp_min(1.0)


def spectral_angle_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    prediction = (prediction.float() * mask.float()).flatten(1)
    target = (target.float() * mask.float()).flatten(1)
    dot = (prediction * target).sum(dim=1)
    denominator = prediction.norm(dim=1) * target.norm(dim=1)
    cosine = (dot / denominator.clamp_min(eps)).clamp(-1.0 + eps, 1.0 - eps)
    return (2.0 * torch.arccos(cosine) / math.pi).mean()


def masked_occurrence_bce(
    logits: torch.Tensor,
    observed_mask: torch.Tensor,
    theoretical_mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), observed_mask.float(), reduction="none"
    )
    return (loss * theoretical_mask).sum() / theoretical_mask.sum().clamp_min(1.0)


def loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: LossConfig,
    phase: str = "stage2",
    stage2_distribution_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute the phase-specific objective and all monitoring components.

    Stage 1 fits the hurdle distribution with BCE and conditional MVN NLL.
    Stage 2 optimizes SA of E[Y|x] = pi * mu and can retain a weighted copy
    of the Stage-1 objective to preserve occurrence/intensity calibration.
    """

    if phase not in {"stage1", "stage2"}:
        raise ValueError(f"unknown training phase: {phase}")
    if stage2_distribution_weight < 0.0:
        raise ValueError("stage2_distribution_weight must be non-negative")

    theoretical = make_theoretical_mask(batch)
    observed = batch["ion_mask"] * theoretical
    mvn = masked_low_rank_mvn_nll(
        output["conditional_mean"],
        output["residual_scale"],
        output["low_rank_factor"],
        batch["y_intensity"],
        observed,
        jitter=cfg.jitter,
    )
    sa = spectral_angle_loss(
        output["expected_intensity"], batch["y_intensity"], theoretical
    )
    bce = masked_occurrence_bce(
        output["occurrence_logits"], observed, theoretical
    )
    distribution = cfg.w_mvn * mvn + cfg.w_bce * bce
    task = cfg.w_sa * sa
    if phase == "stage1":
        total = distribution
    elif stage2_distribution_weight == 0.0:
        # Keep covariance-head gradients as None in the strict task-only mode;
        # multiplying by zero would still let AdamW decay those parameters.
        total = task
    else:
        total = task + stage2_distribution_weight * distribution
    return {
        "total": total,
        "distribution": distribution,
        "task": task,
        "sa": sa,
        "mvn_nll": mvn,
        "bce": bce,
    }


def mvn_diagnostics(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize direct-MVN fit and its probability outside [0, 1]."""

    observed = observed_mask.bool()
    if not observed.any():
        zero = output["conditional_mean"].sum() * 0.0
        return {
            "conditional_mae": zero,
            "mean_residual_scale": zero,
            "factor_rms": zero,
            "outside_unit_probability": zero,
        }

    mean = output["conditional_mean"][observed]
    standard_deviation = output["marginal_variance"][observed].sqrt().clamp_min(1e-8)
    lower_tail = torch.special.ndtr(-mean / standard_deviation)
    upper_tail = 1.0 - torch.special.ndtr((1.0 - mean) / standard_deviation)
    return {
        "conditional_mae": (mean - target[observed]).abs().mean(),
        "mean_residual_scale": output["residual_scale"][observed].mean(),
        "factor_rms": output["low_rank_factor"][observed].square().mean().sqrt(),
        "outside_unit_probability": (lower_tail + upper_tail).mean(),
    }
