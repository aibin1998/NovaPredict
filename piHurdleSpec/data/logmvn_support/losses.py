"""Masked low-rank multivariate log-normal likelihood."""
from __future__ import annotations

import math
from typing import Dict

import torch


def masked_low_rank_log_mvn_nll(
    location: torch.Tensor,
    scale: torch.Tensor,
    factor: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_eps: float = 1e-4,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Exact masked MVN NLL for log(target) with covariance D + U U^T.

    The output positions are flattened within each spectrum. Missing and zero
    positions are removed algebraically by zeroing their precision weights.
    The determinant lemma and Woodbury identity reduce the only matrix
    factorization from peak_count x peak_count to rank x rank.
    """

    batch = target.shape[0]
    rank = factor.shape[-1]
    location = location.float().reshape(batch, -1)
    scale = scale.float().reshape(batch, -1)
    factor = factor.float().reshape(batch, -1, rank)
    target = target.float().reshape(batch, -1).clamp_min(target_eps)
    mask = mask.float().reshape(batch, -1)

    residual = (torch.log(target) - location) * mask
    diagonal_variance = scale.square().clamp_min(jitter)
    inverse_diagonal = mask / diagonal_variance
    masked_factor = factor * mask.unsqueeze(-1)

    weighted_factor = masked_factor * inverse_diagonal.unsqueeze(-1)
    small_matrix = torch.eye(
        rank, device=target.device, dtype=target.dtype
    ).unsqueeze(0) + torch.einsum(
        "bnr,bns->brs", masked_factor, weighted_factor
    )
    eye = torch.eye(rank, device=target.device, dtype=target.dtype).unsqueeze(0)
    chol = torch.linalg.cholesky(small_matrix + jitter * eye)

    projected = torch.einsum(
        "bnr,bn->br", masked_factor, inverse_diagonal * residual
    )
    solved = torch.cholesky_solve(projected.unsqueeze(-1), chol).squeeze(-1)
    diagonal_quadratic = (residual.square() * inverse_diagonal).sum(dim=1)
    correction = (projected * solved).sum(dim=1)
    quadratic = (diagonal_quadratic - correction).clamp_min(0.0)

    log_determinant = (
        mask * torch.log(diagonal_variance)
    ).sum(dim=1) + 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)
    ).sum(dim=1)
    observed_count = mask.sum(dim=1)
    nll = 0.5 * (
        observed_count * math.log(2.0 * math.pi)
        + log_determinant
        + quadratic
    )
    return nll.sum() / observed_count.sum().clamp(min=1.0)


def probability_summary(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    theoretical_mask: torch.Tensor,
    target_eps: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    observed = observed_mask.bool()
    theoretical = theoretical_mask.bool()
    break_probability = torch.sigmoid(output["break_logits"])
    squared_break_error = (
        break_probability - observed_mask.to(break_probability.dtype)
    ).square()

    if observed.any():
        conditional_error = (
            output["conditional_mean"] - target
        ).abs()[observed].mean()
        mean_scale = output["log_scale"][observed].mean()
        factor_rms = output["low_rank_factor"][observed].square().mean().sqrt()
    else:
        zero = output["conditional_mean"].sum() * 0.0
        conditional_error = zero
        mean_scale = zero
        factor_rms = zero

    return {
        "break_brier": squared_break_error[theoretical].mean(),
        "conditional_mae": conditional_error,
        "mean_log_scale": mean_scale,
        "factor_rms": factor_rms,
        "log_mvn_nll": masked_low_rank_log_mvn_nll(
            output["log_location"],
            output["log_scale"],
            output["low_rank_factor"],
            target,
            observed_mask,
            target_eps=target_eps,
        ),
    }
