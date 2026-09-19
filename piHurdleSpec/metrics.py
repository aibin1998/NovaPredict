"""Validation metrics for sparse charge-resolved spectra."""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from losses import make_theoretical_mask, mvn_diagnostics


def _masked_pcc(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    mask = mask.float().flatten(1)
    count = mask.sum(dim=1)
    pred_mean = (prediction * mask).sum(dim=1) / count.clamp_min(1.0)
    target_mean = (target * mask).sum(dim=1) / count.clamp_min(1.0)
    pred_centered = (prediction - pred_mean[:, None]) * mask
    target_centered = (target - target_mean[:, None]) * mask
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)
    ).sqrt()
    valid = (count >= 2) & (denominator > 1e-12)
    values = numerator / denominator.clamp_min(1e-12)
    return values, valid


def _masked_cosine(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = (prediction.float() * mask.float()).flatten(1)
    target = (target.float() * mask.float()).flatten(1)
    pred_norm = prediction.norm(dim=1)
    target_norm = target.norm(dim=1)
    valid = (pred_norm > 1e-12) & (target_norm > 1e-12)
    cosine = (prediction * target).sum(dim=1) / (pred_norm * target_norm).clamp_min(1e-12)
    return cosine.clamp(-1.0, 1.0), valid


def _spectral_angle_similarity(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, valid = _masked_cosine(prediction, target, mask)
    similarity = 1.0 - 2.0 * torch.arccos(cosine) / math.pi
    return similarity, valid


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.scalar_sums: dict[str, float] = defaultdict(float)
        self.scalar_counts: dict[str, int] = defaultdict(int)
        self.tp = torch.zeros(30, dtype=torch.long)
        self.fp = torch.zeros(30, dtype=torch.long)
        self.fn = torch.zeros(30, dtype=torch.long)
        self.tn = torch.zeros(30, dtype=torch.long)

    def _add_vector(self, name: str, values: torch.Tensor, valid: torch.Tensor) -> None:
        values = values.detach()[valid].float().cpu()
        self.scalar_sums[name] += float(values.sum())
        self.scalar_counts[name] += int(values.numel())

    @torch.no_grad()
    def update(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        theoretical = make_theoretical_mask(batch)
        observed = batch["ion_mask"] * theoretical
        expected = output["expected_intensity"]
        target = batch["y_intensity"]

        pcc, valid = _masked_pcc(expected, target, theoretical)
        self._add_vector("charge_resolved_pcc", pcc, valid)
        cosine, valid = _masked_cosine(expected, target, theoretical)
        self._add_vector("charge_resolved_cosine", cosine, valid)
        sa, valid = _spectral_angle_similarity(expected, target, theoretical)
        self._add_vector("charge_resolved_sa", sa, valid)

        expected_aggregated = (expected * theoretical).sum(dim=-1)
        target_aggregated = (target * theoretical).sum(dim=-1)
        aggregate_mask = batch["cleavage_mask"][:, :, None].expand_as(
            expected_aggregated
        )
        pcc, valid = _masked_pcc(
            expected_aggregated, target_aggregated, aggregate_mask
        )
        self._add_vector("charge_aggregated_pcc", pcc, valid)
        cosine, valid = _masked_cosine(
            expected_aggregated, target_aggregated, aggregate_mask
        )
        self._add_vector("charge_aggregated_cosine", cosine, valid)
        sa, valid = _spectral_angle_similarity(
            expected_aggregated, target_aggregated, aggregate_mask
        )
        self._add_vector("charge_aggregated_sa", sa, valid)

        pcc, valid = _masked_pcc(output["conditional_mean"], target, observed)
        self._add_vector("conditional_observed_pcc", pcc, valid)

        probability = output["occurrence_probability"]
        prediction = probability >= self.threshold
        truth = observed.bool()
        valid_theoretical = theoretical.bool()
        reduce_dims = (0, 1, 2)
        self.tp += (prediction & truth & valid_theoretical).sum(reduce_dims).cpu()
        self.fp += (prediction & ~truth & valid_theoretical).sum(reduce_dims).cpu()
        self.fn += (~prediction & truth & valid_theoretical).sum(reduce_dims).cpu()
        self.tn += (~prediction & ~truth & valid_theoretical).sum(reduce_dims).cpu()

        brier = (probability - truth.float()).square()
        self.scalar_sums["occurrence_brier"] += float(
            (brier * theoretical).sum().cpu()
        )
        self.scalar_counts["occurrence_brier"] += int(theoretical.sum().item())

        diagnostics = mvn_diagnostics(output, target, observed)
        batch_size = int(target.shape[0])
        for key, value in diagnostics.items():
            self.scalar_sums[key] += float(value.cpu()) * batch_size
            self.scalar_counts[key] += batch_size

    def compute(self) -> dict:
        result = {
            key: self.scalar_sums[key] / max(1, self.scalar_counts[key])
            for key in self.scalar_sums
        }
        tp, fp, fn, tn = (int(x.sum()) for x in (self.tp, self.fp, self.fn, self.tn))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        result["occurrence"] = {
            "threshold": self.threshold,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        per_charge = []
        for charge in range(30):
            charge_tp = int(self.tp[charge])
            charge_fp = int(self.fp[charge])
            charge_fn = int(self.fn[charge])
            charge_precision = charge_tp / max(1, charge_tp + charge_fp)
            charge_recall = charge_tp / max(1, charge_tp + charge_fn)
            per_charge.append(
                {
                    "charge": charge + 1,
                    "support": charge_tp + charge_fn,
                    "precision": charge_precision,
                    "recall": charge_recall,
                    "f1": 2.0
                    * charge_precision
                    * charge_recall
                    / max(1e-12, charge_precision + charge_recall),
                }
            )
        result["occurrence_per_charge"] = per_charge
        return result
