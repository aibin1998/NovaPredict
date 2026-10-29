"""Numerical checks for the low-rank masked MVN likelihood."""
from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LossConfig  # noqa: E402
from losses import loss_components, masked_low_rank_mvn_nll  # noqa: E402


class LowRankMVNTest(unittest.TestCase):
    def test_matches_explicit_masked_covariance(self):
        torch.manual_seed(7)
        batch, variables, rank = 3, 7, 2
        mean = torch.randn(batch, variables) * 0.1
        scale = 0.05 + torch.rand(batch, variables) * 0.2
        factor = torch.randn(batch, variables, rank) * 0.03
        target = torch.randn(batch, variables) * 0.1
        mask = torch.tensor(
            [
                [1, 1, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0],
                [1, 0, 1, 1, 1, 0, 1],
            ],
            dtype=torch.float32,
        )
        actual = masked_low_rank_mvn_nll(
            mean, scale, factor, target, mask, jitter=1e-8
        )

        explicit_nll = 0.0
        observed_total = 0
        for row in range(batch):
            keep = mask[row].bool()
            covariance = torch.diag(scale[row, keep].square())
            covariance = covariance + factor[row, keep] @ factor[row, keep].T
            residual = target[row, keep] - mean[row, keep]
            sign, logdet = torch.linalg.slogdet(covariance)
            self.assertEqual(float(sign), 1.0)
            explicit_nll += 0.5 * (
                keep.sum() * math.log(2.0 * math.pi)
                + logdet
                + residual @ torch.linalg.solve(covariance, residual)
            )
            observed_total += int(keep.sum())
        expected = explicit_nll / observed_total
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))


class TwoStageObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        batch_size, steps, ion_types, charges, rank = 2, 3, 2, 2, 2
        shape = (batch_size, steps, ion_types, charges)
        mean = 0.1 + 0.2 * torch.rand(shape)
        scale = 0.05 + 0.05 * torch.rand(shape)
        factor = 0.01 * torch.randn(*shape, rank)
        logits = torch.randn(shape)
        ion_mask = torch.tensor(
            [
                [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 0]]],
                [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 0]]],
            ],
            dtype=torch.float32,
        )
        self.output = {
            "conditional_mean": mean,
            "residual_scale": scale,
            "low_rank_factor": factor,
            "occurrence_logits": logits,
            "expected_intensity": torch.sigmoid(logits) * mean,
        }
        self.batch = {
            "cleavage_mask": torch.ones(batch_size, steps),
            "charge_valid_mask": torch.ones(batch_size, ion_types, charges),
            "ion_mask": ion_mask,
            "y_intensity": ion_mask * (0.05 + 0.4 * torch.rand(shape)),
        }
        self.cfg = LossConfig(w_sa=1.3, w_mvn=0.2, w_bce=0.7)

    def test_stage1_uses_distribution_terms_only(self):
        losses = loss_components(
            self.output, self.batch, self.cfg, phase="stage1"
        )
        expected = self.cfg.w_mvn * losses["mvn_nll"] + self.cfg.w_bce * losses["bce"]
        self.assertTrue(torch.allclose(losses["total"], expected))
        self.assertTrue(torch.allclose(losses["total"], losses["distribution"]))

    def test_stage2_adds_expected_spectrum_sa(self):
        weight = 0.25
        losses = loss_components(
            self.output,
            self.batch,
            self.cfg,
            phase="stage2",
            stage2_distribution_weight=weight,
        )
        expected = losses["task"] + weight * losses["distribution"]
        self.assertTrue(torch.allclose(losses["total"], expected))

    def test_zero_distribution_weight_is_strict_task_only(self):
        output = {
            key: value.detach().requires_grad_(True)
            for key, value in self.output.items()
            if key != "expected_intensity"
        }
        output["expected_intensity"] = (
            torch.sigmoid(output["occurrence_logits"])
            * output["conditional_mean"]
        )
        losses = loss_components(
            output,
            self.batch,
            self.cfg,
            phase="stage2",
            stage2_distribution_weight=0.0,
        )
        self.assertTrue(torch.allclose(losses["total"], losses["task"]))
        losses["total"].backward()
        self.assertIsNotNone(output["conditional_mean"].grad)
        self.assertIsNotNone(output["occurrence_logits"].grad)
        self.assertIsNone(output["residual_scale"].grad)
        self.assertIsNone(output["low_rank_factor"].grad)


if __name__ == "__main__":
    unittest.main()
