"""Checks for the direct pi-mean-covariance decision decoder."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LossConfig  # noqa: E402
from decision_model import DecisionConfig, DistributionToSpectrumDecoder  # noqa: E402
from losses import loss_components  # noqa: E402


class DecisionDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.shape = (2, 5, 2, 3)
        probability = torch.sigmoid(torch.randn(self.shape)).requires_grad_()
        mean = (0.05 + 0.8 * torch.rand(self.shape)).requires_grad_()
        residual_scale = (0.01 + 0.1 * torch.rand(self.shape)).requires_grad_()
        low_rank_factor = (0.05 * torch.randn(*self.shape, 2)).requires_grad_()
        occurrence_logits = torch.randn(self.shape, requires_grad=True)
        self.distribution = {
            "occurrence_probability": probability,
            "conditional_mean": mean,
            "residual_scale": residual_scale,
            "low_rank_factor": low_rank_factor,
            "marginal_variance": (
                residual_scale.square() + low_rank_factor.square().sum(-1)
            ),
            "occurrence_logits": occurrence_logits,
        }
        self.mask = torch.ones(self.shape)
        self.mask[:, -1] = 0.0
        self.cfg = DecisionConfig(
            max_fragment_charge=3,
            covariance_rank=2,
            hidden_size=16,
            residual_blocks=1,
            dropout=0.0,
        )

    def test_direct_output_is_nonnegative_masked_and_normalized(self):
        decoder = DistributionToSpectrumDecoder(self.cfg).eval()
        predicted = decoder(self.distribution, self.mask)["predicted_intensity"]
        self.assertTrue(torch.all(predicted >= 0.0))
        self.assertEqual(float(predicted[:, -1].detach().abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(predicted.flatten(1).amax(1), torch.ones(self.shape[0]))
        )

    def test_output_is_direct_not_a_pi_mu_correction(self):
        decoder = DistributionToSpectrumDecoder(self.cfg).eval()
        torch.nn.init.zeros_(decoder.output_head[-1].weight)
        torch.nn.init.zeros_(decoder.output_head[-1].bias)
        output = decoder(self.distribution, self.mask)
        valid = self.mask.bool()
        self.assertTrue(
            torch.allclose(
                output["predicted_intensity"][valid],
                torch.ones_like(output["predicted_intensity"][valid]),
            )
        )
        self.assertFalse(
            torch.allclose(output["predicted_intensity"], output["base_intensity"])
        )

    def test_sa_gradient_reaches_all_distribution_parameters(self):
        decoder = DistributionToSpectrumDecoder(self.cfg)
        decoder(self.distribution, self.mask)["predicted_intensity"].sum().backward()
        for name in (
            "occurrence_probability",
            "conditional_mean",
            "residual_scale",
            "low_rank_factor",
        ):
            self.assertIsNotNone(self.distribution[name].grad, name)

    def test_inputs_are_only_pi_mean_and_covariance_parameters(self):
        decoder = DistributionToSpectrumDecoder(self.cfg)
        features, expected = decoder.distribution_features(
            self.distribution, self.mask
        )
        outputs = self.cfg.num_ion_types * self.cfg.max_fragment_charge
        self.assertEqual(
            features.shape,
            (self.shape[0], self.shape[1], outputs * (3 + self.cfg.covariance_rank)),
        )
        self.assertEqual(expected.shape, self.shape)
        self.assertEqual(
            decoder.feature_names,
            (
                "fragmentation_probability",
                "conditional_mean",
                "conditional_covariance",
            ),
        )

    def test_low_weight_distribution_loss_formula_is_finite(self):
        decoder = DistributionToSpectrumDecoder(self.cfg)
        output = decoder(self.distribution, self.mask)
        target = torch.rand(self.shape) * self.mask
        model_output = dict(self.distribution)
        model_output["expected_intensity"] = output["predicted_intensity"]
        batch = {
            "y_intensity": target,
            "ion_mask": (target > 0.0).float(),
            "cleavage_mask": self.mask.any(dim=(-1, -2)).float(),
            "charge_valid_mask": torch.ones(self.shape[0], *self.shape[2:]),
        }
        losses = loss_components(
            model_output,
            batch,
            LossConfig(w_sa=1.0, w_mvn=0.1, w_bce=0.5),
            phase="stage2",
            stage2_distribution_weight=0.01,
        )
        expected_total = losses["sa"] + 0.01 * losses["distribution"]
        self.assertTrue(torch.allclose(losses["total"], expected_total))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))


if __name__ == "__main__":
    unittest.main()
