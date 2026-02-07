#!/usr/bin/env python3
"""Regression tests for gradient accumulation token weighting."""

import copy
import unittest

import torch
import torch.nn.functional as F

from neobert.pretraining.trainer import _gradient_token_scale, _scale_gradients


class TestGradientAccumulationTokenWeighting(unittest.TestCase):
    """Ensure token-weighted GA matches full-batch gradients."""

    def test_token_weighted_ga_matches_full_batch(self):
        """Verify GA scaling matches full-batch normalization."""
        torch.manual_seed(0)
        vocab_size = 11
        hidden_size = 5
        ga_steps = 2

        model_full = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        model_accum = copy.deepcopy(model_full)

        inputs_1 = torch.randn(6, hidden_size)
        labels_1 = torch.tensor([1, 2, -100, 3, -100, 4])
        inputs_2 = torch.randn(4, hidden_size)
        labels_2 = torch.tensor([5, -100, 6, 7])

        inputs_full = torch.cat([inputs_1, inputs_2], dim=0)
        labels_full = torch.cat([labels_1, labels_2], dim=0)
        total_tokens = (labels_full != -100).sum()

        logits_full = model_full(inputs_full)
        loss_full = F.cross_entropy(
            logits_full,
            labels_full,
            ignore_index=-100,
            reduction="sum",
        )
        (loss_full / total_tokens).backward()
        full_grads = [param.grad.detach().clone() for param in model_full.parameters()]

        for inputs, labels in ((inputs_1, labels_1), (inputs_2, labels_2)):
            logits = model_accum(inputs)
            loss_sum = F.cross_entropy(
                logits,
                labels,
                ignore_index=-100,
                reduction="sum",
            )
            (loss_sum / ga_steps).backward()

        scale = ga_steps / total_tokens.float()
        for param in model_accum.parameters():
            param.grad.mul_(scale)

        for expected, actual in zip(full_grads, model_accum.parameters()):
            self.assertTrue(torch.allclose(expected, actual.grad, atol=1e-6))

    def test_scale_gradients_casts_to_grad_dtype(self):
        """Ensure gradient scaling casts the scalar to the grad dtype."""
        model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
        for param in model.parameters():
            param.grad = torch.ones_like(param, dtype=torch.bfloat16)

        scale = torch.tensor(0.5, dtype=torch.float32)
        _scale_gradients(model, scale)

        for param in model.parameters():
            self.assertEqual(param.grad.dtype, torch.bfloat16)
            self.assertTrue(
                torch.allclose(param.grad, torch.full_like(param.grad, 0.5))
            )

    def test_gradient_token_scale_clamps_low_token_updates(self):
        """Ensure tiny masked-token counts cannot amplify gradients."""
        scale, clamped = _gradient_token_scale(
            torch.tensor(1, dtype=torch.long),
            num_processes=8,
            grad_accumulation_steps=4,
        )

        self.assertIsNotNone(scale)
        self.assertTrue(clamped)
        assert scale is not None
        self.assertAlmostEqual(float(scale.item()), 1.0, places=6)

    def test_gradient_token_scale_matches_standard_formula_above_floor(self):
        """Ensure scaling is standard token-mean normalization on normal updates."""
        scale, clamped = _gradient_token_scale(
            torch.tensor(64, dtype=torch.long),
            num_processes=8,
            grad_accumulation_steps=2,
        )

        self.assertFalse(clamped)
        assert scale is not None
        # standard scale: (num_processes * grad_accumulation_steps) / tokens_global
        self.assertAlmostEqual(float(scale.item()), 16.0 / 64.0, places=6)

    def test_gradient_token_scale_zero_tokens_clamps_to_safe_scale(self):
        """Ensure empty masked batches use a safe clamped scale."""
        scale, clamped = _gradient_token_scale(
            torch.tensor(0, dtype=torch.long),
            num_processes=2,
            grad_accumulation_steps=2,
        )

        self.assertIsNotNone(scale)
        self.assertTrue(clamped)
        assert scale is not None
        self.assertAlmostEqual(float(scale.item()), 1.0, places=6)
