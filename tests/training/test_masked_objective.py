#!/usr/bin/env python3
"""Tests for masked-only MLM objective."""

import unittest

import torch
import torch.nn.functional as F

from neobert.pretraining.masked_objective import (
    MaskedPositionsOnlyMLMObjective,
    gather_masked_index_select,
)


class TestMaskedObjective(unittest.TestCase):
    """Unit tests for masked-only objective paths."""

    def test_gather_masked_index_select(self):
        """Ensure masked token gather returns expected rows and targets."""
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        labels = torch.tensor([[0, -100], [2, -100]])

        masked_hidden, masked_targets, num_masked = gather_masked_index_select(
            hidden,
            labels,
            ignore_index=-100,
        )

        self.assertEqual(num_masked, 2)
        self.assertTrue(torch.equal(masked_targets, torch.tensor([0, 2])))
        self.assertTrue(
            torch.equal(masked_hidden, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
        )

    def test_train_checkpointed_fallback_matches_reference_ce(self):
        """Ensure masked-only train fallback matches PyTorch CE sum."""
        torch.manual_seed(0)
        hidden = torch.randn(2, 4, 6, requires_grad=True)
        lm_weight = torch.randn(13, 6, requires_grad=True)
        labels = torch.tensor(
            [[1, -100, 3, -100], [2, 4, -100, 0]],
            dtype=torch.long,
        )

        objective = MaskedPositionsOnlyMLMObjective()
        out = objective(hidden, labels, lm_weight, compute_accuracy=True)

        ref_logits = F.linear(hidden, lm_weight)  # full-logits reference
        ref_loss = F.cross_entropy(
            ref_logits.reshape(-1, lm_weight.size(0)).float(),
            labels.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )

        self.assertEqual(out.used_path, "train_checkpointed_masked_ce")
        self.assertEqual(
            out.num_masked_local.item(), int((labels != -100).sum().item())
        )
        self.assertTrue(
            torch.allclose(out.loss_sum_local, ref_loss, atol=1e-6, rtol=1e-6)
        )
        self.assertIsNotNone(out.num_correct_local)

        out.loss_sum_local.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(lm_weight.grad)

    def test_zero_masked_path_keeps_gradients_connected(self):
        """Ensure zero-masked path returns zero loss with valid autograd graph."""
        hidden = torch.randn(2, 3, 5, requires_grad=True)
        lm_weight = torch.randn(11, 5, requires_grad=True)
        labels = torch.full((2, 3), -100, dtype=torch.long)

        objective = MaskedPositionsOnlyMLMObjective()
        out = objective(hidden, labels, lm_weight)
        out.loss_sum_local.backward()

        self.assertEqual(out.used_path, "zero_masked")
        self.assertEqual(out.num_masked_local.item(), 0)
        self.assertEqual(out.loss_sum_local.item(), 0.0)
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(lm_weight.grad)

    def test_fp16_inputs_rejected(self):
        """Ensure fp16 tensors are rejected by the masked-only objective."""
        hidden = torch.randn(1, 2, 3, dtype=torch.float16)
        lm_weight = torch.randn(9, 3, dtype=torch.float32)
        labels = torch.tensor([[1, -100]], dtype=torch.long)

        objective = MaskedPositionsOnlyMLMObjective()
        with self.assertRaises(RuntimeError):
            objective(hidden, labels, lm_weight)

    def test_eval_streaming_close_to_masked_logits(self):
        """Ensure streaming eval CE stays close to masked-logits eval CE."""
        torch.manual_seed(7)
        hidden = torch.randn(2, 5, 6)
        lm_weight = torch.randn(17, 6)
        labels = torch.tensor(
            [[1, -100, 3, -100, 5], [2, 4, -100, 0, -100]],
            dtype=torch.long,
        )

        objective_masked_logits = MaskedPositionsOnlyMLMObjective(
            eval_loss_mode="masked_logits",
        )
        objective_streaming = MaskedPositionsOnlyMLMObjective(
            eval_loss_mode="streaming",
            token_chunk_eval=2,
            vocab_chunk_eval=4,
        )

        with torch.no_grad():
            ref_out = objective_masked_logits(
                hidden,
                labels,
                lm_weight,
                compute_accuracy=True,
            )
            streaming_out = objective_streaming(
                hidden,
                labels,
                lm_weight,
                compute_accuracy=True,
            )

        self.assertEqual(streaming_out.used_path, "eval_streaming_ce")
        self.assertTrue(
            torch.allclose(
                ref_out.loss_sum_local,
                streaming_out.loss_sum_local,
                atol=5e-4,
                rtol=5e-4,
            )
        )
        self.assertEqual(
            ref_out.num_correct_local.item(), streaming_out.num_correct_local.item()
        )

    def test_eval_streaming_rejects_out_of_vocab_targets(self):
        """Ensure streaming eval raises on labels outside vocabulary range."""
        hidden = torch.randn(1, 2, 4)
        lm_weight = torch.randn(5, 4)
        labels = torch.tensor([[1, 7]], dtype=torch.long)
        objective_streaming = MaskedPositionsOnlyMLMObjective(
            eval_loss_mode="streaming"
        )

        with torch.no_grad():
            with self.assertRaisesRegex(IndexError, "out of bounds"):
                objective_streaming(hidden, labels, lm_weight, compute_accuracy=False)


if __name__ == "__main__":
    unittest.main()
