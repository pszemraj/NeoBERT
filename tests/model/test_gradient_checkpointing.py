#!/usr/bin/env python3
"""Gradient checkpointing unit tests."""

import unittest

import torch

from neobert.model.model import NeoBERTConfig, NeoBERTLMHead


class TestGradientCheckpointing(unittest.TestCase):
    """Verify gradient checkpointing toggles work on NeoBERT."""

    def test_forward_backward_with_checkpointing(self):
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0.0,
            vocab_size=128,
            max_length=64,
            flash_attention=False,
            rope=True,
        )

        model = NeoBERTLMHead(config)
        model.train()
        model.gradient_checkpointing_enable()

        self.assertTrue(getattr(model.model, "gradient_checkpointing", False))

        batch_size, seq_len = 2, 12
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)
        loss = outputs["logits"].mean()
        loss.backward()

        self.assertTrue(
            any(param.grad is not None for param in model.parameters()),
            "Expected gradients when checkpointing is enabled",
        )

    def test_checkpointing_preserves_rng_with_dropout(self):
        """Ensure checkpointed gradients match non-checkpointed when dropout is enabled."""
        torch.manual_seed(123)
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            dropout=0.1,
            vocab_size=128,
            max_length=32,
            flash_attention=False,
            rope=True,
        )

        model_ckpt = NeoBERTLMHead(config)
        model_ref = NeoBERTLMHead(config)
        model_ref.load_state_dict(model_ckpt.state_dict())

        model_ckpt.train()
        model_ref.train()
        model_ckpt.gradient_checkpointing_enable()

        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        torch.manual_seed(999)
        loss_ref = model_ref(input_ids)["logits"].mean()
        loss_ref.backward()
        ref_grads = [
            p.grad.clone() for p in model_ref.parameters() if p.grad is not None
        ]

        torch.manual_seed(999)
        loss_ckpt = model_ckpt(input_ids)["logits"].mean()
        loss_ckpt.backward()
        ckpt_grads = [
            p.grad.clone() for p in model_ckpt.parameters() if p.grad is not None
        ]

        self.assertEqual(len(ref_grads), len(ckpt_grads))
        for ref_grad, ckpt_grad in zip(ref_grads, ckpt_grads):
            self.assertTrue(torch.allclose(ref_grad, ckpt_grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
