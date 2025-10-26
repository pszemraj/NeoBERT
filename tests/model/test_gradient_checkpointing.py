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
            dropout_prob=0.0,
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


if __name__ == "__main__":
    unittest.main()
