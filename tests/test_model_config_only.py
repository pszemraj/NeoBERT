#!/usr/bin/env python3
"""Test model config creation without importing the full model."""

import unittest
from pathlib import Path

from neobert.config import ConfigLoader


class TestModelConfigOnly(unittest.TestCase):
    """Test model configuration without importing full model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_dir = Path(__file__).parent / "configs"

    def test_model_config_validation(self):
        """Test model config validation without importing model classes."""
        # Load config
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        config = ConfigLoader.load(str(config_path))

        # Test NeoBERTConfig-like validation manually
        hidden_size = config.model.hidden_size
        num_attention_heads = config.model.num_attention_heads

        # Test divisibility constraint (critical for transformer models)
        self.assertEqual(
            hidden_size % num_attention_heads,
            0,
            "Hidden size must be divisible by number of attention heads",
        )

        # Test reasonable ranges
        self.assertGreater(hidden_size, 0)
        self.assertGreater(num_attention_heads, 0)
        self.assertGreater(config.model.num_hidden_layers, 0)
        self.assertGreater(config.model.intermediate_size, 0)
        self.assertGreater(config.model.vocab_size, 0)

        # Test dropout range
        self.assertGreaterEqual(config.model.dropout_prob, 0.0)
        self.assertLess(config.model.dropout_prob, 1.0)

        # Test that we can compute head dimension
        head_dim = hidden_size // num_attention_heads
        self.assertGreater(head_dim, 0)
        self.assertEqual(head_dim * num_attention_heads, hidden_size)

    def test_all_model_configs_valid(self):
        """Test that all model configs meet basic requirements."""
        test_configs = [
            "pretraining/test_tiny_pretrain.yaml",
            "evaluation/test_tiny_glue.yaml",
            "contrastive/test_tiny_contrastive.yaml",
        ]

        for config_name in test_configs:
            with self.subTest(config=config_name):
                config_path = self.config_dir / config_name
                config = ConfigLoader.load(str(config_path))

                # Test transformer architecture constraints
                self.assertEqual(
                    config.model.hidden_size % config.model.num_attention_heads,
                    0,
                    f"Invalid attention head configuration in {config_name}",
                )

                # Test that intermediate size is reasonable
                self.assertGreaterEqual(
                    config.model.intermediate_size,
                    config.model.hidden_size,
                    f"Intermediate size should be >= hidden size in {config_name}",
                )

    def test_model_parameter_counts(self):
        """Test rough model parameter counts for tiny models."""
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        config = ConfigLoader.load(str(config_path))

        # Estimate parameters for our tiny model
        vocab_size = config.model.vocab_size
        hidden_size = config.model.hidden_size
        num_layers = config.model.num_hidden_layers
        intermediate_size = config.model.intermediate_size

        # Embedding parameters
        embedding_params = vocab_size * hidden_size

        # Per-layer parameters (rough estimate)
        # QKV projection: hidden_size * (hidden_size * 3)
        # Output projection: hidden_size * hidden_size
        # FFN: hidden_size * intermediate_size * 2
        attention_params_per_layer = (
            hidden_size * (hidden_size * 3) + hidden_size * hidden_size
        )
        ffn_params_per_layer = hidden_size * intermediate_size * 2
        total_params_per_layer = attention_params_per_layer + ffn_params_per_layer

        total_estimated_params = embedding_params + (
            num_layers * total_params_per_layer
        )

        # For our tiny model (64 hidden, 2 layers, 1000 vocab), should be manageable
        self.assertLess(
            total_estimated_params, 10_000_000, "Model should be small for testing"
        )
        self.assertGreater(
            total_estimated_params, 100_000, "Model should have reasonable size"
        )

        print(f"Estimated parameters for tiny model: {total_estimated_params:,}")


if __name__ == "__main__":
    unittest.main()
