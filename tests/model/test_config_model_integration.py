#!/usr/bin/env python3
"""Test model and config integration without heavy dependencies."""

import unittest
from pathlib import Path

import torch

from neobert.config import ConfigLoader


class TestConfigModelIntegration(unittest.TestCase):
    """Test integration between config system and model creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_dir = Path(__file__).parent.parent / "configs"

    def test_config_to_model_config_conversion(self):
        """Test that config can be converted to model config."""
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        config = ConfigLoader.load(str(config_path))

        # Test that we can extract model parameters
        model_params = {
            "hidden_size": config.model.hidden_size,
            "num_hidden_layers": config.model.num_hidden_layers,
            "num_attention_heads": config.model.num_attention_heads,
            "intermediate_size": config.model.intermediate_size,
            "dropout_prob": config.model.dropout_prob,
            "vocab_size": config.model.vocab_size,
        }

        # Verify all parameters are reasonable
        self.assertEqual(model_params["hidden_size"], 64)
        self.assertEqual(model_params["num_hidden_layers"], 2)
        self.assertEqual(model_params["num_attention_heads"], 2)
        self.assertEqual(model_params["intermediate_size"], 128)
        self.assertEqual(model_params["dropout_prob"], 0.1)
        self.assertEqual(model_params["vocab_size"], 30522)  # BERT vocab size

        # Test that hidden_size is divisible by num_attention_heads
        self.assertEqual(
            model_params["hidden_size"] % model_params["num_attention_heads"], 0
        )

    def test_basic_tensor_operations(self):
        """Test basic tensor operations work (PyTorch available)."""
        # Test basic PyTorch functionality
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        z = x + y
        self.assertEqual(z.shape, (2, 3, 4))

        # Test that we can create embeddings
        embedding = torch.nn.Embedding(1000, 64)
        input_ids = torch.randint(0, 1000, (2, 10))

        embedded = embedding(input_ids)
        self.assertEqual(embedded.shape, (2, 10, 64))

    def test_all_test_configs_are_valid(self):
        """Test that all test configs have valid model parameters."""
        test_configs = [
            "pretraining/test_tiny_pretrain.yaml",
            "evaluation/test_tiny_glue.yaml",
            "contrastive/test_tiny_contrastive.yaml",
        ]

        for config_name in test_configs:
            with self.subTest(config=config_name):
                config_path = self.config_dir / config_name
                config = ConfigLoader.load(str(config_path))

                # Check model config validity
                self.assertGreater(config.model.hidden_size, 0)
                self.assertGreater(config.model.num_hidden_layers, 0)
                self.assertGreater(config.model.num_attention_heads, 0)
                self.assertGreater(config.model.intermediate_size, 0)
                self.assertGreaterEqual(config.model.dropout_prob, 0.0)
                self.assertLess(config.model.dropout_prob, 1.0)
                self.assertGreater(config.model.vocab_size, 0)

                # Check divisibility constraint
                self.assertEqual(
                    config.model.hidden_size % config.model.num_attention_heads,
                    0,
                    f"hidden_size must be divisible by num_attention_heads in {config_name}",
                )

    def test_config_compatibility_across_tasks(self):
        """Test that configs are compatible across different tasks."""
        pretrain_config = ConfigLoader.load(
            str(self.config_dir / "pretraining" / "test_tiny_pretrain.yaml")
        )
        glue_config = ConfigLoader.load(
            str(self.config_dir / "evaluation" / "test_tiny_glue.yaml")
        )
        contrastive_config = ConfigLoader.load(
            str(self.config_dir / "contrastive" / "test_tiny_contrastive.yaml")
        )

        # Model architectures should be compatible (same dimensions)
        self.assertEqual(
            pretrain_config.model.hidden_size, glue_config.model.hidden_size
        )
        self.assertEqual(
            pretrain_config.model.hidden_size, contrastive_config.model.hidden_size
        )

        self.assertEqual(pretrain_config.model.vocab_size, glue_config.model.vocab_size)
        self.assertEqual(
            pretrain_config.model.vocab_size, contrastive_config.model.vocab_size
        )

        # Tasks should be different
        self.assertEqual(pretrain_config.task, "pretraining")
        self.assertEqual(glue_config.task, "glue")
        self.assertEqual(contrastive_config.task, "contrastive")

    def test_optimizer_scheduler_config_validity(self):
        """Test optimizer and scheduler configs are valid."""
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        config = ConfigLoader.load(str(config_path))

        # Optimizer checks
        self.assertIn(config.optimizer.name, ["adam", "adamw", "sgd"])

        # Handle lr as either float or string
        lr_value = (
            float(config.optimizer.lr)
            if isinstance(config.optimizer.lr, str)
            else config.optimizer.lr
        )
        self.assertGreater(lr_value, 0)

        # Handle weight_decay as either float or string
        wd_value = (
            float(config.optimizer.weight_decay)
            if isinstance(config.optimizer.weight_decay, str)
            else config.optimizer.weight_decay
        )
        self.assertGreaterEqual(wd_value, 0)

        # Scheduler checks
        self.assertIn(
            config.scheduler.name, ["linear", "cosine", "cosine_decay", "linear_decay"]
        )
        self.assertGreaterEqual(config.scheduler.warmup_steps, 0)

    def test_trainer_config_cpu_compatibility(self):
        """Test trainer configs are suitable for CPU-only testing."""
        configs = [
            ConfigLoader.load(
                str(self.config_dir / "pretraining" / "test_tiny_pretrain.yaml")
            ),
            ConfigLoader.load(
                str(self.config_dir / "evaluation" / "test_tiny_glue.yaml")
            ),
            ConfigLoader.load(
                str(self.config_dir / "contrastive" / "test_tiny_contrastive.yaml")
            ),
        ]

        for config in configs:
            # Batch sizes should be small for CPU
            self.assertLessEqual(config.trainer.per_device_train_batch_size, 4)
            self.assertLessEqual(config.trainer.per_device_eval_batch_size, 4)

            # Should have CPU-friendly settings
            self.assertEqual(config.trainer.output_dir, "./test_outputs")


if __name__ == "__main__":
    unittest.main()
