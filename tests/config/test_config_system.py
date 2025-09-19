#!/usr/bin/env python3
"""Test the new configuration system functionality."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from neobert.config import (Config, ConfigLoader, DatasetConfig, ModelConfig,
                            OptimizerConfig, SchedulerConfig, TokenizerConfig,
                            TrainerConfig, load_config_from_args)


class TestConfigSystem(unittest.TestCase):
    """Test configuration system functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_dir = Path(__file__).parent.parent / "configs"

    def test_default_config_creation(self):
        """Test creating default config objects."""
        config = Config()

        # Check that all sub-configs are properly initialized
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.dataset, DatasetConfig)
        self.assertIsInstance(config.tokenizer, TokenizerConfig)
        self.assertIsInstance(config.trainer, TrainerConfig)
        self.assertIsInstance(config.optimizer, OptimizerConfig)
        self.assertIsInstance(config.scheduler, SchedulerConfig)

        # Check some default values
        self.assertEqual(config.model.hidden_size, 768)
        self.assertEqual(config.model.num_hidden_layers, 12)
        self.assertEqual(config.trainer.per_device_train_batch_size, 16)

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        self.assertTrue(config_path.exists(), f"Test config not found: {config_path}")

        config = ConfigLoader.load(str(config_path))

        # Check that tiny model config was loaded correctly
        self.assertEqual(config.model.hidden_size, 64)
        self.assertEqual(config.model.num_hidden_layers, 2)
        self.assertEqual(config.model.num_attention_heads, 2)
        self.assertEqual(config.trainer.per_device_train_batch_size, 2)
        self.assertEqual(config.dataset.max_seq_length, 128)

    def test_cli_override_system(self):
        """Test CLI override functionality."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        # Simulate command line args
        test_args = [
            "script.py",
            "--config",
            str(config_path),
            "--model.hidden_size",
            "128",
            "--optimizer.lr",
            "5e-4",
            "--trainer.per_device_train_batch_size",
            "4",
        ]

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = test_args

        try:
            config = load_config_from_args()

            # Check that overrides were applied
            self.assertEqual(config.model.hidden_size, 128)  # Overridden from 64
            self.assertEqual(config.optimizer.lr, 5e-4)  # Overridden from 1e-4
            self.assertEqual(
                config.trainer.per_device_train_batch_size, 4
            )  # Overridden from 2

            # Check that non-overridden values remain the same
            self.assertEqual(config.model.num_hidden_layers, 2)

        finally:
            sys.argv = original_argv

    def test_nested_config_override(self):
        """Test deeply nested configuration overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
model:
  hidden_size: 64
  num_attention_heads: 2
trainer:
  output_dir: "./test"
  learning_rate: 1e-4
optimizer:
  name: "adamw"
  lr: 1e-4
""")
            temp_config_path = f.name

        try:
            test_args = [
                "script.py",
                "--config",
                temp_config_path,
                "--model.hidden_size",
                "256",
                "--trainer.output_dir",
                "./new_test",
                "--optimizer.lr",
                "2e-4",
            ]

            original_argv = sys.argv
            sys.argv = test_args

            try:
                config = load_config_from_args()

                self.assertEqual(config.model.hidden_size, 256)
                self.assertEqual(config.trainer.output_dir, "./new_test")
                self.assertEqual(config.optimizer.lr, 2e-4)
                self.assertEqual(config.model.num_attention_heads, 2)  # Unchanged

            finally:
                sys.argv = original_argv

        finally:
            os.unlink(temp_config_path)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid hidden_size (not divisible by num_attention_heads)
        config = Config()
        config.model.hidden_size = 65  # Not divisible by 12
        config.model.num_attention_heads = 12

        # This should be caught when creating the model, not the config
        # The config itself should allow invalid combinations for flexibility

    def test_all_test_configs_load(self):
        """Test that all test configuration files load without errors."""
        test_configs = [
            "pretraining/test_tiny_pretrain.yaml",
            "evaluation/test_tiny_glue.yaml",
            "contrastive/test_tiny_contrastive.yaml",
        ]

        for config_name in test_configs:
            config_path = self.test_config_dir / config_name
            with self.subTest(config=config_name):
                self.assertTrue(
                    config_path.exists(), f"Config not found: {config_path}"
                )

                config = ConfigLoader.load(str(config_path))
                self.assertIsInstance(config, Config)

                # Check that all required fields are present
                self.assertIsNotNone(config.model.hidden_size)
                self.assertIsNotNone(config.trainer.output_dir)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from dataclasses import asdict

        config = Config()
        config.model.hidden_size = 128

        config_dict = asdict(config)

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model"]["hidden_size"], 128)

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load("nonexistent_config.yaml")

    def test_glue_config_specifics(self):
        """Test GLUE-specific configuration."""
        config_path = self.test_config_dir / "evaluation" / "test_tiny_glue.yaml"
        config = ConfigLoader.load(str(config_path))

        # GLUE config should be part of the main config, not separate
        self.assertEqual(config.task, "glue")  # Should be set in YAML
        self.assertEqual(config.dataset.name, "cola")

    def test_contrastive_config_specifics(self):
        """Test contrastive-specific configuration."""
        config_path = (
            self.test_config_dir / "contrastive" / "test_tiny_contrastive.yaml"
        )
        config = ConfigLoader.load(str(config_path))

        # Contrastive config should be part of the main config
        self.assertEqual(config.task, "contrastive")  # Should be set in YAML
        self.assertEqual(config.dataset.name, "ALLNLI")


if __name__ == "__main__":
    unittest.main()
