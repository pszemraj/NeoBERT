#!/usr/bin/env python3
"""Test the new configuration system functionality."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.config import (
    Config,
    ConfigLoader,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TransformerEngineConfig,
    TorchAOConfig,
    TokenizerConfig,
    TrainerConfig,
    load_config_from_args,
    round_up_to_multiple,
)


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
        self.assertIsInstance(config.torchao, TorchAOConfig)
        self.assertIsInstance(config.transformer_engine, TransformerEngineConfig)

        # Check some default values
        self.assertEqual(config.model.hidden_size, 768)
        self.assertEqual(config.model.num_hidden_layers, 12)
        self.assertEqual(config.trainer.per_device_train_batch_size, 16)
        self.assertFalse(config.trainer.torch_compile)
        self.assertEqual(config.trainer.torch_compile_backend, "inductor")
        self.assertTrue(config.trainer.enforce_full_packed_batches)
        self.assertFalse(config.torchao.enable)
        self.assertEqual(config.torchao.recipe, "none")
        self.assertFalse(config.transformer_engine.enable)
        self.assertEqual(config.transformer_engine.recipe, "none")

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
            str(config_path),
            "--model.hidden_size",
            "128",
            "--optimizer.lr",
            "5e-4",
            "--trainer.per_device_train_batch_size",
            "4",
            "--dataset.streaming",
            "false",
            "--datacollator.pack_sequences",
            "true",
            "--trainer.torch_compile",
            "true",
            "--trainer.torch_compile_backend",
            "aot_eager",
            "--trainer.enforce_full_packed_batches",
            "false",
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
            self.assertFalse(config.dataset.streaming)
            self.assertTrue(config.datacollator.pack_sequences)
            self.assertTrue(config.trainer.torch_compile)
            self.assertEqual(config.trainer.torch_compile_backend, "aot_eager")
            self.assertFalse(config.trainer.enforce_full_packed_batches)

            # Check that non-overridden values remain the same
            self.assertEqual(config.model.num_hidden_layers, 2)

        finally:
            sys.argv = original_argv

    def test_cli_masked_logits_only_loss_strict_bool(self):
        """Ensure CLI parsing accepts explicit boolean tokens for loss path."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        test_args = [
            "script.py",
            str(config_path),
            "--trainer.masked_logits_only_loss",
            "off",
        ]

        original_argv = sys.argv
        sys.argv = test_args
        try:
            config = load_config_from_args()
            self.assertFalse(config.trainer.masked_logits_only_loss)
        finally:
            sys.argv = original_argv

    def test_cli_masked_logits_only_loss_rejects_typos(self):
        """Ensure invalid CLI boolean tokens fail fast for loss-path selection."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        test_args = [
            "script.py",
            str(config_path),
            "--trainer.masked_logits_only_loss",
            "ture",
        ]

        original_argv = sys.argv
        sys.argv = test_args
        try:
            with self.assertRaises(SystemExit):
                load_config_from_args()
        finally:
            sys.argv = original_argv

    def test_cli_dataset_eval_samples_override(self):
        """Ensure CLI parsing accepts dataset.eval_samples integer override."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        test_args = [
            "script.py",
            str(config_path),
            "--dataset.eval_samples",
            "2048",
        ]

        original_argv = sys.argv
        sys.argv = test_args
        try:
            config = load_config_from_args()
            self.assertEqual(config.dataset.eval_samples, 2048)
        finally:
            sys.argv = original_argv

    def test_cli_torchao_overrides(self):
        """Ensure CLI parsing supports torchao section overrides."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        test_args = [
            "script.py",
            str(config_path),
            "--trainer.torch_compile",
            "true",
            "--torchao.enable",
            "true",
            "--torchao.recipe",
            "mxfp8_emulated",
            "--torchao.filter_fqns",
            "decoder,head",
            "--torchao.skip_first_last_linear",
            "false",
            "--torchao.auto_filter_small_kn",
            "false",
            "--torchao.require_compile",
            "true",
        ]

        original_argv = sys.argv
        sys.argv = test_args
        try:
            config = load_config_from_args()
            self.assertTrue(config.torchao.enable)
            self.assertEqual(config.torchao.recipe, "mxfp8_emulated")
            self.assertEqual(config.torchao.filter_fqns, ["decoder", "head"])
            self.assertFalse(config.torchao.skip_first_last_linear)
            self.assertFalse(config.torchao.auto_filter_small_kn)
            self.assertTrue(config.torchao.require_compile)
        finally:
            sys.argv = original_argv

    def test_cli_transformer_engine_overrides(self):
        """Ensure CLI parsing supports transformer_engine section overrides."""
        config_path = self.test_config_dir / "pretraining" / "test_tiny_pretrain.yaml"
        test_args = [
            "script.py",
            str(config_path),
            "--trainer.torch_compile",
            "true",
            "--transformer_engine.enable",
            "true",
            "--transformer_engine.recipe",
            "nvfp4",
            "--transformer_engine.filter_fqns",
            "decoder,qkv",
            "--transformer_engine.skip_first_last_linear",
            "false",
            "--transformer_engine.require_compile",
            "true",
            "--transformer_engine.disable_stochastic_rounding",
            "true",
        ]

        original_argv = sys.argv
        sys.argv = test_args
        try:
            config = load_config_from_args()
            self.assertTrue(config.transformer_engine.enable)
            self.assertEqual(config.transformer_engine.recipe, "nvfp4")
            self.assertEqual(config.transformer_engine.filter_fqns, ["decoder", "qkv"])
            self.assertFalse(config.transformer_engine.skip_first_last_linear)
            self.assertTrue(config.transformer_engine.require_compile)
            self.assertTrue(config.transformer_engine.disable_stochastic_rounding)
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
  use_cpu: true
optimizer:
  name: "adamw"
  lr: 1e-4
""")
            temp_config_path = f.name

        try:
            test_args = [
                "script.py",
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

    def test_config_save_includes_task_sections(self):
        """Ensure saved configs include glue and contrastive sections."""
        config = Config()
        config.glue.task_name = "sst2"
        config.contrastive.temperature = 0.1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            ConfigLoader.save(config, path)
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)

            self.assertIn("glue", data)
            self.assertIn("contrastive", data)
            self.assertEqual(data["glue"]["task_name"], "sst2")
            self.assertEqual(data["contrastive"]["temperature"], 0.1)
        finally:
            os.unlink(path)

    def test_config_save_includes_torchao_section(self):
        """Ensure saved configs include torchao settings."""
        config = Config()
        config.torchao.enable = True
        config.torchao.recipe = "float8_rowwise"
        config.torchao.filter_fqns = ["decoder", "embed"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            ConfigLoader.save(config, path)
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            self.assertIn("torchao", data)
            self.assertTrue(data["torchao"]["enable"])
            self.assertEqual(data["torchao"]["recipe"], "float8_rowwise")
            self.assertEqual(data["torchao"]["filter_fqns"], ["decoder", "embed"])
        finally:
            os.unlink(path)

    def test_config_save_includes_transformer_engine_section(self):
        """Ensure saved configs include transformer_engine settings."""
        config = Config()
        config.transformer_engine.enable = True
        config.transformer_engine.recipe = "nvfp4"
        config.transformer_engine.filter_fqns = ["decoder", "qkv"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            ConfigLoader.save(config, path)
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            self.assertIn("transformer_engine", data)
            self.assertTrue(data["transformer_engine"]["enable"])
            self.assertEqual(data["transformer_engine"]["recipe"], "nvfp4")
            self.assertEqual(
                data["transformer_engine"]["filter_fqns"], ["decoder", "qkv"]
            )
        finally:
            os.unlink(path)

    def test_unknown_keys_raise(self):
        """Ensure unknown config keys raise a clear error."""
        config_data = {
            "model": {"hidden_size": 32, "unknown_key": 123},
            "trainer": {"output_dir": "./tmp"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            yaml.safe_dump(config_data, f)

        try:
            with self.assertRaises(ValueError):
                ConfigLoader.load(path)
        finally:
            os.unlink(path)

    def test_preprocess_uses_tokenizer_path(self):
        """Ensure tokenizer path is preferred when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "hello": 3}
            raw_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
                mask_token="[MASK]",
            )
            tokenizer.save_pretrained(tmpdir)

            config = Config()
            config.trainer.use_cpu = False
            config.tokenizer.name = "non-existent-tokenizer"
            config.tokenizer.path = tmpdir
            config.tokenizer.vocab_size = len(tokenizer)
            config.model.vocab_size = len(tokenizer)

            processed = ConfigLoader.preprocess_config(config, resolve_vocab_size=True)

            expected_vocab_size = round_up_to_multiple(len(tokenizer), 128)
            self.assertEqual(processed.model.vocab_size, expected_vocab_size)

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
