#!/usr/bin/env python3
"""End-to-end integration tests for the NeoBERT system."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

from neobert.config import ConfigLoader, load_config_from_args


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end functionality of the NeoBERT system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_to_model_pipeline(self):
        """Test full pipeline from config loading to model creation."""
        # Test pretrain config
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        # Simulate command line args
        test_args = [
            "script.py",
            "--config",
            str(config_path),
            "--trainer.output_dir",
            self.temp_dir,
        ]

        original_argv = sys.argv
        sys.argv = test_args

        try:
            config = load_config_from_args()

            # Create model from config
            from neobert.model import NeoBERTConfig, NeoBERTLMHead

            model_config = NeoBERTConfig(
                hidden_size=config.model.hidden_size,
                num_hidden_layers=config.model.num_hidden_layers,
                num_attention_heads=config.model.num_attention_heads,
                intermediate_size=config.model.intermediate_size,
                dropout=config.model.dropout_prob,
                vocab_size=config.model.vocab_size,
                max_length=config.model.max_position_embeddings,
                flash_attention=config.model.flash_attention,
                ngpt=config.model.ngpt,
                hidden_act=config.model.hidden_act,
            )

            model = NeoBERTLMHead(model_config)

            # Test forward pass
            batch_size, seq_len = 2, 8
            input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                outputs = model(input_ids)

            self.assertIn("logits", outputs)
            self.assertIn("hidden_representation", outputs)

        finally:
            sys.argv = original_argv

    def test_different_model_modes(self):
        """Test different model configurations (regular vs nGPT)."""
        base_config = {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 100,
            "flash_attention": False,
            "hidden_act": "gelu",
        }

        from neobert.model import NeoBERT, NeoBERTConfig, NormNeoBERT

        # Test regular NeoBERT
        regular_config = NeoBERTConfig(ngpt=False, **base_config)
        regular_model = NeoBERT(regular_config)

        # Test nGPT-style NeoBERT
        ngpt_config = NeoBERTConfig(ngpt=True, **base_config)
        ngpt_model = NormNeoBERT(ngpt_config)

        # Test both models
        input_ids = torch.randint(0, 100, (1, 5))
        pad_mask = torch.zeros(1, 5)

        with torch.no_grad():
            regular_output = regular_model(input_ids, pad_mask)
            ngpt_output = ngpt_model(input_ids, pad_mask)

        # Both should produce valid outputs
        self.assertEqual(regular_output.shape, (1, 5, 32))
        self.assertEqual(ngpt_output.shape, (1, 5, 32))

        # Outputs should be different due to different architectures
        self.assertFalse(torch.allclose(regular_output, ngpt_output, atol=1e-3))

    def test_all_task_configs_compatibility(self):
        """Test that all task configs are compatible with the system."""
        task_configs = [
            "pretraining/test_tiny_pretrain.yaml",
            "evaluation/test_tiny_glue.yaml",
            "contrastive/test_tiny_contrastive.yaml",
        ]

        for config_name in task_configs:
            with self.subTest(config=config_name):
                config_path = self.config_dir / config_name

                config = ConfigLoader.load(str(config_path))

                # Test model creation
                from neobert.model import NeoBERTConfig

                model_config = NeoBERTConfig(
                    hidden_size=config.model.hidden_size,
                    num_hidden_layers=config.model.num_hidden_layers,
                    num_attention_heads=config.model.num_attention_heads,
                    intermediate_size=config.model.intermediate_size,
                    dropout=config.model.dropout_prob,
                    vocab_size=config.model.vocab_size,
                    max_length=config.model.max_position_embeddings,
                    flash_attention=config.model.flash_attention,
                    ngpt=config.model.ngpt,
                    hidden_act=config.model.hidden_act,
                )

                # Choose appropriate model based on task
                if "glue" in config_name:
                    from neobert.model import NeoBERTHFForSequenceClassification

                    model_config.num_labels = config.glue.num_labels
                    model = NeoBERTHFForSequenceClassification(model_config)

                    # Test classification forward pass
                    input_ids = torch.randint(0, config.model.vocab_size, (1, 5))
                    attention_mask = torch.ones(1, 5)

                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )

                    self.assertTrue(hasattr(outputs, "logits"))

                else:
                    from neobert.model import NeoBERT

                    model = NeoBERT(model_config)

                    # Test encoder forward pass
                    input_ids = torch.randint(0, config.model.vocab_size, (1, 5))
                    pad_mask = torch.zeros(1, 5)

                    with torch.no_grad():
                        outputs = model(input_ids, pad_mask)

                    expected_shape = (1, 5, config.model.hidden_size)
                    self.assertEqual(outputs.shape, expected_shape)

    def test_optimizer_scheduler_integration(self):
        """Test optimizer and scheduler creation from configs."""
        try:
            from neobert.model import NeoBERT, NeoBERTConfig
            from neobert.optimizer import get_optimizer
            from neobert.scheduler import get_scheduler
        except ImportError as e:
            self.skipTest(f"Model dependencies not available: {e}")

        # Create tiny model
        model_config = NeoBERTConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=50,
            flash_attention=False,
            hidden_act="gelu",
        )
        model = NeoBERT(model_config)

        # Test different optimizer/scheduler combinations
        test_cases = [
            ("pretraining/test_tiny_pretrain.yaml", "adamw", "cosine"),
            ("evaluation/test_tiny_glue.yaml", "adamw", "linear"),
            ("contrastive/test_tiny_contrastive.yaml", "adamw", "linear"),
        ]

        for config_file, expected_opt, expected_sched in test_cases:
            with self.subTest(config=config_file):
                config = ConfigLoader.load(str(self.config_dir / config_file))

                # Test optimizer creation
                from accelerate.utils import DistributedType

                optimizer = get_optimizer(
                    model,
                    DistributedType.NO,  # No distributed for test
                    name=config.optimizer.name,
                    lr=config.optimizer.lr,
                    weight_decay=config.optimizer.weight_decay,
                )
                self.assertIsNotNone(optimizer)

                # Test scheduler creation
                scheduler = get_scheduler(
                    optimizer=optimizer,
                    lr=config.optimizer.lr,
                    decay=config.scheduler.name.replace("_decay", ""),
                    warmup_steps=config.scheduler.warmup_steps,
                    decay_steps=getattr(
                        config.scheduler,
                        "decay_steps",
                        config.scheduler.total_steps or 1000,
                    ),
                )
                self.assertIsNotNone(scheduler)

    def test_tokenizer_model_compatibility(self):
        """Test tokenizer and model vocab_size compatibility."""
        config = ConfigLoader.load(
            str(self.config_dir / "pretraining/test_tiny_pretrain.yaml")
        )

        # Model and tokenizer should have matching vocab sizes
        self.assertEqual(config.model.vocab_size, config.tokenizer.vocab_size)

        # Test that model can handle full vocab range
        from neobert.model import NeoBERT, NeoBERTConfig

        model_config = NeoBERTConfig(
            hidden_size=config.model.hidden_size,
            vocab_size=config.model.vocab_size,
            num_attention_heads=config.model.num_attention_heads,
            flash_attention=False,
            hidden_act="gelu",
        )
        model = NeoBERT(model_config)

        # Test with tokens across full vocab range
        input_ids = torch.tensor([[0, 1, config.model.vocab_size - 1, 2]])
        pad_mask = torch.zeros(1, 4)

        with torch.no_grad():
            outputs = model(input_ids, pad_mask)

        self.assertFalse(torch.isnan(outputs).any())

    def test_training_arguments_compatibility(self):
        """Test that training arguments are compatible across tasks."""
        configs = []
        for config_file in [
            "pretraining/test_tiny_pretrain.yaml",
            "evaluation/test_tiny_glue.yaml",
            "contrastive/test_tiny_contrastive.yaml",
        ]:
            config = ConfigLoader.load(str(self.config_dir / config_file))
            configs.append(config)

        # All configs should have CPU-friendly settings
        for config in configs:
            self.assertEqual(config.trainer.dataloader_num_workers, 0)
            self.assertTrue(config.trainer.use_cpu)
            self.assertEqual(config.trainer.report_to, [])

            # Batch sizes should be small for CPU testing
            self.assertTrue(config.trainer.per_device_train_batch_size <= 4)

    def test_config_override_system_robustness(self):
        """Test that config override system handles edge cases."""
        config_path = self.config_dir / "pretraining" / "test_tiny_pretrain.yaml"

        # Test various override patterns
        test_cases = [
            # Basic override
            ["--model.hidden_size", "128"],
            # Multiple overrides
            ["--model.hidden_size", "64", "--optimizer.lr", "1e-3"],
            # Nested overrides
            ["--trainer.per_device_train_batch_size", "1"],
            # Boolean overrides
            ["--model.flash_attention", "true"],
            # Float overrides
            ["--optimizer.weight_decay", "0.02"],
        ]

        for overrides in test_cases:
            with self.subTest(overrides=overrides):
                test_args = ["script.py", "--config", str(config_path)] + overrides

                original_argv = sys.argv
                sys.argv = test_args

                try:
                    config = load_config_from_args()

                    # Config should load without errors
                    self.assertIsNotNone(config)
                    self.assertIsNotNone(config.model.hidden_size)

                finally:
                    sys.argv = original_argv

    def test_error_handling_robustness(self):
        """Test system robustness to common errors."""
        # Test missing config file
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load("nonexistent.yaml")

        # Test invalid model config (hidden_size not divisible by num_heads)
        from neobert.model import NeoBERTConfig

        # This should raise an error when creating the config
        with self.assertRaises(ValueError):
            NeoBERTConfig(
                hidden_size=65,  # Not divisible by 12
                num_attention_heads=12,
                flash_attention=False,
            )


if __name__ == "__main__":
    unittest.main()
