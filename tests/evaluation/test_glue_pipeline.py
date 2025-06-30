#!/usr/bin/env python3
"""Test GLUE evaluation pipeline functionality."""

import tempfile
import unittest
from pathlib import Path

import torch

from neobert.config import ConfigLoader


class TestGLUEPipeline(unittest.TestCase):
    """Test GLUE evaluation pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "evaluation"
            / "test_tiny_glue.yaml"
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_glue_config_loading(self):
        """Test that GLUE config loads correctly."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Check GLUE-specific settings
        self.assertEqual(config.glue.task_name, "cola")
        self.assertEqual(config.glue.num_labels, 2)
        self.assertEqual(config.dataset.name, "cola")

    def test_glue_model_setup(self):
        """Test GLUE model setup for sequence classification."""
        config = ConfigLoader.load(str(self.test_config_path))

        from neobert.model import NeoBERTConfig, NeoBERTHFForSequenceClassification

        # Create model config
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
            num_labels=config.glue.num_labels,
            hidden_act=config.model.hidden_act,
        )

        # Test model creation
        model = NeoBERTHFForSequenceClassification(model_config)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

        # Check outputs
        self.assertTrue(hasattr(outputs, "logits"))
        expected_shape = (batch_size, config.glue.num_labels)
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_glue_data_processing(self):
        """Test GLUE data processing functionality."""
        try:
            from neobert.glue.process import process_dataset

            # This may fail due to network/dataset access on CI
            # We test the function exists and handles basic cases
            self.assertTrue(callable(process_dataset))

        except ImportError:
            self.skipTest("GLUE processing module not available")
        except Exception as e:
            # Expected on CPU-only systems without network access
            expected_errors = ["Connection", "404", "HfApi", "disk"]
            if any(err.lower() in str(e).lower() for err in expected_errors):
                self.skipTest(f"Expected network/dataset error: {e}")
            else:
                raise e

    def test_glue_metrics(self):
        """Test GLUE evaluation metrics."""
        import numpy as np

        # Test basic accuracy calculation (simulated)
        predictions = np.array([0, 1, 0, 1, 1])
        labels = np.array([0, 1, 1, 1, 0])

        accuracy = np.mean(predictions == labels)
        expected_accuracy = 3 / 5  # 3 correct out of 5

        self.assertAlmostEqual(accuracy, expected_accuracy)

    def test_different_glue_tasks(self):
        """Test different GLUE task configurations."""
        # Test different task setups that might be in configs
        glue_tasks = {
            "cola": {"num_labels": 2, "metric": "matthews_correlation"},
            "mnli": {"num_labels": 3, "metric": "accuracy"},
            "qnli": {"num_labels": 2, "metric": "accuracy"},
            "rte": {"num_labels": 2, "metric": "accuracy"},
        }

        for task, info in glue_tasks.items():
            with self.subTest(task=task):
                # Test that we can handle different label numbers
                from neobert.model import (
                    NeoBERTConfig,
                    NeoBERTHFForSequenceClassification,
                )

                model_config = NeoBERTConfig(
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    vocab_size=100,
                    num_labels=info["num_labels"],
                    flash_attention=False,
                    hidden_act="gelu",
                )

                model = NeoBERTHFForSequenceClassification(model_config)

                # Test forward pass
                input_ids = torch.randint(0, 100, (1, 5))
                attention_mask = torch.ones(1, 5)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                expected_shape = (1, info["num_labels"])
                self.assertEqual(outputs.logits.shape, expected_shape)

    def test_loss_computation(self):
        """Test loss computation for classification."""
        from neobert.model import NeoBERTConfig, NeoBERTHFForSequenceClassification

        model_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            num_labels=2,
            flash_attention=False,
            hidden_act="gelu",
        )

        model = NeoBERTHFForSequenceClassification(model_config)

        # Test with labels
        input_ids = torch.randint(0, 100, (2, 5))
        attention_mask = torch.ones(2, 5)
        labels = torch.tensor([0, 1])

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # Should compute loss automatically
        self.assertTrue(hasattr(outputs, "loss"))
        self.assertIsNotNone(outputs.loss)
        self.assertFalse(torch.isnan(outputs.loss))

    def test_training_argument_compatibility(self):
        """Test that training arguments are compatible with GLUE."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Check training settings suitable for classification
        self.assertTrue(config.trainer.per_device_train_batch_size > 0)
        self.assertTrue(config.optimizer.lr > 0)
        self.assertEqual(config.optimizer.name, "adamw")

        # Should have appropriate scheduler for fine-tuning
        self.assertEqual(config.scheduler.name, "linear_decay")


class TestGLUETaskSpecific(unittest.TestCase):
    """Test GLUE task-specific functionality."""

    def test_cola_specifics(self):
        """Test CoLA task specifics (grammar acceptability)."""
        # CoLA is a single sentence task with binary classification
        from neobert.model import NeoBERTConfig, NeoBERTHFForSequenceClassification

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            num_labels=2,  # Binary classification
            flash_attention=False,
            hidden_act="gelu",
        )

        model = NeoBERTHFForSequenceClassification(config)

        # Test single sentence input
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

        self.assertEqual(outputs.logits.shape, (1, 2))

    def test_sentence_pair_tasks(self):
        """Test sentence pair tasks (like RTE, MRPC)."""
        # These tasks typically use [CLS] token_type_ids [SEP] setup
        from neobert.model import NeoBERTConfig, NeoBERTHFForSequenceClassification

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            num_labels=2,
            flash_attention=False,
            hidden_act="gelu",
        )

        model = NeoBERTHFForSequenceClassification(config)

        # Simulate sentence pair: [CLS] sent1 [SEP] sent2 [SEP]
        input_ids = torch.randint(0, 100, (1, 15))
        attention_mask = torch.ones(1, 15)

        # Note: NeoBERT doesn't use token_type_ids, but should still work
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

        self.assertEqual(outputs.logits.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
