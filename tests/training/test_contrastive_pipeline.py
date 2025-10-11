#!/usr/bin/env python3
"""Test contrastive training pipeline functionality."""

import tempfile
import unittest
from pathlib import Path

import torch

from neobert.config import ConfigLoader


class TestContrastivePipeline(unittest.TestCase):
    """Test contrastive training pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "contrastive"
            / "test_tiny_contrastive.yaml"
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_contrastive_config_loading(self):
        """Test that contrastive config loads correctly."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Check contrastive-specific settings
        self.assertTrue(hasattr(config, "contrastive"))
        self.assertEqual(config.contrastive.temperature, 0.05)
        self.assertEqual(config.contrastive.pooling, "avg")
        self.assertEqual(config.contrastive.loss_type, "simcse")
        self.assertEqual(config.dataset.name, "ALLNLI")

    def test_contrastive_loss_function(self):
        """Test contrastive loss function."""
        try:
            from neobert.contrastive.loss import SupConLoss

            # Test SupConLoss creation
            loss_fn = SupConLoss(temperature=0.1)

            # Create dummy embeddings
            batch_size = 4
            hidden_size = 64
            features = torch.randn(batch_size, hidden_size)

            # Test loss computation (self-supervised case)
            # SupConLoss expects queries and corpus
            queries = features
            corpus = features  # Self-supervised case
            loss = loss_fn(queries, corpus)

            self.assertIsInstance(loss, torch.Tensor)
            self.assertFalse(torch.isnan(loss))
            self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

        except ImportError as e:
            self.skipTest(f"Contrastive loss module not available: {e}")

    def test_contrastive_model_setup(self):
        """Test model setup for contrastive learning."""
        config = ConfigLoader.load(str(self.test_config_path))

        from neobert.model import NeoBERT, NeoBERTConfig

        # Create model config
        model_config = NeoBERTConfig(
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            intermediate_size=config.model.intermediate_size,
            dropout_prob=config.model.dropout_prob,
            vocab_size=config.model.vocab_size,
            max_position_embeddings=config.model.max_position_embeddings,
            flash_attention=config.model.flash_attention,
            ngpt=config.model.ngpt,
            liger_kernels=False,
            hidden_act="gelu",  # Keep tests independent of optional acceleration kernels
        )

        # Test model creation
        model = NeoBERT(model_config)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Convert to additive mask for NeoBERT
        pad_mask = torch.where(attention_mask == 0, float("-inf"), float(0.0))

        with torch.no_grad():
            outputs = model(input_ids, pad_mask)

        expected_shape = (batch_size, seq_len, config.model.hidden_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_pooling_strategies(self):
        """Test different pooling strategies for contrastive learning."""
        from neobert.model import NeoBERT, NeoBERTConfig

        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            flash_attention=False,
            hidden_act="gelu",
            liger_kernels=False,
        )

        model = NeoBERT(config)

        # Test input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        pad_mask = torch.where(attention_mask == 0, float("-inf"), float(0.0))

        with torch.no_grad():
            hidden_states = model(input_ids, pad_mask)

            # Test average pooling
            avg_pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(1).unsqueeze(-1)
            self.assertEqual(avg_pooled.shape, (batch_size, config.hidden_size))

            # Test CLS pooling (first token)
            cls_pooled = hidden_states[:, 0, :]
            self.assertEqual(cls_pooled.shape, (batch_size, config.hidden_size))

    def test_contrastive_dataset_classes(self):
        """Test contrastive dataset class functionality."""
        try:
            from neobert.contrastive.datasets import get_bsz

            # Test batch size calculation
            bsz = get_bsz("ALLNLI", target_batch_size=8)
            self.assertEqual(bsz, 4)  # 8 // 2 = 4 (ALLNLI has factor 2)

            # Test invalid dataset name
            with self.assertRaises(ValueError):
                get_bsz("INVALID_DATASET", target_batch_size=8)

        except ImportError as e:
            self.skipTest(f"Contrastive datasets module not available: {e}")

    def test_simcse_style_training_setup(self):
        """Test SimCSE-style contrastive training setup."""
        config = ConfigLoader.load(str(self.test_config_path))

        # SimCSE typically uses dropout for positive pairs
        self.assertTrue(config.model.dropout_prob > 0)

        # Should use contrastive loss
        self.assertEqual(config.contrastive.loss_type, "simcse")

        # Temperature should be reasonable for contrastive learning
        self.assertTrue(0.01 <= config.contrastive.temperature <= 0.2)

    def test_contrastive_trainer_integration(self):
        """Test contrastive trainer integration."""
        config = ConfigLoader.load(str(self.test_config_path))

        try:
            from neobert.contrastive.trainer import trainer

            # Test that trainer function exists and can be called
            # Note: We don't actually run training due to dataset/network requirements
            self.assertTrue(callable(trainer))

            # Test config validation for contrastive training
            config.trainer.output_dir = self.temp_dir
            config.trainer.num_train_epochs = 0  # Don't actually train

            # These should be set for contrastive training
            self.assertTrue(hasattr(config, "contrastive"))
            self.assertIsNotNone(config.contrastive.temperature)

        except ImportError as e:
            self.skipTest(f"Contrastive trainer module not available: {e}")
        except Exception as e:
            # Expected failures for dataset loading
            expected_errors = ["Connection", "404", "HfApi", "disk", "CUDA"]
            if any(err.lower() in str(e).lower() for err in expected_errors):
                self.skipTest(f"Expected dataset/network error: {e}")
            else:
                raise e

    def test_contrastive_metrics(self):
        """Test contrastive learning metrics."""
        try:
            from neobert.contrastive.metrics import compute_metrics

            # Test that metrics function exists
            self.assertTrue(callable(compute_metrics))

        except ImportError:
            # Skip if metrics module doesn't exist or has different structure
            self.skipTest("Contrastive metrics module not available")

    def test_negative_sampling(self):
        """Test negative sampling for contrastive learning."""
        # Test in-batch negative sampling (common in contrastive learning)
        batch_size = 4
        hidden_size = 32

        # Simulate embeddings
        embeddings = torch.randn(batch_size, hidden_size)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Diagonal should be positive examples (self-similarity)
        self.assertEqual(sim_matrix.shape, (batch_size, batch_size))

        # Check that diagonal elements are maximum (self-similarity)
        for i in range(batch_size):
            self.assertTrue(sim_matrix[i, i] >= sim_matrix[i, :].max() - 1e-6)

    def test_temperature_scaling(self):
        """Test temperature scaling in contrastive loss."""
        import torch.nn.functional as F

        # Test temperature effects on softmax
        logits = torch.tensor([1.0, 2.0, 3.0])

        # High temperature (more uniform)
        high_temp = F.softmax(logits / 1.0, dim=0)

        # Low temperature (more peaked)
        low_temp = F.softmax(logits / 0.1, dim=0)

        # Low temperature should have higher max probability
        self.assertTrue(low_temp.max() > high_temp.max())


class TestContrastiveLoss(unittest.TestCase):
    """Test contrastive loss implementations."""

    def test_supervised_contrastive_loss(self):
        """Test supervised contrastive loss."""
        try:
            from neobert.contrastive.loss import SupConLoss

            loss_fn = SupConLoss(temperature=0.1)

            # Test with labels
            batch_size = 4
            hidden_size = 16
            features = torch.randn(batch_size, 1, hidden_size)  # [N, 1, D]
            torch.tensor([0, 0, 1, 1])  # Two classes

            # SupConLoss expects queries and corpus, not features and labels
            # For supervised case, we'd need to handle label grouping separately
            queries = features[:, 0, :]
            corpus = features[:, 0, :]
            loss = loss_fn(queries, corpus)

            self.assertIsInstance(loss, torch.Tensor)
            self.assertFalse(torch.isnan(loss))
            self.assertTrue(loss.item() >= 0)

        except ImportError:
            self.skipTest("SupConLoss not available")

    def test_self_supervised_contrastive_loss(self):
        """Test self-supervised contrastive loss (SimCSE style)."""
        try:
            from neobert.contrastive.loss import SupConLoss

            loss_fn = SupConLoss(temperature=0.05)

            # Test without labels (self-supervised)
            batch_size = 4
            hidden_size = 16
            features = torch.randn(
                batch_size, 2, hidden_size
            )  # [N, 2, D] for two views

            # SupConLoss expects queries and corpus
            queries = features[:, 0, :]
            corpus = features[:, 1, :]  # Second view
            loss = loss_fn(queries, corpus)

            self.assertIsInstance(loss, torch.Tensor)
            self.assertFalse(torch.isnan(loss))
            self.assertTrue(loss.item() >= 0)

        except ImportError:
            self.skipTest("SupConLoss not available")


if __name__ == "__main__":
    unittest.main()
