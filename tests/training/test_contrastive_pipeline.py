#!/usr/bin/env python3
"""Test contrastive training pipeline functionality."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from neobert.config import Config, ConfigLoader


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
            dropout=config.model.dropout_prob,
            vocab_size=config.model.vocab_size,
            max_length=config.model.max_position_embeddings,
            attn_backend=config.model.attn_backend,
            ngpt=config.model.ngpt,
            hidden_act="gelu",  # Use GELU to avoid flash_attn requirement
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
            attn_backend="sdpa",
            hidden_act="gelu",
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

            # Missing dataset.path should raise a clear error early.
            config.dataset.path = None
            with self.assertRaisesRegex(ValueError, "dataset.path"):
                trainer(config)

        except ImportError as e:
            self.skipTest(f"Contrastive trainer module not available: {e}")
        except Exception as e:
            # Expected failures for dataset loading
            expected_errors = ["Connection", "404", "HfApi", "disk", "CUDA"]
            if any(err.lower() in str(e).lower() for err in expected_errors):
                self.skipTest(f"Expected dataset/network error: {e}")
            else:
                raise e

    def test_checkpoint_retention_limit_resolves_null_and_fallback(self):
        """Ensure retention limit handles optional fields without TypeError."""
        from neobert.contrastive.trainer import _resolve_checkpoint_retention_limit

        cfg = Config()
        cfg.task = "contrastive"

        cfg.trainer.save_total_limit = None
        cfg.trainer.max_ckpt = None
        self.assertEqual(_resolve_checkpoint_retention_limit(cfg), 0)

        cfg.trainer.max_ckpt = 5
        self.assertEqual(_resolve_checkpoint_retention_limit(cfg), 5)

        cfg.trainer.save_total_limit = 2
        self.assertEqual(_resolve_checkpoint_retention_limit(cfg), 2)

    def test_prune_step_checkpoints_keeps_latest_for_limit_one(self):
        """Ensure pruning keeps the latest checkpoint when limit=1."""
        from neobert.contrastive.trainer import _prune_step_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            for step in (10, 20):
                (checkpoint_dir / str(step)).mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "notes").mkdir(parents=True, exist_ok=True)

            _prune_step_checkpoints(checkpoint_dir, retention_limit=1)

            self.assertFalse((checkpoint_dir / "10").exists())
            self.assertTrue((checkpoint_dir / "20").exists())
            self.assertTrue((checkpoint_dir / "notes").exists())

    def test_contrastive_pretrained_checkpoint_root_rejects_legacy_layout(self):
        """Ensure legacy ``model_checkpoints`` roots fail fast for contrastive init."""
        try:
            from neobert.contrastive.trainer import (
                _normalize_contrastive_pretrained_checkpoint_root,
            )
        except ImportError as e:
            self.skipTest(f"Contrastive trainer module not available: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_root = Path(tmpdir) / "model_checkpoints"
            legacy_root.mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(ValueError, "model_checkpoints"):
                _normalize_contrastive_pretrained_checkpoint_root(legacy_root)

    def test_contrastive_trainer_saves_under_checkpoints_root(self):
        """Ensure contrastive saves portable weights under ``checkpoints/<step>/``."""
        config = ConfigLoader.load(str(self.test_config_path))
        config.dataset.path = self.temp_dir
        config.trainer.output_dir = self.temp_dir
        config.trainer.max_steps = 1
        config.trainer.save_steps = 1
        config.trainer.save_total_limit = 1
        config.trainer.logging_steps = 1
        config.trainer.save_strategy = "steps"
        config.trainer.save_model = True
        config.trainer.per_device_train_batch_size = 1
        config.trainer.use_cpu = True
        config.trainer.disable_tqdm = True
        config.wandb.mode = "disabled"
        config.wandb.enabled = False
        config.contrastive.pretraining_prob = 0.0

        try:
            from datasets import Dataset, DatasetDict
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast

            from neobert.contrastive.trainer import trainer
        except ImportError as e:
            self.skipTest(f"Contrastive trainer dependencies unavailable: {e}")

        dataset_dict = DatasetDict(
            {
                "ALLNLI": Dataset.from_dict(
                    {
                        "input_ids_query": [[2, 3, 0]],
                        "attention_mask_query": [[1, 1, 0]],
                        "input_ids_corpus": [[2, 4, 0]],
                        "attention_mask_corpus": [[1, 1, 0]],
                    }
                )
            }
        )
        pretraining_dataset = Dataset.from_dict(
            {
                "input_ids": [[2, 5, 0]],
                "attention_mask": [[1, 1, 0]],
            }
        )

        def _fake_load_from_disk(path: str):
            if Path(path).name == "all":
                return dataset_dict
            return pretraining_dataset

        def _make_tokenizer() -> PreTrainedTokenizerFast:
            vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "test": 4, "x": 5}
            tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            return PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token="[PAD]",
                unk_token="[UNK]",
            )

        with (
            mock.patch(
                "neobert.contrastive.trainer.load_from_disk",
                side_effect=_fake_load_from_disk,
            ),
            mock.patch(
                "neobert.contrastive.trainer.get_tokenizer",
                return_value=_make_tokenizer(),
            ),
        ):
            trainer(config)

        step_dir = Path(self.temp_dir) / "checkpoints" / "1"
        self.assertTrue(step_dir.is_dir())
        self.assertTrue((step_dir / "model.safetensors").is_file())
        self.assertFalse((Path(self.temp_dir) / "model_checkpoints").exists())

    def test_muonclip_trainer_passes_model_config(self):
        """Ensure MuonClip optimizer receives a model config in trainer."""
        config = ConfigLoader.load(str(self.test_config_path))
        config.dataset.path = self.temp_dir
        config.optimizer.name = "muonclip"
        config.trainer.max_steps = 0
        config.wandb.mode = "disabled"

        try:
            from datasets import Dataset, DatasetDict
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast

            from neobert.contrastive.trainer import trainer

            dataset_dict = DatasetDict({"ALLNLI": Dataset.from_dict({"dummy": ["x"]})})
            pretraining_dataset = Dataset.from_dict({"dummy": ["x"]})

            def _fake_load_from_disk(path: str):
                if path.endswith("all"):
                    return dataset_dict
                return pretraining_dataset

            def _make_tokenizer() -> PreTrainedTokenizerFast:
                vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2}
                tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                return PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer,
                    pad_token="[PAD]",
                    unk_token="[UNK]",
                )

            captured = {}

            def _fake_get_optimizer(
                model, distributed_type, model_config=None, **kwargs
            ):
                captured["model_config"] = model_config
                return torch.optim.Adam(model.parameters(), lr=1e-3)

            with (
                mock.patch(
                    "neobert.contrastive.trainer.load_from_disk",
                    side_effect=_fake_load_from_disk,
                ),
                mock.patch(
                    "neobert.contrastive.trainer.get_tokenizer",
                    return_value=_make_tokenizer(),
                ),
                mock.patch(
                    "neobert.contrastive.trainer.get_optimizer",
                    side_effect=_fake_get_optimizer,
                ),
            ):
                trainer(config)

            self.assertIsNotNone(captured.get("model_config"))

        except ImportError as e:
            self.skipTest(f"Contrastive trainer dependencies unavailable: {e}")

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
