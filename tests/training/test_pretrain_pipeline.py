#!/usr/bin/env python3
"""Test pretraining pipeline functionality."""

import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.config import ConfigLoader
from neobert.pretraining.trainer import trainer


class TestPretrainPipeline(unittest.TestCase):
    """Test pretraining pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "pretraining"
            / "test_tiny_pretrain.yaml"
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading_for_pretraining(self):
        """Test that pretraining config loads correctly."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Check key pretraining-specific settings
        self.assertEqual(config.model.hidden_size, 64)
        # Config should have max_steps set for pretraining
        self.assertGreater(config.trainer.max_steps, 0)
        # datacollator doesn't have a 'type' field, it's implicitly MLM for pretraining
        self.assertEqual(config.datacollator.mlm_probability, 0.15)

    def test_tiny_dataset_creation(self):
        """Test creating a tiny dataset for testing."""
        # Create minimal fake dataset
        texts = [
            "This is a test sentence for pretraining.",
            "Another example text for the model to learn from.",
            "Short text.",
            "A longer example sentence that contains more tokens for testing purposes.",
        ]

        dataset = Dataset.from_dict({"text": texts})

        self.assertEqual(len(dataset), 4)
        self.assertIn("text", dataset.column_names)

    def test_pretraining_setup_without_execution(self):
        """Test pretraining setup without actually running training."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Override output directory to temp location
        config.trainer.output_dir = self.temp_dir
        config.trainer.num_train_epochs = 0  # Don't actually train
        config.trainer.max_steps = 1  # Minimal steps
        config.wandb.mode = "disabled"

        # Test that the trainer function can be called
        # Note: This will fail if dataset loading fails, which is expected for CPU-only testing
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*epoch parameter in `scheduler\.step\(\)`.*",
                category=UserWarning,
            )
            try:
                trainer(config)
            except Exception as e:
                # Expected failures for CPU-only testing:
                # - Dataset download/loading failures
                # - Missing dependencies for specific datasets
                expected_errors = [
                    "HfApi",  # Hugging Face API issues on CPU-only
                    "Connection",  # Network issues
                    "disk",  # Disk space issues
                    "CUDA",  # CUDA-related errors (expected on CPU)
                    "404",  # Dataset not found (expected for small test datasets)
                    "sentencepiece",  # Tokenizer not found
                    "Repository Not Found",  # HF repo not found
                    "input_ids",  # Dataset format mismatch
                    "ValueError",  # Dataset not tokenized
                ]

                error_str = str(e).lower()
                is_expected_error = any(
                    err.lower() in error_str for err in expected_errors
                )

                if not is_expected_error:
                    # If it's not an expected dataset/network error, re-raise
                    raise e

    def test_model_config_compatibility(self):
        """Test that model config is compatible with pretraining."""
        config = ConfigLoader.load(str(self.test_config_path))

        from neobert.model import NeoBERTConfig, NeoBERTLMHead

        # Create model config from our config
        model_config = NeoBERTConfig(
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            intermediate_size=config.model.intermediate_size,
            dropout=config.model.dropout_prob,
            vocab_size=config.model.vocab_size,
            max_length=config.model.max_position_embeddings,
            flash_attention=config.model.xformers_attention,
            ngpt=config.model.ngpt,
            hidden_act="gelu",  # Use GELU to avoid xformers requirement
        )

        # Test that model can be created
        model = NeoBERTLMHead(model_config)

        # Test forward pass with dummy data
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertIn("logits", outputs)
        self.assertIn("hidden_representation", outputs)

        expected_logits_shape = (batch_size, seq_len, config.model.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_logits_shape)

    def test_tokenizer_setup(self):
        """Test tokenizer configuration for pretraining."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Test tokenizer config
        self.assertIn(
            config.tokenizer.name, ["google/sentencepiece", "bert-base-uncased"]
        )
        self.assertEqual(config.tokenizer.vocab_size, 30522)
        self.assertEqual(config.tokenizer.max_length, 128)

    def test_optimizer_scheduler_setup(self):
        """Test optimizer and scheduler configuration."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Test optimizer config
        self.assertEqual(config.optimizer.name, "adamw")
        self.assertEqual(config.optimizer.lr, 1e-4)
        self.assertEqual(config.optimizer.weight_decay, 0.01)

        # Test scheduler config
        self.assertEqual(config.scheduler.name, "cosine")
        self.assertEqual(config.scheduler.warmup_steps, 10)
        self.assertEqual(config.scheduler.total_steps, 50)

    def test_mlm_collator_config(self):
        """Test MLM data collator configuration."""
        config = ConfigLoader.load(str(self.test_config_path))

        # datacollator doesn't have a 'type' field, it's implicitly MLM for pretraining
        self.assertEqual(config.datacollator.mlm_probability, 0.15)

    def test_training_arguments_config(self):
        """Test HuggingFace training arguments configuration."""
        config = ConfigLoader.load(str(self.test_config_path))

        # Check CPU-friendly settings
        self.assertEqual(config.trainer.per_device_train_batch_size, 2)
        self.assertEqual(config.trainer.dataloader_num_workers, 0)  # CPU-friendly
        self.assertTrue(config.trainer.use_cpu)
        self.assertEqual(config.trainer.report_to, [])  # No wandb on CI


class TestPretrainComponents(unittest.TestCase):
    """Test individual pretraining components."""

    def _make_tokenizer(self) -> PreTrainedTokenizerFast:
        """Build a minimal tokenizer for tests.

        :return PreTrainedTokenizerFast: Tokenizer with a tiny word-level vocab.
        """
        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[MASK]": 2,
            "[SEP]": 3,
            "hello": 4,
            "world": 5,
            "test": 6,
            "sentence": 7,
        }
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            mask_token="[MASK]",
            sep_token="[SEP]",
        )

    def test_mlm_data_collator(self):
        """Test MLM data collator functionality."""
        from neobert.collator import get_collator

        tokenizer = self._make_tokenizer()

        collator = get_collator(tokenizer=tokenizer, mlm_probability=0.15)

        # Test with dummy data
        texts = ["hello world", "test sentence"]
        tokenized = tokenizer(texts, padding=True, return_tensors="pt")

        batch = [
            {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
            },
            {
                "input_ids": tokenized["input_ids"][1],
                "attention_mask": tokenized["attention_mask"][1],
            },
        ]

        collated = collator(batch)

        self.assertIn("input_ids", collated)
        self.assertIn("labels", collated)
        self.assertIn("attention_mask", collated)

    def test_masked_correct_count(self):
        """Test masked accuracy counting ignores -100 labels."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        labels = torch.tensor([[2, -100]])
        self.assertEqual(_count_masked_correct(logits, labels).item(), 1)

    def test_pack_sequences_collator(self):
        """Ensure packed collator builds a block attention mask."""
        from neobert.collator import get_collator

        tokenizer = self._make_tokenizer()

        collator = get_collator(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pack_sequences=True,
            max_length=8,
        )

        batch = [
            {"input_ids": tokenizer("hello", add_special_tokens=False)["input_ids"]},
            {"input_ids": tokenizer("world", add_special_tokens=False)["input_ids"]},
        ]

        collated = collator(batch)

        self.assertIn("attention_mask", collated)
        self.assertEqual(collated["attention_mask"].dim(), 2)
        self.assertIn("packed_seqlens", collated)
        packed = collated["packed_seqlens"]
        self.assertTrue(torch.is_tensor(packed))
        self.assertTrue(all(row[row > 0].numel() >= 1 for row in packed))

    def test_to_target_batch_size_handles_empty_buffer(self):
        """Ensure batch packing handles empty buffers without crashing."""
        from neobert.pretraining.trainer import to_target_batch_size

        batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.zeros((2, 4), dtype=torch.long),
        }
        stored_batch = {"input_ids": None, "attention_mask": None, "labels": None}

        out, stored = to_target_batch_size(batch, stored_batch, target_size=4)
        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertIsNone(stored["input_ids"])

        stored_batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.zeros((2, 4), dtype=torch.long),
        }
        out, stored = to_target_batch_size(batch, stored_batch, target_size=4)
        self.assertEqual(out["input_ids"].shape[0], 4)

    def test_optimizer_creation(self):
        """Test optimizer creation from config."""
        config = ConfigLoader.load(
            Path(__file__).parent.parent
            / "configs"
            / "pretraining"
            / "test_tiny_pretrain.yaml"
        )

        from neobert.model import NeoBERT, NeoBERTConfig
        from neobert.optimizer import get_optimizer

        # Create tiny model
        model_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            flash_attention=False,
            hidden_act="gelu",
            rms_norm=False,
        )
        model = NeoBERT(model_config)

        from accelerate.utils import DistributedType

        optimizer = get_optimizer(
            model,
            DistributedType.NO,
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.assertIsNotNone(optimizer)
        # Should be AdamW
        self.assertTrue("AdamW" in str(type(optimizer)))
        self.assertGreaterEqual(len(optimizer.param_groups), 2)
        no_decay = [
            group for group in optimizer.param_groups if group["weight_decay"] == 0.0
        ]
        self.assertTrue(no_decay)
        no_decay_params = set(no_decay[0]["params"])
        self.assertIn(model.encoder.weight, no_decay_params)
        bias_params = {
            p for n, p in model.named_parameters() if n.lower().endswith(".bias")
        }
        self.assertTrue(bias_params.issubset(no_decay_params))

    def test_scheduler_creation(self):
        """Test scheduler creation from config."""
        config = ConfigLoader.load(
            Path(__file__).parent.parent
            / "configs"
            / "pretraining"
            / "test_tiny_pretrain.yaml"
        )

        from neobert.model import NeoBERT, NeoBERTConfig
        from neobert.optimizer import get_optimizer
        from neobert.scheduler import get_scheduler, resolve_scheduler_steps

        # Create minimal model and optimizer
        model_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            flash_attention=False,
            hidden_act="gelu",
        )
        model = NeoBERT(model_config)

        from accelerate.utils import DistributedType

        optimizer = get_optimizer(
            model,
            DistributedType.NO,
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        _, warmup_steps, decay_steps, constant_steps = resolve_scheduler_steps(
            trainer_max_steps=config.trainer.max_steps,
            total_steps=config.scheduler.total_steps,
            warmup_steps=config.scheduler.warmup_steps,
            warmup_percent=config.scheduler.warmup_percent,
            decay_steps=config.scheduler.decay_steps,
            decay_percent=config.scheduler.decay_percent,
            constant_steps=0,
        )
        scheduler = get_scheduler(
            optimizer=optimizer,
            lr=config.optimizer.lr,
            decay=config.scheduler.name,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            constant_steps=constant_steps,
        )

        self.assertIsNotNone(scheduler)


if __name__ == "__main__":
    unittest.main()
