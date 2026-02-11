#!/usr/bin/env python3
"""Test GLUE evaluation pipeline functionality."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from accelerate.utils import DistributedType

from neobert.config import Config, ConfigLoader


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
            attn_backend=config.model.attn_backend,
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
                    attn_backend="sdpa",
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
            attn_backend="sdpa",
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
        self.assertEqual(config.scheduler.name, "linear")


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
            attn_backend="sdpa",
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

    def test_from_hub_tokenizer_disables_mlm_special_token_enforcement(self):
        """Ensure hub tokenizer loading does not require MLM mask-token semantics."""
        from neobert.glue.train import _load_from_hub_tokenizer

        cfg = Config()
        cfg.task = "glue"
        cfg.model.name = "dummy/model"
        cfg.glue.max_seq_length = 128
        cfg.tokenizer.trust_remote_code = False
        cfg.tokenizer.revision = "main"
        cfg.tokenizer.allow_special_token_rewrite = False

        with mock.patch(
            "neobert.glue.train.get_tokenizer",
            return_value=object(),
        ) as mocked_get_tokenizer:
            _load_from_hub_tokenizer(cfg)

        call_kwargs = mocked_get_tokenizer.call_args.kwargs
        self.assertFalse(call_kwargs["enforce_mlm_special_tokens"])

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
            attn_backend="sdpa",
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

    def test_forward_classifier_logits_passes_token_type_ids_for_hf_models(self):
        """Ensure HF path forwards token_type_ids when they are present."""
        from neobert.glue.train import _forward_classifier_logits

        class DummyHFModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_kwargs = None

            def forward(self, **kwargs):
                self.last_kwargs = kwargs
                return {"logits": torch.zeros((2, 2))}

        model = DummyHFModel()
        input_ids = torch.ones((2, 4), dtype=torch.long)
        attention_mask = torch.ones((2, 4), dtype=torch.long)
        token_type_ids = torch.zeros((2, 4), dtype=torch.long)

        _forward_classifier_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_hf_signature=True,
        )

        self.assertIsNotNone(model.last_kwargs)
        self.assertIn("token_type_ids", model.last_kwargs)
        self.assertTrue(
            torch.equal(model.last_kwargs["token_type_ids"], token_type_ids)
        )

    def test_build_glue_attention_mask_preserves_hf_binary_mask(self):
        """Ensure HF signature keeps 0/1 attention masks unchanged."""
        from neobert.glue.train import _build_glue_attention_mask

        binary_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        out = _build_glue_attention_mask(
            binary_mask,
            use_hf_signature=True,
            dtype_pad_mask=torch.float32,
        )
        self.assertTrue(torch.equal(out, binary_mask))

    def test_create_glue_data_collator_uses_config_pad_multiple(self):
        """Ensure GLUE collator honors cfg.datacollator.pad_to_multiple_of."""
        from neobert.glue.train import _create_glue_data_collator

        cfg = Config()
        cfg.datacollator.pad_to_multiple_of = 16

        tokenizer = mock.MagicMock()
        with mock.patch("neobert.glue.train.DataCollatorWithPadding") as collator_ctor:
            _create_glue_data_collator(tokenizer, cfg)

        collator_ctor.assert_called_once_with(tokenizer, pad_to_multiple_of=16)

    def test_load_glue_metric_uses_expected_mapping(self):
        """Ensure helper maps task aliases to the intended evaluate metric."""
        from neobert.glue.train import _load_glue_metric

        with mock.patch("neobert.glue.train.evaluate.load") as load_fn:
            _load_glue_metric("multirc", "glue", "exp")
        load_fn.assert_called_once_with("accuracy", experiment_id="exp")

        with mock.patch("neobert.glue.train.evaluate.load") as load_fn:
            _load_glue_metric("snli", "glue", "exp")
        load_fn.assert_called_once_with("glue", "mnli", experiment_id="exp")

    def test_load_glue_metric_returns_independent_instances(self):
        """Ensure train/eval trackers can hold separate evaluate state."""
        from neobert.glue.train import _load_glue_metric

        created = []

        def _fake_load(*args, **kwargs):
            del args, kwargs
            metric = mock.MagicMock()
            created.append(metric)
            return metric

        with mock.patch("neobert.glue.train.evaluate.load", side_effect=_fake_load):
            train_tracker = _load_glue_metric("cola", "glue", "exp")
            eval_tracker = _load_glue_metric("cola", "glue", "exp")

        self.assertEqual(len(created), 2)
        self.assertIs(train_tracker, created[0])
        self.assertIs(eval_tracker, created[1])
        self.assertIsNot(train_tracker, eval_tracker)

    def test_save_training_checkpoint_zero_limit_still_saves(self):
        """Ensure save_total_limit=0 keeps all GLUE checkpoints while still saving."""
        from neobert.glue.train import save_training_checkpoint

        class DummyAccelerator:
            distributed_type = DistributedType.NO

            @staticmethod
            def unwrap_model(model):
                return model

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config()
            cfg.trainer.output_dir = tmpdir
            cfg.trainer.save_total_limit = 0
            cfg.trainer.max_ckpt = None

            model = torch.nn.Linear(8, 2)
            accelerator = DummyAccelerator()

            save_training_checkpoint(cfg, model, accelerator, completed_steps=10)
            save_training_checkpoint(cfg, model, accelerator, completed_steps=20)

            ckpt_root = Path(tmpdir) / "model_checkpoints"
            self.assertTrue((ckpt_root / "10").exists())
            self.assertTrue((ckpt_root / "20").exists())


if __name__ == "__main__":
    unittest.main()
