#!/usr/bin/env python3
"""Test GLUE evaluation pipeline functionality."""

import tempfile
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

from neobert.config import Config


class TestGLUETaskSpecific:
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

        assert outputs.logits.shape == (1, 2)

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
        assert not call_kwargs["enforce_mlm_special_tokens"]

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
        assert outputs.logits.shape == (1, 2)

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

        assert model.last_kwargs is not None
        assert "token_type_ids" in model.last_kwargs
        assert torch.equal(model.last_kwargs["token_type_ids"], token_type_ids)

    def test_build_glue_attention_mask_preserves_hf_binary_mask(self):
        """Ensure HF signature keeps 0/1 attention masks unchanged."""
        from neobert.glue.train import _build_glue_attention_mask

        binary_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        out = _build_glue_attention_mask(
            binary_mask,
            use_hf_signature=True,
            dtype_pad_mask=torch.float32,
        )
        assert torch.equal(out, binary_mask)

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

        assert len(created) == 2
        assert train_tracker is created[0]
        assert eval_tracker is created[1]
        assert train_tracker is not eval_tracker

    def test_save_training_checkpoint_zero_limit_still_saves(self):
        """Ensure save_total_limit=0 keeps all GLUE checkpoints while still saving."""
        from neobert.glue.train import save_training_checkpoint
        from neobert.checkpointing import MODEL_WEIGHTS_NAME

        class DummyAccelerator:
            is_main_process = True

            @staticmethod
            def save_state(output_dir):
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            @staticmethod
            def wait_for_everyone():
                return None

            @staticmethod
            def unwrap_model(model):
                return model

            @staticmethod
            def get_state_dict(model, unwrap=True):
                del unwrap
                return model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config()
            cfg.trainer.output_dir = tmpdir
            cfg.trainer.save_total_limit = 0
            cfg.trainer.max_ckpt = None

            model = torch.nn.Linear(8, 2)
            accelerator = DummyAccelerator()

            save_training_checkpoint(cfg, model, accelerator, completed_steps=10)
            save_training_checkpoint(cfg, model, accelerator, completed_steps=20)

            checkpoint_root = Path(tmpdir) / "checkpoints"
            assert (checkpoint_root / "10").exists()
            assert (checkpoint_root / "20").exists()
            assert (checkpoint_root / "10" / MODEL_WEIGHTS_NAME).exists()
            assert (checkpoint_root / "20" / MODEL_WEIGHTS_NAME).exists()
            assert not (Path(tmpdir) / "model_checkpoints").exists()

    def test_save_training_checkpoint_prunes_only_above_limit(self):
        """Ensure retention keeps exactly N checkpoints after each save."""
        from neobert.glue.train import save_training_checkpoint

        class DummyAccelerator:
            is_main_process = True

            @staticmethod
            def save_state(output_dir):
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            @staticmethod
            def wait_for_everyone():
                return None

            @staticmethod
            def unwrap_model(model):
                return model

            @staticmethod
            def get_state_dict(model, unwrap=True):
                del unwrap
                return model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config()
            cfg.trainer.output_dir = tmpdir
            cfg.trainer.save_total_limit = 1
            cfg.trainer.max_ckpt = None

            model = torch.nn.Linear(8, 2)
            accelerator = DummyAccelerator()

            with mock.patch("neobert.glue.train.logger.info"):
                save_training_checkpoint(cfg, model, accelerator, completed_steps=10)
                save_training_checkpoint(cfg, model, accelerator, completed_steps=20)

            checkpoint_root = Path(tmpdir) / "checkpoints"
            assert not (checkpoint_root / "10").exists()
            assert (checkpoint_root / "20").exists()
            assert not (Path(tmpdir) / "model_checkpoints").exists()

    def test_resolve_glue_training_schedule_handles_epoch_and_step_modes(self):
        """Ensure schedule helper computes steps/epochs from prepared loader length."""
        from neobert.glue.train import _resolve_glue_training_schedule

        cfg = Config()
        cfg.trainer.gradient_accumulation_steps = 2
        cfg.trainer.num_train_epochs = 3
        cfg.trainer.max_steps = -1

        updates, max_steps, epochs = _resolve_glue_training_schedule(
            cfg, batches_per_process=8
        )
        assert updates == 4
        assert max_steps == 12
        assert epochs == 3

        cfg.trainer.max_steps = 11
        updates, max_steps, epochs = _resolve_glue_training_schedule(
            cfg, batches_per_process=8
        )
        assert updates == 4
        assert max_steps == 11
        assert epochs == 3

    def test_validate_glue_config_allows_from_hub_without_pretrained_checkpoints(self):
        """Ensure from-hub fine-tuning bypasses local checkpoint requirements."""
        from neobert.validation.validators import validate_glue_config

        cfg = Config()
        cfg.glue.task_name = "sst2"
        cfg.model.from_hub = True
        cfg.glue.pretrained_checkpoint_dir = None
        cfg.glue.pretrained_checkpoint = None

        validate_glue_config(cfg)

    def test_validate_glue_config_allows_zero_checkpoint_id(self):
        """Ensure integer checkpoint id 0 is treated as a valid explicit value."""
        from neobert.validation.validators import validate_glue_config

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            cfg = Config()
            cfg.glue.task_name = "sst2"
            cfg.model.from_hub = False
            cfg.glue.allow_random_weights = False
            cfg.glue.pretrained_checkpoint_dir = checkpoint_dir
            cfg.glue.pretrained_checkpoint = 0

            validate_glue_config(cfg)

    def test_should_save_glue_checkpoint_respects_strategy_semantics(self):
        """Ensure step/epoch saves are independent of evaluation cadence."""
        from neobert.glue.train import _should_save_glue_checkpoint

        assert _should_save_glue_checkpoint(
            save_strategy="steps",
            completed_steps=10,
            num_update_steps_per_epoch=8,
            save_steps=5,
            eval_ran_this_step=False,
            metric_improved_this_eval=False,
        )
        assert _should_save_glue_checkpoint(
            save_strategy="epoch",
            completed_steps=16,
            num_update_steps_per_epoch=8,
            save_steps=None,
            eval_ran_this_step=False,
            metric_improved_this_eval=False,
        )
        assert not _should_save_glue_checkpoint(
            save_strategy="best",
            completed_steps=16,
            num_update_steps_per_epoch=8,
            save_steps=None,
            eval_ran_this_step=False,
            metric_improved_this_eval=True,
        )
        assert _should_save_glue_checkpoint(
            save_strategy="best",
            completed_steps=16,
            num_update_steps_per_epoch=8,
            save_steps=None,
            eval_ran_this_step=True,
            metric_improved_this_eval=True,
        )

    def test_sync_runtime_cfg_from_pretraining_uses_pretrained_values(self):
        """Ensure runtime GLUE config mirrors loaded pretraining architecture/tokenizer."""
        from neobert.glue.train import _sync_runtime_cfg_from_pretraining

        cfg = Config()
        cfg.model.hidden_size = 1024
        cfg.model.attn_backend = "flash_attn_varlen"
        cfg.tokenizer.max_length = 128
        cfg.tokenizer.revision = "some-user-rev"

        pretraining_cfg = Config()
        pretraining_cfg.model.hidden_size = 256
        pretraining_cfg.model.norm_eps = 2e-5
        pretraining_cfg.model.attn_backend = "flash_attn_varlen"
        pretraining_cfg.tokenizer.max_length = 512
        pretraining_cfg.tokenizer.revision = "checkpoint-rev"

        _sync_runtime_cfg_from_pretraining(cfg, pretraining_cfg)

        assert cfg.model.hidden_size == 256
        assert cfg.model.norm_eps == 2e-5
        assert cfg.model.attn_backend == "sdpa"
        assert cfg.tokenizer.max_length == 512
        assert cfg.tokenizer.revision == "checkpoint-rev"

    def test_get_evaluation_regression_keeps_vector_shapes(self):
        """Ensure STS-B style regression keeps predictions/labels 1D for batch=1."""
        from neobert.glue.train import get_evaluation

        class TinyRegressionDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                del idx
                return {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                    "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                    "labels": torch.tensor([0.5], dtype=torch.float32),
                }

        class TinyRegressionModel(torch.nn.Module):
            def forward(self, src, pad_mask):
                del pad_mask
                return {"logits": src[:, :1].to(torch.float32)}

        class ShapeCheckingMetric:
            def __init__(self):
                self.pred_shape = None
                self.ref_shape = None

            def add_batch(self, predictions, references):
                self.pred_shape = tuple(predictions.shape)
                self.ref_shape = tuple(references.shape)

            def compute(self):
                return {"pearson": 1.0}

        def _collate(batch):
            keys = batch[0].keys()
            return {
                key: torch.stack([item[key] for item in batch], dim=0) for key in keys
            }

        dataloader = DataLoader(
            TinyRegressionDataset(), batch_size=1, collate_fn=_collate
        )
        metric = ShapeCheckingMetric()
        eval_out = get_evaluation(
            model=TinyRegressionModel(),
            dataloader=dataloader,
            is_regression=True,
            metric=metric,
            accelerator=None,
            dtype_pad_mask=torch.float32,
            return_predictions=False,
            compute_metric=True,
            use_hf_signature=False,
            disable_tqdm=True,
        )

        assert metric.pred_shape == (1,)
        assert metric.ref_shape == (1,)
        assert "pearson" in eval_out["eval_metric"]

    def test_get_evaluation_respects_disable_tqdm_flag(self):
        """Ensure evaluation progress bars honor the disable_tqdm runtime flag."""
        from neobert.glue.train import get_evaluation

        class TinyDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                del idx
                return {
                    "input_ids": torch.tensor([1, 2], dtype=torch.long),
                    "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                    "labels": torch.tensor(0, dtype=torch.long),
                }

        class TinyClassifier(torch.nn.Module):
            def forward(self, src, pad_mask):
                del pad_mask
                batch_size = src.shape[0]
                return {"logits": torch.zeros((batch_size, 2), dtype=torch.float32)}

        def _collate(batch):
            keys = batch[0].keys()
            return {
                key: torch.stack([item[key] for item in batch], dim=0) for key in keys
            }

        with mock.patch(
            "neobert.glue.train.tqdm",
            side_effect=lambda iterable, **kwargs: iterable,
        ) as mocked_tqdm:
            get_evaluation(
                model=TinyClassifier(),
                dataloader=DataLoader(TinyDataset(), batch_size=1, collate_fn=_collate),
                is_regression=False,
                metric=None,
                accelerator=None,
                dtype_pad_mask=torch.float32,
                return_predictions=False,
                compute_metric=False,
                use_hf_signature=False,
                disable_tqdm=True,
            )

        assert mocked_tqdm.call_args.kwargs["disable"]
