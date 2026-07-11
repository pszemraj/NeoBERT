#!/usr/bin/env python3
"""Test GLUE evaluation pipeline functionality."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from neobert.config import Config, ConfigLoader


class TestGLUETaskSpecific:
    """Test GLUE task-specific functionality."""

    def test_glue_helper_factories_and_metric_loading(self):
        """Ensure tokenizer/collator/metric helpers honor expected GLUE wiring."""
        from neobert.glue.train import (
            _create_glue_data_collator,
            _load_from_hub_tokenizer,
            _load_glue_metric,
            _resolve_glue_runtime_policy,
        )

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

        cfg.datacollator.pad_to_multiple_of = 16
        tokenizer = mock.MagicMock()
        with mock.patch("neobert.glue.train.DataCollatorWithPadding") as collator_ctor:
            _create_glue_data_collator(tokenizer, cfg)
        collator_ctor.assert_called_once_with(tokenizer, pad_to_multiple_of=16)

        with mock.patch("neobert.glue.train.evaluate.load") as load_fn:
            _load_glue_metric("snli", "exp")
        load_fn.assert_called_once_with("glue", "mnli", experiment_id="exp")

        created = []

        def _fake_load(*args, **kwargs):
            del args, kwargs
            metric = mock.MagicMock()
            created.append(metric)
            return metric

        with mock.patch("neobert.glue.train.evaluate.load", side_effect=_fake_load):
            train_tracker = _load_glue_metric("cola", "exp")
            eval_tracker = _load_glue_metric("cola", "exp")
        assert len(created) == 2
        assert train_tracker is created[0]
        assert eval_tracker is created[1]
        assert train_tracker is not eval_tracker

        cfg.trainer.mixed_precision = "bf16"
        cfg.model.attn_backend = "flash_attn_varlen"
        with (
            mock.patch(
                "neobert.glue.train._bootstrap_logger.warning"
            ) as warning_logger,
        ):
            mixed_precision, attn_backend = _resolve_glue_runtime_policy(cfg)
        warning_logger.assert_called_once()
        assert mixed_precision == "bf16"
        assert attn_backend == "sdpa"

    def test_hf_logits_and_attention_mask_passthrough_helpers(self):
        """Ensure HF helper paths preserve token_type_ids and binary masks."""
        from neobert.glue.train import (
            _build_glue_attention_mask,
            _forward_classifier_logits,
        )

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

        binary_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        out = _build_glue_attention_mask(
            binary_mask,
            use_hf_signature=True,
            dtype_pad_mask=torch.float32,
        )
        assert torch.equal(out, binary_mask)

    def test_save_training_checkpoint_retention_behaviors(self):
        """GLUE checkpoints handle retention and write the optimizer manifest.

        The optimizer parameter-name manifest guards resume against positional
        state corruption; GLUE saves lacked it (unlike pretraining/contrastive),
        so this also pins that each kept checkpoint carries the manifest.
        """
        from neobert.glue.train import save_training_checkpoint
        from neobert.checkpointing import MODEL_WEIGHTS_NAME
        from neobert.training_utils import (
            OPTIMIZER_PARAM_NAMES_MANIFEST,
            attach_optimizer_param_names,
        )

        class DummyAccelerator:
            is_main_process = True

            @staticmethod
            def save_state(output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for filename in (
                    "model.safetensors",
                    "optimizer.bin",
                    "scheduler.bin",
                    "random_states_0.pkl",
                ):
                    (output_dir / filename).write_bytes(b"x")
                (output_dir / "custom_checkpoint_0.pkl").write_bytes(b"x")

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

        class DummyTokenizer:
            @staticmethod
            def save_pretrained(output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "tokenizer_config.json").write_text(
                    "{}", encoding="utf-8"
                )

        cases = [
            (0, {"10": True, "20": True}),
            (1, {"10": False, "20": True}),
        ]
        for save_total_limit, existence in cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = Config()
                cfg.task = "glue"
                cfg.trainer.output_dir = tmpdir
                cfg.trainer.save_total_limit = save_total_limit

                model = torch.nn.Linear(8, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                attach_optimizer_param_names(model, optimizer)
                accelerator = DummyAccelerator()
                tokenizer = DummyTokenizer()

                with mock.patch("neobert.glue.train.logger.info"):
                    save_training_checkpoint(
                        cfg,
                        tokenizer,
                        model,
                        optimizer,
                        accelerator,
                        completed_steps=10,
                    )
                    save_training_checkpoint(
                        cfg,
                        tokenizer,
                        model,
                        optimizer,
                        accelerator,
                        completed_steps=20,
                    )

                checkpoint_root = Path(tmpdir) / "checkpoints"
                for step, should_exist in existence.items():
                    step_dir = checkpoint_root / step
                    assert step_dir.exists() is should_exist
                    if should_exist:
                        assert (step_dir / MODEL_WEIGHTS_NAME).exists()
                        assert (step_dir / OPTIMIZER_PARAM_NAMES_MANIFEST).exists()
                        assert (step_dir / "config.yaml").exists()
                        assert (step_dir / "tokenizer").is_dir()
                        assert (step_dir / "checkpoint_complete.json").exists()
                assert not (Path(tmpdir) / "model_checkpoints").exists()

    def test_glue_loop_state_preserves_negative_best_score_and_resume_counters(self):
        """Ensure first negative task scores improve and state roundtrips exactly."""
        from neobert.glue.state import GlueLoopState

        state = GlueLoopState(world_size=2)
        assert state.update_selection_score(-0.4)
        assert state.best_selection_score == pytest.approx(-0.4)
        assert state.early_stopping_counter == 0
        assert not state.update_selection_score(-0.6)
        assert state.early_stopping_counter == 1
        with pytest.raises(ValueError, match="must be finite"):
            state.update_selection_score(float("nan"))
        state.record_update(
            completed_steps=7,
            epoch=1,
            microbatches_in_epoch=3,
            total_loss=2.5,
        )
        state.last_val_metrics = {"matthews_correlation": -0.6}

        restored = GlueLoopState(world_size=2)
        restored.load_state_dict(state.state_dict())

        assert restored.state_dict() == state.state_dict()
        with pytest.raises(ValueError, match="different world size"):
            GlueLoopState(world_size=1).load_state_dict(state.state_dict())

    def test_glue_loop_state_roundtrips_through_accelerate(self, tmp_path):
        """Ensure Accelerate persists the custom GLUE state implementation."""
        from accelerate import Accelerator

        from neobert.glue.state import GlueLoopState

        accelerator = Accelerator(cpu=True)
        state = GlueLoopState(world_size=accelerator.num_processes)
        accelerator.register_for_checkpointing(state)
        state.record_update(
            completed_steps=3,
            epoch=1,
            microbatches_in_epoch=2,
            total_loss=1.25,
        )

        accelerator.save_state(tmp_path)
        state.completed_steps = 0
        accelerator.load_state(tmp_path)

        assert state.completed_steps == 3
        assert state.epoch == 1
        assert state.microbatches_in_epoch == 2

    def test_glue_output_setup_preserves_resume_artifacts(self, tmp_path):
        """Ensure continuation never deletes prior metrics or checkpoint trees."""
        from neobert.glue.train import _prepare_glue_output_dir

        metrics_path = tmp_path / "all_results.json"
        metrics_path.write_text("{}", encoding="utf-8")
        checkpoint_path = tmp_path / "checkpoints" / "10"
        checkpoint_path.mkdir(parents=True)

        _prepare_glue_output_dir(
            tmp_path,
            resume_checkpoint_path=checkpoint_path,
            overwrite=True,
        )
        assert metrics_path.exists()
        assert checkpoint_path.exists()

        with pytest.raises(FileExistsError, match="not empty"):
            _prepare_glue_output_dir(
                tmp_path,
                resume_checkpoint_path=None,
                overwrite=False,
            )

        _prepare_glue_output_dir(
            tmp_path,
            resume_checkpoint_path=None,
            overwrite=True,
        )
        assert list(tmp_path.iterdir()) == []

    def test_glue_dataloader_uses_epoch_seeded_sampler(self):
        """Ensure GLUE map-style shuffling can reconstruct a saved epoch order."""
        from neobert.training_utils import build_dataloader_config

        cfg = Config()
        cfg.seed = 123
        dataloader_config = build_dataloader_config(seed=cfg.seed)

        assert dataloader_config.use_seedable_sampler
        assert dataloader_config.data_seed == 123

    def test_glue_terminal_resume_state_cannot_run_an_extra_update(self):
        """Ensure completed budgets and early-stop thresholds remain terminal."""
        from neobert.glue.state import GlueLoopState
        from neobert.glue.train import _glue_terminal_resume_reason

        state = GlueLoopState(world_size=1, completed_steps=10)
        assert "max_steps" in _glue_terminal_resume_reason(
            state, max_steps=10, early_stopping=0
        )

        state.completed_steps = 4
        state.early_stopping_counter = 3
        assert "early-stopping" in _glue_terminal_resume_reason(
            state, max_steps=10, early_stopping=3
        )
        assert (
            _glue_terminal_resume_reason(state, max_steps=10, early_stopping=4) is None
        )

    def test_glue_schedule_and_save_strategy_semantics(self):
        """Ensure training schedule and checkpoint-save strategy semantics are stable."""
        from neobert.glue.train import (
            _resolve_glue_training_schedule,
            _should_save_glue_checkpoint,
        )

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

        from neobert.glue.train import _resolve_glue_scheduler_steps

        cfg.scheduler.total_steps = 80
        cfg.scheduler.warmup_percent = 10
        cfg.scheduler.decay_percent = 75
        cfg.scheduler.warmup_steps = 999
        cfg.scheduler.decay_steps = 999
        warmup, decay, constant = _resolve_glue_scheduler_steps(cfg)
        assert (warmup, decay, constant) == (8, 60, 0)
        assert cfg.scheduler.warmup_steps == 999
        assert cfg.scheduler.decay_steps == 999

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

    def test_validate_glue_config_accepts_from_hub_and_zero_checkpoint(self):
        """Accept valid sources and percentage warmup without a default warning."""
        from neobert.glue.validation import validate_glue_config

        cfg = Config()
        cfg.task = "glue"
        cfg.glue.task_name = "sst2"
        cfg.model.from_hub = True
        cfg.glue.pretrained_checkpoint_dir = None
        cfg.glue.pretrained_checkpoint = None
        cfg.scheduler.warmup_percent = 10
        assert validate_glue_config(cfg) == ()

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            pretrained_config = Path(checkpoint_dir) / "config.yaml"
            pretrained_config.write_text("task: pretraining\n", encoding="utf-8")
            cfg = Config()
            cfg.task = "glue"
            cfg.glue.task_name = "sst2"
            cfg.model.from_hub = False
            cfg.glue.allow_random_weights = False
            cfg.glue.pretrained_checkpoint_dir = checkpoint_dir
            cfg.glue.pretrained_checkpoint = 0
            cfg.glue.pretrained_model_path = str(pretrained_config)

            validate_glue_config(cfg)

    def test_validate_glue_config_rejects_nonportable_remote_code_checkpoints(self):
        """Ensure custom Hub modeling code cannot masquerade as self-contained."""
        from neobert.glue.validation import GlueValidationError, validate_glue_config

        cfg = Config()
        cfg.task = "glue"
        cfg.glue.task_name = "sst2"
        cfg.model.from_hub = True
        cfg.tokenizer.trust_remote_code = True

        with pytest.raises(GlueValidationError, match="trust_remote_code=true"):
            validate_glue_config(cfg)

        cfg.trainer.save_model = False
        validate_glue_config(cfg)

    def test_validate_glue_config_is_side_effect_free(self, tmp_path):
        """Ensure validation rejects bad labels without mutating config or paths."""
        from neobert.glue.validation import GlueValidationError, validate_glue_config

        cfg = Config()
        cfg.task = "glue"
        cfg.model.from_hub = True
        cfg.glue.task_name = "sst2"
        cfg.glue.num_labels = 3
        cfg.trainer.output_dir = str(tmp_path / "not-created")

        with pytest.raises(GlueValidationError, match="expects glue.num_labels=2"):
            validate_glue_config(cfg)

        assert cfg.glue.num_labels == 3
        assert not (tmp_path / "not-created").exists()

    def test_suite_dry_run_uses_production_validation(
        self, tmp_path, monkeypatch, capsys
    ):
        """Ensure suite dry-runs reject invalid configs after launch overrides."""
        from scripts.evaluation.glue import run_glue_suite

        cfg = Config()
        cfg.task = "glue"
        cfg.glue.task_name = "rte"
        cfg.glue.num_labels = 2
        cfg.glue.pretrained_checkpoint_dir = None
        cfg.glue.pretrained_checkpoint = None
        cfg.glue.pretrained_model_path = None
        config_path = tmp_path / "rte.yaml"
        ConfigLoader.save(cfg, config_path)

        monkeypatch.setattr(run_glue_suite, "QUICK_TASKS", ("rte",))
        args = SimpleNamespace(
            config_dir=tmp_path,
            suite="quick",
            model_name_or_path=None,
            log_dir=None,
            dry_run=True,
        )
        assert run_glue_suite.run_suite(args) == 1
        assert "GLUE requires pretrained weights" in capsys.readouterr().err

        args.model_name_or_path = "example/model"
        assert run_glue_suite.run_suite(args) == 0
        output = capsys.readouterr()
        assert "DRY-RUN rte" in output.out
        assert "--model_name_or_path example/model" in output.out

    def test_glue_resume_preflight_is_checkpoint_self_contained(self, tmp_path):
        """Ensure continuation does not require the original pretraining source."""
        from neobert.checkpointing import mark_checkpoint_complete
        from neobert.glue.validation import load_validated_glue_config

        output_dir = tmp_path / "run"
        checkpoint_path = output_dir / "checkpoints" / "10"
        (checkpoint_path / "accelerate").mkdir(parents=True)
        for filename in (
            "model.safetensors",
            "optimizer.bin",
            "scheduler.bin",
            "random_states_0.pkl",
        ):
            (checkpoint_path / "accelerate" / filename).write_bytes(b"x")
        (checkpoint_path / "accelerate" / "custom_checkpoint_0.pkl").write_bytes(b"x")
        tokenizer_dir = checkpoint_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        model_config_dir = checkpoint_path / "model_config"
        model_config_dir.mkdir()
        (model_config_dir / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint_path / "model.safetensors").write_bytes(b"x")
        (checkpoint_path / "optimizer_param_names.json").write_text(
            '{"schema_version":1,"state_semantics":"adamw-v1","param_name_groups":[]}',
            encoding="utf-8",
        )

        checkpoint_cfg = Config()
        checkpoint_cfg.task = "glue"
        checkpoint_cfg.glue.task_name = "sst2"
        checkpoint_cfg.tokenizer.max_length = checkpoint_cfg.glue.max_seq_length
        checkpoint_cfg.glue.pretrained_model_path = None
        checkpoint_cfg.glue.pretrained_checkpoint_dir = None
        checkpoint_cfg.glue.pretrained_checkpoint = None
        ConfigLoader.save(checkpoint_cfg, checkpoint_path / "config.yaml")
        mark_checkpoint_complete(checkpoint_path, task="glue")

        launch_cfg = Config()
        launch_cfg.task = "glue"
        launch_cfg.trainer.output_dir = str(output_dir)
        launch_cfg.trainer.resume_from_checkpoint = "latest"
        launch_cfg.glue.pretrained_model_path = None
        launch_cfg.glue.pretrained_checkpoint_dir = None
        launch_cfg.glue.pretrained_checkpoint = None
        launch_path = tmp_path / "launch.yaml"
        ConfigLoader.save(launch_cfg, launch_path)

        loaded, _ = load_validated_glue_config(launch_path)

        assert loaded.glue.task_name == "sst2"
        assert loaded.tokenizer.path == str(checkpoint_path / "tokenizer")

    def test_task_registry_separates_selection_and_official_scores(self):
        """Ensure checkpoint selection never masquerades as an official GLUE score."""
        from neobert.glue.tasks import (
            compute_official_glue_score,
            get_checkpoint_selection_score,
            get_glue_task_spec,
        )

        mrpc = get_glue_task_spec("mrpc")
        assert mrpc.num_labels == 2
        assert mrpc.sentence_keys == ("sentence1", "sentence2")
        assert mrpc.checkpoint_metric == "f1"
        assert (
            get_checkpoint_selection_score(
                "mrpc", {"eval_accuracy": 0.9, "eval_f1": 0.8}
            )
            == 0.8
        )
        assert compute_official_glue_score(
            "mrpc", {"eval_accuracy": 0.9, "eval_f1": 0.8}
        ) == pytest.approx(0.85)
        assert compute_official_glue_score("mrpc", {"eval_f1": 0.8}) is None
        assert compute_official_glue_score(
            "mnli", {"accuracy": 0.8, "accuracy_mm": 0.6}
        ) == pytest.approx(0.7)

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
        assert cfg.tokenizer.max_length == 128
        assert cfg.tokenizer.revision == "checkpoint-rev"

    def test_glue_preprocessing_uses_task_context_length(self):
        """Ensure checkpoint tokenizer defaults cannot override GLUE tokenization."""
        from neobert.glue.process import process_function

        cfg = Config()
        cfg.task = "glue"
        cfg.mode = "train"
        cfg.glue.task_name = "sst2"
        cfg.glue.max_seq_length = 96
        cfg.tokenizer.max_length = 512
        tokenizer = mock.MagicMock(return_value={"input_ids": [[1, 2]]})

        result = process_function(
            {"sentence": ["example"], "label": [1]}, cfg, tokenizer
        )

        assert result["labels"] == [1]
        assert tokenizer.call_args.kwargs["max_length"] == 96

    def test_glue_preflight_rejects_oversized_learned_position_context(self, tmp_path):
        """Ensure dry validation uses checkpoint architecture for context limits."""
        from neobert.glue.validation import (
            GlueValidationError,
            load_validated_glue_config,
        )

        pretrained_cfg = Config()
        pretrained_cfg.model.rope = False
        pretrained_cfg.model.max_position_embeddings = 128
        pretrained_cfg.dataset.max_seq_length = 128
        pretrained_cfg.tokenizer.max_length = 128
        pretrained_path = tmp_path / "pretraining.yaml"
        ConfigLoader.save(pretrained_cfg, pretrained_path)

        checkpoint_root = tmp_path / "checkpoints"
        checkpoint_root.mkdir()
        cfg = Config()
        cfg.task = "glue"
        cfg.glue.task_name = "sst2"
        cfg.glue.max_seq_length = 256
        cfg.glue.pretrained_model_path = str(pretrained_path)
        cfg.glue.pretrained_checkpoint_dir = str(checkpoint_root)
        cfg.glue.pretrained_checkpoint = 10
        config_path = tmp_path / "glue.yaml"
        ConfigLoader.save(cfg, config_path)

        with pytest.raises(GlueValidationError, match="learned position table"):
            load_validated_glue_config(config_path)

        pretrained_cfg.model.rope = True
        ConfigLoader.save(pretrained_cfg, pretrained_path)
        loaded, _ = load_validated_glue_config(config_path)
        assert loaded.glue.max_seq_length == 256

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
