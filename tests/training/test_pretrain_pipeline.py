#!/usr/bin/env python3
"""Test pretraining pipeline functionality."""

import os
import tempfile
import warnings
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from accelerate.utils import DistributedType
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.checkpointing import MODEL_WEIGHTS_NAME, load_model_safetensors
from neobert.config import Config, ConfigLoader
from neobert.pretraining.masked_objective import MaskedObjectiveOut
from neobert.pretraining.trainer import (
    _compute_weight_norm_for_logging,
    _ensure_pinned_cpu_batch,
    _gather_decoder_weight_for_masked_objective,
    _infer_eval_split_name,
    _resolve_eval_samples,
    _resolve_loader_perf_settings,
    _resolve_streaming_eval_budget,
    _save_portable_checkpoint_weights,
    _run_masked_objective_step,
    _split_train_dataset_for_eval_samples,
    _should_backward_inside_gathered_decoder_weight,
    _sync_tokenizer_derived_config,
    _write_deepspeed_latest_file,
    trainer,
)


class TestPretrainPipeline:
    """Test pretraining pipeline functionality."""

    def test_pretraining_rejects_fp16(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Ensure pretraining trainer rejects fp16 mixed precision."""
        config = ConfigLoader.load(str(tiny_pretrain_config_path))
        config.trainer.output_dir = temp_output_dir
        config.trainer.mixed_precision = "fp16"

        with pytest.raises(ValueError, match="fp16"):
            trainer(config)

    def test_pretraining_rejects_invalid_masked_logits_only_loss(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Ensure invalid loss-path config fails before tokenizer/network setup."""
        config = ConfigLoader.load(str(tiny_pretrain_config_path))
        config.trainer.output_dir = temp_output_dir
        config.trainer.masked_logits_only_loss = "something_else"

        with patch("neobert.pretraining.trainer.get_tokenizer") as mocked_tokenizer:
            with pytest.raises(ValueError, match="masked_logits_only_loss"):
                trainer(config)
            mocked_tokenizer.assert_not_called()

    def test_pretraining_rejects_fsdp1_before_tokenizer_setup(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Ensure FSDP1 fails fast before tokenizer/dataset initialization."""
        config = ConfigLoader.load(str(tiny_pretrain_config_path))
        config.trainer.output_dir = temp_output_dir

        class _FSDPPluginStub:
            fsdp_version = 1

        class _StateStub:
            fsdp_plugin = _FSDPPluginStub()

        class _AcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

        with patch(
            "neobert.pretraining.trainer.Accelerator",
            return_value=_AcceleratorStub(),
        ):
            with patch("neobert.pretraining.trainer.get_tokenizer") as mocked_tokenizer:
                with pytest.raises(RuntimeError, match="FSDP2-first"):
                    trainer(config)
                mocked_tokenizer.assert_not_called()

    def test_pretraining_setup_without_execution(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Test pretraining setup path without requiring full training success."""
        config = ConfigLoader.load(str(tiny_pretrain_config_path))
        config.trainer.output_dir = temp_output_dir
        config.trainer.num_train_epochs = 0
        config.trainer.max_steps = 1
        config.wandb.mode = "disabled"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*epoch parameter in `scheduler\\.step\\(\\)`.*",
                category=UserWarning,
            )
            try:
                trainer(config)
            except Exception as exc:
                expected_errors = [
                    "HfApi",
                    "Connection",
                    "disk",
                    "CUDA",
                    "404",
                    "sentencepiece",
                    "Repository Not Found",
                    "input_ids",
                    "ValueError",
                ]
                error_str = str(exc).lower()
                if not any(err.lower() in error_str for err in expected_errors):
                    raise


class TestPretrainComponents:
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
        assert "input_ids" in collated
        assert "labels" in collated
        assert "attention_mask" in collated

    def test_ensure_pinned_cpu_batch_repins_unpinned_tensors(self):
        """Ensure trainer repins stitched CPU batches before async H2D transfer."""
        batch = {
            "input_ids": torch.randint(0, 10, (2, 4), dtype=torch.long),
            "labels": torch.randint(0, 10, (2, 4), dtype=torch.long),
            "meta": ["a", "b"],
        }
        try:
            out = _ensure_pinned_cpu_batch(batch)
        except RuntimeError as exc:
            pytest.skip(f"pin_memory not supported in this environment: {exc}")
            return

        assert out["input_ids"].is_pinned()
        assert out["labels"].is_pinned()
        assert out["meta"] == batch["meta"]

        out_again = _ensure_pinned_cpu_batch(out)
        assert out_again is out

    def test_ensure_pinned_cpu_batch_handles_nested_structures(self):
        """Ensure nested tensor containers are repinned recursively."""
        batch = {
            "input_ids": torch.randint(0, 10, (2, 4), dtype=torch.long),
            "nested": {
                "labels": torch.randint(0, 10, (2, 4), dtype=torch.long),
                "meta": ("a", torch.randint(0, 10, (1,), dtype=torch.long)),
            },
        }
        try:
            out = _ensure_pinned_cpu_batch(batch)
        except RuntimeError as exc:
            pytest.skip(f"pin_memory not supported in this environment: {exc}")
            return

        assert out["input_ids"].is_pinned()
        assert out["nested"]["labels"].is_pinned()
        assert out["nested"]["meta"][1].is_pinned()

    def test_sync_tokenizer_derived_config_pads_vocab_and_pad_id(self):
        """Ensure config is synchronized with tokenizer-derived vocab/pad fields."""
        cfg = Config()
        cfg.model.vocab_size = 17
        cfg.tokenizer.vocab_size = 17
        tokenizer = self._make_tokenizer()

        original, resolved, added = _sync_tokenizer_derived_config(cfg, tokenizer)

        assert original == 8
        assert resolved == 128
        assert added == 120
        assert len(tokenizer) == 128
        assert cfg.model.vocab_size == 128
        assert cfg.tokenizer.vocab_size == 128
        assert cfg.model.pad_token_id == tokenizer.pad_token_id

    def test_write_deepspeed_latest_file(self):
        """Ensure DeepSpeed root latest indirection is refreshed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_deepspeed_latest_file(root, "12345")
            latest_path = root / "latest"
            assert latest_path.exists()
            assert latest_path.read_text(encoding="utf-8") == "12345\n"

    def test_save_portable_checkpoint_weights_from_accelerator_state_dict(self):
        """Ensure portable safetensors is emitted from accelerator state dicts."""

        class _AcceleratorStub:
            is_main_process = True

            @staticmethod
            def get_state_dict(model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                return model.state_dict()

        model = torch.nn.Linear(4, 3, bias=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)
            saved = _save_portable_checkpoint_weights(
                model,
                _AcceleratorStub(),
                checkpoint_path,
            )
            assert saved
            assert (checkpoint_path / MODEL_WEIGHTS_NAME).exists()
            state_dict = load_model_safetensors(checkpoint_path, map_location="cpu")
            assert "weight" in state_dict

    def test_save_portable_checkpoint_weights_handles_export_failure(self):
        """Ensure portable save failures are non-fatal and return False."""

        class _AcceleratorStub:
            is_main_process = True

            @staticmethod
            def get_state_dict(_model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                raise ValueError("simulated failure")

        model = torch.nn.Linear(4, 3, bias=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)
            saved = _save_portable_checkpoint_weights(
                model,
                _AcceleratorStub(),
                checkpoint_path,
            )
            assert not saved
            assert not (checkpoint_path / MODEL_WEIGHTS_NAME).exists()

    def test_save_portable_checkpoint_weights_collective_call_non_main(self):
        """Ensure non-main ranks still participate in state-dict collection."""

        class _AcceleratorStub:
            is_main_process = False
            get_state_dict_called = False

            @classmethod
            def get_state_dict(cls, model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                cls.get_state_dict_called = True
                return model.state_dict()

        model = torch.nn.Linear(4, 3, bias=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)
            saved = _save_portable_checkpoint_weights(
                model,
                _AcceleratorStub(),
                checkpoint_path,
            )
            assert not saved
            assert _AcceleratorStub.get_state_dict_called
            assert not (checkpoint_path / MODEL_WEIGHTS_NAME).exists()

    def test_gather_decoder_weight_passthrough_outside_deepspeed(self):
        """Ensure decoder weight passthrough when not running DeepSpeed."""
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(4, 6, bias=False)

        class _AcceleratorStub:
            distributed_type = DistributedType.NO

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with patch("deepspeed.zero.GatheredParameters") as gathered:
            with _gather_decoder_weight_for_masked_objective(
                model, _AcceleratorStub()
            ) as lm_weight:
                assert lm_weight is model.decoder.weight
            gathered.assert_not_called()

    def test_gather_decoder_weight_uses_deepspeed_gather_for_zero3_param(self):
        """Ensure ZeRO-sharded decoder weights are gathered for masked objective."""
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(4, 6, bias=False)
        setattr(model.decoder.weight, "ds_id", 123)

        class _AcceleratorStub:
            distributed_type = DistributedType.DEEPSPEED

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ) as gathered:
            with _gather_decoder_weight_for_masked_objective(
                model, _AcceleratorStub()
            ) as lm_weight:
                assert lm_weight is model.decoder.weight

        gathered.assert_called_once()
        assert gathered.call_args.args[0] == [model.decoder.weight]
        assert gathered.call_args.kwargs["modifier_rank"] is None
        assert gathered.call_args.kwargs["fwd_module"] is model

    def test_gather_decoder_weight_rejects_fsdp1(self):
        """Ensure FSDP1 is rejected for masked-objective decoder-weight access."""
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(4, 6, bias=False)

        class _FSDPPluginStub:
            fsdp_version = 1

        class _StateStub:
            fsdp_plugin = _FSDPPluginStub()

        class _AcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with pytest.raises(RuntimeError, match="FSDP v1"):
            with _gather_decoder_weight_for_masked_objective(model, _AcceleratorStub()):
                pass

    def test_gather_decoder_weight_uses_fsdp2_unshard_reshard(self):
        """Ensure FSDP2 unshards and reshards around decoder-weight access."""

        class _WaitHandle:
            def __init__(self, calls: list[tuple[str, bool | None]]) -> None:
                self._calls = calls

            def wait(self) -> None:
                self._calls.append(("wait", None))

        class _FSDP2ModelStub(torch.nn.Module):
            def __init__(self, calls: list[tuple[str, bool | None]]) -> None:
                super().__init__()
                self.decoder = torch.nn.Linear(4, 6, bias=False)
                self._calls = calls

            def unshard(self, async_op: bool = False) -> _WaitHandle:
                self._calls.append(("unshard", async_op))
                return _WaitHandle(self._calls)

            def reshard(self) -> None:
                self._calls.append(("reshard", None))

        calls: list[tuple[str, bool | None]] = []
        model = _FSDP2ModelStub(calls)

        class _FSDPPluginStub:
            fsdp_version = 2

        class _StateStub:
            fsdp_plugin = _FSDPPluginStub()

        class _AcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params"
        ) as summon_full_params:
            with _gather_decoder_weight_for_masked_objective(
                model, _AcceleratorStub()
            ) as lm_weight:
                assert lm_weight is model.decoder.weight
        summon_full_params.assert_not_called()

        assert calls == [
            ("unshard", True),
            ("wait", None),
            ("reshard", None),
        ]

    def test_gather_decoder_weight_fsdp2_owner_search_prefers_wrapped_model(self):
        """Ensure owner search still works when unwrap_model strips FSDP2 hooks."""

        class _WaitHandle:
            def __init__(self, calls: list[tuple[str, bool | None]]) -> None:
                self._calls = calls

            def wait(self) -> None:
                self._calls.append(("wait", None))

        class _CoreModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = torch.nn.Linear(4, 6, bias=False)

        class _WrappedFSDP2Model(torch.nn.Module):
            def __init__(self, calls: list[tuple[str, bool | None]]) -> None:
                super().__init__()
                self.module = _CoreModel()
                self._calls = calls

            def unshard(self, async_op: bool = False) -> _WaitHandle:
                self._calls.append(("unshard", async_op))
                return _WaitHandle(self._calls)

            def reshard(self) -> None:
                self._calls.append(("reshard", None))

        calls: list[tuple[str, bool | None]] = []
        wrapped_model = _WrappedFSDP2Model(calls)

        class _FSDPPluginStub:
            fsdp_version = 2

        class _StateStub:
            fsdp_plugin = _FSDPPluginStub()

        class _AcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                # Mimic Accelerate unwrapping that removes a top-level wrapper.
                return module.module

        with _gather_decoder_weight_for_masked_objective(
            wrapped_model, _AcceleratorStub()
        ) as lm_weight:
            assert lm_weight is wrapped_model.module.decoder.weight

        assert calls == [
            ("unshard", True),
            ("wait", None),
            ("reshard", None),
        ]

    def test_run_masked_objective_step_backprops_inside_gather_on_zero3(self):
        """Ensure ZeRO-3 objective step runs backward before gather context exits."""

        class _ModelStub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = torch.nn.Linear(2, 3, bias=False)

            def forward(
                self,
                src: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                packed_seqlens: torch.Tensor | None = None,
                *,
                return_logits: bool = True,
            ) -> dict[str, torch.Tensor]:
                _ = pad_mask, packed_seqlens, return_logits
                hidden = torch.ones(
                    src.shape[0],
                    src.shape[1],
                    2,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                return {"hidden_representation": hidden}

        class _ObjectiveStub:
            def __call__(
                self,
                hidden_states: torch.Tensor,
                labels: torch.Tensor,
                lm_weight: torch.Tensor,
                *,
                compute_accuracy: bool = False,
            ) -> MaskedObjectiveOut:
                _ = labels, compute_accuracy
                loss = hidden_states.sum() * 0.0 + lm_weight.sum() * 0.0
                return MaskedObjectiveOut(
                    loss_sum_local=loss.float(),
                    num_masked_local=torch.tensor(0, dtype=torch.long),
                    used_path="train_checkpointed_masked_ce",
                    num_correct_local=torch.tensor(0, dtype=torch.long),
                )

        class _AcceleratorStub:
            distributed_type = DistributedType.DEEPSPEED

            def __init__(self, marker: dict[str, bool]) -> None:
                self._marker = marker
                self.backward_calls = 0

            def backward(self, _loss: torch.Tensor) -> None:
                self.backward_calls += 1
                if not self._marker["active"]:
                    raise AssertionError(
                        "backward must run while gather context is still active"
                    )

        class _GatherMarkerContext:
            def __init__(self, state: dict[str, bool], value: torch.Tensor) -> None:
                self._state = state
                self._value = value

            def __enter__(self) -> torch.Tensor:
                self._state["active"] = True
                return self._value

            def __exit__(self, exc_type, exc, tb) -> bool:
                self._state["active"] = False
                return False

        model = _ModelStub()
        setattr(model.decoder.weight, "ds_id", 7)
        marker = {"active": False}
        accelerator = _AcceleratorStub(marker)
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.full((1, 2), -100, dtype=torch.long),
        }

        with patch(
            "neobert.pretraining.trainer._gather_decoder_weight_for_masked_objective",
            side_effect=lambda *_args, **_kwargs: _GatherMarkerContext(
                marker, model.decoder.weight
            ),
        ):
            _objective_out, _loss_sum, backward_done = _run_masked_objective_step(
                model=model,
                batch=batch,
                pad_mask=None,
                packed_seqlens=None,
                masked_objective=_ObjectiveStub(),
                accelerator=accelerator,
                log_train_accuracy=False,
            )

        assert backward_done
        assert accelerator.backward_calls == 1

    def test_run_masked_objective_step_backprops_inside_gather_on_fsdp(self):
        """Ensure FSDP2 objective step runs backward before gather exits."""

        class _ModelStub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = torch.nn.Linear(2, 3, bias=False)

            def forward(
                self,
                src: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                packed_seqlens: torch.Tensor | None = None,
                *,
                return_logits: bool = True,
            ) -> dict[str, torch.Tensor]:
                _ = pad_mask, packed_seqlens, return_logits
                hidden = torch.ones(
                    src.shape[0],
                    src.shape[1],
                    2,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                return {"hidden_representation": hidden}

        class _ObjectiveStub:
            def __call__(
                self,
                hidden_states: torch.Tensor,
                labels: torch.Tensor,
                lm_weight: torch.Tensor,
                *,
                compute_accuracy: bool = False,
            ) -> MaskedObjectiveOut:
                _ = labels, compute_accuracy
                loss = hidden_states.sum() * 0.0 + lm_weight.sum() * 0.0
                return MaskedObjectiveOut(
                    loss_sum_local=loss.float(),
                    num_masked_local=torch.tensor(0, dtype=torch.long),
                    used_path="train_checkpointed_masked_ce",
                    num_correct_local=torch.tensor(0, dtype=torch.long),
                )

        class _AcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = type(
                "_StateStub",
                (),
                {"fsdp_plugin": type("_FSDPPluginStub", (), {"fsdp_version": 2})()},
            )()

            def __init__(self, marker: dict[str, bool]) -> None:
                self._marker = marker
                self.backward_calls = 0

            def backward(self, _loss: torch.Tensor) -> None:
                self.backward_calls += 1
                if not self._marker["active"]:
                    raise AssertionError(
                        "backward must run while gather context is still active"
                    )

        class _GatherMarkerContext:
            def __init__(self, state: dict[str, bool], value: torch.Tensor) -> None:
                self._state = state
                self._value = value

            def __enter__(self) -> torch.Tensor:
                self._state["active"] = True
                return self._value

            def __exit__(self, exc_type, exc, tb) -> bool:
                self._state["active"] = False
                return False

        model = _ModelStub()
        marker = {"active": False}
        accelerator = _AcceleratorStub(marker)
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.full((1, 2), -100, dtype=torch.long),
        }

        with patch(
            "neobert.pretraining.trainer._gather_decoder_weight_for_masked_objective",
            side_effect=lambda *_args, **_kwargs: _GatherMarkerContext(
                marker, model.decoder.weight
            ),
        ):
            _objective_out, _loss_sum, backward_done = _run_masked_objective_step(
                model=model,
                batch=batch,
                pad_mask=None,
                packed_seqlens=None,
                masked_objective=_ObjectiveStub(),
                accelerator=accelerator,
                log_train_accuracy=False,
            )

        assert backward_done
        assert accelerator.backward_calls == 1

    def test_should_backward_inside_gather_fsdp_depends_on_version(self):
        """Ensure gather-scoped backward policy only applies to FSDP2+."""

        class _FSDP1Accel:
            distributed_type = DistributedType.FSDP
            state = type(
                "_StateStub",
                (),
                {"fsdp_plugin": type("_FSDPPluginStub", (), {"fsdp_version": 1})()},
            )()

        class _FSDP2Accel:
            distributed_type = DistributedType.FSDP
            state = type(
                "_StateStub",
                (),
                {"fsdp_plugin": type("_FSDPPluginStub", (), {"fsdp_version": 2})()},
            )()

        lm_weight = torch.nn.Linear(2, 3, bias=False).weight
        assert not _should_backward_inside_gathered_decoder_weight(
            _FSDP1Accel(), lm_weight
        )
        assert _should_backward_inside_gathered_decoder_weight(_FSDP2Accel(), lm_weight)

    def test_run_masked_objective_step_defers_backward_when_not_zero3(self):
        """Ensure non-ZeRO paths defer backward to the outer training loop."""

        class _ModelStub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = torch.nn.Linear(2, 3, bias=False)

            def forward(
                self,
                src: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                packed_seqlens: torch.Tensor | None = None,
                *,
                return_logits: bool = True,
            ) -> dict[str, torch.Tensor]:
                _ = src, pad_mask, packed_seqlens, return_logits
                hidden = torch.ones((1, 2, 2), dtype=torch.float32, requires_grad=True)
                return {"hidden_representation": hidden}

        class _ObjectiveStub:
            def __call__(
                self,
                hidden_states: torch.Tensor,
                labels: torch.Tensor,
                lm_weight: torch.Tensor,
                *,
                compute_accuracy: bool = False,
            ) -> MaskedObjectiveOut:
                _ = labels, compute_accuracy
                return MaskedObjectiveOut(
                    loss_sum_local=(hidden_states.sum() * 0.0 + lm_weight.sum() * 0.0),
                    num_masked_local=torch.tensor(0, dtype=torch.long),
                    used_path="train_checkpointed_masked_ce",
                )

        class _AcceleratorStub:
            distributed_type = DistributedType.NO

            def __init__(self) -> None:
                self.backward_calls = 0

            def backward(self, _loss: torch.Tensor) -> None:
                self.backward_calls += 1

        model = _ModelStub()
        accelerator = _AcceleratorStub()
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.full((1, 2), -100, dtype=torch.long),
        }

        _objective_out, _loss_sum, backward_done = _run_masked_objective_step(
            model=model,
            batch=batch,
            pad_mask=None,
            packed_seqlens=None,
            masked_objective=_ObjectiveStub(),
            accelerator=accelerator,
            log_train_accuracy=False,
        )
        assert not backward_done
        assert accelerator.backward_calls == 0

    def test_compute_weight_norm_for_logging_deepspeed_skips_missing_full_params(self):
        """Ensure DeepSpeed weight-norm logging ignores params without full mappings."""
        model = torch.nn.Linear(3, 2, bias=True)

        class _AcceleratorStub:
            distributed_type = DistributedType.DEEPSPEED

        params = list(model.parameters())
        weight_full = torch.full_like(params[0], 2.0)

        def _fake_safe_get_full_fp32_param(param: torch.nn.Parameter):
            if param is params[0]:
                return weight_full
            return None

        with patch(
            "neobert.pretraining.trainer.safe_get_full_fp32_param",
            side_effect=_fake_safe_get_full_fp32_param,
        ):
            norm = _compute_weight_norm_for_logging(model, _AcceleratorStub())

        assert norm is not None
        assert round(abs(norm - weight_full.norm(2).item()), 6) == 0

    def test_compute_weight_norm_for_logging_deepspeed_returns_none_when_unavailable(
        self,
    ):
        """Ensure DeepSpeed weight-norm logging returns None without mapped params."""
        model = torch.nn.Linear(3, 2, bias=False)

        class _AcceleratorStub:
            distributed_type = DistributedType.DEEPSPEED

        with patch(
            "neobert.pretraining.trainer.safe_get_full_fp32_param",
            return_value=None,
        ):
            norm = _compute_weight_norm_for_logging(model, _AcceleratorStub())

        assert norm is None

    def test_resolve_loader_perf_settings_cuda_defaults(self):
        """Ensure CUDA runs get throughput-friendly loader defaults."""
        cfg = Config()
        cfg.dataset.num_workers = 4
        cfg.dataset.pin_memory = False
        cfg.dataset.persistent_workers = True
        cfg.dataset.prefetch_factor = None

        pin_memory, persistent_workers, prefetch_factor, notes = (
            _resolve_loader_perf_settings(cfg, device=torch.device("cuda"))
        )
        assert pin_memory
        assert persistent_workers
        assert prefetch_factor == 4
        assert len(notes) > 0

    def test_resolve_loader_perf_settings_cpu_respects_config(self):
        """Ensure CPU runs keep user-configured loader values."""
        cfg = Config()
        cfg.dataset.num_workers = 3
        cfg.dataset.pin_memory = False
        cfg.dataset.persistent_workers = True
        cfg.dataset.prefetch_factor = 2

        pin_memory, persistent_workers, prefetch_factor, notes = (
            _resolve_loader_perf_settings(cfg, device=torch.device("cpu"))
        )
        assert not pin_memory
        assert persistent_workers
        assert prefetch_factor == 2
        assert notes == []

    def test_mlm_collator_returns_packed_seqlens_metadata(self):
        """Ensure non-packed collator can emit packed_seqlens metadata."""
        from neobert.collator import get_collator

        tokenizer = self._make_tokenizer()
        collator = get_collator(
            tokenizer=tokenizer, mlm_probability=0.15, return_packed_seqlens=True
        )

        texts = ["hello world", "test"]
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
        packed = collated["packed_seqlens"]
        assert torch.is_tensor(packed)

        attention_mask = collated["attention_mask"]
        keep = torch.isfinite(attention_mask) & (attention_mask == 0)
        lengths = keep.sum(dim=1, keepdim=True).to(torch.int32)
        assert torch.equal(packed, lengths)

    def test_mlm_collator_skips_packed_seqlens_for_left_padding(self):
        """Ensure left-padded batches do not emit packed_seqlens metadata."""
        from neobert.collator import get_collator

        tokenizer = self._make_tokenizer()
        tokenizer.padding_side = "left"
        collator = get_collator(
            tokenizer=tokenizer, mlm_probability=0.15, return_packed_seqlens=True
        )

        texts = ["hello world", "test"]
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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            collated = collator(batch)
        assert any("Skipping packed_seqlens" in str(w.message) for w in caught)
        assert "attention_mask" in collated
        assert "packed_seqlens" not in collated

    def test_resolve_tokenize_num_proc_falls_back_to_cpu_count(self):
        """Ensure tokenization num_proc falls back when affinity is unavailable."""
        from neobert.pretraining.trainer import _resolve_tokenize_num_proc

        with patch("os.sched_getaffinity", side_effect=AttributeError, create=True):
            with patch("os.cpu_count", return_value=8):
                resolved = _resolve_tokenize_num_proc(
                    requested=None, num_processes=1, is_main_process=True
                )

        assert resolved == 8

    def test_resolve_eval_max_batches_per_rank(self):
        """Ensure eval max_batches is split across ranks."""
        from neobert.pretraining.trainer import _resolve_eval_max_batches

        assert _resolve_eval_max_batches(None, num_processes=2) is None
        assert _resolve_eval_max_batches(0, num_processes=2) is None
        assert _resolve_eval_max_batches(10, num_processes=1) == 10
        assert _resolve_eval_max_batches(10, num_processes=2) == 5
        assert _resolve_eval_max_batches(11, num_processes=2) == 6

    def test_resolve_eval_samples_normalization(self):
        """Ensure eval_samples resolves to positive integer or None."""
        assert _resolve_eval_samples(128) == 128
        assert _resolve_eval_samples("256") == 256
        assert _resolve_eval_samples(None) is None
        assert _resolve_eval_samples(0) is None
        assert _resolve_eval_samples(-5) is None
        with pytest.raises(ValueError):
            _resolve_eval_samples(True)
        with pytest.raises(ValueError):
            _resolve_eval_samples("not_an_int")

    def test_resolve_streaming_eval_budget_with_explicit_max_batches(self):
        """Ensure explicit trainer.eval_max_batches is respected."""
        resolved, source = _resolve_streaming_eval_budget(
            eval_max_batches=320,
            eval_samples=None,
            per_device_eval_batch_size=32,
        )
        assert resolved == 320
        assert source == "trainer.eval_max_batches"

    def test_resolve_streaming_eval_budget_derived_from_eval_samples(self):
        """Ensure dataset.eval_samples derives eval batch budget when needed."""
        resolved, source = _resolve_streaming_eval_budget(
            eval_max_batches=None,
            eval_samples=1000,
            per_device_eval_batch_size=64,
        )
        assert resolved == 16
        assert source == "dataset.eval_samples"

    def test_resolve_streaming_eval_budget_requires_explicit_budget(self):
        """Ensure streaming eval fails fast without explicit budget settings."""
        with pytest.raises(ValueError):
            _resolve_streaming_eval_budget(
                eval_max_batches=None,
                eval_samples=None,
                per_device_eval_batch_size=32,
            )

    def test_infer_eval_split_name_prefers_validation(self):
        """Ensure eval split inference chooses validation-style splits."""

        class _BuilderInfo:
            splits = {"train": object(), "validation": object(), "test": object()}

        class _Builder:
            info = _BuilderInfo()

        with patch(
            "neobert.pretraining.trainer.load_dataset_builder",
            return_value=_Builder(),
        ):
            inferred = _infer_eval_split_name(
                "dummy_dataset",
                {},
                train_split="train",
            )

        assert inferred == "validation"

    def test_split_train_dataset_for_eval_samples_non_streaming(self):
        """Ensure eval samples are carved out from non-streaming train dataset."""
        train_dataset = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})

        remaining_train, eval_dataset = _split_train_dataset_for_eval_samples(
            train_dataset,
            eval_samples=2,
            is_streaming=False,
        )

        assert len(eval_dataset) == 2
        assert len(remaining_train) == 3
        assert eval_dataset["text"] == ["a", "b"]
        assert remaining_train["text"] == ["c", "d", "e"]

    def test_split_train_dataset_for_eval_samples_streaming(self):
        """Ensure streaming split helper uses take/skip without materialization."""

        class _StreamingStub:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def take(self, n: int):
                self.calls.append(("take", n))
                return ("eval", n)

            def skip(self, n: int):
                self.calls.append(("skip", n))
                return ("train", n)

        train_dataset = _StreamingStub()
        remaining_train, eval_dataset = _split_train_dataset_for_eval_samples(
            train_dataset, eval_samples=321, is_streaming=True
        )

        assert eval_dataset == ("eval", 321)
        assert remaining_train == ("train", 321)
        assert train_dataset.calls == [("take", 321), ("skip", 321)]

    def test_select_train_split_from_datasetdict(self):
        """Ensure DatasetDict splits resolve to a Dataset."""
        from neobert.pretraining.trainer import _select_train_split

        dataset = Dataset.from_dict({"text": ["a", "b"]})
        dataset_dict = DatasetDict(train=dataset, validation=dataset)

        resolved = _select_train_split(dataset_dict, None)

        assert isinstance(resolved, Dataset)

    def test_masked_correct_count(self):
        """Test masked accuracy counting ignores -100 labels."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        labels = torch.tensor([[2, -100]])
        assert _count_masked_correct(logits, labels).item() == 1

    def test_masked_correct_count_all_ignored(self):
        """Test masked accuracy count returns zero when all labels are ignored."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        labels = torch.full((1, 2), -100, dtype=torch.long)
        assert _count_masked_correct(logits, labels).item() == 0

    def test_set_default_worker_env_respects_user_overrides(self):
        """Ensure worker env defaults only apply when vars are unset."""
        from neobert.pretraining.trainer import _set_default_worker_env

        with patch.dict(os.environ, {"OMP_NUM_THREADS": "8"}, clear=False):
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
            os.environ.pop("MKL_NUM_THREADS", None)
            _set_default_worker_env(num_workers=4)
            assert os.environ["OMP_NUM_THREADS"] == "8"
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            assert os.environ["MKL_NUM_THREADS"] == "1"

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

        assert "attention_mask" in collated
        assert collated["attention_mask"].dim() == 2
        assert "packed_seqlens" in collated
        packed = collated["packed_seqlens"]
        assert torch.is_tensor(packed)
        assert all(row[row > 0].numel() >= 1 for row in packed)

    def test_normalize_packed_seqlens_tensor(self):
        """Ensure packed_seqlens tensors normalize to CPU int32 tensors."""
        from neobert.model.model import _normalize_packed_seqlens

        packed = torch.tensor([[3, 0, 0], [2, 1, 0]], dtype=torch.int32)
        normalized = _normalize_packed_seqlens(packed)
        assert torch.is_tensor(normalized)
        assert normalized.dtype == torch.int32
        assert normalized.device.type == "cpu"
        assert torch.equal(
            normalized,
            torch.tensor([[3, 0, 0], [2, 1, 0]], dtype=torch.int32),
        )

    def test_clear_stored_batch_drops_mode_transition_fragments(self):
        """Ensure stale buffered fragments are cleared on packed-mode transitions."""
        from neobert.pretraining.trainer import _clear_stored_batch

        stored_batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.float32),
            "labels": torch.zeros((2, 4), dtype=torch.long),
            "packed_seqlens": None,
        }
        _clear_stored_batch(stored_batch)
        assert all(value is None for value in stored_batch.values())

    def test_promote_tmp_checkpoint_dir_keeps_old_until_swap(self):
        """Ensure tmp checkpoint promotion does not delete final dir pre-swap."""
        from neobert.pretraining.trainer import _promote_tmp_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tmp_ckpt = root / "100.tmp"
            final_ckpt = root / "100"
            tmp_ckpt.mkdir()
            final_ckpt.mkdir()
            (tmp_ckpt / "model.safetensors").write_text("new")
            (final_ckpt / "model.safetensors").write_text("old")

            _promote_tmp_checkpoint_dir(tmp_ckpt, final_ckpt)

            assert not tmp_ckpt.exists()
            assert final_ckpt.exists()
            assert (final_ckpt / "model.safetensors").read_text() == "new"
            assert not (root / "100.old").exists()

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
            attn_backend="sdpa",
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

        assert optimizer is not None
        # Should be AdamW
        assert "AdamW" in str(type(optimizer))
        assert len(optimizer.param_groups) >= 2
        no_decay = [
            group for group in optimizer.param_groups if group["weight_decay"] == 0.0
        ]
        assert no_decay
        no_decay_params = set(no_decay[0]["params"])
        assert model.encoder.weight in no_decay_params
        bias_params = {
            p for n, p in model.named_parameters() if n.lower().endswith(".bias")
        }
        assert bias_params.issubset(no_decay_params)

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
            attn_backend="sdpa",
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

        assert scheduler is not None

    def test_scheduler_case_insensitive_decay(self):
        """Scheduler decay selection should be case-insensitive."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from neobert.model import NeoBERT, NeoBERTConfig
        from neobert.scheduler import get_scheduler

        model_config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
            attn_backend="sdpa",
            hidden_act="gelu",
        )
        model = NeoBERT(model_config)
        optimizer = AdamW(model.parameters(), lr=1e-4)

        scheduler = get_scheduler(
            optimizer=optimizer,
            lr=1e-4,
            decay="CoSiNe",
            warmup_steps=0,
            decay_steps=10,
            constant_steps=0,
        )

        assert any(isinstance(s, CosineAnnealingLR) for s in scheduler._schedulers)
