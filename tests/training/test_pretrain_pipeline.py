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

    def test_pretraining_fail_fast_validation_paths(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Ensure invalid pretraining settings fail before expensive setup."""

        def _base_config() -> Config:
            cfg = ConfigLoader.load(str(tiny_pretrain_config_path))
            cfg.trainer.output_dir = temp_output_dir
            return cfg

        fp16_cfg = _base_config()
        fp16_cfg.trainer.mixed_precision = "fp16"
        with pytest.raises(ValueError, match="fp16"):
            trainer(fp16_cfg)

        invalid_loss_cfg = _base_config()
        invalid_loss_cfg.trainer.masked_logits_only_loss = "something_else"
        with patch("neobert.pretraining.trainer.get_tokenizer") as mocked_tokenizer:
            with pytest.raises(ValueError, match="masked_logits_only_loss"):
                trainer(invalid_loss_cfg)
            mocked_tokenizer.assert_not_called()

        fsdp_cfg = _base_config()

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
                    trainer(fsdp_cfg)
                mocked_tokenizer.assert_not_called()

    def test_pretraining_setup_smoke_without_full_execution(
        self, tiny_pretrain_config_path: Path, temp_output_dir: str
    ):
        """Exercise trainer setup path without requiring a full successful run."""
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
                    "hfapi",
                    "connection",
                    "disk",
                    "cuda",
                    "404",
                    "sentencepiece",
                    "repository not found",
                    "input_ids",
                    "valueerror",
                ]
                error_str = str(exc).lower()
                if not any(err in error_str for err in expected_errors):
                    raise

    def test_pretraining_calls_dion2_finalize_hook_after_prepare(
        self,
        tiny_pretrain_config_path: Path,
        temp_output_dir: str,
        make_wordlevel_tokenizer,
    ):
        """Ensure Dion2 mesh-finalization hook is invoked in pretraining runtime."""
        config = ConfigLoader.load(str(tiny_pretrain_config_path))
        config.trainer.output_dir = temp_output_dir
        config.trainer.max_steps = 0
        config.trainer.use_cpu = True
        config.wandb.mode = "disabled"
        config.dataset.streaming = False
        config.dataset.num_workers = 0
        config.dataset.num_proc = 1
        config.dataset.train_split = None
        config.dataset.eval_split = None
        config.dataset.eval_samples = 2
        config.datacollator.pack_sequences = False

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset"
            Dataset.from_dict(
                {"text": ["hello world", "hello test", "test sentence", "world test"]}
            ).save_to_disk(str(dataset_path))
            config.dataset.path = str(dataset_path)
            config.dataset.name = ""
            config.dataset.text_column = "text"

            tokenizer = make_wordlevel_tokenizer()
            tokenizer_path = Path(tmpdir) / "tokenizer"
            tokenizer.save_pretrained(str(tokenizer_path))
            config.tokenizer.path = str(tokenizer_path)
            config.tokenizer.name = "local-test-tokenizer"
            config.tokenizer.max_length = 32
            config.dataset.max_seq_length = 32
            config.model.max_position_embeddings = 32

            with (
                patch(
                    "neobert.pretraining.trainer.finalize_dion2_distributed_mesh"
                ) as finalize_mock,
                patch(
                    "neobert.pretraining.trainer.finalize_dion2_qk_clipping_runtime"
                ) as finalize_qk_mock,
            ):
                trainer(config)

            finalize_mock.assert_called_once()
            finalize_qk_mock.assert_called_once()


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

    def test_ensure_pinned_cpu_batch_repins_flat_and_nested_tensors(self):
        """Ensure CPU repinning covers flat and nested tensor containers."""
        cases = [
            {
                "batch": {
                    "input_ids": torch.randint(0, 10, (2, 4), dtype=torch.long),
                    "labels": torch.randint(0, 10, (2, 4), dtype=torch.long),
                    "meta": ["a", "b"],
                },
                "checks": [
                    lambda out: out["input_ids"].is_pinned(),
                    lambda out: out["labels"].is_pinned(),
                    lambda out: out["meta"] == ["a", "b"],
                ],
            },
            {
                "batch": {
                    "input_ids": torch.randint(0, 10, (2, 4), dtype=torch.long),
                    "nested": {
                        "labels": torch.randint(0, 10, (2, 4), dtype=torch.long),
                        "meta": ("a", torch.randint(0, 10, (1,), dtype=torch.long)),
                    },
                },
                "checks": [
                    lambda out: out["input_ids"].is_pinned(),
                    lambda out: out["nested"]["labels"].is_pinned(),
                    lambda out: out["nested"]["meta"][1].is_pinned(),
                ],
            },
        ]
        for case in cases:
            try:
                out = _ensure_pinned_cpu_batch(case["batch"])
            except RuntimeError as exc:
                pytest.skip(f"pin_memory not supported in this environment: {exc}")
                return

            for check in case["checks"]:
                assert check(out)

            out_again = _ensure_pinned_cpu_batch(out)
            assert out_again is out

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

    def test_save_portable_checkpoint_weights_paths(self):
        """Ensure portable checkpoint save behavior for success/failure/rank paths."""

        class _MainSuccessAccelerator:
            is_main_process = True

            @staticmethod
            def get_state_dict(model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                return model.state_dict()

        class _MainFailureAccelerator:
            is_main_process = True

            @staticmethod
            def get_state_dict(_model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                raise ValueError("simulated failure")

        class _NonMainAccelerator:
            is_main_process = False
            get_state_dict_called = False

            @classmethod
            def get_state_dict(cls, model: torch.nn.Module, unwrap: bool = True):
                assert unwrap is True
                cls.get_state_dict_called = True
                return model.state_dict()

        cases = [
            (_MainSuccessAccelerator(), True, True),
            (_MainFailureAccelerator(), False, False),
            (_NonMainAccelerator(), False, False),
        ]
        for accelerator, expected_saved, expected_file in cases:
            model = torch.nn.Linear(4, 3, bias=False)
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir)
                saved = _save_portable_checkpoint_weights(
                    model,
                    accelerator,
                    checkpoint_path,
                )
                assert saved is expected_saved
                assert (checkpoint_path / MODEL_WEIGHTS_NAME).exists() is expected_file
                if expected_file:
                    state_dict = load_model_safetensors(
                        checkpoint_path, map_location="cpu"
                    )
                    assert "weight" in state_dict

        assert _NonMainAccelerator.get_state_dict_called

    def test_gather_decoder_weight_context_matrix(self):
        """Ensure decoder-weight gather behavior is correct across distributed modes."""
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(4, 6, bias=False)

        class _NoDistAccelerator:
            distributed_type = DistributedType.NO

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with patch("deepspeed.zero.GatheredParameters") as gathered:
            with _gather_decoder_weight_for_masked_objective(
                model, _NoDistAccelerator()
            ) as lm_weight:
                assert lm_weight is model.decoder.weight
            gathered.assert_not_called()

        setattr(model.decoder.weight, "ds_id", 123)

        class _DeepSpeedAccelerator:
            distributed_type = DistributedType.DEEPSPEED

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ) as gathered:
            with _gather_decoder_weight_for_masked_objective(
                model, _DeepSpeedAccelerator()
            ) as lm_weight:
                assert lm_weight is model.decoder.weight
        gathered.assert_called_once()
        assert gathered.call_args.args[0] == [model.decoder.weight]
        assert gathered.call_args.kwargs["modifier_rank"] is None
        assert gathered.call_args.kwargs["fwd_module"] is model

        class _FSDPPluginStub:
            fsdp_version = 1

        class _StateStub:
            fsdp_plugin = _FSDPPluginStub()

        class _FSDP1Accelerator:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module

        with pytest.raises(RuntimeError, match="FSDP v1"):
            with _gather_decoder_weight_for_masked_objective(
                model, _FSDP1Accelerator()
            ):
                pass

    def test_gather_decoder_weight_fsdp2_unshard_and_owner_search_paths(self):
        """Ensure FSDP2 gather uses unshard/reshard and owner lookup works when unwrapped."""

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

        calls: list[tuple[str, bool | None]] = []
        model = _FSDP2ModelStub(calls)
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

        class _UnwrappedAcceleratorStub:
            distributed_type = DistributedType.FSDP
            state = _StateStub()

            @staticmethod
            def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
                return module.module

        wrapped_calls: list[tuple[str, bool | None]] = []
        wrapped_model = _WrappedFSDP2Model(wrapped_calls)
        with _gather_decoder_weight_for_masked_objective(
            wrapped_model, _UnwrappedAcceleratorStub()
        ) as lm_weight:
            assert lm_weight is wrapped_model.module.decoder.weight
        assert wrapped_calls == [
            ("unshard", True),
            ("wait", None),
            ("reshard", None),
        ]

    def test_masked_objective_backward_policy_matrix(self):
        """Ensure masked-objective backward timing matches distributed policy."""

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
                    num_correct_local=torch.tensor(0, dtype=torch.long),
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

        class _DeepSpeedBackwardAccel:
            distributed_type = DistributedType.DEEPSPEED

            def __init__(self, marker: dict[str, bool]) -> None:
                self._marker = marker
                self.backward_calls = 0

            def backward(self, _loss: torch.Tensor) -> None:
                self.backward_calls += 1
                assert self._marker["active"]

        class _FSDP2BackwardAccel:
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
                assert self._marker["active"]

        for accelerator_cls, zero3 in [
            (_DeepSpeedBackwardAccel, True),
            (_FSDP2BackwardAccel, False),
        ]:
            model = _ModelStub()
            if zero3:
                setattr(model.decoder.weight, "ds_id", 7)
            marker = {"active": False}
            accelerator = accelerator_cls(marker)
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

        class _NoDistAccelerator:
            distributed_type = DistributedType.NO

            def __init__(self) -> None:
                self.backward_calls = 0

            def backward(self, _loss: torch.Tensor) -> None:
                self.backward_calls += 1

        model = _ModelStub()
        accelerator = _NoDistAccelerator()
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

    def test_compute_weight_norm_for_logging_deepspeed(self):
        """Ensure DeepSpeed norm logging handles mapped and unmapped params."""

        class _AcceleratorStub:
            distributed_type = DistributedType.DEEPSPEED

        model = torch.nn.Linear(3, 2, bias=True)
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

        with patch(
            "neobert.pretraining.trainer.safe_get_full_fp32_param",
            return_value=None,
        ):
            norm_none = _compute_weight_norm_for_logging(model, _AcceleratorStub())
        assert norm_none is None

    def test_resolve_loader_perf_settings_cuda_and_cpu(self):
        """Ensure loader perf settings differ appropriately across CUDA and CPU."""
        cuda_cfg = Config()
        cuda_cfg.dataset.num_workers = 4
        cuda_cfg.dataset.pin_memory = False
        cuda_cfg.dataset.persistent_workers = True
        cuda_cfg.dataset.prefetch_factor = None
        pin_memory, persistent_workers, prefetch_factor, notes = (
            _resolve_loader_perf_settings(cuda_cfg, device=torch.device("cuda"))
        )
        assert pin_memory
        assert persistent_workers
        assert prefetch_factor == 4
        assert len(notes) > 0

        cpu_cfg = Config()
        cpu_cfg.dataset.num_workers = 3
        cpu_cfg.dataset.pin_memory = False
        cpu_cfg.dataset.persistent_workers = True
        cpu_cfg.dataset.prefetch_factor = 2
        pin_memory, persistent_workers, prefetch_factor, notes = (
            _resolve_loader_perf_settings(cpu_cfg, device=torch.device("cpu"))
        )
        assert not pin_memory
        assert persistent_workers
        assert prefetch_factor == 2
        assert notes == []

    def test_resolve_tokenize_num_proc_falls_back_to_cpu_count(self):
        """Ensure tokenization num_proc falls back when affinity is unavailable."""
        from neobert.pretraining.trainer import _resolve_tokenize_num_proc

        with patch("os.sched_getaffinity", side_effect=AttributeError, create=True):
            with patch("os.cpu_count", return_value=8):
                resolved = _resolve_tokenize_num_proc(
                    requested=None, num_processes=1, is_main_process=True
                )

        assert resolved == 8

    def test_eval_budget_resolution_helpers(self):
        """Ensure eval max-batch/sample budget helpers normalize consistently."""
        from neobert.pretraining.trainer import _resolve_eval_max_batches

        assert _resolve_eval_max_batches(None, num_processes=2) is None
        assert _resolve_eval_max_batches(0, num_processes=2) is None
        assert _resolve_eval_max_batches(10, num_processes=1) == 10
        assert _resolve_eval_max_batches(10, num_processes=2) == 5
        assert _resolve_eval_max_batches(11, num_processes=2) == 6

        assert _resolve_eval_samples(128) == 128
        assert _resolve_eval_samples("256") == 256
        assert _resolve_eval_samples(None) is None
        assert _resolve_eval_samples(0) is None
        assert _resolve_eval_samples(-5) is None
        with pytest.raises(ValueError):
            _resolve_eval_samples(True)
        with pytest.raises(ValueError):
            _resolve_eval_samples("not_an_int")

        valid_cases = [
            (
                {"eval_max_batches": 320, "eval_samples": None, "batch_size": 32},
                (320, "trainer.eval_max_batches"),
            ),
            (
                {"eval_max_batches": None, "eval_samples": 1000, "batch_size": 64},
                (16, "dataset.eval_samples"),
            ),
        ]
        for case, expected in valid_cases:
            resolved, source = _resolve_streaming_eval_budget(
                eval_max_batches=case["eval_max_batches"],
                eval_samples=case["eval_samples"],
                per_device_eval_batch_size=case["batch_size"],
            )
            assert (resolved, source) == expected

        with pytest.raises(ValueError):
            _resolve_streaming_eval_budget(
                eval_max_batches=None,
                eval_samples=None,
                per_device_eval_batch_size=32,
            )

    def test_eval_split_selection_helpers(self):
        """Ensure eval split discovery and train/eval partition helpers stay stable."""
        from neobert.pretraining.trainer import _select_train_split

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

        class _StreamingStub:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def take(self, n: int):
                self.calls.append(("take", n))
                return ("eval", n)

            def skip(self, n: int):
                self.calls.append(("skip", n))
                return ("train", n)

        stream_dataset = _StreamingStub()
        remaining_train, eval_dataset = _split_train_dataset_for_eval_samples(
            stream_dataset, eval_samples=321, is_streaming=True
        )
        assert eval_dataset == ("eval", 321)
        assert remaining_train == ("train", 321)
        assert stream_dataset.calls == [("take", 321), ("skip", 321)]

        dataset = Dataset.from_dict({"text": ["a", "b"]})
        dataset_dict = DatasetDict(train=dataset, validation=dataset)
        resolved = _select_train_split(dataset_dict, None)
        assert isinstance(resolved, Dataset)

    def test_masked_correct_count(self):
        """Test masked accuracy count with mixed and all-ignored labels."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        assert _count_masked_correct(logits, torch.tensor([[2, -100]])).item() == 1
        assert (
            _count_masked_correct(
                logits, torch.full((1, 2), -100, dtype=torch.long)
            ).item()
            == 0
        )

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

    def test_optimizer_and_scheduler_creation_with_decay_name_normalization(self):
        """Ensure optimizer grouping and scheduler decay-name normalization both work."""
        config = ConfigLoader.load(
            Path(__file__).parent.parent
            / "configs"
            / "pretraining"
            / "test_tiny_pretrain.yaml"
        )
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from neobert.model import NeoBERT, NeoBERTConfig
        from neobert.optimizer import get_optimizer
        from neobert.scheduler import get_scheduler, resolve_scheduler_steps

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
        assert optimizer is not None
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

        mixed_case_optimizer = AdamW(model.parameters(), lr=1e-4)

        scheduler = get_scheduler(
            optimizer=mixed_case_optimizer,
            lr=1e-4,
            decay="CoSiNe",
            warmup_steps=0,
            decay_steps=10,
            constant_steps=0,
        )

        assert any(isinstance(s, CosineAnnealingLR) for s in scheduler._schedulers)

    def test_scheduler_accepts_optimizers_with_function_step(self):
        """Ensure scheduler factory normalizes function-style optimizer.step."""
        from torch.optim import Optimizer

        from neobert.scheduler import get_scheduler

        param = torch.nn.Parameter(torch.tensor([1.0]))

        class _FunctionStepOptimizer(Optimizer):
            def __init__(self, params):
                super().__init__(params, defaults={"lr": 1e-3})

                def _plain_step(*args, **kwargs):
                    del args, kwargs
                    return None

                self.step = _plain_step

            def zero_grad(self, set_to_none: bool = False):
                del set_to_none
                return None

        optimizer = _FunctionStepOptimizer([param])
        assert callable(optimizer.step)
        assert not hasattr(optimizer.step, "__func__")

        scheduler = get_scheduler(
            optimizer=optimizer,
            lr=1e-3,
            decay="linear",
            warmup_steps=0,
            decay_steps=10,
            constant_steps=0,
        )

        assert scheduler is not None
        assert hasattr(optimizer.step, "__func__")
