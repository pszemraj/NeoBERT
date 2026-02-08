#!/usr/bin/env python3
"""Test pretraining pipeline functionality."""

import os
import tempfile
import unittest
import warnings
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path

import torch
from accelerate.utils import DistributedType
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neobert.config import Config, ConfigLoader
from neobert.pretraining.masked_objective import MaskedObjectiveOut
from neobert.pretraining.trainer import (
    _ensure_pinned_cpu_batch,
    _gather_decoder_weight_for_masked_objective,
    _resolve_loader_perf_settings,
    _run_masked_objective_step,
    _should_backward_inside_gathered_decoder_weight,
    _sync_tokenizer_derived_config,
    _write_deepspeed_latest_file,
    trainer,
)


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

    def test_pretraining_rejects_fp16(self):
        """Ensure pretraining trainer rejects fp16 mixed precision."""
        config = ConfigLoader.load(str(self.test_config_path))
        config.trainer.output_dir = self.temp_dir
        config.trainer.mixed_precision = "fp16"

        with self.assertRaisesRegex(ValueError, "fp16"):
            trainer(config)

    def test_pretraining_rejects_invalid_masked_logits_only_loss(self):
        """Ensure invalid loss-path config fails before tokenizer/network setup."""
        config = ConfigLoader.load(str(self.test_config_path))
        config.trainer.output_dir = self.temp_dir
        config.trainer.masked_logits_only_loss = "something_else"

        with patch("neobert.pretraining.trainer.get_tokenizer") as mocked_tokenizer:
            with self.assertRaisesRegex(ValueError, "masked_logits_only_loss"):
                trainer(config)
            mocked_tokenizer.assert_not_called()

    def test_pretraining_rejects_fsdp1_before_tokenizer_setup(self):
        """Ensure FSDP1 fails fast before tokenizer/dataset initialization."""
        config = ConfigLoader.load(str(self.test_config_path))
        config.trainer.output_dir = self.temp_dir

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
                with self.assertRaisesRegex(RuntimeError, "FSDP2-first"):
                    trainer(config)
                mocked_tokenizer.assert_not_called()

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
            attn_backend=config.model.attn_backend,
            ngpt=config.model.ngpt,
            hidden_act="gelu",  # Use GELU to avoid flash_attn requirement
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
            self.skipTest(f"pin_memory not supported in this environment: {exc}")
            return

        self.assertTrue(out["input_ids"].is_pinned())
        self.assertTrue(out["labels"].is_pinned())
        self.assertEqual(out["meta"], batch["meta"])

        out_again = _ensure_pinned_cpu_batch(out)
        self.assertIs(out_again, out)

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
            self.skipTest(f"pin_memory not supported in this environment: {exc}")
            return

        self.assertTrue(out["input_ids"].is_pinned())
        self.assertTrue(out["nested"]["labels"].is_pinned())
        self.assertTrue(out["nested"]["meta"][1].is_pinned())

    def test_sync_tokenizer_derived_config_pads_vocab_and_pad_id(self):
        """Ensure config is synchronized with tokenizer-derived vocab/pad fields."""
        cfg = Config()
        cfg.model.vocab_size = 17
        cfg.tokenizer.vocab_size = 17
        tokenizer = self._make_tokenizer()

        original, resolved, added = _sync_tokenizer_derived_config(cfg, tokenizer)

        self.assertEqual(original, 8)
        self.assertEqual(resolved, 128)
        self.assertEqual(added, 120)
        self.assertEqual(len(tokenizer), 128)
        self.assertEqual(cfg.model.vocab_size, 128)
        self.assertEqual(cfg.tokenizer.vocab_size, 128)
        self.assertEqual(cfg.model.pad_token_id, tokenizer.pad_token_id)

    def test_write_deepspeed_latest_file(self):
        """Ensure DeepSpeed root latest indirection is refreshed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_deepspeed_latest_file(root, "12345")
            latest_path = root / "latest"
            self.assertTrue(latest_path.exists())
            self.assertEqual(latest_path.read_text(encoding="utf-8"), "12345\n")

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
                self.assertIs(lm_weight, model.decoder.weight)
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
                self.assertIs(lm_weight, model.decoder.weight)

        gathered.assert_called_once()
        self.assertEqual(gathered.call_args.args[0], [model.decoder.weight])
        self.assertIsNone(gathered.call_args.kwargs["modifier_rank"])
        self.assertIs(gathered.call_args.kwargs["fwd_module"], model)

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

        with self.assertRaisesRegex(RuntimeError, "FSDP v1"):
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
                self.assertIs(lm_weight, model.decoder.weight)
        summon_full_params.assert_not_called()

        self.assertEqual(
            calls,
            [
                ("unshard", True),
                ("wait", None),
                ("reshard", None),
            ],
        )

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

        self.assertTrue(backward_done)
        self.assertEqual(accelerator.backward_calls, 1)

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

        self.assertTrue(backward_done)
        self.assertEqual(accelerator.backward_calls, 1)

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
        self.assertFalse(
            _should_backward_inside_gathered_decoder_weight(_FSDP1Accel(), lm_weight)
        )
        self.assertTrue(
            _should_backward_inside_gathered_decoder_weight(_FSDP2Accel(), lm_weight)
        )

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
        self.assertFalse(backward_done)
        self.assertEqual(accelerator.backward_calls, 0)

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
        self.assertTrue(pin_memory)
        self.assertTrue(persistent_workers)
        self.assertEqual(prefetch_factor, 4)
        self.assertGreater(len(notes), 0)

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
        self.assertFalse(pin_memory)
        self.assertTrue(persistent_workers)
        self.assertEqual(prefetch_factor, 2)
        self.assertEqual(notes, [])

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
        self.assertTrue(torch.is_tensor(packed))

        attention_mask = collated["attention_mask"]
        keep = torch.isfinite(attention_mask) & (attention_mask == 0)
        lengths = keep.sum(dim=1, keepdim=True).to(torch.int32)
        self.assertTrue(torch.equal(packed, lengths))

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
        self.assertTrue(
            any("Skipping packed_seqlens" in str(w.message) for w in caught)
        )
        self.assertIn("attention_mask", collated)
        self.assertNotIn("packed_seqlens", collated)

    def test_resolve_tokenize_num_proc_falls_back_to_cpu_count(self):
        """Ensure tokenization num_proc falls back when affinity is unavailable."""
        from neobert.pretraining.trainer import _resolve_tokenize_num_proc

        with patch("os.sched_getaffinity", side_effect=AttributeError, create=True):
            with patch("os.cpu_count", return_value=8):
                resolved = _resolve_tokenize_num_proc(
                    requested=None, num_processes=1, is_main_process=True
                )

        self.assertEqual(resolved, 8)

    def test_resolve_eval_max_batches_per_rank(self):
        """Ensure eval max_batches is split across ranks."""
        from neobert.pretraining.trainer import _resolve_eval_max_batches

        self.assertIsNone(_resolve_eval_max_batches(None, num_processes=2))
        self.assertIsNone(_resolve_eval_max_batches(0, num_processes=2))
        self.assertEqual(_resolve_eval_max_batches(10, num_processes=1), 10)
        self.assertEqual(_resolve_eval_max_batches(10, num_processes=2), 5)
        self.assertEqual(_resolve_eval_max_batches(11, num_processes=2), 6)

    def test_select_train_split_from_datasetdict(self):
        """Ensure DatasetDict splits resolve to a Dataset."""
        from neobert.pretraining.trainer import _select_train_split

        dataset = Dataset.from_dict({"text": ["a", "b"]})
        dataset_dict = DatasetDict(train=dataset, validation=dataset)

        resolved = _select_train_split(dataset_dict, None)

        self.assertIsInstance(resolved, Dataset)

    def test_masked_correct_count(self):
        """Test masked accuracy counting ignores -100 labels."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        labels = torch.tensor([[2, -100]])
        self.assertEqual(_count_masked_correct(logits, labels).item(), 1)

    def test_masked_correct_count_all_ignored(self):
        """Test masked accuracy count returns zero when all labels are ignored."""
        from neobert.pretraining.trainer import _count_masked_correct

        logits = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]]])
        labels = torch.full((1, 2), -100, dtype=torch.long)
        self.assertEqual(_count_masked_correct(logits, labels).item(), 0)

    def test_set_default_worker_env_respects_user_overrides(self):
        """Ensure worker env defaults only apply when vars are unset."""
        from neobert.pretraining.trainer import _set_default_worker_env

        with patch.dict(os.environ, {"OMP_NUM_THREADS": "8"}, clear=False):
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
            os.environ.pop("MKL_NUM_THREADS", None)
            _set_default_worker_env(num_workers=4)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "8")
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")

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

    def test_normalize_packed_seqlens_tensor(self):
        """Ensure packed_seqlens tensors normalize to CPU int32 tensors."""
        from neobert.model.model import _normalize_packed_seqlens

        packed = torch.tensor([[3, 0, 0], [2, 1, 0]], dtype=torch.int32)
        normalized = _normalize_packed_seqlens(packed)
        self.assertTrue(torch.is_tensor(normalized))
        self.assertEqual(normalized.dtype, torch.int32)
        self.assertEqual(normalized.device.type, "cpu")
        self.assertTrue(
            torch.equal(
                normalized,
                torch.tensor([[3, 0, 0], [2, 1, 0]], dtype=torch.int32),
            )
        )

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

    def test_to_target_batch_size_handles_none_packed_seqlens(self):
        """Ensure None packed_seqlens are ignored when resizing batches."""
        from neobert.pretraining.trainer import to_target_batch_size

        batch = {
            "input_ids": torch.zeros((3, 4), dtype=torch.long),
            "attention_mask": torch.ones((3, 4), dtype=torch.long),
            "labels": torch.zeros((3, 4), dtype=torch.long),
            "packed_seqlens": None,
        }
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seqlens": None,
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=2)
        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertIsNone(out["packed_seqlens"])
        self.assertIsNone(stored["packed_seqlens"])

    def test_to_target_batch_size_handles_tensor_packed_seqlens(self):
        """Ensure packed_seqlens tensors split and buffer with batch resizing."""
        from neobert.pretraining.trainer import to_target_batch_size

        batch = {
            "input_ids": torch.zeros((3, 4), dtype=torch.long),
            "attention_mask": torch.ones((3, 4), dtype=torch.long),
            "labels": torch.zeros((3, 4), dtype=torch.long),
            "packed_seqlens": torch.tensor(
                [[2, 2, 0], [3, 1, 0], [4, 0, 0]], dtype=torch.int32
            ),
        }
        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seqlens": None,
        }

        out, stored = to_target_batch_size(batch, stored_batch, target_size=2)
        self.assertEqual(out["input_ids"].shape[0], 2)
        self.assertTrue(torch.is_tensor(out["packed_seqlens"]))
        self.assertEqual(tuple(out["packed_seqlens"].shape), (2, 3))
        self.assertTrue(
            torch.equal(
                out["packed_seqlens"][0], torch.tensor([2, 2, 0], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(
                out["packed_seqlens"][1], torch.tensor([3, 1, 0], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.is_tensor(stored["packed_seqlens"]))
        self.assertEqual(tuple(stored["packed_seqlens"].shape), (1, 3))
        self.assertTrue(
            torch.equal(
                stored["packed_seqlens"][0], torch.tensor([4, 0, 0], dtype=torch.int32)
            )
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
        self.assertTrue(all(value is None for value in stored_batch.values()))

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

            self.assertFalse(tmp_ckpt.exists())
            self.assertTrue(final_ckpt.exists())
            self.assertEqual((final_ckpt / "model.safetensors").read_text(), "new")
            self.assertFalse((root / "100.old").exists())

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

        self.assertIsNotNone(scheduler)

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

        self.assertTrue(
            any(isinstance(s, CosineAnnealingLR) for s in scheduler._schedulers)
        )


if __name__ == "__main__":
    unittest.main()
