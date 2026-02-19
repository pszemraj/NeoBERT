#!/usr/bin/env python3
"""Tests for FP8 pretraining accelerator wiring."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
from accelerate.utils import DistributedType

from neobert.config import ConfigLoader
from neobert.pretraining.trainer import trainer


class _StopAfterWiring(RuntimeError):
    """Internal sentinel used to stop trainer after accelerator wiring."""


class _FSDPPluginStub:
    fsdp_version = 2
    cpu_ram_efficient_loading = False


class _FSDPStateStub:
    fsdp_plugin = _FSDPPluginStub()


class _AcceleratorStub:
    """Minimal accelerator stub for wiring-level tests."""

    def __init__(self, distributed_type: DistributedType, fp8_backend: str = ""):
        self.distributed_type = distributed_type
        self.state = _FSDPStateStub() if distributed_type is DistributedType.FSDP else object()
        self.fp8_backend = fp8_backend
        self.is_main_process = False
        self.num_processes = 1

    @staticmethod
    def register_for_checkpointing(_obj) -> None:
        return None

    @staticmethod
    def print(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def main_process_first():
        return nullcontext()


def test_fp8_enabled_passes_ao_handler_and_dynamo_plugin(
    tiny_pretrain_config_path: Path,
    temp_output_dir: str,
) -> None:
    """FP8 path should pass AO handler + dynamo plugin to create_accelerator."""
    cfg = ConfigLoader.load(str(tiny_pretrain_config_path))
    cfg.trainer.output_dir = temp_output_dir
    cfg.trainer.mixed_precision = "fp8"
    cfg.trainer.torch_compile = True
    cfg.wandb.enabled = False

    sentinel_ao = object()
    sentinel_dynamo = object()

    class _FilterStub:
        @staticmethod
        def bind_model(_model) -> None:
            return None

    captured: dict[str, object] = {}

    def _fake_create_accelerator(**kwargs):
        captured.update(kwargs)
        return _AcceleratorStub(DistributedType.FSDP, fp8_backend="AO")

    with patch(
        "neobert.pretraining.trainer._build_fp8_accelerator_components",
        return_value=(sentinel_ao, sentinel_dynamo, _FilterStub()),
    ):
        with patch(
            "neobert.pretraining.trainer.create_accelerator",
            side_effect=_fake_create_accelerator,
        ):
            with patch(
                "neobert.pretraining.trainer.get_tokenizer",
                side_effect=_StopAfterWiring("stop after wiring"),
            ):
                with pytest.raises(_StopAfterWiring):
                    trainer(cfg)

    assert captured.get("dynamo_plugin") is sentinel_dynamo
    kwargs_handlers = captured.get("kwargs_handlers")
    assert isinstance(kwargs_handlers, list)
    assert len(kwargs_handlers) == 2
    assert sentinel_ao in kwargs_handlers


def test_fp8_disabled_keeps_legacy_accelerator_wiring(
    tiny_pretrain_config_path: Path,
    temp_output_dir: str,
) -> None:
    """Non-FP8 path should keep existing create_accelerator signature."""
    cfg = ConfigLoader.load(str(tiny_pretrain_config_path))
    cfg.trainer.output_dir = temp_output_dir
    cfg.trainer.mixed_precision = "bf16"
    cfg.wandb.enabled = False

    captured: dict[str, object] = {}

    def _fake_create_accelerator(**kwargs):
        captured.update(kwargs)
        return _AcceleratorStub(DistributedType.NO)

    with patch(
        "neobert.pretraining.trainer._build_fp8_accelerator_components",
        side_effect=AssertionError("FP8 helper should not be called when disabled"),
    ):
        with patch(
            "neobert.pretraining.trainer.create_accelerator",
            side_effect=_fake_create_accelerator,
        ):
            with patch(
                "neobert.pretraining.trainer.get_tokenizer",
                side_effect=_StopAfterWiring("stop after wiring"),
            ):
                with pytest.raises(_StopAfterWiring):
                    trainer(cfg)

    assert "dynamo_plugin" not in captured
    kwargs_handlers = captured.get("kwargs_handlers")
    assert isinstance(kwargs_handlers, list)
    assert len(kwargs_handlers) == 1
