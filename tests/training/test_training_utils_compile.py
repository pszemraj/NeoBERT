"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.training_utils import (
    _maybe_compile_model,
    resolve_wandb_watch_mode,
    validate_muon_distributed_compatibility,
)


def _make_cfg() -> Config:
    """Build a minimal config for compile helper tests."""
    cfg = Config()
    cfg.trainer.torch_compile = True
    cfg.trainer.torch_compile_dynamic = None
    cfg.trainer.torch_compile_backend = "inductor"
    cfg.optimizer.name = "adamw"
    cfg.datacollator.pack_sequences = False
    cfg.model.attn_backend = "sdpa"
    return cfg


def _make_accelerator() -> SimpleNamespace:
    """Build a minimal accelerator stub."""
    return SimpleNamespace(distributed_type=DistributedType.NO)


def test_maybe_compile_model_allows_muonclip_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure MuonClip clipping does not block torch.compile."""
    cfg = _make_cfg()
    cfg.optimizer.name = "muonclip"
    model = torch.nn.Linear(8, 8)

    called = {"count": 0}

    def _fake_compile(module: torch.nn.Module, **_: object) -> torch.nn.Module:
        called["count"] += 1
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert called["count"] == 1


def test_maybe_compile_model_uses_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure compile backend from config is forwarded to torch.compile."""
    cfg = _make_cfg()
    cfg.trainer.torch_compile_backend = "aot_eager"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["backend"] == "aot_eager"
    assert captured["dynamic"] is False


def test_maybe_compile_model_invalid_backend_falls_back_to_inductor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unsupported backend names fall back to inductor."""
    cfg = _make_cfg()
    cfg.trainer.torch_compile_backend = "bad_backend"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["backend"] == "inductor"


def test_maybe_compile_model_defaults_dynamic_false_for_packed_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure packed flash-attn still defaults to static compile."""
    cfg = _make_cfg()
    cfg.datacollator.pack_sequences = True
    cfg.model.attn_backend = "flash_attn_varlen"
    model = torch.nn.Linear(8, 8)

    captured: dict[str, object] = {}

    def _fake_compile(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)

    out = _maybe_compile_model(
        model=model,
        cfg=cfg,
        accelerator=_make_accelerator(),
        log=logging.getLogger("test"),
    )

    assert out is model
    assert captured["dynamic"] is False


def test_resolve_wandb_watch_mode_defaults_to_gradients_online() -> None:
    """Default to gradient watching when online and WANDB_WATCH is unset."""
    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="gradients",
        env_value=None,
    )
    assert mode == "gradients"
    assert warning is None


def test_resolve_wandb_watch_mode_disabled_offline() -> None:
    """Do not watch by default for non-online runs."""
    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="offline",
        config_value="gradients",
        env_value=None,
    )
    assert mode is None
    assert warning is None


def test_resolve_wandb_watch_mode_env_override_and_validation() -> None:
    """Honor env overrides and return warnings for unsupported values."""
    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="gradients",
        env_value="all",
    )
    assert mode == "all"
    assert warning is None

    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="gradients",
        env_value="weights",
    )
    assert mode == "parameters"
    assert warning is None

    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="gradients",
        env_value="off",
    )
    assert mode is None
    assert warning is None

    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="gradients",
        env_value="bad",
    )
    assert mode is None
    assert warning is not None


def test_resolve_wandb_watch_mode_uses_config_when_env_missing() -> None:
    """Use config value when WANDB_WATCH is unset."""
    mode, warning = resolve_wandb_watch_mode(
        wandb_mode="online",
        config_value="parameters",
        env_value=None,
    )
    assert mode == "parameters"
    assert warning is None


def test_validate_muon_distributed_compatibility_rejects_fsdp() -> None:
    """MuonClip must fail fast under FSDP-sharded training."""
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP)
    with pytest.raises(RuntimeError, match="not compatible with FSDP"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )
