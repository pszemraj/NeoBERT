"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.training_utils import _maybe_compile_model


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
