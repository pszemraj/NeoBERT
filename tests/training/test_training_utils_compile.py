"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.training_utils import (
    _maybe_compile_model,
    resolve_wandb_watch_mode,
    stabilize_cuda_mixed_precision,
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


def test_resolve_wandb_watch_mode_matrix() -> None:
    """Ensure WANDB watch-mode defaults and env/config override rules stay stable."""
    cases = [
        ("online", "gradients", None, "gradients", False),
        ("offline", "gradients", None, None, False),
        ("online", "parameters", None, "parameters", False),
        ("online", "gradients", "all", "all", False),
        ("online", "gradients", "weights", "parameters", False),
        ("online", "gradients", "off", None, False),
        ("online", "gradients", "bad", None, True),
    ]
    for wandb_mode, config_value, env_value, expected_mode, expect_warning in cases:
        mode, warning = resolve_wandb_watch_mode(
            wandb_mode=wandb_mode,
            config_value=config_value,
            env_value=env_value,
        )
        assert mode == expected_mode
        if expect_warning:
            assert warning is not None
        else:
            assert warning is None


def test_stabilize_cuda_mixed_precision_passthrough_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-CUDA runtimes must keep the configured mixed precision unchanged."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = stabilize_cuda_mixed_precision(
        mixed_precision="bf16",
        log=logging.getLogger("test"),
    )
    assert out == "bf16"


def test_stabilize_cuda_mixed_precision_switches_to_cublaslt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When default bf16 probe fails, helper should switch to cuBLASLt."""
    import neobert.training_utils as training_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    probe_results = iter([False, True])
    monkeypatch.setattr(
        training_utils,
        "_probe_cuda_linear_dtype",
        lambda _dtype: next(probe_results),
    )

    backend_state = {"name": "cublas"}

    def _preferred_blas_library(
        requested: str | None = None,
    ) -> object:
        if requested is None:
            return backend_state["name"]
        backend_state["name"] = str(requested).lower()
        return backend_state["name"]

    monkeypatch.setattr(
        torch.backends.cuda, "preferred_blas_library", _preferred_blas_library
    )

    with caplog.at_level(logging.WARNING):
        out = stabilize_cuda_mixed_precision(
            mixed_precision="bf16",
            log=logging.getLogger("test"),
        )

    assert out == "bf16"
    assert backend_state["name"] == "cublaslt"
    assert "switched torch.backends.cuda.preferred_blas_library('cublaslt')" in (
        caplog.text
    )


def test_stabilize_cuda_mixed_precision_falls_back_to_fp32(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If bf16 probe still fails after switch, helper must disable bf16."""
    import neobert.training_utils as training_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_utils, "_probe_cuda_linear_dtype", lambda _dtype: False
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "preferred_blas_library",
        lambda _requested=None: "cublaslt",
    )

    with caplog.at_level(logging.WARNING):
        out = stabilize_cuda_mixed_precision(
            mixed_precision="bf16",
            log=logging.getLogger("test"),
        )

    assert out == "no"
    assert "falling back to mixed_precision='no'" in caplog.text


def test_validate_muon_distributed_compatibility_rejects_fsdp1() -> None:
    """MuonClip must fail fast when FSDP v1 is active."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=1)),
    )
    with pytest.raises(RuntimeError, match="requires FSDP v2"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_validate_muon_distributed_compatibility_allows_fsdp2() -> None:
    """MuonClip should allow FSDP2 runtime."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
    )
    validate_muon_distributed_compatibility(
        accelerator=accelerator,
        optimizer_name="muonclip",
        log=logging.getLogger("test"),
        context="unit-test",
    )


def test_validate_muon_distributed_compatibility_rejects_unknown_fsdp() -> None:
    """Unknown FSDP version metadata should default to v1-style rejection."""
    accelerator = SimpleNamespace(distributed_type=DistributedType.FSDP)
    with pytest.raises(RuntimeError, match="requires FSDP v2"):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_get_optimizer_disables_muonclip_clipping_under_fsdp(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """FSDP MuonClip builds must force clipping off and emit a warning once."""
    import neobert.optimizer.optimizer as optimizer_module

    monkeypatch.setattr(
        optimizer_module, "_WARNED_MUONCLIP_FSDP_CLIPPING_DISABLE", False
    )
    model_cfg = NeoBERTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_length=32,
        attn_backend="sdpa",
        hidden_act="gelu",
        rope=False,
    )
    model = NeoBERT(model_cfg)

    with caplog.at_level(logging.WARNING):
        optimizer = get_optimizer(
            model,
            DistributedType.FSDP,
            model_config=model_cfg,
            name="muonclip",
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_config={"enable_clipping": True},
        )

    assert hasattr(optimizer, "config")
    assert not optimizer.config.enable_clipping
    assert "Auto-disabling clipping" in caplog.text
