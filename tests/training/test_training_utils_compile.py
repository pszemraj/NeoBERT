"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.training_utils import (
    _compute_l2_norm_for_logging,
    _maybe_compile_model,
    create_accelerator,
    resolve_runtime_mixed_precision_and_attn_backend,
    resolve_wandb_watch_mode,
    stabilize_cuda_mixed_precision,
    validate_distributed_runtime_policy,
    validate_muon_distributed_compatibility,
    validate_muon_runtime_topology,
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


def test_probe_cuda_linear_dtype_uses_local_rank_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe must use rank-local CUDA device before Accelerator initialization."""
    import neobert.training_utils as training_utils

    training_utils._LOW_PRECISION_LINEAR_PROBE_CACHE.clear()
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.backends.cuda, "preferred_blas_library", lambda _requested=None: "cublas"
    )

    seen_devices: list[str] = []
    sync_devices: list[str] = []

    def _fake_randn(
        shape: tuple[int, int], *, device: object = None, dtype: object = None
    ) -> torch.Tensor:
        seen_devices.append(str(torch.device(device)))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _fake_randn)
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda x, w: torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: sync_devices.append(str(torch.device(device))),
    )

    assert training_utils._probe_cuda_linear_dtype(torch.bfloat16)
    assert seen_devices == ["cuda:2", "cuda:2"]
    assert sync_devices == ["cuda:2"]


def test_probe_cuda_linear_dtype_falls_back_to_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If local-rank metadata is invalid, probe should use current CUDA device."""
    import neobert.training_utils as training_utils

    training_utils._LOW_PRECISION_LINEAR_PROBE_CACHE.clear()
    monkeypatch.setenv("LOCAL_RANK", "99")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.backends.cuda, "preferred_blas_library", lambda _requested=None: "cublas"
    )

    seen_devices: list[str] = []

    def _fake_randn(
        shape: tuple[int, int], *, device: object = None, dtype: object = None
    ) -> torch.Tensor:
        seen_devices.append(str(torch.device(device)))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(torch, "randn", _fake_randn)
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda x, w: torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)

    assert training_utils._probe_cuda_linear_dtype(torch.bfloat16)
    assert seen_devices == ["cuda:1", "cuda:1"]


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


def test_create_accelerator_recreates_state_for_mixed_precision_reuse() -> None:
    """Sequential trainer runs should honor updated mixed precision settings."""
    GradientState._reset_state()
    AcceleratorState._reset_state(reset_partial_state=True)

    try:
        first = create_accelerator(
            use_cpu=True,
            log=logging.getLogger("test"),
            mixed_precision="bf16",
        )
        assert first.device.type == "cpu"
        assert first.state.mixed_precision == "bf16"

        second = create_accelerator(
            use_cpu=True,
            log=logging.getLogger("test"),
            mixed_precision="no",
        )
        assert second.device.type == "cpu"
        assert second.state.mixed_precision == "no"
    finally:
        GradientState._reset_state()
        AcceleratorState._reset_state(reset_partial_state=True)


def test_create_accelerator_resets_on_state_mismatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State-mismatch errors should reset Accelerate and retry once."""
    import neobert.training_utils as training_utils

    reset_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        training_utils.AcceleratorState,
        "_reset_state",
        lambda reset_partial_state=False: reset_calls.append(
            ("accelerator", bool(reset_partial_state))
        ),
    )
    monkeypatch.setattr(
        training_utils.GradientState,
        "_reset_state",
        lambda: reset_calls.append(("gradient", None)),
    )

    calls: list[dict[str, object]] = []

    def _fake_factory(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError(
                "AcceleratorState has already been initialized and cannot be "
                "changed, restart your runtime completely and pass `cpu=True` "
                "to `Accelerator()`."
            )
        return SimpleNamespace(**kwargs)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=_fake_factory,
        mixed_precision="bf16",
    )

    assert out.cpu is True
    assert out.mixed_precision == "bf16"
    assert calls == [
        {"mixed_precision": "bf16", "cpu": True},
        {"mixed_precision": "bf16", "cpu": True},
    ]
    assert reset_calls == [("gradient", None), ("accelerator", True)]


def test_create_accelerator_reraises_unrelated_value_errors() -> None:
    """Non-state errors from Accelerator construction must propagate unchanged."""

    def _boom(**_: object) -> SimpleNamespace:
        raise ValueError("different accelerator failure")

    with pytest.raises(ValueError, match="different accelerator failure"):
        create_accelerator(
            use_cpu=False,
            log=logging.getLogger("test"),
            accelerator_factory=_boom,
            mixed_precision="bf16",
        )


def test_stabilize_cuda_mixed_precision_skips_probe_for_explicit_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-targeted runs should not touch CUDA probe paths on GPU hosts."""
    called = {"probe": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(_dtype: torch.dtype) -> bool:
        called["probe"] += 1
        raise AssertionError("CUDA probe should not run for trainer.use_cpu=true")

    monkeypatch.setattr("neobert.training_utils._probe_cuda_linear_dtype", _boom)

    out = stabilize_cuda_mixed_precision(
        mixed_precision="bf16",
        log=logging.getLogger("test"),
        use_cpu=True,
    )

    assert out == "bf16"
    assert called["probe"] == 0


def test_resolve_runtime_mixed_precision_and_attn_backend_forces_sdpa_on_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Explicit CPU runs must disable flash-attn even when CUDA is present."""
    called = {"probe": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(_dtype: torch.dtype) -> bool:
        called["probe"] += 1
        raise AssertionError("CUDA probe should not run for trainer.use_cpu=true")

    monkeypatch.setattr("neobert.training_utils._probe_cuda_linear_dtype", _boom)

    with caplog.at_level(logging.WARNING):
        mixed_precision, attn_backend = (
            resolve_runtime_mixed_precision_and_attn_backend(
                mixed_precision="bf16",
                attn_backend="flash_attn_varlen",
                log=logging.getLogger("test"),
                use_cpu=True,
            )
        )

    assert mixed_precision == "bf16"
    assert attn_backend == "sdpa"
    assert called["probe"] == 0
    assert "trainer.use_cpu=true" in caplog.text


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


def test_validate_muon_distributed_compatibility_rejects_fsdp2_tp_mesh() -> None:
    """MuonClip should fail fast when FSDP2 is combined with extra mesh axes."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(fsdp_version=2),
            parallelism_config=SimpleNamespace(tp_enabled=True, cp_enabled=False),
        ),
    )
    with pytest.raises(RuntimeError, match="1D row-sharded device mesh"):
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


@pytest.mark.parametrize("zero_stage", [None, 0, 1, 2, 3])
def test_validate_muon_distributed_compatibility_rejects_deepspeed(
    zero_stage: int | None,
) -> None:
    """MuonClip should reject all DeepSpeed runtimes, not just ZeRO-2/3."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage),
        ),
    )

    match = "FSDP2-only" if zero_stage is None else f"ZeRO stage {zero_stage}"
    with pytest.raises(RuntimeError, match=match):
        validate_muon_distributed_compatibility(
            accelerator=accelerator,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


@pytest.mark.parametrize("zero_stage", [None, 0, 1, 2, 3])
def test_validate_distributed_runtime_policy_rejects_deepspeed(
    zero_stage: int | None,
) -> None:
    """Repo runtime policy should reject DeepSpeed regardless of optimizer."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(zero_stage=zero_stage),
        ),
    )

    match = "unsupported" if zero_stage is None else f"ZeRO stage {zero_stage}"
    with pytest.raises(RuntimeError, match=match):
        validate_distributed_runtime_policy(
            accelerator=accelerator,
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_validate_muon_runtime_topology_rejects_multidim_mesh() -> None:
    """Prepared MuonClip DTensor params must reject unsupported mesh rank."""

    class _FakeShard:
        def __init__(self, dim: int):
            self.dim = dim

    class _FakeMesh:
        ndim = 2

    class _FakeDTensorParam:
        device_mesh = _FakeMesh()
        placements = (_FakeShard(0),)

        def to_local(self) -> torch.Tensor:
            return torch.zeros(1, 1)

    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = SimpleNamespace(
        param_groups=[{"use_muon": True, "params": [_FakeDTensorParam()]}]
    )

    with pytest.raises(RuntimeError, match="device_mesh.ndim=2"):
        validate_muon_runtime_topology(
            accelerator=accelerator,
            optimizer=optimizer,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_validate_muon_runtime_topology_accepts_row_shard_layout() -> None:
    """Prepared MuonClip DTensor params should allow 1D Shard(0) layouts."""

    class _FakeShard:
        def __init__(self, dim: int):
            self.dim = dim

    class _FakeMesh:
        ndim = 1

    class _FakeDTensorParam:
        device_mesh = _FakeMesh()
        placements = (_FakeShard(0),)

        def to_local(self) -> torch.Tensor:
            return torch.zeros(1, 1)

    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = SimpleNamespace(
        param_groups=[{"use_muon": True, "params": [_FakeDTensorParam()]}]
    )

    validate_muon_runtime_topology(
        accelerator=accelerator,
        optimizer=optimizer,
        optimizer_name="muonclip",
        log=logging.getLogger("test"),
        context="unit-test",
    )


def test_validate_muon_runtime_topology_rejects_missing_dtensor_params() -> None:
    """Prepared multi-rank MuonClip runs must not continue without DTensor params."""
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
    )
    optimizer = SimpleNamespace(
        param_groups=[
            {"use_muon": True, "params": [torch.nn.Parameter(torch.zeros(1, 1))]}
        ]
    )

    with pytest.raises(RuntimeError, match="expected DTensor Muon parameters"):
        validate_muon_runtime_topology(
            accelerator=accelerator,
            optimizer=optimizer,
            optimizer_name="muonclip",
            log=logging.getLogger("test"),
            context="unit-test",
        )


def test_compute_l2_norm_for_logging_reduces_only_sharded_dtensors() -> None:
    """Global logged norms must reduce shard contributions without double-counting replicas."""

    class _FakeShard:
        def __init__(self, dim: int):
            self.dim = dim

    class _FakeReplicate:
        pass

    class _FakeDTensor:
        device_mesh = SimpleNamespace(ndim=1)

        def __init__(self, local_value: torch.Tensor, placements: tuple[object, ...]):
            self._local_value = local_value
            self.placements = placements

        def to_local(self) -> torch.Tensor:
            return self._local_value

    reduce_calls: list[tuple[float, str]] = []

    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
        reduce=lambda tensor, reduction="sum": (
            reduce_calls.append((float(tensor.item()), str(reduction))) or tensor * 2
        ),
    )
    parameters = [
        _FakeDTensor(torch.tensor([3.0, 4.0]), (_FakeShard(0),)),
        _FakeDTensor(torch.tensor([1.0, 2.0]), (_FakeReplicate(),)),
    ]

    norm = _compute_l2_norm_for_logging(parameters, accelerator)

    assert norm is not None
    assert math.isclose(norm, math.sqrt(55.0), rel_tol=0.0, abs_tol=1e-8)
    assert reduce_calls == [(25.0, "sum")]


def test_compute_l2_norm_for_logging_uses_dtensor_owner_for_gradients() -> None:
    """Gradient logging must reduce local grads when the owning param is sharded."""

    class _FakeShard:
        def __init__(self, dim: int):
            self.dim = dim

    class _FakeShardedParam:
        device_mesh = SimpleNamespace(ndim=1)
        placements = (_FakeShard(0),)

        def __init__(self, grad: torch.Tensor):
            self.grad = grad

        def to_local(self) -> torch.Tensor:
            return torch.zeros(0)

    reduce_calls: list[tuple[float, str]] = []
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        num_processes=2,
        reduce=lambda tensor, reduction="sum": (
            reduce_calls.append((float(tensor.item()), str(reduction))) or tensor * 2
        ),
    )

    norm = _compute_l2_norm_for_logging(
        [_FakeShardedParam(torch.tensor([6.0, 8.0]))],
        accelerator,
        grad=True,
    )

    assert norm is not None
    assert math.isclose(norm, math.sqrt(200.0), rel_tol=0.0, abs_tol=1e-8)
    assert reduce_calls == [(100.0, "sum")]


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


def test_get_optimizer_rejects_muonclip_under_deepspeed() -> None:
    """Optimizer factory should fail fast on unsupported DeepSpeed MuonClip."""
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

    with pytest.raises(RuntimeError, match="FSDP2-only"):
        get_optimizer(
            model,
            DistributedType.DEEPSPEED,
            model_config=model_cfg,
            name="muonclip",
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_config={"enable_clipping": False},
        )
