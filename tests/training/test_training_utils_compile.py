"""Unit tests for torch.compile setup helpers."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DataLoaderConfiguration, DistributedType

from neobert.config import Config, ConfigLoader
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.training_utils import (
    _compute_l2_norm_for_logging,
    _maybe_compile_model,
    _update_global_norm_metric_for_logging,
    attach_optimizer_param_names,
    create_accelerator,
    resolve_runtime_mixed_precision_and_attn_backend,
    resolve_wandb_watch_mode,
    save_optimizer_param_name_manifest,
    sync_resume_source_of_truth,
    validate_optimizer_param_name_manifest,
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


def test_sync_resume_source_of_truth_uses_checkpoint_config(
    tmp_path: Path,
) -> None:
    """Resume should use checkpoint tokenizer/model/objective fields."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)

    checkpoint_cfg = Config()
    checkpoint_cfg.model.hidden_size = 128
    checkpoint_cfg.model.dropout_prob = 0.2
    checkpoint_cfg.tokenizer.name = "checkpoint-tokenizer"
    checkpoint_cfg.tokenizer.max_length = 256
    checkpoint_cfg.dataset.name = "checkpoint-dataset"
    checkpoint_cfg.dataset.path = "checkpoint-data"
    checkpoint_cfg.dataset.max_seq_length = 256
    checkpoint_cfg.datacollator.mlm_probability = 0.3
    checkpoint_cfg.datacollator.pack_sequences = True
    checkpoint_cfg.contrastive.pooling = "max"
    checkpoint_cfg.contrastive.pretraining_prob = 0.4
    ConfigLoader.save(checkpoint_cfg, str(checkpoint_dir / "config.yaml"))

    runtime_cfg = Config()
    runtime_cfg.model.hidden_size = 64
    runtime_cfg.model.dropout_prob = 0.0
    runtime_cfg.tokenizer.name = "runtime-tokenizer"
    runtime_cfg.tokenizer.max_length = 128
    runtime_cfg.dataset.name = "runtime-dataset"
    runtime_cfg.dataset.path = "runtime-data"
    runtime_cfg.dataset.max_seq_length = 128
    runtime_cfg.datacollator.mlm_probability = 0.15
    runtime_cfg.datacollator.pack_sequences = False
    runtime_cfg.contrastive.pooling = "avg"
    runtime_cfg.contrastive.pretraining_prob = 0.0

    sync_resume_source_of_truth(
        runtime_cfg,
        checkpoint_dir,
        task="contrastive",
        log=logging.getLogger("test"),
    )

    assert runtime_cfg.model.hidden_size == 128
    assert runtime_cfg.model.dropout_prob == 0.2
    assert runtime_cfg.tokenizer.path == str(tokenizer_dir)
    assert runtime_cfg.tokenizer.max_length == 256
    assert runtime_cfg.dataset.name == "checkpoint-dataset"
    assert runtime_cfg.dataset.path == "checkpoint-data"
    assert runtime_cfg.dataset.max_seq_length == 256
    assert runtime_cfg.datacollator.mlm_probability == 0.3
    assert runtime_cfg.datacollator.pack_sequences is True
    assert runtime_cfg.contrastive.pooling == "max"
    assert runtime_cfg.contrastive.pretraining_prob == 0.4


def test_sync_resume_source_of_truth_rejects_missing_config(tmp_path: Path) -> None:
    """Resume without checkpoint config must fail before runtime state load."""
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="config.yaml"):
        sync_resume_source_of_truth(
            Config(),
            checkpoint_dir,
            task="pretraining",
            log=logging.getLogger("test"),
        )


def test_optimizer_param_name_manifest_rejects_reordered_groups(
    tmp_path: Path,
) -> None:
    """Same-shaped parameter reordering must not load optimizer state silently."""

    class OrderedPair(torch.nn.Module):
        def __init__(self, reverse: bool = False) -> None:
            super().__init__()
            if reverse:
                self.second = torch.nn.Linear(2, 2, bias=False)
                self.first = torch.nn.Linear(2, 2, bias=False)
            else:
                self.first = torch.nn.Linear(2, 2, bias=False)
                self.second = torch.nn.Linear(2, 2, bias=False)

    model = OrderedPair(reverse=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    attach_optimizer_param_names(model, optimizer)
    save_optimizer_param_name_manifest(optimizer, tmp_path)

    validate_optimizer_param_name_manifest(optimizer, tmp_path)

    reordered_model = OrderedPair(reverse=True)
    reordered_optimizer = torch.optim.AdamW(reordered_model.parameters(), lr=1e-3)
    attach_optimizer_param_names(reordered_model, reordered_optimizer)
    with pytest.raises(RuntimeError, match="parameter order changed"):
        validate_optimizer_param_name_manifest(reordered_optimizer, tmp_path)


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


def test_create_accelerator_resets_when_cpu_request_reuses_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU requests should not silently reuse stale CUDA accelerator state."""
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
        device = torch.device("cuda" if len(calls) == 1 else "cpu")
        return SimpleNamespace(device=device, **kwargs)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=_fake_factory,
        mixed_precision="bf16",
    )

    assert out.device.type == "cpu"
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


def test_create_accelerator_binds_local_cuda_device_before_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA runs should bind LOCAL_RANK before constructing Accelerator."""
    import neobert.training_utils as training_utils

    bound_devices: list[int] = []

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(training_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_utils.torch.cuda,
        "set_device",
        lambda device: bound_devices.append(int(device)),
    )

    out = create_accelerator(
        use_cpu=False,
        log=logging.getLogger("test"),
        accelerator_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        mixed_precision="bf16",
    )

    assert out.mixed_precision == "bf16"
    assert bound_devices == [3]


def test_create_accelerator_preserves_dataloader_config() -> None:
    """Explicit dataloader config should be forwarded unchanged."""
    dataloader_config = DataLoaderConfiguration(even_batches=False)

    out = create_accelerator(
        use_cpu=True,
        log=logging.getLogger("test"),
        accelerator_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        dataloader_config=dataloader_config,
    )

    assert out.cpu is True
    assert out.dataloader_config is dataloader_config
    assert out.dataloader_config.even_batches is False


def test_update_global_norm_metric_for_logging_computes_on_non_main_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FSDP norm logging must run on every rank even if only rank 0 emits."""
    import neobert.training_utils as training_utils

    calls: list[tuple[tuple[object, ...], bool]] = []

    def _fake_compute(
        parameters: object,
        accelerator: object,
        *,
        grad: bool = False,
    ) -> float:
        del accelerator
        calls.append((tuple(parameters), grad))
        return 12.5

    monkeypatch.setattr(
        training_utils,
        "_compute_l2_norm_for_logging",
        _fake_compute,
    )

    metrics = {"train/weight_norm": 99.0}
    accelerator = SimpleNamespace(is_main_process=False)
    params = (object(), object())

    _update_global_norm_metric_for_logging(
        metrics,
        key="train/weight_norm",
        parameters=params,
        accelerator=accelerator,
        enabled=True,
    )

    assert calls == [(params, False)]
    assert "train/weight_norm" not in metrics


def test_update_global_norm_metric_for_logging_sets_main_process_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main rank should publish the already-collected global norm."""
    import neobert.training_utils as training_utils

    monkeypatch.setattr(
        training_utils,
        "_compute_l2_norm_for_logging",
        lambda *args, **kwargs: 8.5,
    )

    metrics: dict[str, float] = {}
    accelerator = SimpleNamespace(is_main_process=True)

    _update_global_norm_metric_for_logging(
        metrics,
        key="train/weight_norm",
        parameters=(object(),),
        accelerator=accelerator,
        enabled=True,
    )

    assert metrics["train/weight_norm"] == 8.5


def test_resolve_runtime_mixed_precision_and_attn_backend_forces_sdpa_on_cpu(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit CPU runs must disable flash-attn even when CUDA is present."""
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
    assert "trainer.use_cpu=true" in caplog.text


def test_resolve_runtime_mixed_precision_and_attn_backend_forces_sdpa_on_fp32(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flash attention should be disabled when the run is explicitly fp32."""
    with caplog.at_level(logging.WARNING):
        mixed_precision, attn_backend = (
            resolve_runtime_mixed_precision_and_attn_backend(
                mixed_precision="no",
                attn_backend="flash_attn_varlen",
                log=logging.getLogger("test"),
                use_cpu=False,
            )
        )

    assert mixed_precision == "no"
    assert attn_backend == "sdpa"
    assert "mixed_precision='no'" in caplog.text


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
