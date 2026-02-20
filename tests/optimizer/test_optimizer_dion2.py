"""Unit tests for Dion2 optimizer factory wiring."""

from __future__ import annotations

import builtins
import sys
import types

import pytest
import torch
from accelerate.utils import DistributedType

import neobert.optimizer.optimizer as optimizer_module
from neobert.optimizer import get_optimizer


class _ToyModel(torch.nn.Module):
    """Small module that exercises matrix/scalar/embedding parameter grouping."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.scalar = torch.nn.Parameter(torch.ones(4))
        self.matrix = torch.nn.Parameter(torch.ones(4, 4))


def test_dion2_missing_dependency_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dion2 should fail with a clear install hint when optional dep is missing."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "dion":
            raise ImportError("forced missing dion")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match=r"\.\[dion\]|microsoft/dion"):
        get_optimizer(
            _ToyModel(),
            DistributedType.NO,
            name="dion2",
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
        )


def test_dion2_param_group_split_and_config_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dion2 factory should build expected groups and forward config knobs."""

    class _FakeDion2:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = list(param_groups)
            self.kwargs = kwargs

    fake_dion_module = types.SimpleNamespace(Dion2=_FakeDion2)
    monkeypatch.setitem(sys.modules, "dion", fake_dion_module)

    world_sentinel = object()
    monkeypatch.setattr(optimizer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(optimizer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        optimizer_module.dist.group,
        "WORLD",
        world_sentinel,
        raising=False,
    )

    model = _ToyModel()
    optimizer = get_optimizer(
        model,
        DistributedType.NO,
        name="dion2",
        lr=2e-4,
        weight_decay=0.05,
        betas=(0.91, 0.97),
        eps=1e-7,
        dion2_config={
            "fraction": 0.5,
            "ef_decay": 0.9,
            "adjust_lr": "rms_norm",
            "orthogonalization": "polar_express",
            "ns_steps": 3,
            "flatten": True,
            "use_triton": True,
            "verbose": True,
            "scalar_algorithm": "lion",
        },
    )

    assert optimizer.__class__.__name__ == "_FakeDion2"
    assert getattr(optimizer, "_is_neobert_dion2", False)
    assert optimizer.kwargs["distributed_mesh"] is world_sentinel
    assert optimizer.kwargs["fraction"] == 0.5
    assert optimizer.kwargs["ef_decay"] == 0.9
    assert optimizer.kwargs["adjust_lr"] == "rms_norm"
    assert optimizer.kwargs["flatten"] is True
    assert optimizer.kwargs["use_triton"] is False
    assert optimizer.kwargs["verbose"] is True
    assert optimizer.kwargs["betas"] == (0.91, 0.97)
    assert optimizer.kwargs["epsilon"] == 1e-7
    assert callable(optimizer.kwargs["newton_schulz_func"])

    groups = optimizer.param_groups
    matrix_group = next(g for g in groups if g["algorithm"] == "dion2")
    scalar_decay = next(
        g for g in groups if g["algorithm"] == "lion" and g["weight_decay"] > 0
    )
    scalar_no_decay = next(
        g for g in groups if g["algorithm"] == "lion" and g["weight_decay"] == 0
    )

    matrix_ids = {id(param) for param in matrix_group["params"]}
    scalar_decay_ids = {id(param) for param in scalar_decay["params"]}
    scalar_no_decay_ids = {id(param) for param in scalar_no_decay["params"]}

    assert id(model.embedding.weight) not in matrix_ids
    assert id(model.linear.weight) in matrix_ids
    assert id(model.matrix) in matrix_ids
    assert id(model.scalar) in scalar_decay_ids
    assert id(model.linear.bias) in scalar_no_decay_ids
    assert id(model.norm.weight) in scalar_no_decay_ids
    assert id(model.embedding.weight) in scalar_no_decay_ids
    assert scalar_decay["beta1"] == pytest.approx(0.91)
    assert scalar_decay["beta2"] == pytest.approx(0.97)
    assert scalar_decay["epsilon"] == pytest.approx(1e-7)


def test_dion2_qk_clipping_requires_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QK clipping on Dion2 requires model config to register hooks."""

    class _FakeDion2:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = list(param_groups)
            self.kwargs = kwargs

        def step(self):
            return None

    monkeypatch.setitem(sys.modules, "dion", types.SimpleNamespace(Dion2=_FakeDion2))

    with pytest.raises(ValueError, match="enable_clipping=true requires model_config"):
        get_optimizer(
            _ToyModel(),
            DistributedType.NO,
            name="dion2",
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            dion2_config={"enable_clipping": True},
        )


def test_dion2_qk_clipping_runtime_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dion2 can run MuonClip QK clipping hook path after optimizer.step()."""

    class _FakeDion2:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = list(param_groups)
            self.kwargs = kwargs
            self.step_calls = 0

        def step(self):
            self.step_calls += 1
            return "dion2-step"

    class _HookSystem:
        def __init__(self):
            self.enable_calls = []
            self.clear_calls = 0

        def has_captured_inputs(self):
            return True

        def clear(self):
            self.clear_calls += 1

        def set_enabled(self, enabled, *, clear_cache_when_disabling=True):
            self.enable_calls.append((enabled, clear_cache_when_disabling))

    class _FakeMuonClipOptimizer:
        def __init__(self, model, model_config, config):
            del model, model_config
            self.config = config
            self._step = 0
            self._last_metrics = {}
            self.hook_system = _HookSystem()
            self.prepare_calls = []
            self.apply_calls = 0
            self.apply_grad_enabled_states = []

        def should_clip_update(self, update_step):
            return int(update_step) % 2 == 1

        def prepare_for_forward(self, *, update_step, is_last_microbatch):
            self.prepare_calls.append((int(update_step), bool(is_last_microbatch)))
            return True

        def _apply_qk_clipping(self):
            self.apply_grad_enabled_states.append(torch.is_grad_enabled())
            self.apply_calls += 1
            self._last_metrics["train/max_attention_logit"] = 12.3

        def get_metrics(self):
            metrics = dict(self._last_metrics)
            self._last_metrics.clear()
            return metrics

    monkeypatch.setitem(sys.modules, "dion", types.SimpleNamespace(Dion2=_FakeDion2))
    monkeypatch.setattr(
        optimizer_module,
        "MuonClipOptimizer",
        _FakeMuonClipOptimizer,
        raising=True,
    )

    optimizer = get_optimizer(
        _ToyModel(),
        DistributedType.NO,
        model_config=types.SimpleNamespace(),
        name="dion2",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        dion2_config={
            "enable_clipping": True,
            "clipping_threshold": 60.0,
            "clipping_alpha": 0.6,
            "clipping_interval": 2,
        },
    )

    assert hasattr(optimizer, "prepare_for_forward")
    assert hasattr(optimizer, "get_metrics")
    assert optimizer.prepare_for_forward(update_step=1, is_last_microbatch=True)
    assert optimizer.step() == "dion2-step"

    runtime = getattr(optimizer, "_neobert_dion2_qk_runtime")
    clipper = runtime._qk_clipper
    assert clipper.apply_calls == 1
    assert clipper.apply_grad_enabled_states == [False]
    assert clipper.prepare_calls == [(1, True)]
    assert clipper.hook_system.clear_calls == 1
    assert clipper.hook_system.enable_calls[-1] == (False, False)

    assert optimizer.get_metrics()["train/max_attention_logit"] == pytest.approx(12.3)
    assert optimizer.get_metrics() == {}


def test_dion2_qk_step_wrapper_is_scheduler_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dion2 QK step wrapper must remain a bound method for LR schedulers."""

    class _FakeDion2(torch.optim.Optimizer):
        def __init__(self, param_groups, **kwargs):
            defaults = {"lr": float(kwargs.get("lr", 1e-4))}
            super().__init__(param_groups, defaults)

        def step(self, closure=None):
            del closure
            return None

    class _FakeHookSystem:
        def has_captured_inputs(self):
            return False

        def clear(self):
            return None

        def set_enabled(self, enabled, *, clear_cache_when_disabling=True):
            del enabled, clear_cache_when_disabling
            return None

    class _FakeMuonClipOptimizer:
        def __init__(self, model, model_config, config):
            del model, model_config
            self.config = config
            self._step = 0
            self._last_metrics = {}
            self.hook_system = _FakeHookSystem()

        def should_clip_update(self, update_step):
            return False

        def prepare_for_forward(self, *, update_step, is_last_microbatch):
            del update_step, is_last_microbatch
            return True

        def get_metrics(self):
            return {}

    monkeypatch.setitem(sys.modules, "dion", types.SimpleNamespace(Dion2=_FakeDion2))
    monkeypatch.setattr(
        optimizer_module,
        "MuonClipOptimizer",
        _FakeMuonClipOptimizer,
        raising=True,
    )

    optimizer = get_optimizer(
        _ToyModel(),
        DistributedType.NO,
        model_config=types.SimpleNamespace(),
        name="dion2",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        dion2_config={"enable_clipping": True},
    )

    step_method = optimizer.step
    assert hasattr(step_method, "__func__")
    assert getattr(step_method, "__self__", None) is optimizer
    _ = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1.0,
        total_iters=1,
    )
