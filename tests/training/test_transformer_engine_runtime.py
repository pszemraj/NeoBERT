#!/usr/bin/env python3
"""Unit tests for Transformer Engine pretraining runtime integration."""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.quantization import apply_transformer_engine_pretraining_quantization


class _AcceleratorStub:
    """Minimal accelerator stub for runtime quantization tests."""

    def __init__(self, distributed_type: DistributedType = DistributedType.NO) -> None:
        self.distributed_type = distributed_type


class _TinyModel(nn.Module):
    """Small model with multiple linear/layernorm layers for conversion tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.block = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.decoder = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x)
        x = self.block(x)
        return self.decoder(x)


def _replace_module(model: nn.Module, fqn: str, new_module: nn.Module) -> None:
    """Replace a submodule by its fully-qualified name."""
    if "." in fqn:
        parent_fqn, leaf = fqn.rsplit(".", 1)
        parent = model.get_submodule(parent_fqn)
    else:
        parent = model
        leaf = fqn
    setattr(parent, leaf, new_module)


def _build_fake_transformer_engine_modules(state: dict) -> dict:
    """Build fake Transformer Engine modules for runtime adapter tests."""
    te_mod = types.ModuleType("transformer_engine")
    te_common_mod = types.ModuleType("transformer_engine.common")
    te_recipe_mod = types.ModuleType("transformer_engine.common.recipe")
    te_pytorch_mod = types.ModuleType("transformer_engine.pytorch")
    te_quant_mod = types.ModuleType("transformer_engine.pytorch.quantization")

    class Format(Enum):
        E2M1 = "E2M1"
        E4M3 = "E4M3"
        E5M2 = "E5M2"
        HYBRID = "HYBRID"

    @dataclass
    class DelayedScaling:
        margin: int = 0
        fp8_format: Format = Format.HYBRID
        amax_history_len: int = 1024
        amax_compute_algo: str = "max"

    @dataclass
    class Float8CurrentScaling:
        fp8_format: Format = Format.HYBRID

    @dataclass
    class MXFP8BlockScaling:
        margin: int = 0
        fp8_format: Format = Format.E4M3

    @dataclass
    class NVFP4BlockScaling:
        disable_rht: bool = False
        disable_stochastic_rounding: bool = False
        disable_2d_quantization: bool = False

    class TELinear(nn.Linear):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            params_dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(in_features, out_features, bias=bias)
            if params_dtype is not None:
                self.to(dtype=params_dtype)

    class TELayerNorm(nn.LayerNorm):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-5,
            params_dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(hidden_size, eps=eps)
            if params_dtype is not None:
                self.to(dtype=params_dtype)

    @contextmanager
    def fp8_autocast(enabled: bool = True, fp8_recipe=None, recipe=None):
        state["autocast_calls"] = state.get("autocast_calls", 0) + 1
        state["autocast_enabled"] = enabled
        state["autocast_recipe"] = fp8_recipe if fp8_recipe is not None else recipe
        yield

    def _support(flag_name: str, return_reason: bool = False):
        supported = bool(state.get(flag_name, True))
        reason = "" if supported else f"{flag_name} unavailable"
        if return_reason:
            return supported, reason
        return supported

    def is_fp8_available(return_reason: bool = False):
        return _support("fp8_available", return_reason=return_reason)

    def is_mxfp8_available(return_reason: bool = False):
        return _support("mxfp8_available", return_reason=return_reason)

    def is_nvfp4_available(return_reason: bool = False):
        return _support("nvfp4_available", return_reason=return_reason)

    te_recipe_mod.Format = Format
    te_recipe_mod.DelayedScaling = DelayedScaling
    te_recipe_mod.Float8CurrentScaling = Float8CurrentScaling
    te_recipe_mod.MXFP8BlockScaling = MXFP8BlockScaling
    te_recipe_mod.NVFP4BlockScaling = NVFP4BlockScaling

    te_quant_mod.is_fp8_available = is_fp8_available
    te_quant_mod.is_mxfp8_available = is_mxfp8_available
    te_quant_mod.is_nvfp4_available = is_nvfp4_available

    te_pytorch_mod.Linear = TELinear
    te_pytorch_mod.LayerNorm = TELayerNorm
    te_pytorch_mod.fp8_autocast = fp8_autocast
    te_pytorch_mod.autocast = fp8_autocast
    te_pytorch_mod.quantization = te_quant_mod

    te_common_mod.recipe = te_recipe_mod
    te_mod.common = te_common_mod
    te_mod.pytorch = te_pytorch_mod

    return {
        "transformer_engine": te_mod,
        "transformer_engine.common": te_common_mod,
        "transformer_engine.common.recipe": te_recipe_mod,
        "transformer_engine.pytorch": te_pytorch_mod,
        "transformer_engine.pytorch.quantization": te_quant_mod,
    }


def _build_fake_accelerate_te_module(state: dict) -> dict:
    """Build fake accelerate Transformer Engine helper module."""
    accelerate_te_mod = types.ModuleType("accelerate.utils.transformer_engine")

    def contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval=False):
        state["accelerate_helper_calls"] = state.get("accelerate_helper_calls", 0) + 1

        def _forward(self, *args, **kwargs):
            _ = self
            enabled = use_during_eval or self.training
            state["accelerate_forward_calls"] = state.get("accelerate_forward_calls", 0) + 1
            state["accelerate_autocast_enabled"] = enabled
            state["accelerate_recipe"] = fp8_recipe
            return model_forward(*args, **kwargs)

        _forward.__wrapped__ = model_forward
        return _forward

    accelerate_te_mod.contextual_fp8_autocast = contextual_fp8_autocast
    return {"accelerate.utils.transformer_engine": accelerate_te_mod}


class TestTransformerEngineRuntime(unittest.TestCase):
    """Behavioral tests for Transformer Engine runtime adapter."""

    def _base_cfg(self) -> Config:
        cfg = Config()
        cfg.task = "pretraining"
        cfg.trainer.torch_compile = True
        cfg.transformer_engine.enable = True
        cfg.transformer_engine.recipe = "fp8_delayed"
        cfg.transformer_engine.filter_fqns = []
        cfg.transformer_engine.skip_first_last_linear = False
        return cfg

    def test_runtime_noop_when_disabled(self):
        """Disabled transformer_engine config should produce a no-op runtime state."""
        cfg = Config()
        model = _TinyModel()
        state = apply_transformer_engine_pretraining_quantization(
            model,
            cfg,
            accelerator=_AcceleratorStub(),
        )
        self.assertFalse(state.enabled)
        self.assertEqual(state.recipe, "none")
        self.assertEqual(state.converted_linear_count, 0)
        self.assertEqual(state.converted_layernorm_count, 0)

    def test_runtime_requires_compile_by_default(self):
        """Transformer Engine should require torch.compile unless disabled."""
        cfg = self._base_cfg()
        cfg.trainer.torch_compile = False
        model = _TinyModel()
        with self.assertRaisesRegex(ValueError, "torch_compile"):
            apply_transformer_engine_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

    def test_fp8_delayed_uses_accelerate_autocast_helper_when_available(self):
        """FP8 delayed path should prefer accelerate contextual helper when present."""
        cfg = self._base_cfg()
        model = _TinyModel()
        fake_state: dict = {}

        modules = _build_fake_transformer_engine_modules(fake_state)
        modules.update(_build_fake_accelerate_te_module(fake_state))
        with patch.dict(sys.modules, modules, clear=False):
            state = apply_transformer_engine_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )
            _ = model(torch.randn(2, 16))

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "fp8_delayed")
        self.assertTrue(state.used_accelerate_helper)
        self.assertGreater(state.converted_linear_count, 0)
        self.assertGreater(fake_state.get("accelerate_helper_calls", 0), 0)
        self.assertGreater(fake_state.get("accelerate_forward_calls", 0), 0)

    def test_fp8_delayed_falls_back_to_native_autocast_wrapper(self):
        """When accelerate TE helper is unavailable, runtime should use native wrapper."""
        cfg = self._base_cfg()
        model = _TinyModel()
        fake_state: dict = {}
        broken_accelerate_te = types.ModuleType("accelerate.utils.transformer_engine")

        with patch.dict(
            sys.modules,
            {
                **_build_fake_transformer_engine_modules(fake_state),
                "accelerate.utils.transformer_engine": broken_accelerate_te,
            },
            clear=False,
        ):
            state = apply_transformer_engine_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )
            _ = model(torch.randn(2, 16))

        self.assertTrue(state.enabled)
        self.assertFalse(state.used_accelerate_helper)
        self.assertGreater(fake_state.get("autocast_calls", 0), 0)

    def test_nvfp4_support_check_is_enforced(self):
        """NVFP4 recipe should fail fast when backend reports unsupported."""
        cfg = self._base_cfg()
        cfg.transformer_engine.recipe = "nvfp4"
        model = _TinyModel()
        fake_state = {"nvfp4_available": False}

        with patch.dict(
            sys.modules,
            _build_fake_transformer_engine_modules(fake_state),
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "nvfp4"):
                apply_transformer_engine_pretraining_quantization(
                    model,
                    cfg,
                    accelerator=_AcceleratorStub(),
                )

    def test_skip_first_last_linear_applies_to_te_conversion(self):
        """First/last linear skipping should keep edge layers in nn.Linear form."""
        cfg = self._base_cfg()
        cfg.transformer_engine.skip_first_last_linear = True
        cfg.transformer_engine.convert_layernorm = False
        model = _TinyModel()
        fake_state: dict = {}

        with patch.dict(
            sys.modules,
            _build_fake_transformer_engine_modules(fake_state),
            clear=False,
        ):
            state = apply_transformer_engine_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

        self.assertEqual(state.converted_linear_count, 2)
        self.assertIsInstance(model.embed, nn.Linear)
        self.assertIsInstance(model.decoder, nn.Linear)


if __name__ == "__main__":
    unittest.main()
