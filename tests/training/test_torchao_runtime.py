#!/usr/bin/env python3
"""Unit tests for TorchAO pretraining runtime integration."""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.quantization import apply_torchao_pretraining_quantization


class _AcceleratorStub:
    """Minimal accelerator stub for runtime quantization tests."""

    def __init__(self, distributed_type: DistributedType = DistributedType.NO) -> None:
        """Initialize accelerator stub.

        :param DistributedType distributed_type: Distributed runtime type.
        """
        self.distributed_type = distributed_type


class _TinyModel(nn.Module):
    """Small model with multiple linear layers for conversion tests."""

    def __init__(self) -> None:
        """Build a tiny module graph with linear projections."""
        super().__init__()
        self.embed = nn.Linear(16, 16)
        self.block = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.decoder = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Model output tensor.
        """
        x = self.embed(x)
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


def _build_fake_float8_modules(state: dict) -> dict:
    """Build fake torchao.float8 and accelerate.utils.ao modules."""
    torchao_mod = types.ModuleType("torchao")
    float8_mod = types.ModuleType("torchao.float8")
    accelerate_ao_mod = types.ModuleType("accelerate.utils.ao")

    @dataclass
    class Float8LinearConfig:
        """Fake float8 config."""

        enable_fsdp_float8_all_gather: bool = False
        emulate: bool = False
        recipe_name: str | None = None

        @staticmethod
        def from_recipe_name(recipe_name: str) -> "Float8LinearConfig":
            return Float8LinearConfig(recipe_name=recipe_name)

    class Float8Linear(nn.Linear):
        """Fake Float8Linear module for conversion counting."""

    def _convert_impl(
        model: nn.Module,
        config: Float8LinearConfig | None = None,
        module_filter_fn=None,
    ) -> None:
        _ = config
        state["float8_convert_calls"] = state.get("float8_convert_calls", 0) + 1
        for fqn, mod in list(model.named_modules()):
            if not fqn or not isinstance(mod, nn.Linear):
                continue
            should_convert = True if module_filter_fn is None else module_filter_fn(mod, fqn)
            if not should_convert:
                continue
            new_mod = Float8Linear(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
            with torch.no_grad():
                new_mod.weight.copy_(mod.weight)
                if mod.bias is not None:
                    new_mod.bias.copy_(mod.bias)
            _replace_module(model, fqn, new_mod)

    def convert_to_float8_training(
        model: nn.Module,
        config: Float8LinearConfig | None = None,
        module_filter_fn=None,
    ) -> None:
        _convert_impl(model, config=config, module_filter_fn=module_filter_fn)

    def convert_model_to_fp8_ao(
        model: nn.Module,
        config: Float8LinearConfig | None = None,
        module_filter_func=None,
    ) -> None:
        state["accelerate_calls"] = state.get("accelerate_calls", 0) + 1
        _convert_impl(model, config=config, module_filter_fn=module_filter_func)

    def precompute_float8_dynamic_scale_for_fsdp(model: nn.Module) -> None:
        _ = model
        state["precompute_calls"] = state.get("precompute_calls", 0) + 1

    def _auto_filter_for_recipe(recipe_name: str, filter_fqns: list[str]):
        _ = recipe_name

        def _fn(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            return not any(skip in fqn for skip in filter_fqns)

        return _fn

    float8_mod.Float8LinearConfig = Float8LinearConfig
    float8_mod.Float8Linear = Float8Linear
    float8_mod.convert_to_float8_training = convert_to_float8_training
    float8_mod.precompute_float8_dynamic_scale_for_fsdp = (
        precompute_float8_dynamic_scale_for_fsdp
    )
    float8_mod._auto_filter_for_recipe = _auto_filter_for_recipe

    accelerate_ao_mod.convert_model_to_fp8_ao = convert_model_to_fp8_ao

    torchao_mod.float8 = float8_mod
    return {
        "torchao": torchao_mod,
        "torchao.float8": float8_mod,
        "accelerate.utils.ao": accelerate_ao_mod,
    }


def _build_fake_mx_modules(state: dict) -> dict:
    """Build fake torchao MX modules and quantize_ entrypoint."""
    torchao_mod = types.ModuleType("torchao")
    prototype_mod = types.ModuleType("torchao.prototype")
    mx_formats_mod = types.ModuleType("torchao.prototype.mx_formats")
    mx_config_mod = types.ModuleType("torchao.prototype.mx_formats.config")
    quantization_mod = types.ModuleType("torchao.quantization")

    class MXFP8Dim1CastKernelChoice(Enum):
        """Fake MX dim1 kernel choice enum."""

        TRITON = "triton"
        CUDA = "cuda"
        TORCH = "torch"

    @dataclass
    class MXLinearConfig:
        """Fake MXLinearConfig with recipe loader."""

        recipe_name: str
        mxfp8_dim1_cast_kernel_choice: MXFP8Dim1CastKernelChoice = (
            MXFP8Dim1CastKernelChoice.TORCH
        )

        @staticmethod
        def from_recipe_name(recipe_name: str) -> "MXLinearConfig":
            return MXLinearConfig(recipe_name=recipe_name)

    class MXLinear(nn.Linear):
        """Fake MXLinear module for conversion counting."""

    def quantize_(model: nn.Module, config: MXLinearConfig, filter_fn=None) -> None:
        _ = config
        state["mx_quantize_calls"] = state.get("mx_quantize_calls", 0) + 1
        for fqn, mod in list(model.named_modules()):
            if not fqn or not isinstance(mod, nn.Linear):
                continue
            should_convert = True if filter_fn is None else filter_fn(mod, fqn)
            if not should_convert:
                continue
            new_mod = MXLinear(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
            with torch.no_grad():
                new_mod.weight.copy_(mod.weight)
                if mod.bias is not None:
                    new_mod.bias.copy_(mod.bias)
            _replace_module(model, fqn, new_mod)

    mx_config_mod.MXFP8Dim1CastKernelChoice = MXFP8Dim1CastKernelChoice
    mx_config_mod.MXLinearConfig = MXLinearConfig
    quantization_mod.quantize_ = quantize_

    mx_formats_mod.config = mx_config_mod
    prototype_mod.mx_formats = mx_formats_mod
    torchao_mod.prototype = prototype_mod
    torchao_mod.quantization = quantization_mod
    return {
        "torchao": torchao_mod,
        "torchao.prototype": prototype_mod,
        "torchao.prototype.mx_formats": mx_formats_mod,
        "torchao.prototype.mx_formats.config": mx_config_mod,
        "torchao.quantization": quantization_mod,
    }


class TestTorchAORuntime(unittest.TestCase):
    """Behavioral tests for TorchAO runtime adapter."""

    def _base_cfg(self) -> Config:
        """Build base config with TorchAO enabled and compile on."""
        cfg = Config()
        cfg.task = "pretraining"
        cfg.trainer.torch_compile = True
        cfg.torchao.enable = True
        cfg.torchao.filter_fqns = []
        cfg.torchao.skip_first_last_linear = False
        cfg.torchao.auto_filter_small_kn = False
        return cfg

    def test_runtime_noop_when_disabled(self):
        """Disabled torchao config should produce a no-op runtime state."""
        cfg = Config()
        model = _TinyModel()
        state = apply_torchao_pretraining_quantization(
            model,
            cfg,
            accelerator=_AcceleratorStub(),
        )
        self.assertFalse(state.enabled)
        self.assertEqual(state.recipe, "none")
        self.assertEqual(state.converted_linear_count, 0)
        self.assertIsNone(state.post_optimizer_hook)

    def test_runtime_requires_compile_by_default(self):
        """TorchAO should require torch.compile unless explicitly disabled."""
        cfg = self._base_cfg()
        cfg.trainer.torch_compile = False
        cfg.torchao.recipe = "float8_tensorwise"
        model = _TinyModel()
        with self.assertRaisesRegex(ValueError, "torch_compile"):
            apply_torchao_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

    def test_float8_prefers_accelerate_helper_when_available(self):
        """Float8 recipe should use Accelerate helper path when present."""
        cfg = self._base_cfg()
        cfg.torchao.recipe = "float8_tensorwise"
        model = _TinyModel()
        fake_state: dict = {}
        with patch.dict(
            sys.modules,
            _build_fake_float8_modules(fake_state),
            clear=False,
        ):
            state = apply_torchao_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "float8_tensorwise")
        self.assertTrue(state.used_accelerate_helper)
        self.assertGreater(state.converted_linear_count, 0)
        self.assertGreater(fake_state.get("accelerate_calls", 0), 0)

    def test_float8_fsdp_precompute_hook_is_registered(self):
        """FSDP tensorwise all-gather path should attach post-optimizer hook."""
        cfg = self._base_cfg()
        cfg.torchao.recipe = "float8_tensorwise"
        cfg.torchao.enable_fsdp_float8_all_gather = True
        cfg.torchao.precompute_float8_dynamic_scale_for_fsdp = True
        model = _TinyModel()
        fake_state: dict = {}
        with patch.dict(
            sys.modules,
            _build_fake_float8_modules(fake_state),
            clear=False,
        ):
            state = apply_torchao_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(DistributedType.FSDP),
            )
            self.assertIsNotNone(state.post_optimizer_hook)
            state.post_optimizer_hook(model)

        self.assertEqual(fake_state.get("precompute_calls", 0), 1)

    def test_mx_recipe_mxfp8_emulated_supported(self):
        """MX path should accept mxfp8_emulated and convert linear modules."""
        cfg = self._base_cfg()
        cfg.torchao.recipe = "mxfp8_emulated"
        cfg.torchao.mxfp8_dim1_cast_kernel_choice = "cuda"
        model = _TinyModel()
        fake_state: dict = {}
        with patch.dict(
            sys.modules,
            _build_fake_mx_modules(fake_state),
            clear=False,
        ):
            state = apply_torchao_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "mxfp8_emulated")
        self.assertGreater(state.converted_linear_count, 0)
        self.assertGreater(fake_state.get("mx_quantize_calls", 0), 0)


if __name__ == "__main__":
    unittest.main()
