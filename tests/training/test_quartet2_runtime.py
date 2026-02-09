#!/usr/bin/env python3
"""Unit tests for Quartet-II pretraining runtime integration."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

from neobert.config import Config
from neobert.quantization import apply_quartet2_pretraining_quantization


class _AcceleratorStub:
    """Minimal accelerator stub for runtime quantization tests."""

    def __init__(self, distributed_type: DistributedType = DistributedType.NO) -> None:
        self.distributed_type = distributed_type


class _TinyModel(nn.Module):
    """Small model with multiple linear layers for conversion tests."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(128, 128, bias=False)
        self.block = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, 128, bias=False),
        )
        self.decoder = nn.Linear(128, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block(x)
        return self.decoder(x)


class _BiasTinyModel(nn.Module):
    """Model variant with one biased linear for skip-behavior checks."""

    def __init__(self) -> None:
        super().__init__()
        self.unbiased = nn.Linear(128, 128, bias=False)
        self.biased = nn.Linear(128, 128, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unbiased(x)
        return self.biased(x)


def _build_fake_quartet_modules(state: dict) -> dict:
    """Build fake quartet2.linear module for runtime adapter tests."""
    quartet2_mod = types.ModuleType("quartet2")
    linear_mod = types.ModuleType("quartet2.linear")

    class Quartet_II_linear(nn.Linear):
        def __init__(
            self,
            *args,
            four_over_six: bool = True,
            dtype: torch.dtype = torch.bfloat16,
            **kwargs,
        ) -> None:
            super().__init__(*args, dtype=dtype, **kwargs)
            self.four_over_six = four_over_six
            self.weight_abs_max = None

        def forward(
            self,
            x: torch.Tensor,
            disable_backward_quant: bool = False,
            input_abs_max=None,
        ) -> torch.Tensor:
            _ = input_abs_max
            state["forward_calls"] = state.get("forward_calls", 0) + 1
            state["disable_backward_quant"] = disable_backward_quant
            return torch.nn.functional.linear(x, self.weight, None)

    linear_mod.Quartet_II_linear = Quartet_II_linear
    quartet2_mod.linear = linear_mod
    return {
        "quartet2": quartet2_mod,
        "quartet2.linear": linear_mod,
    }


class TestQuartet2Runtime(unittest.TestCase):
    """Behavioral tests for Quartet-II runtime adapter."""

    def _base_cfg(self) -> Config:
        cfg = Config()
        cfg.task = "pretraining"
        cfg.trainer.torch_compile = True
        cfg.trainer.mixed_precision = "bf16"
        cfg.quartet2.enable = True
        cfg.quartet2.recipe = "quartet_ii"
        cfg.quartet2.filter_fqns = []
        cfg.quartet2.skip_first_last_linear = False
        cfg.quartet2.required_dim_multiple = 128
        return cfg

    def test_runtime_noop_when_disabled(self):
        """Disabled quartet2 config should produce a no-op runtime state."""
        cfg = Config()
        model = _TinyModel()
        state = apply_quartet2_pretraining_quantization(
            model,
            cfg,
            accelerator=_AcceleratorStub(),
        )
        self.assertFalse(state.enabled)
        self.assertEqual(state.recipe, "none")
        self.assertEqual(state.converted_linear_count, 0)

    def test_runtime_requires_compile_by_default(self):
        """Quartet-II should require torch.compile unless explicitly disabled."""
        cfg = self._base_cfg()
        cfg.trainer.torch_compile = False
        model = _TinyModel()
        with self.assertRaisesRegex(ValueError, "torch_compile"):
            apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

    def test_runtime_requires_bf16_mixed_precision(self):
        """Quartet-II path should reject non-bf16 mixed precision settings."""
        cfg = self._base_cfg()
        cfg.trainer.mixed_precision = "fp32"
        model = _TinyModel()
        with self.assertRaisesRegex(ValueError, "BF16"):
            apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

    def test_quartet_recipe_converts_linear_layers(self):
        """Quartet-II path should convert linear modules when kernels are available."""
        cfg = self._base_cfg()
        model = _TinyModel()
        fake_state: dict = {}
        with (
            patch.dict(
                sys.modules,
                _build_fake_quartet_modules(fake_state),
                clear=False,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            state = apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "quartet_ii")
        self.assertGreater(state.converted_linear_count, 0)

    def test_biased_linears_are_skipped(self):
        """Quartet-II conversion should skip biased linear modules."""
        cfg = self._base_cfg()
        model = _BiasTinyModel()
        fake_state: dict = {}
        with (
            patch.dict(
                sys.modules,
                _build_fake_quartet_modules(fake_state),
                clear=False,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            state = apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )

        self.assertGreaterEqual(state.skipped_bias_linear_count, 1)
        self.assertEqual(state.converted_linear_count, 1)

    def test_disable_backward_quant_flag_is_applied(self):
        """Quartet-II disable_backward_quant should be propagated to converted layers."""
        cfg = self._base_cfg()
        cfg.quartet2.disable_backward_quant = True
        model = _TinyModel()
        fake_state: dict = {}
        with (
            patch.dict(
                sys.modules,
                _build_fake_quartet_modules(fake_state),
                clear=False,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            _ = apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(),
            )
            _ = model(torch.randn(2, 128, dtype=torch.bfloat16))

        self.assertTrue(bool(fake_state.get("disable_backward_quant", False)))

    def test_fsdp_normalizes_floating_param_dtypes(self):
        """FSDP Quartet path should enforce uniform BF16 floating param dtype."""
        cfg = self._base_cfg()
        model = _TinyModel()
        fake_state: dict = {}
        with (
            patch.dict(
                sys.modules,
                _build_fake_quartet_modules(fake_state),
                clear=False,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            _ = apply_quartet2_pretraining_quantization(
                model,
                cfg,
                accelerator=_AcceleratorStub(DistributedType.FSDP),
            )

        floating_dtypes = {
            param.dtype for param in model.parameters() if param.is_floating_point()
        }
        self.assertEqual(floating_dtypes, {torch.bfloat16})


if __name__ == "__main__":
    unittest.main()
