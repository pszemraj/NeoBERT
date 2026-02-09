"""Quartet-II pretraining integration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

LOGGER = logging.getLogger(__name__)

_CANONICAL_RECIPES = {"none", "quartet_ii"}
_RECIPE_ALIASES = {
    "none": "none",
    "quartet_ii": "quartet_ii",
    "quartet-ii": "quartet_ii",
    "quartet2": "quartet_ii",
    "quartet_ii_linear": "quartet_ii",
    "nvfp4": "quartet_ii",
}


@dataclass
class QuartetIIRuntimeState:
    """Runtime state for Quartet-II pretraining integration."""

    enabled: bool = False
    recipe: str = "none"
    converted_linear_count: int = 0
    skipped_bias_linear_count: int = 0
    skipped_dim_linear_count: int = 0
    post_optimizer_hook: Optional[Callable[[nn.Module], None]] = None


def _normalize_recipe(recipe: Any) -> str:
    """Normalize a user recipe token."""
    normalized = str(recipe or "none").strip().lower()
    return _RECIPE_ALIASES.get(normalized, normalized)


def _get_first_last_linear_fqns(model: nn.Module) -> tuple[Optional[str], Optional[str]]:
    """Return first and last linear-module FQNs in model traversal order."""
    linear_fqns = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if not linear_fqns:
        return None, None
    return linear_fqns[0], linear_fqns[-1]


def _replace_module(model: nn.Module, fqn: str, new_module: nn.Module) -> None:
    """Replace a submodule by its fully-qualified name."""
    if "." in fqn:
        parent_fqn, leaf = fqn.rsplit(".", 1)
        parent = model.get_submodule(parent_fqn)
    else:
        parent = model
        leaf = fqn
    setattr(parent, leaf, new_module)


def _patch_quartet_forward_runtime(
    module: nn.Module,
    *,
    force_disable_backward_quant: bool,
) -> None:
    """Patch Quartet forward for robust BF16 runtime behavior.

    Some distributed runtime stacks can upcast buffers/params. Quartet kernels
    expect BF16 activations and BF16 Hadamard buffers, so we normalize these at
    call time and optionally force ``disable_backward_quant``.

    Important: patch at class scope (once) and use per-instance attributes for
    runtime flags. This avoids per-module closure identities that can trigger
    excessive ``torch.compile`` recompiles across transformer blocks.
    """
    module_cls = type(module)
    if not bool(getattr(module_cls, "_neobert_runtime_patch_applied", False)):
        original_forward = module_cls.forward

        def _forward(
            self: nn.Module,
            x: torch.Tensor,
            disable_backward_quant: bool = False,
            input_abs_max: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if x.dtype != torch.bfloat16:
                x = x.to(dtype=torch.bfloat16)

            had = getattr(self, "had", None)
            if isinstance(had, torch.Tensor) and had.dtype != torch.bfloat16:
                had_bf16 = had.to(dtype=torch.bfloat16)
                if (
                    isinstance(getattr(self, "_buffers", None), dict)
                    and "had" in self._buffers
                ):
                    self._buffers["had"] = had_bf16
                else:
                    setattr(self, "had", had_bf16)

            if bool(getattr(self, "_neobert_force_disable_backward_quant", False)):
                disable_backward_quant = True

            return original_forward(
                self,
                x,
                disable_backward_quant=disable_backward_quant,
                input_abs_max=input_abs_max,
            )

        setattr(module_cls, "forward", _forward)
        setattr(module_cls, "_neobert_runtime_patch_applied", True)

    setattr(
        module,
        "_neobert_force_disable_backward_quant",
        bool(force_disable_backward_quant),
    )


def _normalize_floating_dtypes_for_fsdp(
    model: nn.Module,
    target_dtype: torch.dtype,
    log: logging.Logger,
) -> None:
    """Normalize floating parameter/buffer dtypes for FSDP uniformity checks.

    FSDP2 currently expects uniform original parameter dtype at lazy init.
    Quartet conversion materializes BF16 linears while untouched modules remain
    FP32 by default, which can trip this invariant.
    """
    observed_param_dtypes = {
        param.dtype for param in model.parameters() if param.is_floating_point()
    }
    observed_buffer_dtypes = {
        buffer.dtype for buffer in model.buffers() if buffer.is_floating_point()
    }
    if observed_param_dtypes == {target_dtype} and (
        not observed_buffer_dtypes or observed_buffer_dtypes == {target_dtype}
    ):
        return

    log.info(
        "Quartet-II FSDP dtype normalization: casting floating params/buffers to %s "
        "(params before=%s, buffers before=%s).",
        target_dtype,
        sorted(str(dtype) for dtype in observed_param_dtypes),
        sorted(str(dtype) for dtype in observed_buffer_dtypes),
    )
    model.to(dtype=target_dtype)


def _validate_qutlass_kernel_symbols() -> None:
    """Validate that required qutlass symbols for Quartet-II are available."""
    try:
        import qutlass  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(
            "Quartet-II kernels require the qutlass Python module. Install with: "
            "pip install --no-build-isolation "
            "'git+https://github.com/IST-DASLab/qutlass.git' "
            "and then install Quartet-II kernels."
        ) from exc

    required_symbols = ("matmul_nvf4_bf16_tn",)
    missing_symbols = [
        symbol for symbol in required_symbols if not hasattr(qutlass, symbol)
    ]
    if missing_symbols:
        qutlass_path = getattr(qutlass, "__file__", "<unknown>")
        raise RuntimeError(
            "Quartet-II runtime check failed: qutlass is missing required kernel "
            f"symbols {missing_symbols} (loaded from {qutlass_path}). This usually "
            "means a Quartet-II/qutlass version mismatch or stale installation. "
            "Reinstall qutlass from GitHub with: pip install --no-build-isolation "
            "'git+https://github.com/IST-DASLab/qutlass.git' "
            "and then reinstall Quartet-II kernels with: pip install "
            "--no-build-isolation "
            "'git+https://github.com/IST-DASLab/Quartet-II.git#subdirectory=kernels'. "
            "Also ensure no older PyPI qutlass package shadows this environment."
        )


def apply_quartet2_pretraining_quantization(
    model: nn.Module,
    cfg: Any,
    *,
    accelerator: Any,
    logger: Optional[logging.Logger] = None,
) -> QuartetIIRuntimeState:
    """Apply Quartet-II quantized-training transforms for pretraining."""
    log = logger or LOGGER
    q_cfg = getattr(cfg, "quartet2", None)
    if q_cfg is None or not bool(getattr(q_cfg, "enable", False)):
        return QuartetIIRuntimeState(enabled=False, recipe="none")

    recipe = _normalize_recipe(getattr(q_cfg, "recipe", "none"))
    if recipe not in _CANONICAL_RECIPES:
        valid = sorted(_CANONICAL_RECIPES)
        raise ValueError(
            f"Unknown quartet2.recipe='{recipe}'. Supported values: {valid}."
        )
    if recipe == "none":
        return QuartetIIRuntimeState(enabled=False, recipe="none")

    if str(getattr(cfg, "task", "pretraining")).strip().lower() != "pretraining":
        raise ValueError(
            "Quartet-II integration is currently supported for task='pretraining' only."
        )

    if bool(getattr(q_cfg, "require_compile", True)) and not bool(
        getattr(cfg.trainer, "torch_compile", False)
    ):
        raise ValueError(
            "Quartet-II quantized training requires trainer.torch_compile=true. "
            "Set trainer.torch_compile=true or quartet2.require_compile=false."
        )
    if not bool(getattr(cfg.trainer, "torch_compile", False)):
        log.warning(
            "Quartet-II is enabled with trainer.torch_compile=false. This is typically "
            "slower and can be less stable than compiled mode."
        )

    mixed_precision = str(getattr(cfg.trainer, "mixed_precision", "bf16")).strip().lower()
    if mixed_precision not in {"bf16", "bfloat16"}:
        raise ValueError(
            "Quartet-II requires BF16 mixed precision. Set "
            "trainer.mixed_precision='bf16'."
        )

    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        raise RuntimeError(
            "Quartet-II quantized pretraining is not supported with DeepSpeed in this "
            "release. Use DDP or FSDP2."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Quartet-II kernels require CUDA. No CUDA device is available in this runtime."
        )
    device_capability = torch.cuda.get_device_capability()
    if tuple(device_capability) < (12, 0):
        raise RuntimeError(
            "Quartet-II kernels target Blackwell-class GPUs. Detected compute "
            f"capability {device_capability}, but >= (12, 0) is required."
        )

    try:
        from quartet2.linear import Quartet_II_linear
    except Exception as exc:
        raise ImportError(
            "Quartet-II recipe requested but quartet2 kernels are unavailable. Install "
            "with: pip install --no-build-isolation "
            "'git+https://github.com/IST-DASLab/Quartet-II.git#subdirectory=kernels' "
            "(and ensure qutlass/scipy/nvtx are available)."
        ) from exc
    _validate_qutlass_kernel_symbols()

    filter_fqns = list(getattr(q_cfg, "filter_fqns", []) or [])
    first_linear, last_linear = _get_first_last_linear_fqns(model)
    skip_first_last_linear = bool(getattr(q_cfg, "skip_first_last_linear", True))
    required_dim_multiple = int(getattr(q_cfg, "required_dim_multiple", 128))
    four_over_six = bool(getattr(q_cfg, "four_over_six", True))
    disable_backward_quant = bool(getattr(q_cfg, "disable_backward_quant", False))

    converted_linear_count = 0
    skipped_bias_linear_count = 0
    skipped_dim_linear_count = 0

    for fqn, mod in list(model.named_modules()):
        if not fqn or not isinstance(mod, nn.Linear):
            continue
        if skip_first_last_linear and fqn in {first_linear, last_linear}:
            continue
        if any(skip and skip in fqn for skip in filter_fqns):
            continue
        if mod.bias is not None:
            skipped_bias_linear_count += 1
            continue
        if (
            mod.in_features % required_dim_multiple != 0
            or mod.out_features % required_dim_multiple != 0
        ):
            skipped_dim_linear_count += 1
            continue

        quartet_linear = Quartet_II_linear(
            mod.in_features,
            mod.out_features,
            bias=False,
            four_over_six=four_over_six,
            dtype=torch.bfloat16,
            device=mod.weight.device,
        )
        with torch.no_grad():
            quartet_linear.weight.copy_(mod.weight.to(dtype=torch.bfloat16))
        _patch_quartet_forward_runtime(
            quartet_linear,
            force_disable_backward_quant=disable_backward_quant,
        )

        _replace_module(model, fqn, quartet_linear)
        converted_linear_count += 1

    if skipped_bias_linear_count > 0:
        log.warning(
            "Quartet-II skipped %s biased linear layers (Quartet_II_linear currently "
            "does not preserve bias in forward).",
            skipped_bias_linear_count,
        )
    if skipped_dim_linear_count > 0:
        log.warning(
            "Quartet-II skipped %s linear layers that are not divisible by %s.",
            skipped_dim_linear_count,
            required_dim_multiple,
        )

    if converted_linear_count <= 0:
        log.warning(
            "Quartet-II recipe '%s' enabled but converted 0 linear modules. "
            "Check filter rules and layer dimensions.",
            recipe,
        )
    else:
        if accelerator.distributed_type is DistributedType.FSDP:
            _normalize_floating_dtypes_for_fsdp(model, torch.bfloat16, log)
        log.info(
            "Quartet-II recipe '%s' active: converted %s linear modules.",
            recipe,
            converted_linear_count,
        )

    return QuartetIIRuntimeState(
        enabled=True,
        recipe=recipe,
        converted_linear_count=converted_linear_count,
        skipped_bias_linear_count=skipped_bias_linear_count,
        skipped_dim_linear_count=skipped_dim_linear_count,
        post_optimizer_hook=None,
    )
