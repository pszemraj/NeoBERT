"""TorchAO pretraining integration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

LOGGER = logging.getLogger(__name__)

_FLOAT8_RECIPES = {
    "float8_tensorwise": "tensorwise",
    "float8_rowwise": "rowwise",
    "float8_rowwise_with_gw_hp": "rowwise_with_gw_hp",
}
_MX_LINEAR_RECIPES = {
    "mxfp8_emulated",
    "mxfp8_cublas",
    "mxfp8_cublas_rceil",
    "mxfp4_cutlass",
    "mxfp4_emulated",
}
_QAT_RECIPES = {"nvfp4_qat", "mxfp4_qat"}
_SUPPORTED_RECIPES = {"none", *_FLOAT8_RECIPES.keys(), *_MX_LINEAR_RECIPES, *_QAT_RECIPES}


@dataclass
class TorchAORuntimeState:
    """Runtime state for TorchAO pretraining integration."""

    enabled: bool = False
    recipe: str = "none"
    converted_linear_count: int = 0
    post_optimizer_hook: Optional[Callable[[torch.nn.Module], None]] = None
    used_accelerate_helper: bool = False


def _normalize_recipe(recipe: Any) -> str:
    """Normalize a user recipe token."""
    return str(recipe or "none").strip().lower()


def _get_first_last_linear_fqns(model: nn.Module) -> tuple[Optional[str], Optional[str]]:
    """Return first and last linear-module FQNs in model traversal order."""
    linear_fqns = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if not linear_fqns:
        return None, None
    return linear_fqns[0], linear_fqns[-1]


def _count_module_class_names(model: nn.Module, class_names: set[str]) -> int:
    """Count module instances by class name."""
    return sum(1 for mod in model.modules() if mod.__class__.__name__ in class_names)


def _has_cuda_kernel_for_mxfp8_quantize() -> Optional[bool]:
    """Return whether torchao::mxfp8_quantize has a CUDA kernel.

    Returns ``False`` when the op is missing, ``True`` when CUDA kernel is present,
    and ``None`` when dispatch metadata cannot be queried in this runtime.
    """
    try:
        _ = torch.ops.torchao.mxfp8_quantize.default
    except Exception:
        return False

    has_kernel_for_dispatch = getattr(
        torch._C, "_dispatch_has_kernel_for_dispatch_key", None
    )
    if not callable(has_kernel_for_dispatch):
        return None
    try:
        return bool(has_kernel_for_dispatch("torchao::mxfp8_quantize", "CUDA"))
    except Exception:
        return None


def _build_linear_filter(
    *,
    filter_fqns: list[str],
    first_linear_fqn: Optional[str],
    last_linear_fqn: Optional[str],
    skip_first_last_linear: bool,
    auto_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> Callable[[nn.Module, str], bool]:
    """Build a unified linear-module filter for TorchAO conversion."""

    def _module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if skip_first_last_linear and fqn in {first_linear_fqn, last_linear_fqn}:
            return False
        if any(skip_fqn in fqn for skip_fqn in filter_fqns):
            return False
        if auto_filter_fn is not None and not auto_filter_fn(mod, fqn):
            return False
        return True

    return _module_filter_fn


def _build_float8_config(torchao_cfg: Any, recipe_name: str) -> Any:
    """Build torchao.float8 config for requested float8 recipe."""
    from torchao.float8 import Float8LinearConfig

    if recipe_name == "tensorwise":
        return Float8LinearConfig(
            enable_fsdp_float8_all_gather=bool(
                getattr(torchao_cfg, "enable_fsdp_float8_all_gather", False)
            ),
            emulate=bool(getattr(torchao_cfg, "emulate", False)),
        )

    if getattr(torchao_cfg, "enable_fsdp_float8_all_gather", False):
        raise ValueError(
            "torchao.enable_fsdp_float8_all_gather is only supported for "
            "torchao.recipe='float8_tensorwise'."
        )

    base_cfg = Float8LinearConfig.from_recipe_name(recipe_name)
    if getattr(torchao_cfg, "emulate", False):
        base_cfg = replace(base_cfg, emulate=True)
    return base_cfg


def _build_auto_filter_for_float8(
    recipe_name: str,
    auto_filter_small_kn: bool,
    logger: logging.Logger,
) -> Optional[Callable[[nn.Module, str], bool]]:
    """Build TorchAO auto-filter function when available/compatible."""
    if not auto_filter_small_kn:
        return None
    if recipe_name not in {"tensorwise", "rowwise"}:
        logger.warning(
            "torchao.auto_filter_small_kn is ignored for recipe '%s' "
            "(TorchAO auto-filter supports tensorwise/rowwise only).",
            recipe_name,
        )
        return None
    try:
        from torchao.float8 import _auto_filter_for_recipe
    except Exception:
        logger.warning(
            "TorchAO auto-filter requested but unavailable in this torchao build; "
            "using explicit filter rules only."
        )
        return None
    return _auto_filter_for_recipe(recipe_name, filter_fqns=[])


def apply_torchao_pretraining_quantization(
    model: nn.Module,
    cfg: Any,
    *,
    accelerator: Any,
    logger: Optional[logging.Logger] = None,
) -> TorchAORuntimeState:
    """Apply TorchAO quantized-training transforms for pretraining.

    This mutates the model in place before torch.compile / accelerator.prepare.
    """
    log = logger or LOGGER
    torchao_cfg = getattr(cfg, "torchao", None)
    if torchao_cfg is None or not bool(getattr(torchao_cfg, "enable", False)):
        return TorchAORuntimeState(enabled=False, recipe="none")

    recipe = _normalize_recipe(getattr(torchao_cfg, "recipe", "none"))
    if recipe not in _SUPPORTED_RECIPES:
        valid = sorted(_SUPPORTED_RECIPES)
        raise ValueError(
            f"Unknown torchao.recipe='{recipe}'. Supported values: {valid}."
        )
    if recipe == "none":
        return TorchAORuntimeState(enabled=False, recipe="none")

    if str(getattr(cfg, "task", "pretraining")).strip().lower() != "pretraining":
        raise ValueError(
            "TorchAO integration is currently supported for task='pretraining' only."
        )

    if bool(getattr(torchao_cfg, "require_compile", True)) and not bool(
        getattr(cfg.trainer, "torch_compile", False)
    ):
        raise ValueError(
            "TorchAO quantized training requires trainer.torch_compile=true. "
            "Set trainer.torch_compile=true or torchao.require_compile=false."
        )
    if not bool(getattr(cfg.trainer, "torch_compile", False)):
        log.warning(
            "TorchAO is enabled with trainer.torch_compile=false. This is typically "
            "slower and can be less stable than compiled mode."
        )

    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        raise RuntimeError(
            "TorchAO quantized pretraining is not supported with DeepSpeed in this "
            "release. Use DDP or FSDP2."
        )

    filter_fqns = list(getattr(torchao_cfg, "filter_fqns", []) or [])
    attn_backend = str(getattr(cfg.model, "attn_backend", "sdpa")).strip().lower()
    if recipe in _QAT_RECIPES and attn_backend == "flash_attn_varlen":
        if not any("qkv" in skip for skip in filter_fqns):
            raise ValueError(
                "TorchAO recipe "
                f"'{recipe}' with model.attn_backend='flash_attn_varlen' requires "
                "excluding qkv projections from quantization. Add 'qkv' to "
                "torchao.filter_fqns (or switch attn_backend to 'sdpa')."
            )
    first_linear, last_linear = _get_first_last_linear_fqns(model)

    post_optimizer_hook: Optional[Callable[[nn.Module], None]] = None
    converted_linear_count = 0
    used_accelerate_helper = False

    if recipe in _FLOAT8_RECIPES:
        recipe_name = _FLOAT8_RECIPES[recipe]
        auto_filter_fn = _build_auto_filter_for_float8(
            recipe_name,
            bool(getattr(torchao_cfg, "auto_filter_small_kn", True)),
            log,
        )
        module_filter_fn = _build_linear_filter(
            filter_fqns=filter_fqns,
            first_linear_fqn=first_linear,
            last_linear_fqn=last_linear,
            skip_first_last_linear=bool(
                getattr(torchao_cfg, "skip_first_last_linear", True)
            ),
            auto_filter_fn=auto_filter_fn,
        )
        try:
            float8_config = _build_float8_config(torchao_cfg, recipe_name)
            accelerate_fp8_helper_available = False
            try:
                from accelerate.utils.ao import convert_model_to_fp8_ao

                accelerate_fp8_helper_available = True
                convert_model_to_fp8_ao(
                    model,
                    config=float8_config,
                    module_filter_func=module_filter_fn,
                )
                used_accelerate_helper = True
            except ImportError:
                accelerate_fp8_helper_available = False
            except TypeError as exc:
                log.warning(
                    "Accelerate FP8 helper call signature is incompatible with this "
                    "runtime (%s); falling back to torchao conversion.",
                    exc,
                )
            except Exception as exc:
                if accelerate_fp8_helper_available:
                    raise RuntimeError(
                        "Accelerate FP8 helper failed while applying TorchAO float8 "
                        "conversion. This is likely a runtime/config incompatibility."
                    ) from exc
                log.warning(
                    "Accelerate FP8 helper unavailable; falling back to torchao "
                    "convert_to_float8_training path."
                )
            if not used_accelerate_helper:
                from torchao.float8 import convert_to_float8_training

                convert_to_float8_training(
                    model,
                    config=float8_config,
                    module_filter_fn=module_filter_fn,
                )
            converted_linear_count = _count_module_class_names(model, {"Float8Linear"})
        except ImportError as exc:
            raise ImportError(
                "TorchAO float8 recipe requested but torchao is not available. "
                "Install a compatible torchao build (for example: pip install -e .[quant])."
            ) from exc

        wants_precompute = bool(
            getattr(torchao_cfg, "precompute_float8_dynamic_scale_for_fsdp", False)
        )
        has_float8_ag = bool(getattr(torchao_cfg, "enable_fsdp_float8_all_gather", False))
        if wants_precompute and has_float8_ag:
            if accelerator.distributed_type is DistributedType.FSDP:
                from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

                def _post_opt_hook(unwrapped_model: nn.Module) -> None:
                    precompute_float8_dynamic_scale_for_fsdp(unwrapped_model)

                post_optimizer_hook = _post_opt_hook
            else:
                log.warning(
                    "torchao.precompute_float8_dynamic_scale_for_fsdp=true is set, "
                    "but distributed type is %s (not FSDP); skipping precompute hook.",
                    accelerator.distributed_type,
                )
        elif wants_precompute and not has_float8_ag:
            log.warning(
                "torchao.precompute_float8_dynamic_scale_for_fsdp=true is ignored "
                "unless torchao.enable_fsdp_float8_all_gather=true."
            )

    elif recipe in _MX_LINEAR_RECIPES:
        try:
            from torchao.prototype.mx_formats.config import (
                MXFP8Dim1CastKernelChoice,
                MXLinearConfig,
            )
            from torchao.quantization import quantize_

            if recipe in {"mxfp8_cublas", "mxfp8_cublas_rceil"} and torch.cuda.is_available():
                has_cuda_kernel = _has_cuda_kernel_for_mxfp8_quantize()
                if has_cuda_kernel is False:
                    raise RuntimeError(
                        "TorchAO recipe "
                        f"'{recipe}' requires CUDA MXFP8 kernels, but "
                        "torchao::mxfp8_quantize has no CUDA backend in this "
                        "environment. Install a torchao build with MXFP8 CUDA "
                        "extensions (often source/nightly), or use "
                        "torchao.recipe='mxfp8_emulated'."
                    )

            mx_cfg = MXLinearConfig.from_recipe_name(recipe)
            choice_name = str(
                getattr(torchao_cfg, "mxfp8_dim1_cast_kernel_choice", "cuda")
            ).upper()
            try:
                mx_cfg.mxfp8_dim1_cast_kernel_choice = MXFP8Dim1CastKernelChoice[
                    choice_name
                ]
            except KeyError as exc:
                valid = [member.name.lower() for member in MXFP8Dim1CastKernelChoice]
                raise ValueError(
                    "Invalid torchao.mxfp8_dim1_cast_kernel_choice="
                    f"'{choice_name.lower()}'. Expected one of {valid}."
                ) from exc

            module_filter_fn = _build_linear_filter(
                filter_fqns=filter_fqns,
                first_linear_fqn=first_linear,
                last_linear_fqn=last_linear,
                skip_first_last_linear=bool(
                    getattr(torchao_cfg, "skip_first_last_linear", True)
                ),
            )
            quantize_(model, config=mx_cfg, filter_fn=module_filter_fn)
            converted_linear_count = _count_module_class_names(model, {"MXLinear"})
        except ImportError as exc:
            raise ImportError(
                "TorchAO MX recipe requested but required prototype APIs are unavailable. "
                "Install a compatible torchao build."
            ) from exc

    else:
        assert recipe in _QAT_RECIPES
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig

            module_filter_fn = _build_linear_filter(
                filter_fqns=filter_fqns,
                first_linear_fqn=first_linear,
                last_linear_fqn=last_linear,
                skip_first_last_linear=bool(
                    getattr(torchao_cfg, "skip_first_last_linear", True)
                ),
            )
            if recipe == "nvfp4_qat":
                from torchao.prototype.mx_formats import (
                    NVFP4DynamicActivationNVFP4WeightConfig,
                )

                base_config = NVFP4DynamicActivationNVFP4WeightConfig()
                qat_class_names = {"NVFP4FakeQuantizedLinear"}
            else:
                from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

                if not hasattr(torch, "float4_e2m1fn_x2"):
                    raise RuntimeError(
                        "mxfp4_qat requires torch.float4_e2m1fn_x2 support in this "
                        "PyTorch build."
                    )
                base_config = MXDynamicActivationMXWeightConfig(
                    activation_dtype=torch.float4_e2m1fn_x2,
                    weight_dtype=torch.float4_e2m1fn_x2,
                )
                qat_class_names = {"MXFakeQuantizedLinear"}

            quantize_(
                model,
                config=QATConfig(base_config=base_config, step="prepare"),
                filter_fn=module_filter_fn,
            )
            converted_linear_count = _count_module_class_names(model, qat_class_names)
            log.warning(
                "TorchAO recipe '%s' is experimental (QAT prototype path). "
                "Use for controlled experimentation only.",
                recipe,
            )
        except ImportError as exc:
            raise ImportError(
                "TorchAO QAT recipe requested but required QAT/prototype APIs are unavailable."
            ) from exc

    if converted_linear_count <= 0:
        log.warning(
            "TorchAO recipe '%s' enabled but converted 0 linear modules. "
            "Check filter rules and layer dimensions.",
            recipe,
        )
    else:
        log.info(
            "TorchAO recipe '%s' active: converted %s linear modules (%s path).",
            recipe,
            converted_linear_count,
            "accelerate" if used_accelerate_helper else "torchao",
        )

    return TorchAORuntimeState(
        enabled=True,
        recipe=recipe,
        converted_linear_count=converted_linear_count,
        post_optimizer_hook=post_optimizer_hook,
        used_accelerate_helper=used_accelerate_helper,
    )
