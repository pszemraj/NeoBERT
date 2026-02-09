"""Transformer Engine pretraining integration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from accelerate.utils import DistributedType

LOGGER = logging.getLogger(__name__)

_CANONICAL_RECIPES = {
    "none",
    "fp8_delayed",
    "fp8_current",
    "mxfp8",
    "nvfp4",
}
_RECIPE_ALIASES = {
    "none": "none",
    "fp8_delayed": "fp8_delayed",
    "delayed": "fp8_delayed",
    "delayed_scaling": "fp8_delayed",
    "fp8": "fp8_delayed",
    "fp8_current": "fp8_current",
    "current": "fp8_current",
    "current_scaling": "fp8_current",
    "mxfp8": "mxfp8",
    "mxfp8_block": "mxfp8",
    "mxfp8_block_scaling": "mxfp8",
    "nvfp4": "nvfp4",
    "nvfp4_block": "nvfp4",
    "nvfp4_block_scaling": "nvfp4",
}


@dataclass
class TransformerEngineRuntimeState:
    """Runtime state for Transformer Engine pretraining integration."""

    enabled: bool = False
    recipe: str = "none"
    converted_linear_count: int = 0
    converted_layernorm_count: int = 0
    used_accelerate_helper: bool = False
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


def _resolve_format(format_name: Any, recipe_module: Any) -> Any:
    """Resolve a ``transformer_engine.common.recipe.Format`` value."""
    fmt = str(format_name or "HYBRID").strip().upper()
    try:
        return getattr(recipe_module.Format, fmt)
    except AttributeError as exc:
        valid = [member.name for member in recipe_module.Format]
        raise ValueError(
            f"Invalid transformer_engine.fp8_format='{fmt}'. Expected one of {valid}."
        ) from exc


def _query_support(
    support_fn: Optional[Callable[..., Any]],
) -> tuple[Optional[bool], str]:
    """Query backend support check function when available."""
    if support_fn is None:
        return None, ""
    try:
        response = support_fn(return_reason=True)
    except TypeError:
        response = support_fn()
    if isinstance(response, tuple):
        supported, reason = response
        return bool(supported), str(reason or "")
    return bool(response), ""


def _build_recipe(recipe: str, te_cfg: Any, te_recipe: Any) -> Any:
    """Build a Transformer Engine recipe object."""
    fp8_format = _resolve_format(getattr(te_cfg, "fp8_format", "HYBRID"), te_recipe)
    margin = int(getattr(te_cfg, "margin", 0))
    amax_history_len = int(getattr(te_cfg, "amax_history_len", 1024))
    amax_compute_algo = str(getattr(te_cfg, "amax_compute_algo", "most_recent"))

    if recipe == "fp8_delayed":
        return te_recipe.DelayedScaling(
            margin=margin,
            fp8_format=fp8_format,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
    if recipe == "fp8_current":
        return te_recipe.Float8CurrentScaling(fp8_format=fp8_format)
    if recipe == "mxfp8":
        return te_recipe.MXFP8BlockScaling(
            margin=margin,
            fp8_format=fp8_format,
        )
    assert recipe == "nvfp4"
    return te_recipe.NVFP4BlockScaling(
        disable_rht=bool(getattr(te_cfg, "disable_rht", False)),
        disable_stochastic_rounding=bool(
            getattr(te_cfg, "disable_stochastic_rounding", False)
        ),
        disable_2d_quantization=bool(
            getattr(te_cfg, "disable_2d_quantization", False)
        ),
    )


def _make_te_linear(te_pytorch: Any, module: nn.Linear) -> nn.Module:
    """Construct a Transformer Engine linear layer matching ``module``."""
    kwargs = {
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
    }
    try:
        return te_pytorch.Linear(params_dtype=module.weight.dtype, **kwargs)
    except TypeError:
        return te_pytorch.Linear(**kwargs)


def _make_te_layernorm(te_pytorch: Any, module: nn.LayerNorm) -> nn.Module:
    """Construct a Transformer Engine layernorm layer matching ``module``."""
    if isinstance(module.normalized_shape, tuple):
        hidden_size = int(module.normalized_shape[0])
    else:
        hidden_size = int(module.normalized_shape)
    kwargs = {"hidden_size": hidden_size, "eps": module.eps}
    try:
        return te_pytorch.LayerNorm(params_dtype=module.weight.dtype, **kwargs)
    except TypeError:
        return te_pytorch.LayerNorm(**kwargs)


def _copy_module_params(src: nn.Module, dst: nn.Module) -> None:
    """Copy matching ``weight``/``bias`` tensors from ``src`` to ``dst``."""
    with torch.no_grad():
        if hasattr(src, "weight") and hasattr(dst, "weight"):
            src_weight = getattr(src, "weight", None)
            dst_weight = getattr(dst, "weight", None)
            if src_weight is not None and dst_weight is not None:
                dst_weight.copy_(src_weight)
        if hasattr(src, "bias") and hasattr(dst, "bias"):
            src_bias = getattr(src, "bias", None)
            dst_bias = getattr(dst, "bias", None)
            if src_bias is not None and dst_bias is not None:
                dst_bias.copy_(src_bias)


def _convert_model_to_te_layers(
    model: nn.Module,
    *,
    te_pytorch: Any,
    filter_fqns: list[str],
    skip_first_last_linear: bool,
    convert_layernorm: bool,
) -> tuple[int, int]:
    """Convert selected ``nn.Linear``/``nn.LayerNorm`` modules to TE modules."""
    first_linear, last_linear = _get_first_last_linear_fqns(model)
    converted_linear_count = 0
    converted_layernorm_count = 0

    def _should_skip_by_fqn(fqn: str) -> bool:
        return any(skip and skip in fqn for skip in filter_fqns)

    def _convert_in_place(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted_linear_count, converted_layernorm_count
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    continue
                if skip_first_last_linear and child_fqn in {first_linear, last_linear}:
                    continue
                if _should_skip_by_fqn(child_fqn):
                    continue
                te_linear = _make_te_linear(te_pytorch, child)
                _copy_module_params(child, te_linear)
                setattr(parent, child_name, te_linear)
                converted_linear_count += 1
                continue
            if isinstance(child, nn.LayerNorm) and convert_layernorm:
                if getattr(child, "weight", None) is None:
                    continue
                if _should_skip_by_fqn(child_fqn):
                    continue
                te_layernorm = _make_te_layernorm(te_pytorch, child)
                _copy_module_params(child, te_layernorm)
                setattr(parent, child_name, te_layernorm)
                converted_layernorm_count += 1
                continue
            _convert_in_place(child, child_fqn)

    _convert_in_place(model)
    return converted_linear_count, converted_layernorm_count


def _wrap_forward_with_fp8_autocast(
    model: nn.Module,
    *,
    te_pytorch: Any,
    te_recipe_obj: Any,
    use_during_eval: bool,
    prefer_accelerate_helper: bool,
    logger: logging.Logger,
) -> bool:
    """Wrap ``model.forward`` in TE fp8 autocast context."""
    if prefer_accelerate_helper:
        try:
            from accelerate.utils.transformer_engine import contextual_fp8_autocast

            new_forward = contextual_fp8_autocast(
                model.forward,
                te_recipe_obj,
                use_during_eval,
            )
            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
            else:
                model.forward = new_forward
            return True
        except ImportError:
            pass
        except Exception as exc:
            logger.warning(
                "Accelerate TE autocast helper failed (%s); falling back to direct "
                "transformer_engine autocast wrapper.",
                exc,
            )

    fp8_autocast = getattr(te_pytorch, "fp8_autocast", None)
    if fp8_autocast is None:
        fp8_autocast = getattr(te_pytorch, "autocast", None)
    if fp8_autocast is None:
        raise RuntimeError(
            "Transformer Engine runtime does not expose fp8_autocast/autocast."
        )

    model_forward = model.forward

    def _forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
        enabled = use_during_eval or self.training
        try:
            with fp8_autocast(enabled=enabled, fp8_recipe=te_recipe_obj):
                return model_forward(*args, **kwargs)
        except TypeError:
            with fp8_autocast(enabled=enabled, recipe=te_recipe_obj):
                return model_forward(*args, **kwargs)

    _forward.__wrapped__ = model_forward  # type: ignore[attr-defined]
    if hasattr(model_forward, "__func__"):
        model.forward = MethodType(_forward, model)
    else:
        model.forward = _forward  # type: ignore[assignment]
    return False


def apply_transformer_engine_pretraining_quantization(
    model: nn.Module,
    cfg: Any,
    *,
    accelerator: Any,
    logger: Optional[logging.Logger] = None,
) -> TransformerEngineRuntimeState:
    """Apply Transformer Engine quantized-training transforms for pretraining."""
    log = logger or LOGGER
    te_cfg = getattr(cfg, "transformer_engine", None)
    if te_cfg is None or not bool(getattr(te_cfg, "enable", False)):
        return TransformerEngineRuntimeState(enabled=False, recipe="none")

    recipe = _normalize_recipe(getattr(te_cfg, "recipe", "none"))
    if recipe not in _CANONICAL_RECIPES:
        valid = sorted(_CANONICAL_RECIPES)
        raise ValueError(
            f"Unknown transformer_engine.recipe='{recipe}'. Supported values: {valid}."
        )
    if recipe == "none":
        return TransformerEngineRuntimeState(enabled=False, recipe="none")

    if str(getattr(cfg, "task", "pretraining")).strip().lower() != "pretraining":
        raise ValueError(
            "Transformer Engine integration is currently supported for "
            "task='pretraining' only."
        )

    if bool(getattr(te_cfg, "require_compile", True)) and not bool(
        getattr(cfg.trainer, "torch_compile", False)
    ):
        raise ValueError(
            "Transformer Engine quantized training requires trainer.torch_compile=true. "
            "Set trainer.torch_compile=true or transformer_engine.require_compile=false."
        )
    if not bool(getattr(cfg.trainer, "torch_compile", False)):
        log.warning(
            "Transformer Engine is enabled with trainer.torch_compile=false. This is "
            "typically slower and can be less stable than compiled mode."
        )

    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        raise RuntimeError(
            "Transformer Engine quantized pretraining is not supported with DeepSpeed "
            "in this release. Use DDP or FSDP2."
        )

    try:
        import transformer_engine.common.recipe as te_recipe
        import transformer_engine.pytorch as te_pytorch
    except ImportError as exc:
        raise ImportError(
            "Transformer Engine recipe requested but transformer_engine is not "
            "available. Install a compatible build (for example: "
            "pip install --no-build-isolation -e .[quant_te])."
        ) from exc

    support_fn = None
    try:
        from transformer_engine.pytorch import quantization as te_quantization

        if recipe in {"fp8_delayed", "fp8_current"}:
            support_fn = getattr(te_quantization, "is_fp8_available", None)
        elif recipe == "mxfp8":
            support_fn = getattr(te_quantization, "is_mxfp8_available", None)
        else:
            support_fn = getattr(te_quantization, "is_nvfp4_available", None)
    except ImportError:
        support_fn = None

    supported, reason = _query_support(support_fn)
    if supported is False:
        suffix = f" ({reason})" if reason else ""
        raise RuntimeError(
            "Transformer Engine recipe "
            f"'{recipe}' is not available in this environment{suffix}."
        )

    if int(getattr(te_cfg, "interval", 1)) != 1:
        log.warning(
            "transformer_engine.interval=%s is ignored by this runtime adapter.",
            getattr(te_cfg, "interval", 1),
        )

    te_recipe_obj = _build_recipe(recipe, te_cfg, te_recipe)
    prefer_accelerate_helper = True
    if recipe == "nvfp4":
        prefer_accelerate_helper = False
        if bool(getattr(cfg.datacollator, "pack_sequences", False)) and not bool(
            getattr(te_cfg, "disable_2d_quantization", False)
        ):
            log.warning(
                "TE NVFP4 with packed sequences can fail on some Blackwell stacks "
                "when 2D quantization is enabled. Forcing "
                "transformer_engine.disable_2d_quantization=true for this run."
            )
            te_cfg.disable_2d_quantization = True
            te_recipe_obj = _build_recipe(recipe, te_cfg, te_recipe)

    filter_fqns = list(getattr(te_cfg, "filter_fqns", []) or [])
    converted_linear_count, converted_layernorm_count = _convert_model_to_te_layers(
        model,
        te_pytorch=te_pytorch,
        filter_fqns=filter_fqns,
        skip_first_last_linear=bool(getattr(te_cfg, "skip_first_last_linear", True)),
        convert_layernorm=bool(getattr(te_cfg, "convert_layernorm", False)),
    )

    if converted_linear_count <= 0 and converted_layernorm_count <= 0:
        log.warning(
            "Transformer Engine recipe '%s' enabled but converted 0 modules. "
            "Check filter rules and layer dimensions.",
            recipe,
        )
        used_accelerate_helper = False
    else:
        used_accelerate_helper = _wrap_forward_with_fp8_autocast(
            model,
            te_pytorch=te_pytorch,
            te_recipe_obj=te_recipe_obj,
            use_during_eval=bool(getattr(te_cfg, "use_autocast_during_eval", False)),
            prefer_accelerate_helper=prefer_accelerate_helper,
            logger=log,
        )
        log.info(
            "Transformer Engine recipe '%s' active: converted %s linear and %s "
            "layernorm modules (%s autocast wrapper).",
            recipe,
            converted_linear_count,
            converted_layernorm_count,
            "accelerate" if used_accelerate_helper else "native",
        )

    return TransformerEngineRuntimeState(
        enabled=True,
        recipe=recipe,
        converted_linear_count=converted_linear_count,
        converted_layernorm_count=converted_layernorm_count,
        used_accelerate_helper=used_accelerate_helper,
        post_optimizer_hook=None,
    )
