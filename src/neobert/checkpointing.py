"""Checkpoint I/O helpers for NeoBERT training and evaluation."""

import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import nn

MODEL_WEIGHTS_NAME = "model.safetensors"
logger = logging.getLogger(__name__)
_RUNTIME_PREFIXES = ("_orig_mod.", "module.")
_DEEPSPEED_TAG_DIR_PATTERNS = (
    "mp_rank_*_model_states.pt",
    "zero_pp_rank_*_mp_rank_*_optim_states.pt",
    "bf16_zero_pp_rank_*_mp_rank_*_optim_states.pt",
)
_DEEPSPEED_NESTED_TAG_CANDIDATES = ("pytorch_model", "model")


def _unwrap_compile_wrappers(model: nn.Module) -> nn.Module:
    """Return the base module beneath torch.compile wrappers.

    :param nn.Module model: Possibly wrapped model.
    :return nn.Module: Underlying model.
    """
    base = model
    while hasattr(base, "_orig_mod"):
        base = getattr(base, "_orig_mod")
    return base


def _strip_runtime_prefixes(key: str) -> str:
    """Strip runtime wrapper prefixes from a state-dict key.

    :param str key: Raw state-dict key.
    :return str: Canonicalized key.
    """
    while True:
        for prefix in _RUNTIME_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _state_dict_for_safetensors(
    raw_state_dict: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Create a safetensors-ready payload from a raw state dict.

    :param Mapping[str, Any] raw_state_dict: Raw model/state-dict mapping.
    :return dict[str, torch.Tensor]: Canonicalized contiguous CPU tensor payload.
    :raises ValueError: If multiple raw keys normalize to the same canonical key.
    """
    payload: dict[str, torch.Tensor] = {}
    seen_storage_ptrs: set[int] = set()
    for key, value in raw_state_dict.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = _strip_runtime_prefixes(str(key))
        if normalized_key in payload:
            raise ValueError(
                "State dict contains multiple keys that normalize to "
                f"'{normalized_key}' (for example '{key}')."
            )
        tensor = value.detach().cpu().contiguous()
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr in seen_storage_ptrs:
            # ``safetensors`` forbids shared storage; clone alias tensors so all
            # expected keys remain serializable.
            tensor = tensor.clone()
            storage_ptr = tensor.untyped_storage().data_ptr()
        seen_storage_ptrs.add(storage_ptr)
        payload[normalized_key] = tensor
    return payload


def _canonicalize_loaded_state_dict(
    raw_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Canonicalize loaded state-dict keys by stripping runtime wrapper prefixes.

    This keeps checkpoint loading tolerant of portable weight files created by
    generic save paths that may have preserved prefixes such as ``_orig_mod.``
    or ``module.``.

    :param Mapping[str, torch.Tensor] raw_state_dict: Loaded tensor mapping.
    :return dict[str, torch.Tensor]: Canonicalized state dict.
    :raises ValueError: If multiple raw keys normalize to the same canonical key.
    """
    payload: dict[str, torch.Tensor] = {}
    for raw_key, value in raw_state_dict.items():
        normalized_key = _strip_runtime_prefixes(str(raw_key))
        if normalized_key in payload:
            raise ValueError(
                "Loaded state dict contains multiple keys that normalize to "
                f"'{normalized_key}' (for example '{raw_key}')."
            )
        payload[normalized_key] = value
    return payload


def model_state_dict_for_safetensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create a safetensors-ready CPU state dict from a model.

    The payload is fully materialized on CPU and contiguous. This duplicates tied
    tensors intentionally, so all expected keys remain present in the checkpoint.

    :param nn.Module model: Model to serialize.
    :return dict[str, torch.Tensor]: Safetensors payload.
    :raises ValueError: If runtime wrapper prefixes collapse multiple keys.
    """
    base_model = _unwrap_compile_wrappers(model)
    return _state_dict_for_safetensors(base_model.state_dict())


def save_state_dict_safetensors(
    state_dict: Mapping[str, Any],
    checkpoint_dir: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save a pre-collected state dict to ``model.safetensors``.

    :param Mapping[str, Any] state_dict: Raw model state dict to serialize.
    :param str | Path checkpoint_dir: Target checkpoint directory.
    :param Mapping[str, str] | None metadata: Optional safetensors metadata.
    :return Path: Path to the saved safetensors file.
    :raises ValueError:
        If no tensors are present in ``state_dict`` or canonicalization collides.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / MODEL_WEIGHTS_NAME
    payload = _state_dict_for_safetensors(state_dict)
    if not payload:
        raise ValueError("Cannot save empty state dict to safetensors.")
    save_file(payload, str(weights_path), metadata=dict(metadata or {"format": "pt"}))
    return weights_path


def save_model_safetensors(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save model weights to ``model.safetensors``.

    :param nn.Module model: Model to serialize.
    :param str | Path checkpoint_dir: Target checkpoint directory.
    :param Mapping[str, str] | None metadata: Optional safetensors metadata.
    :return Path: Path to the saved safetensors file.
    :raises ValueError: If runtime wrapper prefixes collapse multiple keys.
    """
    return save_state_dict_safetensors(
        model_state_dict_for_safetensors(model),
        checkpoint_dir,
        metadata=metadata,
    )


def _is_deepspeed_tag_dir(path: Path) -> bool:
    """Return whether a path looks like a DeepSpeed ZeRO tag directory.

    :param Path path: Candidate checkpoint tag directory.
    :return bool: True when DeepSpeed shard files are present.
    """
    if not path.is_dir():
        return False
    return any(any(path.glob(pattern)) for pattern in _DEEPSPEED_TAG_DIR_PATTERNS)


def resolve_deepspeed_checkpoint_root_and_tag(
    checkpoint_dir: str | Path,
    *,
    tag: str | int | None = None,
) -> tuple[Path, str]:
    """Resolve DeepSpeed checkpoint root/tag across legacy and nested layouts.

    Supported layouts:
    - ``<root>/<tag>/...zero shards...``
    - ``<root>/<step>/pytorch_model/...zero shards...`` (Accelerate save_state)

    :param str | Path checkpoint_dir: Checkpoint root or step directory.
    :param str | int | None tag: Optional explicit tag/step.
    :return tuple[Path, str]: Resolved ``(root, tag)`` for zero-to-fp32 loaders.
    :raises FileNotFoundError: If no DeepSpeed checkpoint layout is found.
    :raises ValueError: If explicit ``tag`` is empty.
    """
    checkpoint_path = Path(checkpoint_dir)
    if tag is not None:
        tag_text = str(tag).strip()
        if not tag_text:
            raise ValueError("DeepSpeed checkpoint tag cannot be empty.")
        direct_tag_dir = checkpoint_path / tag_text
        if _is_deepspeed_tag_dir(direct_tag_dir):
            return checkpoint_path, tag_text
        for nested_tag in _DEEPSPEED_NESTED_TAG_CANDIDATES:
            nested_dir = direct_tag_dir / nested_tag
            if _is_deepspeed_tag_dir(nested_dir):
                return direct_tag_dir, nested_tag
        raise FileNotFoundError(
            "Unable to resolve DeepSpeed checkpoint tag "
            f"'{tag_text}' under {checkpoint_path}."
        )

    if _is_deepspeed_tag_dir(checkpoint_path):
        return checkpoint_path.parent, checkpoint_path.name

    for nested_tag in _DEEPSPEED_NESTED_TAG_CANDIDATES:
        nested_dir = checkpoint_path / nested_tag
        if _is_deepspeed_tag_dir(nested_dir):
            return checkpoint_path, nested_tag

    latest_path = checkpoint_path / "latest"
    if latest_path.is_file():
        latest_tag = latest_path.read_text(encoding="utf-8").strip()
        if not latest_tag:
            raise ValueError(f"DeepSpeed latest file is empty: {latest_path}")
        return resolve_deepspeed_checkpoint_root_and_tag(
            checkpoint_path,
            tag=latest_tag,
        )

    raise FileNotFoundError(
        "Unable to resolve DeepSpeed checkpoint under "
        f"{checkpoint_path}. Expected either a ZeRO tag dir, "
        f"nested tag ({', '.join(_DEEPSPEED_NESTED_TAG_CANDIDATES)}), or a latest file."
    )


def load_deepspeed_fp32_state_dict(
    checkpoint_dir: str | Path,
    *,
    tag: str | int | None = None,
) -> dict[str, torch.Tensor]:
    """Load fp32 weights from a DeepSpeed ZeRO checkpoint layout.

    :param str | Path checkpoint_dir: Checkpoint root or step directory.
    :param str | int | None tag: Optional explicit root-level tag/step.
    :return dict[str, torch.Tensor]: Materialized fp32 model state dict.
    :raises ModuleNotFoundError:
        If the optional DeepSpeed checkpoint-conversion dependency is missing.
    :raises ValueError: If conversion returns an empty state dict.
    """
    resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(
        checkpoint_dir,
        tag=tag,
    )
    try:
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DeepSpeed checkpoint conversion requires the optional legacy "
            "checkpoint dependency. Install `neobert[legacy-checkpoints]` "
            "before loading DeepSpeed ZeRO checkpoints."
        ) from exc
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        str(resolved_root),
        tag=str(resolved_tag),
    )
    if not state_dict:
        raise ValueError(
            "DeepSpeed checkpoint conversion produced an empty state dict from "
            f"{resolved_root} (tag={resolved_tag})."
        )
    return state_dict


def load_model_safetensors(
    checkpoint_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load model weights from ``model.safetensors``.

    Runtime wrapper prefixes such as ``_orig_mod.`` and ``module.`` are stripped
    on read so callers can consume portable weights produced by either repo
    helpers or generic runtime save paths.

    :param str | Path checkpoint_dir: Checkpoint directory path.
    :param str | torch.device map_location: Device for loaded tensors.
    :return dict[str, torch.Tensor]: Loaded state dict.
    :raises FileNotFoundError: If the safetensors file is missing.
    :raises ValueError: If the loaded payload is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / MODEL_WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"No {MODEL_WEIGHTS_NAME} found at {weights_path}")
    state_dict = load_file(str(weights_path), device=str(map_location))
    if not state_dict:
        raise ValueError(f"Loaded state dict is empty from {weights_path}")
    return _canonicalize_loaded_state_dict(state_dict)


def _checkpoint_path_matches_tag(checkpoint_path: Path, checkpoint: str | int) -> bool:
    """Return whether ``checkpoint_path`` already points at ``checkpoint``.

    This accepts both direct step directories (``.../<tag>``) and nested
    Accelerate DeepSpeed layouts (``.../<tag>/pytorch_model``). Parent-name
    matching is intentionally restricted to known nested DeepSpeed tag dirs so
    numerically named parents such as ``.../<run_id>/checkpoints`` do not
    masquerade as a resolved explicit checkpoint tag.

    :param Path checkpoint_path: Candidate direct checkpoint path.
    :param str | int checkpoint: Requested checkpoint tag/step.
    :return bool: True when the path already targets the requested tag.
    """
    requested_tag = str(checkpoint).strip()
    return bool(requested_tag) and (
        checkpoint_path.name == requested_tag
        or (
            checkpoint_path.name in _DEEPSPEED_NESTED_TAG_CANDIDATES
            and checkpoint_path.parent.name == requested_tag
            and _is_deepspeed_tag_dir(checkpoint_path)
        )
    )


def _resolve_direct_checkpoint_tag(checkpoint_path: Path) -> str | None:
    """Return the step tag when ``checkpoint_path`` already targets one step.

    This accepts direct portable step directories, direct DeepSpeed ZeRO tag
    directories, and nested Accelerate layouts such as ``<step>/pytorch_model``.

    :param Path checkpoint_path: Candidate direct checkpoint path.
    :return str | None: Concrete step/tag when the path is already resolved.
    """
    if (checkpoint_path / MODEL_WEIGHTS_NAME).is_file():
        return checkpoint_path.name

    if _is_deepspeed_tag_dir(checkpoint_path):
        if checkpoint_path.name in _DEEPSPEED_NESTED_TAG_CANDIDATES:
            return checkpoint_path.parent.name
        return checkpoint_path.name

    for nested_tag in _DEEPSPEED_NESTED_TAG_CANDIDATES:
        if _is_deepspeed_tag_dir(checkpoint_path / nested_tag):
            return checkpoint_path.name

    return None


def _is_loadable_step_checkpoint(checkpoint_root: Path, step: int) -> bool:
    """Return whether ``step`` can be loaded from ``checkpoint_root``.

    :param Path checkpoint_root: Root directory containing checkpoint steps.
    :param int step: Numeric checkpoint step to validate.
    :return bool: True when either portable or DeepSpeed weights are loadable.
    """
    step_dir = checkpoint_root / str(step)
    if (step_dir / MODEL_WEIGHTS_NAME).is_file():
        return True
    try:
        resolve_deepspeed_checkpoint_root_and_tag(checkpoint_root, tag=str(step))
    except (FileNotFoundError, ValueError):
        return False
    return True


def resolve_step_checkpoint_selector(
    checkpoint_root: str | Path,
    checkpoint: str | int,
) -> str:
    """Resolve ``checkpoint`` to a concrete step/tag for loading.

    ``latest`` honors an already-selected direct checkpoint path first, then a
    root-level DeepSpeed ``latest`` file when present. When neither exists, scan
    for the highest loadable numbered step so portable checkpoint roots without
    DeepSpeed metadata still work.

    :param str | Path checkpoint_root: Root directory or direct checkpoint path.
    :param str | int checkpoint: Requested checkpoint selector.
    :return str: Concrete checkpoint tag to load.
    :raises ValueError: If a DeepSpeed ``latest`` file is empty.
    """
    checkpoint_root = Path(checkpoint_root)
    requested_tag = str(checkpoint).strip()
    if requested_tag.lower() != "latest":
        return requested_tag

    direct_tag = _resolve_direct_checkpoint_tag(checkpoint_root)
    if direct_tag is not None:
        return direct_tag

    latest_path = checkpoint_root / "latest"
    if latest_path.is_file():
        latest_tag = latest_path.read_text(encoding="utf-8").strip()
        if not latest_tag:
            raise ValueError(f"DeepSpeed latest file is empty: {latest_path}")
        return latest_tag

    candidates = sorted(
        (
            int(path.name)
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        reverse=True,
    )
    for step in candidates:
        if _is_loadable_step_checkpoint(checkpoint_root, step):
            return str(step)
    return requested_tag


def resolve_step_checkpoint_dir(
    checkpoint_path: str | Path,
    checkpoint: str | int,
) -> Path:
    """Resolve the checkpoint directory for portable-weight loading.

    ``checkpoint_path`` may point either at a checkpoint root containing
    ``<tag>/`` subdirectories or at a single step directory already.

    :param str | Path checkpoint_path: User-provided checkpoint path.
    :param str | int checkpoint: Requested checkpoint tag/step.
    :return Path: Resolved candidate checkpoint directory.
    :raises FileNotFoundError:
        If an explicit checkpoint tag is missing beneath a direct checkpoint path
        that already contains portable weights.
    """
    checkpoint_root = Path(checkpoint_path)
    requested_tag = str(checkpoint).strip()
    candidate = checkpoint_root / requested_tag
    if candidate.is_dir():
        return candidate
    if _checkpoint_path_matches_tag(checkpoint_root, requested_tag):
        return checkpoint_root
    if (checkpoint_root / MODEL_WEIGHTS_NAME).is_file():
        raise FileNotFoundError(
            f"Requested checkpoint '{requested_tag}' was not found under "
            f"{checkpoint_root}. Refusing to silently load portable weights from "
            "the root path instead."
        )
    return checkpoint_root


def load_step_checkpoint_state_dict(
    checkpoint_path: str | Path,
    checkpoint: str | int,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load portable or DeepSpeed model weights for a checkpoint selector.

    Portable ``model.safetensors`` payloads are preferred when present. Legacy
    DeepSpeed ZeRO checkpoints remain supported through the optional
    ``neobert[legacy-checkpoints]`` dependency.

    :param str | Path checkpoint_path: Checkpoint root or step directory.
    :param str | int checkpoint: Requested checkpoint tag/step.
    :param str | torch.device map_location: Target device for safetensors loading.
    :return dict[str, torch.Tensor]: Loaded model state dict.
    """
    checkpoint_root = Path(checkpoint_path)
    requested_tag = resolve_step_checkpoint_selector(checkpoint_root, checkpoint)
    checkpoint_dir = resolve_step_checkpoint_dir(checkpoint_root, requested_tag)
    weights_path = checkpoint_dir / MODEL_WEIGHTS_NAME
    if weights_path.is_file():
        return load_model_safetensors(checkpoint_dir, map_location=map_location)

    try:
        return load_deepspeed_fp32_state_dict(checkpoint_root, tag=requested_tag)
    except (FileNotFoundError, ValueError):
        resolved_root = checkpoint_root.resolve()
        if _checkpoint_path_matches_tag(resolved_root, requested_tag):
            return load_deepspeed_fp32_state_dict(resolved_root)
        raise


def resolve_checkpoint_retention_limit(cfg: Any) -> int:
    """Resolve effective checkpoint retention limit from trainer config.

    ``trainer.save_total_limit`` is preferred. Deprecated ``trainer.max_ckpt``
    is used only as a fallback when ``save_total_limit`` is unset.

    :param Any cfg: Runtime config object or ``cfg.trainer``.
    :return int: Maximum number of retained checkpoints (0 disables pruning).
    """
    trainer_cfg = getattr(cfg, "trainer", cfg)
    save_total_limit = getattr(trainer_cfg, "save_total_limit", None)
    if save_total_limit is not None:
        return max(0, int(save_total_limit))
    max_ckpt = getattr(trainer_cfg, "max_ckpt", None)
    if max_ckpt is not None:
        return max(0, int(max_ckpt))
    return 0


def prune_step_checkpoints(checkpoint_dir: str | Path, retention_limit: int) -> None:
    """Prune old numeric step checkpoint folders in ``checkpoint_dir``.

    This helper is best-effort and resilient to concurrent filesystem mutations.
    It never raises on a missing/deleted checkpoint directory.

    :param str | Path checkpoint_dir: Root directory containing ``<step>/`` folders.
    :param int retention_limit: Number of newest numeric checkpoints to keep.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if retention_limit <= 0 or not checkpoint_dir.exists():
        return

    checkpoints: list[tuple[int, Path]] = []
    for item_path in checkpoint_dir.iterdir():
        if not item_path.is_dir():
            continue
        try:
            checkpoints.append((int(item_path.name), item_path))
        except ValueError:
            continue

    if len(checkpoints) <= retention_limit:
        return

    checkpoints.sort(key=lambda item: item[0])
    for _, old_path in checkpoints[: len(checkpoints) - retention_limit]:
        try:
            shutil.rmtree(old_path)
            logger.info(
                "Removed old checkpoint: %s (limit=%d)", old_path, retention_limit
            )
        except FileNotFoundError:
            logger.warning("Checkpoint already removed before prune: %s", old_path)
        except OSError as exc:
            logger.warning("Failed to remove old checkpoint %s: %s", old_path, exc)


def save_portable_checkpoint_weights(
    model: nn.Module,
    accelerator: Any,
    checkpoint_path: str | Path,
    *,
    skip_if_exists: bool = False,
) -> bool:
    """Save backend-agnostic ``model.safetensors`` into a step checkpoint.

    :param nn.Module model: Prepared training model.
    :param Any accelerator: Active accelerator runtime.
    :param str | Path checkpoint_path: Step checkpoint directory path.
    :param bool skip_if_exists: Return early when a portable file already exists.
    :return bool: True when portable weights were saved (or already existed).
    """
    checkpoint_path = Path(checkpoint_path)
    weights_path = checkpoint_path / MODEL_WEIGHTS_NAME
    if skip_if_exists and weights_path.exists():
        return True

    try:
        # Distributed backends (FSDP/FSDP2/DeepSpeed) may require all ranks to
        # participate in state-dict collection collectives even when only rank 0
        # persists the portable safetensors payload.
        state_dict = accelerator.get_state_dict(model, unwrap=True)
    except Exception as exc:
        if getattr(accelerator, "is_main_process", True):
            logger.warning(
                "Unable to export portable checkpoint weights to %s: %s. "
                "Resumable state was still saved.",
                weights_path,
                exc,
            )
        return False

    if not getattr(accelerator, "is_main_process", True):
        return False

    try:
        weights_path = save_state_dict_safetensors(
            state_dict,
            checkpoint_path,
            metadata={"format": "pt", "source": "accelerate.get_state_dict"},
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist portable checkpoint weights at %s: %s. "
            "Resumable state was still saved.",
            weights_path,
            exc,
        )
        return False

    logger.info("Saved portable model weights to %s.", weights_path)
    return True
