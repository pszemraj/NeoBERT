"""Checkpoint I/O helpers for NeoBERT training and evaluation."""

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import nn

MODEL_WEIGHTS_NAME = "model.safetensors"
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
    for prefix in ("_orig_mod.", "module."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _state_dict_for_safetensors(
    raw_state_dict: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Create a safetensors-ready payload from a raw state dict.

    :param Mapping[str, Any] raw_state_dict: Raw model/state-dict mapping.
    :return dict[str, torch.Tensor]: Canonicalized contiguous CPU tensor payload.
    """
    payload: dict[str, torch.Tensor] = {}
    seen_storage_ptrs: set[int] = set()
    for key, value in raw_state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr in seen_storage_ptrs:
            # ``safetensors`` forbids shared storage; clone alias tensors so all
            # expected keys remain serializable.
            tensor = tensor.clone()
            storage_ptr = tensor.untyped_storage().data_ptr()
        seen_storage_ptrs.add(storage_ptr)
        payload[_strip_runtime_prefixes(str(key))] = tensor
    return payload


def model_state_dict_for_safetensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create a safetensors-ready CPU state dict from a model.

    The payload is fully materialized on CPU and contiguous. This duplicates tied
    tensors intentionally, so all expected keys remain present in the checkpoint.

    :param nn.Module model: Model to serialize.
    :return dict[str, torch.Tensor]: Safetensors payload.
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
    :raises ValueError: If no tensors are present in ``state_dict``.
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
    :raises ValueError: If conversion returns an empty state dict.
    """
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(
        checkpoint_dir,
        tag=tag,
    )
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
    return state_dict
