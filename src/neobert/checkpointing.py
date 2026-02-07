"""Checkpoint I/O helpers for NeoBERT training and evaluation."""

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import nn

MODEL_WEIGHTS_NAME = "model.safetensors"


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


def model_state_dict_for_safetensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create a safetensors-ready CPU state dict from a model.

    The payload is fully materialized on CPU and contiguous. This duplicates tied
    tensors intentionally, so all expected keys remain present in the checkpoint.

    :param nn.Module model: Model to serialize.
    :return dict[str, torch.Tensor]: Safetensors payload.
    """
    base_model = _unwrap_compile_wrappers(model)
    raw_state_dict = base_model.state_dict()
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
        payload[_strip_runtime_prefixes(key)] = tensor
    return payload


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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / MODEL_WEIGHTS_NAME
    payload = model_state_dict_for_safetensors(model)
    save_file(payload, str(weights_path), metadata=dict(metadata or {"format": "pt"}))
    return weights_path


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
