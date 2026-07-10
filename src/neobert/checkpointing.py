"""Checkpoint I/O helpers for NeoBERT training and evaluation."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

MODEL_WEIGHTS_NAME = "model.safetensors"
ACCELERATE_STATE_DIR = "accelerate"
OPTIMIZER_PARAM_NAMES_MANIFEST = "optimizer_param_names.json"
CHECKPOINT_COMPLETE_NAME = "checkpoint_complete.json"
CHECKPOINT_COMPLETE_VERSION = 2
logger = logging.getLogger(__name__)
_RUNTIME_PREFIXES = ("_orig_mod.", "module.")
_DEEPSPEED_TAG_DIR_PATTERNS = (
    "mp_rank_*_model_states.pt",
    "zero_pp_rank_*_mp_rank_*_optim_states.pt",
    "bf16_zero_pp_rank_*_mp_rank_*_optim_states.pt",
)
_DEEPSPEED_NESTED_TAG_CANDIDATES = ("pytorch_model", "model")


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a path is a nonempty regular file.

    :param Path path: Artifact path.
    :return bool: True when the file exists and contains bytes.
    """
    return path.is_file() and path.stat().st_size > 0


def is_step_checkpoint_name(value: object) -> bool:
    """Return whether a value is an ASCII decimal checkpoint step name.

    :param object value: Candidate directory name or selector.
    :return bool: True for nonempty names containing only ASCII digits.
    """
    name = str(value)
    return bool(name) and name.isascii() and name.isdigit()


def optimizer_param_name_manifest_schema_errors(payload: object) -> list[str]:
    """Return schema errors for an optimizer parameter-name manifest payload.

    :param object payload: Decoded JSON manifest payload.
    :return list[str]: Human-readable schema validation errors.
    """
    if not isinstance(payload, Mapping):
        return ["manifest must be a JSON object"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(payload.get("state_semantics"), str):
        errors.append("state_semantics must be a string")
    param_groups = payload.get("param_name_groups")
    if not isinstance(param_groups, list) or not all(
        isinstance(group, list) for group in param_groups
    ):
        errors.append("param_name_groups must be a list of lists")
    return errors


def _checkpoint_resume_artifact_errors(checkpoint_path: Path) -> list[str]:
    """Return missing resume-critical artifacts other than the completion marker.

    :param Path checkpoint_path: Candidate step checkpoint directory.
    :return list[str]: Human-readable validation errors.
    """
    errors: list[str] = []
    if not checkpoint_path.is_dir():
        return ["step directory is missing"]
    optimizer_manifest_path = checkpoint_path / OPTIMIZER_PARAM_NAMES_MANIFEST
    if not _is_nonempty_file(optimizer_manifest_path):
        errors.append(f"missing {OPTIMIZER_PARAM_NAMES_MANIFEST}")
    else:
        try:
            optimizer_manifest = json.loads(
                optimizer_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {OPTIMIZER_PARAM_NAMES_MANIFEST}: {exc}")
        else:
            if optimizer_param_name_manifest_schema_errors(optimizer_manifest):
                errors.append(f"invalid {OPTIMIZER_PARAM_NAMES_MANIFEST} schema")
    return errors


def _task_checkpoint_artifact_errors(
    checkpoint_path: Path,
    *,
    task: object,
) -> list[str]:
    """Return missing task metadata required by current self-contained checkpoints.

    :param Path checkpoint_path: Candidate step checkpoint directory.
    :param object task: Task recorded by the completion marker.
    :return list[str]: Human-readable validation errors.
    """
    task_name = str(task)
    if task_name not in {"pretraining", "contrastive", "glue"}:
        return [f"unsupported checkpoint task {task!r}"]

    errors: list[str] = []
    config_payload: Mapping[str, Any] | None = None
    config_path = checkpoint_path / "config.yaml"
    if not config_path.is_file():
        errors.append("missing config.yaml")
    else:
        try:
            config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid config.yaml: {exc}")
        else:
            if not isinstance(config_payload, Mapping):
                config_payload = None
            config_task = config_payload.get("task") if config_payload else None
            if config_task != task_name:
                errors.append(
                    f"config task {config_task!r} does not match marker task "
                    f"{task_name!r}"
                )

    tokenizer_dir = checkpoint_path / "tokenizer"
    if not tokenizer_dir.is_dir() or not any(
        _is_nonempty_file(path) for path in tokenizer_dir.rglob("*")
    ):
        errors.append("missing usable tokenizer/ artifacts")
    if not _is_nonempty_file(checkpoint_path / MODEL_WEIGHTS_NAME):
        errors.append(f"missing {MODEL_WEIGHTS_NAME}")

    accelerate_dir = checkpoint_path / ACCELERATE_STATE_DIR
    accelerate_artifacts = (
        tuple(path for path in accelerate_dir.rglob("*") if _is_nonempty_file(path))
        if accelerate_dir.is_dir()
        else ()
    )
    accelerate_entries = tuple(
        (path, path.relative_to(accelerate_dir).parts) for path in accelerate_artifacts
    )
    has_model_state = any(
        path.name
        in {"model.safetensors", "pytorch_model.bin", "pytorch_model_fsdp.bin"}
        or any(part.startswith("pytorch_model_fsdp_") for part in relative_parts)
        for path, relative_parts in accelerate_entries
    )
    has_optimizer_state = any(
        path.name == "optimizer.bin"
        or (path.name.startswith("optimizer_") and path.suffix == ".bin")
        or any(part.startswith("optimizer_") for part in relative_parts)
        for path, relative_parts in accelerate_entries
    )
    has_scheduler_state = any(
        path.name == "scheduler.bin"
        or (path.name.startswith("scheduler_") and path.suffix == ".bin")
        for path, _ in accelerate_entries
    )
    has_rng_state = any(
        path.name.startswith("random_states_") and path.suffix == ".pkl"
        for path, _ in accelerate_entries
    )
    for present, role in (
        (has_model_state, "model"),
        (has_optimizer_state, "optimizer"),
        (has_scheduler_state, "scheduler"),
        (has_rng_state, "RNG"),
    ):
        if not present:
            errors.append(f"missing Accelerate {role} state")

    custom_state_files = (
        tuple(
            path
            for path in accelerate_dir.rglob("custom_checkpoint_*.pkl")
            if _is_nonempty_file(path)
        )
        if accelerate_dir.is_dir()
        else ()
    )
    expected_custom_states = {"pretraining": 2, "contrastive": 1, "glue": 1}[task_name]
    if len(custom_state_files) != expected_custom_states:
        errors.append(
            "wrong checkpointable-state count in accelerate/: "
            f"expected {expected_custom_states}, found {len(custom_state_files)}"
        )

    model_payload = config_payload.get("model") if config_payload else None
    from_hub = bool(
        model_payload.get("from_hub", False)
        if isinstance(model_payload, Mapping)
        else False
    )
    if task_name == "glue" and from_hub:
        model_config_dir = checkpoint_path / "model_config"
        if not model_config_dir.is_dir() or not any(
            _is_nonempty_file(path) for path in model_config_dir.rglob("*")
        ):
            errors.append("missing GLUE model_config/ artifacts")
    return errors


def _checkpoint_artifact_inventory(checkpoint_path: Path) -> list[dict[str, Any]]:
    """Return a deterministic size inventory for checkpoint artifacts.

    :param Path checkpoint_path: Completed step checkpoint directory.
    :return list[dict[str, Any]]: Relative paths and byte sizes.
    """
    excluded = {CHECKPOINT_COMPLETE_NAME, f".{CHECKPOINT_COMPLETE_NAME}.tmp"}
    return [
        {
            "path": path.relative_to(checkpoint_path).as_posix(),
            "size": path.stat().st_size,
        }
        for path in sorted(checkpoint_path.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _checkpoint_inventory_errors(
    checkpoint_path: Path,
    inventory: object,
) -> list[str]:
    """Validate the completion marker's recorded file inventory.

    :param Path checkpoint_path: Candidate step checkpoint directory.
    :param object inventory: Marker inventory payload.
    :return list[str]: Human-readable validation errors.
    """
    if not isinstance(inventory, list) or not inventory:
        return ["completion marker has no artifact inventory"]
    errors: list[str] = []
    for entry in inventory:
        if not isinstance(entry, Mapping):
            errors.append("completion marker contains an invalid inventory entry")
            continue
        relative_path = Path(str(entry.get("path", "")))
        if (
            not relative_path.parts
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            errors.append(f"completion marker has unsafe artifact path {relative_path}")
            continue
        artifact_path = checkpoint_path / relative_path
        if not artifact_path.is_file():
            errors.append(f"missing inventoried artifact {relative_path.as_posix()}")
            continue
        raw_size = entry.get("size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            errors.append(
                f"completion marker has invalid size for {relative_path.as_posix()}"
            )
            continue
        expected_size = raw_size
        if artifact_path.stat().st_size != expected_size:
            errors.append(
                f"size mismatch for inventoried artifact {relative_path.as_posix()}"
            )
    return errors


def checkpoint_resume_errors(checkpoint_path: str | Path) -> list[str]:
    """Return reasons a training checkpoint cannot be resumed safely.

    :param str | Path checkpoint_path: Candidate step checkpoint directory.
    :return list[str]: Human-readable validation errors.
    """
    checkpoint_path = Path(checkpoint_path)
    errors = _checkpoint_resume_artifact_errors(checkpoint_path)
    marker_path = checkpoint_path / CHECKPOINT_COMPLETE_NAME
    if not marker_path.is_file():
        errors.append(f"missing {CHECKPOINT_COMPLETE_NAME}")
        return errors
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {CHECKPOINT_COMPLETE_NAME}: {exc}")
        return errors
    if not isinstance(marker, Mapping):
        errors.append(f"invalid {CHECKPOINT_COMPLETE_NAME}: expected a JSON object")
        return errors
    if marker.get("format_version") != CHECKPOINT_COMPLETE_VERSION:
        errors.append(
            f"unsupported completion marker version {marker.get('format_version')!r}"
        )
    if marker.get("complete") is not True:
        errors.append("completion marker does not declare complete=true")
    errors.extend(
        _task_checkpoint_artifact_errors(
            checkpoint_path,
            task=marker.get("task"),
        )
    )
    errors.extend(
        _checkpoint_inventory_errors(checkpoint_path, marker.get("artifacts"))
    )
    return errors


def is_resumable_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return whether a step checkpoint has complete resumable state.

    :param str | Path checkpoint_path: Candidate step checkpoint directory.
    :return bool: True when all resume-critical artifacts are complete.
    """
    return not checkpoint_resume_errors(checkpoint_path)


def invalidate_checkpoint_completion(checkpoint_path: str | Path) -> None:
    """Remove completion metadata before creating or overwriting a checkpoint.

    :param str | Path checkpoint_path: Step checkpoint directory.
    """
    checkpoint_path = Path(checkpoint_path)
    for name in (CHECKPOINT_COMPLETE_NAME, f".{CHECKPOINT_COMPLETE_NAME}.tmp"):
        try:
            (checkpoint_path / name).unlink()
        except FileNotFoundError:
            pass


def mark_checkpoint_complete(
    checkpoint_path: str | Path,
    *,
    task: str,
) -> Path:
    """Atomically mark a checkpoint complete after validating resume artifacts.

    :param str | Path checkpoint_path: Step checkpoint directory.
    :param str task: Training task that produced the checkpoint.
    :return Path: Written completion marker path.
    :raises RuntimeError: If resume-critical artifacts are missing.
    """
    checkpoint_path = Path(checkpoint_path)
    errors = _checkpoint_resume_artifact_errors(checkpoint_path)
    errors.extend(_task_checkpoint_artifact_errors(checkpoint_path, task=task))
    if errors:
        raise RuntimeError(
            f"Cannot mark incomplete checkpoint {checkpoint_path}: " + "; ".join(errors)
        )
    marker_path = checkpoint_path / CHECKPOINT_COMPLETE_NAME
    temporary_path = checkpoint_path / f".{CHECKPOINT_COMPLETE_NAME}.tmp"
    payload = {
        "format_version": CHECKPOINT_COMPLETE_VERSION,
        "complete": True,
        "task": str(task),
        "step": checkpoint_path.name,
        "artifacts": _checkpoint_artifact_inventory(checkpoint_path),
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(marker_path)
    return marker_path


def _unwrap_compile_wrappers(model: nn.Module) -> nn.Module:
    """Return the base module beneath torch.compile wrappers.

    :param nn.Module model: Possibly wrapped model.
    :return nn.Module: Underlying model.
    """
    base = model
    while hasattr(base, "_orig_mod"):
        base = getattr(base, "_orig_mod")
    return base


def strip_runtime_prefixes(key: str) -> str:
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
        normalized_key = strip_runtime_prefixes(str(key))
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
        normalized_key = strip_runtime_prefixes(str(raw_key))
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


def _is_loadable_step_checkpoint(checkpoint_root: Path, tag: str) -> bool:
    """Return whether the step directory ``tag`` is loadable from ``checkpoint_root``.

    :param Path checkpoint_root: Root directory containing checkpoint steps.
    :param str tag: Step directory name to validate, kept verbatim so
        zero-padded names written by external tools resolve correctly.
    :return bool: True when either portable or DeepSpeed weights are loadable.
    """
    step_dir = checkpoint_root / tag
    if (step_dir / MODEL_WEIGHTS_NAME).is_file():
        return True
    try:
        resolve_deepspeed_checkpoint_root_and_tag(checkpoint_root, tag=tag)
    except (FileNotFoundError, ValueError):
        return False
    return True


def resolve_step_checkpoint_selector(
    checkpoint_root: str | Path,
    checkpoint: str | int,
) -> str:
    """Resolve ``checkpoint`` to a concrete step/tag for loading.

    ``latest`` honors a root-level ``latest`` file first, then an already-selected
    direct checkpoint path. When neither exists, scan for the highest loadable
    numbered step so portable checkpoint roots without DeepSpeed metadata still
    work.

    :param str | Path checkpoint_root: Root directory or direct checkpoint path.
    :param str | int checkpoint: Requested checkpoint selector.
    :return str: Concrete checkpoint tag to load.
    :raises ValueError: If a DeepSpeed ``latest`` file is empty.
    :raises FileNotFoundError: If ``latest`` cannot resolve a loadable checkpoint.
    """
    checkpoint_root = Path(checkpoint_root)
    requested_tag = str(checkpoint).strip()
    if requested_tag.lower() != "latest":
        return requested_tag

    latest_path = checkpoint_root / "latest"
    if latest_path.is_file():
        latest_tag = latest_path.read_text(encoding="utf-8").strip()
        if not latest_tag:
            raise ValueError(f"DeepSpeed latest file is empty: {latest_path}")
        return latest_tag

    direct_tag = _resolve_direct_checkpoint_tag(checkpoint_root)
    if direct_tag is not None:
        return direct_tag

    candidates = sorted(
        (
            path.name
            for path in checkpoint_root.iterdir()
            if path.is_dir() and is_step_checkpoint_name(path.name)
        ),
        key=int,
        reverse=True,
    )
    for tag in candidates:
        if _is_loadable_step_checkpoint(checkpoint_root, tag):
            return tag
    raise FileNotFoundError(
        f"No loadable numbered checkpoints found under {checkpoint_root}"
    )


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


def resolve_training_checkpoint_artifacts(
    checkpoint_path: str | Path,
    checkpoint: str | int,
) -> tuple[Path, Path, str]:
    """Resolve a run root, checkpoint root, or direct step to one concrete step.

    :param str | Path checkpoint_path: Run root, checkpoint root, or step directory.
    :param str | int checkpoint: Requested checkpoint selector.
    :return tuple[Path, Path, str]: Checkpoint root, step directory, and concrete tag.
    """
    provided_path = Path(checkpoint_path)
    checkpoint_root = (
        provided_path / "checkpoints"
        if (provided_path / "checkpoints").is_dir()
        else provided_path
    )
    concrete_tag = resolve_step_checkpoint_selector(checkpoint_root, checkpoint)
    step_dir = resolve_step_checkpoint_dir(checkpoint_root, concrete_tag)
    if not step_dir.is_dir():
        raise FileNotFoundError(
            f"Resolved checkpoint directory does not exist: {step_dir}"
        )
    return checkpoint_root, step_dir, concrete_tag


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

    :param Any cfg: Runtime config object or ``cfg.trainer``.
    :return int: Maximum number of retained checkpoints (0 disables pruning).
    """
    trainer_cfg = getattr(cfg, "trainer", cfg)
    save_total_limit = getattr(trainer_cfg, "save_total_limit", None)
    if save_total_limit is not None:
        return max(0, int(save_total_limit))
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
        if not is_step_checkpoint_name(item_path.name):
            continue
        checkpoints.append((int(item_path.name), item_path))

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


def resolve_accelerate_state_dir(checkpoint_path: str | Path) -> Path:
    """Resolve where Accelerate ``save_state`` artifacts live for a step checkpoint.

    Accelerate resume state is written to an ``accelerate/`` subdirectory so its
    model payload cannot collide with the portable ``model.safetensors`` export:
    the portable file intentionally duplicates tied tensors for export/eval
    consumers, which safetensors' strict ``load_model`` rejects on resume.

    :param str | Path checkpoint_path: Step checkpoint directory.
    :return Path: Directory to pass to ``Accelerator.load_state``.
    :raises FileNotFoundError: If the checkpoint has no Accelerate state directory.
    """
    checkpoint_path = Path(checkpoint_path)
    state_dir = checkpoint_path / ACCELERATE_STATE_DIR
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint has no Accelerate state directory: {state_dir}"
        )
    return state_dir


def save_accelerate_state(accelerator: Any, checkpoint_path: str | Path) -> Path:
    """Save Accelerate resume state into a step checkpoint's state subdirectory.

    :param Any accelerator: Active accelerator runtime.
    :param str | Path checkpoint_path: Step checkpoint directory.
    :return Path: Directory the state was written to.
    """
    state_dir = Path(checkpoint_path) / ACCELERATE_STATE_DIR
    accelerator.save_state(output_dir=str(state_dir))
    return state_dir


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
    :raises RuntimeError: If state collection or portable serialization fails.
    :return bool: True on the writing rank; false on other ranks.
    """
    checkpoint_path = Path(checkpoint_path)
    weights_path = checkpoint_path / MODEL_WEIGHTS_NAME
    if skip_if_exists and weights_path.exists():
        return True

    local_error: str | None = None
    local_exception: Exception | None = None
    state_dict: dict[str, torch.Tensor] | None = None
    try:
        # Distributed backends (FSDP/FSDP2/DeepSpeed) may require all ranks to
        # participate in state-dict collection collectives even when only rank 0
        # persists the portable safetensors payload.
        state_dict = accelerator.get_state_dict(model, unwrap=True)
    except Exception as exc:
        local_error = (
            f"failed to collect portable model state for {weights_path}: {exc}"
        )
        local_exception = exc

    is_main_process = bool(getattr(accelerator, "is_main_process", True))
    if local_error is None and is_main_process:
        assert state_dict is not None
        try:
            weights_path = save_state_dict_safetensors(
                state_dict,
                checkpoint_path,
                metadata={"format": "pt", "source": "accelerate.get_state_dict"},
            )
        except Exception as exc:
            local_error = (
                f"failed to persist portable model weights at {weights_path}: {exc}"
            )
            local_exception = exc

    reduce_fn = getattr(accelerator, "reduce", None)
    if callable(reduce_fn):
        failed = torch.tensor(
            int(local_error is not None),
            device=accelerator.device,
            dtype=torch.int32,
        )
        failure_count = int(reduce_fn(failed, reduction="sum").item())
    else:
        failure_count = int(local_error is not None)
    if failure_count:
        message = local_error or (
            "portable checkpoint export failed on another rank; inspect the main-rank "
            "error for details"
        )
        raise RuntimeError(message) from local_exception

    if not is_main_process:
        return False

    logger.info("Saved portable model weights to %s.", weights_path)
    return True
