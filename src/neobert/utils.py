"""Utility helpers for logging, TF32 setup, and model summaries."""

import logging
import pprint
import re
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


_TASK_CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "pretraining": (
        "task",
        "seed",
        "debug",
        "config_path",
        "accelerate_config_file",
        "model",
        "dataset",
        "tokenizer",
        "datacollator",
        "trainer",
        "optimizer",
        "scheduler",
        "wandb",
        "pretraining_metadata",
    ),
    "contrastive": (
        "task",
        "seed",
        "debug",
        "config_path",
        "accelerate_config_file",
        "use_deepspeed",
        "model",
        "dataset",
        "tokenizer",
        "datacollator",
        "trainer",
        "optimizer",
        "scheduler",
        "wandb",
        "contrastive",
        "pretraining_metadata",
    ),
    "glue": (
        "task",
        "seed",
        "debug",
        "config_path",
        "accelerate_config_file",
        "model",
        "dataset",
        "tokenizer",
        "trainer",
        "optimizer",
        "scheduler",
        "wandb",
        "glue",
        "_raw_model_dict",
        "pretraining_metadata",
    ),
    "mteb": (
        "task",
        "seed",
        "debug",
        "config_path",
        "accelerate_config_file",
        "model",
        "tokenizer",
        "wandb",
        "pretrained_checkpoint",
        "mteb_task_type",
        "mteb_batch_size",
        "mteb_pooling",
        "mteb_overwrite_results",
        "pretraining_metadata",
    ),
}

_NON_CONTRASTIVE_DATASET_EXCLUDE_FIELDS = {
    "load_all_from_disk",
    "force_redownload",
    "min_length",
    "alpha",
}

_LEGACY_TRAINER_EXCLUDE_FIELDS = {
    "report_to",
    "max_ckpt",
    "train_batch_size",
    "eval_batch_size",
}

_NON_CONTRASTIVE_TRAINER_EXCLUDE_FIELDS = {"dataloader_num_workers"}

_PRETRAINING_TRAINER_EXCLUDE_FIELDS = {
    "disable_tqdm",
    "early_stopping",
    "metric_for_best_model",
    "greater_is_better",
    "load_best_model_at_end",
    "save_model",
}

_SIMPLE_STRING_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:=+-]+$")


def _serialize_config(cfg: Any) -> Dict[str, Any]:
    """Serialize a config-like object into a dictionary.

    :param Any cfg: Config object or dataclass.
    :raises TypeError: If the input cannot be serialized.
    :return dict[str, Any]: Serialized config dictionary.
    """
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    raise TypeError("Unsupported config type for wandb logging")


def _drop_empty_values(value: Any) -> Any:
    """Drop empty/null values recursively from mappings and sequences.

    :param Any value: Value to clean.
    :return Any: Cleaned value.
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, inner in value.items():
            cleaned_inner = _drop_empty_values(inner)
            if cleaned_inner is None:
                continue
            if isinstance(cleaned_inner, (dict, list)) and not cleaned_inner:
                continue
            cleaned[key] = cleaned_inner
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for inner in value:
            cleaned_inner = _drop_empty_values(inner)
            if cleaned_inner is None:
                continue
            if isinstance(cleaned_inner, (dict, list)) and not cleaned_inner:
                continue
            cleaned_list.append(cleaned_inner)
        return cleaned_list
    return value


def _task_filter_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a config dictionary to keys relevant for the active task.

    :param dict[str, Any] config_dict: Full serialized config dictionary.
    :return dict[str, Any]: Task-scoped config dictionary.
    """
    task = str(config_dict.get("task", "pretraining")).strip().lower()
    if task not in _TASK_CONFIG_FIELDS:
        task = "pretraining"

    allowed_keys = _TASK_CONFIG_FIELDS[task]
    filtered: dict[str, Any] = {}
    for key in allowed_keys:
        if key in config_dict:
            filtered[key] = config_dict[key]

    if task != "contrastive":
        dataset_cfg = filtered.get("dataset")
        if isinstance(dataset_cfg, dict):
            filtered["dataset"] = {
                key: value
                for key, value in dataset_cfg.items()
                if key not in _NON_CONTRASTIVE_DATASET_EXCLUDE_FIELDS
            }

    trainer_cfg = filtered.get("trainer")
    if isinstance(trainer_cfg, dict):
        for key in _LEGACY_TRAINER_EXCLUDE_FIELDS:
            trainer_cfg.pop(key, None)
        if task != "contrastive":
            for key in _NON_CONTRASTIVE_TRAINER_EXCLUDE_FIELDS:
                trainer_cfg.pop(key, None)
        if task == "pretraining":
            for key in _PRETRAINING_TRAINER_EXCLUDE_FIELDS:
                trainer_cfg.pop(key, None)

    return _drop_empty_values(filtered)


def _resolve_display_width(width: Optional[int]) -> int:
    """Resolve display width for compact config rendering.

    :param int | None width: Optional explicit width.
    :return int: Effective width clamped to a practical terminal range.
    """
    if width is None:
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
    return max(80, min(int(width), 180))


def _render_display_value(value: Any, printer: pprint.PrettyPrinter) -> str:
    """Render a value as a compact token string for config display.

    :param Any value: Value to render.
    :param pprint.PrettyPrinter printer: Pretty-printer for complex values.
    :return str: Compact one-line representation.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "":
            return "''"
        if _SIMPLE_STRING_TOKEN_RE.fullmatch(value):
            return value
        return repr(value)
    rendered = printer.pformat(value)
    return " ".join(rendered.splitlines())


def _flatten_display_items(
    value: Dict[str, Any],
    *,
    max_depth: int = 2,
) -> List[Tuple[str, Any]]:
    """Flatten nested mappings for compact section display.

    :param dict[str, Any] value: Mapping to flatten.
    :param int max_depth: Maximum nested depth to flatten.
    :return list[tuple[str, Any]]: Flattened dotted-key items.
    """

    def _walk(
        current: Any,
        *,
        prefix: str,
        depth: int,
        out: List[Tuple[str, Any]],
    ) -> None:
        if isinstance(current, dict) and depth < max_depth:
            for key, inner in current.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(inner, prefix=next_prefix, depth=depth + 1, out=out)
            return
        out.append((prefix, current))

    flattened: List[Tuple[str, Any]] = []
    for key, inner in value.items():
        _walk(inner, prefix=str(key), depth=0, out=flattened)
    return flattened


def _wrap_tokens(prefix: str, tokens: List[str], *, width: int) -> List[str]:
    """Wrap token list to the target width with aligned continuation lines.

    :param str prefix: Prefix to place at the start of the first line.
    :param list[str] tokens: Tokens to wrap.
    :param int width: Target display width.
    :return list[str]: Wrapped display lines.
    """
    if not tokens:
        return [prefix.rstrip()]

    lines: List[str] = []
    current = prefix
    continuation_prefix = " " * len(prefix)
    for token in tokens:
        candidate = token if current == prefix else f" {token}"
        if len(current) + len(candidate) <= width or current == prefix:
            current += candidate
            continue
        lines.append(current.rstrip())
        current = f"{continuation_prefix}{token}"
    lines.append(current.rstrip())
    return lines


def format_resolved_config(
    config_dict: Dict[str, Any],
    *,
    width: Optional[int] = None,
) -> str:
    """Format a resolved config dictionary for readable terminal logging.

    :param dict[str, Any] config_dict: Config dictionary to format.
    :param int | None width: Optional target width for wrapped output.
    :return str: Compact sectioned string.
    """
    if not config_dict:
        return "{}"

    resolved_width = _resolve_display_width(width)
    printer = pprint.PrettyPrinter(
        compact=True,
        width=max(40, resolved_width - 16),
        sort_dicts=False,
    )

    lines: List[str] = []
    meta_tokens: List[str] = []
    section_items: List[Tuple[str, Dict[str, Any]]] = []

    for key, value in config_dict.items():
        if isinstance(value, dict):
            section_items.append((key, value))
            continue
        rendered = _render_display_value(value, printer)
        meta_tokens.append(f"{key}={rendered}")

    if meta_tokens:
        lines.extend(_wrap_tokens("[meta] ", meta_tokens, width=resolved_width))

    for section_name, section_value in section_items:
        flattened = _flatten_display_items(section_value)
        tokens = [
            f"{key}={_render_display_value(value, printer)}" for key, value in flattened
        ]
        lines.extend(_wrap_tokens(f"[{section_name}] ", tokens, width=resolved_width))

    return "\n".join(lines)


def prepare_wandb_config(cfg: Any) -> Dict[str, Any]:
    """Prepare task-scoped config payload for tracking backends.

    :param Any cfg: Configuration object (typically a dataclass).
    :return dict[str, Any]: Task-scoped dictionary ready for tracker ingestion.
    """
    config_dict = _serialize_config(cfg)

    # Preserve dynamically attached metadata only for GLUE-oriented runs.
    task = str(config_dict.get("task", "pretraining")).strip().lower()
    if (
        task == "glue"
        and hasattr(cfg, "_raw_model_dict")
        and cfg._raw_model_dict is not None
    ):
        config_dict["_raw_model_dict"] = cfg._raw_model_dict

    return _task_filter_config(config_dict)


def configure_tf32(
    enabled: bool, print_fn: Optional[Callable[[str], Any]] = None
) -> bool:
    """Enable/disable TF32 precision for GPUs with compute capability >= 8.0 (Ampere+).

    :param bool enabled: Whether TF32 should be enabled.
    :param print_fn: Optional function to use for printing messages (default: logging.info).
    :return bool: True if TF32 was enabled, False otherwise.
    """
    log = print_fn if print_fn else logging.info

    if not torch.cuda.is_available():
        log("No GPU detected, running on CPU.")
        return False

    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            log("TF32 disabled by config.")
        except Exception as e:
            error_msg = f"Failed to disable TF32: {e}"
            if print_fn:
                print_fn(error_msg)
            else:
                logging.error(error_msg)
        return False

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        gpu_name = torch.cuda.get_device_name(device)

        if major >= 8:
            # Modern API - replaces both backend flags
            torch.set_float32_matmul_precision("high")
            log(f"{gpu_name} (compute {major}.{minor}) - TF32 enabled")
            return True
        else:
            log(f"{gpu_name} (compute {major}.{minor}) - TF32 not supported")
            return False

    except Exception as e:
        error_msg = f"Failed to configure GPU: {e}"
        if print_fn:
            print_fn(error_msg)
        else:
            logging.error(error_msg)
        return False


@dataclass
class _LayerSummary:
    """Summary statistics for a single layer in the model."""

    name: str
    param_shape: Optional[torch.Size]
    inclusive_total_params: int
    inclusive_trainable_params: int


def model_summary(
    model: nn.Module, max_depth: int = 4, show_param_shapes: bool = False
) -> None:
    """Print hierarchical summary of model with parameter counts.

    :param model: PyTorch model to summarize
    :param max_depth: Maximum depth of hierarchy to display
    :param show_param_shapes: Whether to show parameter shapes
    """

    def _format_number(num: int) -> str:
        """Format integer counts for display.

        :return str: Formatted number string.
        """
        return f"{num:,}" if num > 0 else "--"

    def _format_shape(shape: Optional[torch.Size]) -> str:
        """Format a tensor shape for display.

        :return str: Shape string or ``N/A`` when missing.
        """
        return "x".join(map(str, shape)) if shape else "N/A"

    # Map: id(param) -> (numel, requires_grad)
    param_info: Dict[int, Tuple[int, bool]] = {}
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid not in param_info:
            param_info[pid] = (p.numel(), bool(p.requires_grad))

    if max_depth <= 0:
        total_params = sum(n for (n, _) in param_info.values())
        trainable_params = sum(n for (n, rg) in param_info.values() if rg)
        print("=" * 50)
        print("Total params:", _format_number(total_params))
        print("Trainable params:", _format_number(trainable_params))
        nontrain = total_params - trainable_params
        print("Non-trainable params:", _format_number(nontrain))
        print("=" * 50)
        return

    summary_list: List[_LayerSummary] = []

    def summarize_recursive(module: nn.Module, depth: int, prefix: str) -> Set[int]:
        """Recursively build summary for module subtree.

        :param module: Current module being processed
        :param depth: Current depth in hierarchy
        :param prefix: Indentation prefix for display
        :return: Set of unique parameter IDs in subtree
        """
        # If we're beyond the print depth, just return the deduped set upward
        if depth > max_depth:
            ids = {id(p) for p in module.parameters(recurse=True)}
            return ids

        # Direct parameters of *this* module (non-recursive)
        direct_ids: Set[int] = {id(p) for p in module.parameters(recurse=False)}

        # Recurse into children and union their sets
        child_ids: Set[int] = set()
        for child in module.children():
            child_ids |= summarize_recursive(child, depth + 1, prefix + "  ")

        all_ids = direct_ids | child_ids

        # Inclusive counts from the deduped set
        total = sum(param_info[i][0] for i in all_ids)
        trainable = sum(param_info[i][0] for i in all_ids if param_info[i][1])

        # First direct trainable parameter shape (display purpose only)
        param_shape = next(
            (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
            None,
        )

        summary_list.append(
            _LayerSummary(
                name=f"{prefix}{type(module).__name__}",
                param_shape=param_shape,
                inclusive_total_params=total,
                inclusive_trainable_params=trainable,
            )
        )
        return all_ids

    summarize_recursive(model, 1, "")

    total_params = sum(n for (n, _) in param_info.values())
    trainable_params = sum(n for (n, rg) in param_info.values() if rg)

    name_col_width = max(len("Layer (type)"), max(len(s.name) for s in summary_list))
    shape_col_width = 0
    if show_param_shapes:
        shape_col_width = max(
            len("Param Shape"),
            max(len(_format_shape(s.param_shape)) for s in summary_list),
        )

    params_col_width = 12
    trainable_col_width = 10
    col_spacing = "  "

    header_parts = [f"{'Layer (type)':<{name_col_width}}"]
    if show_param_shapes:
        header_parts.append(f"{'Param Shape':>{shape_col_width}}")
    header_parts.append(f"{'Param #':>{params_col_width}}")
    header_parts.append(f"{'Trainable':>{trainable_col_width}}")
    header = col_spacing.join(header_parts)
    sep = "=" * len(header)

    print(sep)
    print(header)
    print(sep)
    for e in summary_list:
        parts = [f"{e.name:<{name_col_width}}"]
        if show_param_shapes:
            parts.append(f"{_format_shape(e.param_shape):>{shape_col_width}}")
        parts.append(f"{_format_number(e.inclusive_total_params):>{params_col_width}}")
        parts.append(f"{str(e.inclusive_trainable_params > 0):>{trainable_col_width}}")
        print(col_spacing.join(parts))
    print(sep)
    print(f"Total params: {_format_number(total_params)}")
    print(f"Trainable params: {_format_number(trainable_params)}")
    print(f"Non-trainable params: {_format_number(total_params - trainable_params)}")
    print(sep)
