import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


def configure_tf32(print_fn=None) -> bool:
    """Enable TF32 precision for GPUs with compute capability >= 8.0 (Ampere+).

    :param print_fn: Optional function to use for printing messages (default: logging.info)
    :return: True if TF32 was enabled, False otherwise
    """
    # Use provided print function or fall back to logging
    log = print_fn if print_fn else logging.info

    if not torch.cuda.is_available():
        log("No GPU detected, running on CPU.")
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

    # ---------- formatting helpers ----------
    def _format_number(num: int) -> str:
        return f"{num:,}" if num > 0 else "--"

    def _format_shape(shape: Optional[torch.Size]) -> str:
        return "x".join(map(str, shape)) if shape else "N/A"

    # ---------- build param info once ----------
    # Map: id(param) -> (numel, requires_grad)
    param_info: Dict[int, Tuple[int, bool]] = {}
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid not in param_info:
            param_info[pid] = (p.numel(), bool(p.requires_grad))

    # Fast path: totals only
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

    # Build the list (pre-order traversal)
    summarize_recursive(model, 1, "")

    # Totals from the whole model (already deduped)
    total_params = sum(n for (n, _) in param_info.values())
    trainable_params = sum(n for (n, rg) in param_info.values() if rg)

    # ---------- printing ----------
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
