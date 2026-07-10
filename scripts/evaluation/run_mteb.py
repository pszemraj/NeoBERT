"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from mteb import MTEB

from neobert.checkpointing import (
    load_step_checkpoint_state_dict,
    resolve_step_checkpoint_selector,
)
from neobert.config import ConfigLoader
from neobert.mteb_tasks import (
    MTEB_ALL_EXECUTION_TASKS,
    MTEB_EXECUTION_TASKS_BY_TYPE,
    MTEB_TASK_SPECS_BY_EXECUTION_NAME,
    expand_mteb_task_name,
)
from neobert.model import NeoBERTConfig, NeoBERTForMTEB
from neobert.tokenizer import get_tokenizer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


def _resolve_mteb_tasks(cfg: Any) -> list[str]:
    """Resolve selected MTEB task names from config and CLI overrides.

    Resolution order:
    1. ``cfg.task_types`` (CLI ``--task_types``), if provided.
    2. ``cfg.mteb_task_type`` category.

    ``cfg.task_types`` entries can be either category aliases from the registry
    (for example ``classification`` or ``sts``) or explicit task names.

    :param Any cfg: Configuration object.
    :raises ValueError: If any requested task/category is unknown or selection is empty.
    :return list[str]: Ordered deduplicated list of task names.
    """
    requested = getattr(cfg, "task_types", None)
    if requested is None:
        mteb_task_type = str(getattr(cfg, "mteb_task_type", "all")).strip().lower()
        if mteb_task_type not in MTEB_EXECUTION_TASKS_BY_TYPE:
            raise ValueError(
                f"Task type must be one of {sorted(MTEB_EXECUTION_TASKS_BY_TYPE)}."
            )
        return list(MTEB_EXECUTION_TASKS_BY_TYPE[mteb_task_type])

    if isinstance(requested, str):
        requested_tokens = [token.strip() for token in requested.split(",")]
    else:
        requested_tokens = [str(token).strip() for token in requested]

    selected: list[str] = []
    unknown: list[str] = []
    for token in requested_tokens:
        if not token:
            continue
        lowered = token.lower()
        if lowered == "all":
            selected.extend(MTEB_ALL_EXECUTION_TASKS)
            continue
        if lowered in MTEB_EXECUTION_TASKS_BY_TYPE:
            selected.extend(MTEB_EXECUTION_TASKS_BY_TYPE[lowered])
            continue
        explicit_tasks = expand_mteb_task_name(token)
        if explicit_tasks is not None:
            selected.extend(explicit_tasks)
            continue
        unknown.append(token)

    if unknown:
        raise ValueError(
            "Unknown --task_types entries: "
            + ", ".join(sorted(unknown))
            + ". Valid categories: "
            + ", ".join(sorted(MTEB_EXECUTION_TASKS_BY_TYPE))
        )

    # Stable dedupe to preserve user-specified order.
    resolved = list(dict.fromkeys(selected))
    if not resolved:
        raise ValueError(
            "No MTEB tasks selected. Provide at least one task/category via "
            "`--task_types` (for example 'all', 'sts', or 'MSMARCO')."
        )
    return resolved


def _parse_task_type_override(value: str | None) -> list[str] | None:
    """Preserve omission separately from an explicit ``all`` override.

    :param str | None value: Raw CLI task selector.
    :return list[str] | None: Parsed override, or ``None`` when omitted.
    """
    if value is None:
        return None
    return [token.strip() for token in value.split(",")]


def _load_mteb_encoder_weights(
    model: NeoBERTForMTEB,
    state_dict: dict[str, torch.Tensor],
    *,
    source: str,
) -> None:
    """Load checkpoint weights for MTEB with encoder/head key tolerance.

    Pretraining checkpoints commonly include LM-head parameters (for example
    ``decoder.*``) that are not part of ``NeoBERTForMTEB``. We therefore load with
    ``strict=False`` to tolerate known head extras, but still fail fast on any
    non-head key mismatches so MTEB scores are not computed from partially loaded
    encoders.

    :param NeoBERTForMTEB model: MTEB model instance.
    :param dict[str, torch.Tensor] state_dict: Checkpoint state dict.
    :param str source: Human-readable checkpoint source for logs.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible is None:
        # Compatibility for lightweight test doubles that don't return
        # ``_IncompatibleKeys`` from ``load_state_dict``.
        return

    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    missing_keys = list(getattr(incompatible, "missing_keys", []))

    lm_head_prefixes = ("decoder.", "model.decoder.")
    lm_head_unexpected = [
        key for key in unexpected_keys if key.startswith(lm_head_prefixes)
    ]
    remaining_unexpected = [
        key for key in unexpected_keys if key not in lm_head_unexpected
    ]

    if lm_head_unexpected:
        logger.info(
            "Ignoring %d LM-head keys while loading %s for MTEB.",
            len(lm_head_unexpected),
            source,
        )
    if remaining_unexpected or missing_keys:
        mismatch_parts: list[str] = []
        if remaining_unexpected:
            mismatch_parts.append(
                "unexpected_non_head_keys=" + ", ".join(sorted(remaining_unexpected))
            )
        if missing_keys:
            mismatch_parts.append("missing_keys=" + ", ".join(sorted(missing_keys)))
        raise ValueError(
            "MTEB checkpoint/model mismatch while loading "
            f"{source}: {'; '.join(mismatch_parts)}"
        )


def evaluate_mteb(cfg: Any) -> None:
    """Evaluate a model on the MTEB benchmark.

    :param Any cfg: Configuration object with MTEB settings.
    """
    # Get MTEB-specific config (kept at top-level Config for now)
    mteb_batch_size = getattr(cfg, "mteb_batch_size", 32)
    mteb_pooling = getattr(cfg, "mteb_pooling", "avg")
    mteb_overwrite_results = getattr(cfg, "mteb_overwrite_results", False)
    pretrained_checkpoint = getattr(cfg, "pretrained_checkpoint", "latest")
    pretrained_checkpoint_dir = Path(cfg.trainer.output_dir)
    selected_tasks = _resolve_mteb_tasks(cfg)

    # Get checkpoint number
    checkpoint_root = pretrained_checkpoint_dir / "checkpoints"
    ckpt = resolve_step_checkpoint_selector(checkpoint_root, pretrained_checkpoint)

    # Define path to store results
    configured_output = getattr(cfg, "output_folder", None)
    output_folder = (
        Path(configured_output)
        if configured_output
        else pretrained_checkpoint_dir
        / "mteb"
        / str(ckpt)
        / str(cfg.tokenizer.max_length)
    )

    # Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        trust_remote_code=bool(getattr(cfg.tokenizer, "trust_remote_code", False)),
        revision=getattr(cfg.tokenizer, "revision", None),
    )

    # Instantiate model
    model_config = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=cfg.model.max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        attn_backend="sdpa",
    )

    model = NeoBERTForMTEB(
        config=model_config,
        tokenizer=tokenizer,
        batch_size=mteb_batch_size,
        pooling=mteb_pooling,
        max_length=cfg.tokenizer.max_length,
    )

    # Load pretrained weights
    state_dict = load_step_checkpoint_state_dict(
        checkpoint_root,
        ckpt,
        map_location=device,
    )
    _load_mteb_encoder_weights(
        model,
        state_dict,
        source=f"checkpoint step {ckpt}",
    )

    model.to(device)
    model.eval()

    # Evaluate
    for task in selected_tasks:
        logger.info(f"Running task: {task}")
        eval_splits = [MTEB_TASK_SPECS_BY_EXECUTION_NAME[task].evaluation_split]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            evaluation.run(
                model,
                output_folder=output_folder,
                eval_splits=eval_splits,
                overwrite_results=mteb_overwrite_results,
            )


def main() -> None:
    """Run the MTEB evaluation CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Model path"
    )
    parser.add_argument("--task_types", type=str, help="Task types to evaluate")
    parser.add_argument(
        "--output_folder",
        type=Path,
        help="Optional result directory (default: <model>/mteb/<step>/<max-length>)",
    )

    args, remaining = parser.parse_known_args()

    # Load configuration
    config = ConfigLoader.load(args.config, remaining)
    config.trainer.output_dir = args.model_name_or_path
    config.task_types = _parse_task_type_override(args.task_types)
    config.output_folder = args.output_folder

    # Run MTEB evaluation
    evaluate_mteb(config)


if __name__ == "__main__":
    main()
