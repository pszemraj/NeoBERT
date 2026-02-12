"""Comprehensive validation for GLUE configurations."""

import logging
from pathlib import Path
from typing import Any, Dict

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_model_safetensors,
    resolve_deepspeed_checkpoint_root_and_tag,
)
from neobert.config import resolve_mixed_precision

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_glue_config(cfg: Any) -> None:
    """Validate GLUE configuration before training.

    :param Any cfg: Configuration object.
    :raises ValidationError: If configuration is invalid.
    """
    errors = []

    valid_tasks = [
        "cola",
        "sst2",
        "mrpc",
        "stsb",
        "qqp",
        "mnli",
        "qnli",
        "rte",
        "wnli",
        "snli",
        "allnli",
    ]
    task = cfg.glue.task_name if hasattr(cfg, "glue") else getattr(cfg, "task", None)

    if not task:
        errors.append("Task name is required")
    elif task not in valid_tasks:
        errors.append(f"Invalid task: {task}. Must be one of {valid_tasks}")

    def _is_missing(value: Any) -> bool:
        """Return True when a config value should be treated as unset."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    if hasattr(cfg, "model"):
        glue_cfg = getattr(cfg, "glue", None)
        allow_random = bool(getattr(glue_cfg, "allow_random_weights", False))
        checkpoint_dir = getattr(glue_cfg, "pretrained_checkpoint_dir", None)
        checkpoint = getattr(glue_cfg, "pretrained_checkpoint", None)
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            # Legacy fallback only: canonical GLUE schema now stores these under
            # cfg.glue.* and should be used as the source of truth.
            raw_model = cfg._raw_model_dict
            if checkpoint_dir is None and "pretrained_checkpoint_dir" in raw_model:
                checkpoint_dir = raw_model.get("pretrained_checkpoint_dir")
                logger.warning(
                    "Using legacy _raw_model_dict.pretrained_checkpoint_dir; "
                    "migrate to glue.pretrained_checkpoint_dir."
                )
            if checkpoint is None and "pretrained_checkpoint" in raw_model:
                checkpoint = raw_model.get("pretrained_checkpoint")
                logger.warning(
                    "Using legacy _raw_model_dict.pretrained_checkpoint; "
                    "migrate to glue.pretrained_checkpoint."
                )
            if (
                not allow_random
                and "allow_random_weights" in raw_model
                and checkpoint_dir is None
                and checkpoint is None
            ):
                allow_random = bool(raw_model.get("allow_random_weights", False))
                logger.warning(
                    "Using legacy _raw_model_dict.allow_random_weights; "
                    "migrate to glue.allow_random_weights."
                )

        model_cfg = getattr(cfg, "model", None)
        from_hub = bool(getattr(model_cfg, "from_hub", False))
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            from_hub = bool(cfg._raw_model_dict.get("from_hub", from_hub))

        if not allow_random and not from_hub:
            if _is_missing(checkpoint_dir) or _is_missing(checkpoint):
                errors.append(
                    "GLUE requires pretrained weights. Specify "
                    "'glue.pretrained_checkpoint_dir' and "
                    "'glue.pretrained_checkpoint' or set "
                    "'glue.allow_random_weights: true'. "
                    "Use model.from_hub=true for direct HF model fine-tuning."
                )
            elif not Path(str(checkpoint_dir)).exists():
                errors.append(f"Checkpoint directory not found: {checkpoint_dir}")

        if hasattr(cfg.model, "hidden_size") and hasattr(
            cfg.model, "num_attention_heads"
        ):
            if cfg.model.hidden_size % cfg.model.num_attention_heads != 0:
                errors.append(
                    f"hidden_size ({cfg.model.hidden_size}) must be divisible by "
                    f"num_attention_heads ({cfg.model.num_attention_heads})"
                )

        if hasattr(cfg.model, "dropout_prob"):
            if not 0 <= cfg.model.dropout_prob <= 1:
                errors.append(
                    f"dropout_prob must be between 0 and 1, got {cfg.model.dropout_prob}"
                )

    if hasattr(cfg, "trainer"):
        if hasattr(cfg.trainer, "per_device_train_batch_size"):
            if cfg.trainer.per_device_train_batch_size < 1:
                errors.append("per_device_train_batch_size must be at least 1")

        if hasattr(cfg.trainer, "per_device_eval_batch_size"):
            if cfg.trainer.per_device_eval_batch_size < 1:
                errors.append("per_device_eval_batch_size must be at least 1")

        if hasattr(cfg.trainer, "output_dir"):
            output_dir = Path(cfg.trainer.output_dir)
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory {output_dir}: {e}")

        if hasattr(cfg.trainer, "mixed_precision"):
            try:
                cfg.trainer.mixed_precision = resolve_mixed_precision(
                    cfg.trainer.mixed_precision,
                    task="glue",
                )
            except ValueError as exc:
                errors.append(str(exc))

    if hasattr(cfg, "optimizer"):
        if hasattr(cfg.optimizer, "lr"):
            if cfg.optimizer.lr <= 0:
                errors.append(f"Learning rate must be positive, got {cfg.optimizer.lr}")
        elif hasattr(cfg.optimizer, "hparams") and hasattr(cfg.optimizer.hparams, "lr"):
            if cfg.optimizer.hparams.lr <= 0:
                errors.append(
                    f"Learning rate must be positive, got {cfg.optimizer.hparams.lr}"
                )

        if hasattr(cfg.optimizer, "weight_decay"):
            if cfg.optimizer.weight_decay < 0:
                errors.append(
                    f"Weight decay must be non-negative, got {cfg.optimizer.weight_decay}"
                )

    if hasattr(cfg, "scheduler"):
        if hasattr(cfg.scheduler, "warmup_percent") and hasattr(
            cfg.scheduler, "warmup_steps"
        ):
            if (
                cfg.scheduler.warmup_percent is not None
                and cfg.scheduler.warmup_steps is not None
            ):
                logger.warning(
                    "Both warmup_percent and warmup_steps specified. warmup_percent will take precedence."
                )

    if hasattr(cfg, "glue"):
        if hasattr(cfg.glue, "max_seq_length"):
            if cfg.glue.max_seq_length < 1:
                errors.append(
                    f"max_seq_length must be positive, got {cfg.glue.max_seq_length}"
                )
            elif cfg.glue.max_seq_length > 512:
                logger.warning(
                    f"max_seq_length={cfg.glue.max_seq_length} > 512 may cause issues with some models"
                )

        expected_labels = {
            "cola": 2,
            "sst2": 2,
            "mrpc": 2,
            "qqp": 2,
            "mnli": 3,
            "qnli": 2,
            "rte": 2,
            "wnli": 2,
            "stsb": 1,
            "snli": 3,
            "allnli": 2,
        }

        if task in expected_labels:
            expected = expected_labels[task]
            if hasattr(cfg.glue, "num_labels"):
                if cfg.glue.num_labels != expected:
                    logger.warning(
                        f"Task {task} expects {expected} labels but got {cfg.glue.num_labels}. "
                        f"Auto-correcting to {expected}."
                    )
                    cfg.glue.num_labels = expected
            else:
                cfg.glue.num_labels = expected

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValidationError(error_msg)

    logger.info("✓ Configuration validation passed")


def validate_checkpoint_compatibility(
    model_config: Dict[str, Any], checkpoint_path: str
) -> None:
    """Validate that a checkpoint matches the model configuration.

    :param dict[str, Any] model_config: Model configuration dictionary.
    :param str checkpoint_path: Path to checkpoint directory.
    :raises ValidationError: If checkpoint is incompatible.
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise ValidationError(f"Checkpoint not found: {checkpoint_path}")

    try:
        resolved_root, resolved_tag = resolve_deepspeed_checkpoint_root_and_tag(
            checkpoint_dir
        )
    except (FileNotFoundError, ValueError):
        resolved_root = None
        resolved_tag = None

    if resolved_root is not None and resolved_tag is not None:
        logger.info(
            "Detected DeepSpeed checkpoint layout at "
            f"{resolved_root} (tag={resolved_tag})."
        )
    else:
        state_dict_path = checkpoint_dir / MODEL_WEIGHTS_NAME
        if not state_dict_path.exists():
            raise ValidationError(f"No {MODEL_WEIGHTS_NAME} found at {state_dict_path}")

        try:
            state_dict = load_model_safetensors(checkpoint_dir, map_location="cpu")

            has_embeddings = any("embeddings" in k for k in state_dict.keys())
            has_encoder = any("encoder" in k for k in state_dict.keys())

            if not has_embeddings and not has_encoder:
                logger.warning("Checkpoint may not contain expected model weights")

            logger.info(f"Checkpoint contains {len(state_dict)} parameters")

        except Exception as e:
            raise ValidationError(f"Failed to load checkpoint: {e}")
