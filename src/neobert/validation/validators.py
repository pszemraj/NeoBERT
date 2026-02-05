"""Comprehensive validation for GLUE configurations."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

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

    valid_tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
    task = cfg.glue.task_name if hasattr(cfg, "glue") else getattr(cfg, "task", None)

    if not task:
        errors.append("Task name is required")
    elif task not in valid_tasks:
        errors.append(f"Invalid task: {task}. Must be one of {valid_tasks}")

    if hasattr(cfg, "model"):
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            allow_random = cfg._raw_model_dict.get("allow_random_weights", False)
            if not allow_random:
                checkpoint_dir = cfg._raw_model_dict.get("pretrained_checkpoint_dir")
                checkpoint = cfg._raw_model_dict.get("pretrained_checkpoint")

                if not checkpoint_dir or not checkpoint:
                    errors.append(
                        "GLUE requires pretrained weights. Specify 'pretrained_checkpoint_dir' "
                        "and 'pretrained_checkpoint' or set 'allow_random_weights: true'"
                    )
                elif checkpoint_dir and not os.path.exists(checkpoint_dir):
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
            valid_precision = ["no", "bf16", "fp32"]
            mixed_precision = cfg.trainer.mixed_precision
            if isinstance(mixed_precision, bool):
                mixed_precision = "no" if not mixed_precision else "bf16"
            if mixed_precision not in valid_precision:
                errors.append(
                    f"Invalid mixed_precision: {cfg.trainer.mixed_precision}. "
                    f"Must be one of {valid_precision}"
                )

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
    import torch

    if not os.path.exists(checkpoint_path):
        raise ValidationError(f"Checkpoint not found: {checkpoint_path}")

    is_deepspeed = os.path.exists(os.path.join(checkpoint_path, "zero_to_fp32.py"))

    if is_deepspeed:
        required_files = ["zero_to_fp32.py", "latest"]
        for file in required_files:
            if not os.path.exists(os.path.join(checkpoint_path, file)):
                logger.warning(f"DeepSpeed checkpoint missing {file}")
    else:
        state_dict_path = os.path.join(checkpoint_path, "state_dict.pt")
        if not os.path.exists(state_dict_path):
            raise ValidationError(f"No state_dict.pt found at {state_dict_path}")

        try:
            state_dict = torch.load(state_dict_path, map_location="cpu")

            has_embeddings = any("embeddings" in k for k in state_dict.keys())
            has_encoder = any("encoder" in k for k in state_dict.keys())

            if not has_embeddings and not has_encoder:
                logger.warning("Checkpoint may not contain expected model weights")

            logger.info(f"Checkpoint contains {len(state_dict)} parameters")

        except Exception as e:
            raise ValidationError(f"Failed to load checkpoint: {e}")
