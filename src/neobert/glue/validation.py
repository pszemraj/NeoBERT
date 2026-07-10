"""Side-effect-free validation helpers for GLUE fine-tuning."""

import warnings
from pathlib import Path
from typing import Any

from neobert.config import Config, ConfigLoader
from neobert.glue.tasks import SUPPORTED_GLUE_TASK_SPECS, get_glue_task_spec


class GlueValidationError(Exception):
    """Raised when a GLUE configuration is invalid."""


def load_validated_glue_config(
    config_path: str | Path,
    *,
    task_name: str | None = None,
    model_name_or_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[Config, tuple[str, ...]]:
    """Load, apply launch overrides, and validate one GLUE configuration.

    :param str | Path config_path: YAML configuration path.
    :param str | None task_name: Optional task-name override.
    :param str | None model_name_or_path: Optional Hugging Face model override.
    :param str | None output_dir: Optional output-directory override.
    :return tuple[Config, tuple[str, ...]]: Validated config and warning messages.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = ConfigLoader.load(config_path)
    config_warnings = [str(item.message) for item in captured]

    if task_name:
        cfg.glue.task_name = task_name
    if model_name_or_path:
        cfg.model.name = model_name_or_path
        cfg.model.from_hub = True
    if output_dir:
        cfg.trainer.output_dir = output_dir

    effective_model_config: Any | None = None
    if not cfg.model.from_hub:
        if cfg.glue.allow_random_weights:
            effective_model_config = cfg.model
        elif cfg.glue.pretrained_model_path:
            pretrained_config_path = Path(cfg.glue.pretrained_model_path)
            if pretrained_config_path.is_file():
                with warnings.catch_warnings(record=True) as pretrained_warnings:
                    warnings.simplefilter("always")
                    pretrained_cfg = ConfigLoader.load(pretrained_config_path)
                config_warnings.extend(
                    str(item.message) for item in pretrained_warnings
                )
                effective_model_config = pretrained_cfg.model

    glue_warnings = validate_glue_config(
        cfg,
        effective_model_config=effective_model_config,
    )
    return cfg, tuple(config_warnings) + glue_warnings


def validate_glue_config(
    cfg: Any,
    *,
    effective_model_config: Any | None = None,
) -> tuple[str, ...]:
    """Validate GLUE configuration before training.

    :param Any cfg: Configuration object.
    :param Any | None effective_model_config: Resolved model architecture, when known.
    :raises GlueValidationError: If configuration is invalid.
    :return tuple[str, ...]: Non-fatal validation warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    valid_tasks = tuple(sorted(SUPPORTED_GLUE_TASK_SPECS))
    task = cfg.glue.task_name if hasattr(cfg, "glue") else getattr(cfg, "task", None)

    if getattr(cfg, "task", None) != "glue":
        errors.append(
            f"Top-level task must be 'glue', got {getattr(cfg, 'task', None)!r}"
        )

    if not task:
        errors.append("Task name is required")
    elif task not in valid_tasks:
        errors.append(f"Invalid task: {task}. Must be one of {valid_tasks}")

    def _is_missing(value: Any) -> bool:
        """Return whether a config value should be treated as unset.

        :param Any value: Candidate config value.
        :return bool: True when the value is effectively unset.
        """
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    if hasattr(cfg, "model"):
        glue_cfg = getattr(cfg, "glue", None)
        allow_random = bool(getattr(glue_cfg, "allow_random_weights", False))
        pretrained_model_path = getattr(glue_cfg, "pretrained_model_path", None)
        checkpoint_dir = getattr(glue_cfg, "pretrained_checkpoint_dir", None)
        checkpoint = getattr(glue_cfg, "pretrained_checkpoint", None)

        model_cfg = getattr(cfg, "model", None)
        from_hub = bool(getattr(model_cfg, "from_hub", False))

        if not allow_random and not from_hub:
            if (
                _is_missing(pretrained_model_path)
                or _is_missing(checkpoint_dir)
                or _is_missing(checkpoint)
            ):
                errors.append(
                    "GLUE requires pretrained weights. Specify "
                    "'glue.pretrained_model_path', "
                    "'glue.pretrained_checkpoint_dir' and "
                    "'glue.pretrained_checkpoint' or set "
                    "'glue.allow_random_weights: true'. "
                    "Use model.from_hub=true for direct HF model fine-tuning."
                )
            elif not Path(str(pretrained_model_path)).is_file():
                errors.append(
                    f"Pretrained model config not found: {pretrained_model_path}"
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

        if hasattr(cfg.model, "dropout_prob") and not 0 <= cfg.model.dropout_prob <= 1:
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

    if hasattr(cfg, "optimizer"):
        if hasattr(cfg.optimizer, "lr") and cfg.optimizer.lr <= 0:
            errors.append(f"Learning rate must be positive, got {cfg.optimizer.lr}")

        if hasattr(cfg.optimizer, "weight_decay") and cfg.optimizer.weight_decay < 0:
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
                warnings.append(
                    "Both warmup_percent and warmup_steps specified. "
                    "warmup_percent will take precedence."
                )

    if hasattr(cfg, "glue"):
        if hasattr(cfg.glue, "max_seq_length"):
            if cfg.glue.max_seq_length < 1:
                errors.append(
                    f"max_seq_length must be positive, got {cfg.glue.max_seq_length}"
                )
            elif cfg.glue.max_seq_length > 512:
                warnings.append(
                    f"max_seq_length={cfg.glue.max_seq_length} > 512 may cause issues "
                    "with some models"
                )

        if task in SUPPORTED_GLUE_TASK_SPECS:
            expected = get_glue_task_spec(task).num_labels
            if hasattr(cfg.glue, "num_labels"):
                if cfg.glue.num_labels != expected:
                    errors.append(
                        f"Task {task} expects glue.num_labels={expected}, got "
                        f"{cfg.glue.num_labels}."
                    )

        if (
            effective_model_config is not None
            and not bool(getattr(effective_model_config, "rope", False))
            and cfg.glue.max_seq_length
            > int(getattr(effective_model_config, "max_position_embeddings", 0))
        ):
            errors.append(
                "glue.max_seq_length exceeds the checkpoint's learned position "
                f"table: {cfg.glue.max_seq_length} > "
                f"{effective_model_config.max_position_embeddings}."
            )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise GlueValidationError(error_msg)

    return tuple(warnings)
