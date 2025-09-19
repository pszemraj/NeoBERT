"""Configuration validation for NeoBERT training and evaluation."""

from typing import Any
from accelerate.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_glue_config(cfg: Any) -> None:
    """Validate GLUE-specific configuration with improved error messages"""
    # Task validation
    valid_tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
    if cfg.glue.task_name not in valid_tasks:
        raise ValidationError(f"Invalid GLUE task: {cfg.glue.task_name}. Must be one of {valid_tasks}")
    
    # Check that required fields are present
    if cfg.trainer.output_dir is None:
        raise ValidationError("trainer.output_dir is required for GLUE evaluation")
    
    # Validate num_labels matches task
    expected_labels = {
        "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, 
        "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2,
        "stsb": 1  # Regression task
    }
    
    if cfg.glue.task_name in expected_labels:
        expected = expected_labels[cfg.glue.task_name]
        if cfg.glue.num_labels != expected:
            logger.warning(
                f"Task {cfg.glue.task_name} expects {expected} labels, "
                f"but config has {cfg.glue.num_labels}. Auto-correcting."
            )
            cfg.glue.num_labels = expected
    
    # Validate pretrained model configuration
    has_pretrained_info = False
    
    # Check GLUEConfig
    if hasattr(cfg.glue, 'pretrained_checkpoint_dir') and cfg.glue.pretrained_checkpoint_dir:
        has_pretrained_info = True
    
    # Check raw model dict
    if hasattr(cfg, '_raw_model_dict') and cfg._raw_model_dict:
        if cfg._raw_model_dict.get('pretrained_checkpoint_dir'):
            has_pretrained_info = True
        if cfg._raw_model_dict.get('allow_random_weights', False):
            has_pretrained_info = True  # Random weights allowed for testing
    
    # Check model config
    if hasattr(cfg.model, 'pretrained_checkpoint_dir') and cfg.model.pretrained_checkpoint_dir:
        has_pretrained_info = True
    
    if not has_pretrained_info:
        if not getattr(cfg.glue, 'allow_random_weights', False):
            raise ValidationError(
                "GLUE evaluation requires pretrained model weights!\n"
                "Please provide one of the following:\n"
                "  1. Set 'pretrained_checkpoint_dir' and 'pretrained_checkpoint' in the model config\n"
                "  2. Set 'allow_random_weights: true' in glue config for testing with random weights\n"
                "  3. Use a pretrained model from HuggingFace Hub with 'from_hub: true'"
            )
    
    # Validate training parameters
    if cfg.trainer.num_train_epochs <= 0:
        raise ValidationError("trainer.num_train_epochs must be > 0")
    
    if cfg.trainer.per_device_train_batch_size <= 0:
        raise ValidationError("trainer.per_device_train_batch_size must be > 0")
    
    # Warn about common configuration issues
    if cfg.trainer.eval_steps > 10000:
        logger.warning(
            f"eval_steps is very high ({cfg.trainer.eval_steps}). "
            "This may result in infrequent evaluation. Consider reducing it."
        )
    
    if cfg.trainer.save_total_limit == 0:
        logger.warning(
            "save_total_limit is 0, which means no checkpoints will be saved during training."
        )