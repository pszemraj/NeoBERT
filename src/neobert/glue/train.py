"""Fine-tune NeoBERT sequence classifiers on GLUE and GLUE-like NLI tasks."""

import json
import logging
import math
import os
import random
import re
import shutil
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Optional

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import ClassLabel, load_dataset
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_deepspeed_fp32_state_dict,
    load_model_safetensors,
    save_state_dict_safetensors,
)
from neobert.model import NeoBERTConfig, NeoBERTForSequenceClassification
from neobert.tokenizer import get_tokenizer

from neobert.config import Config, resolve_mixed_precision
from neobert.glue.process import process_function
from neobert.optimizer import get_optimizer
from neobert.scheduler import get_scheduler
from neobert.training_utils import (
    _maybe_compile_model,
    _maybe_prepare_for_forward,
    _resolve_resume_checkpoint,
    _unwrap_optimizer,
    create_accelerator,
    validate_muon_distributed_compatibility,
)
from neobert.utils import configure_tf32, format_resolved_config, prepare_wandb_config
from neobert.validation import ValidationError, validate_glue_config

logger = get_logger(__name__)

TASK_TO_METRIC = {
    "stsb": "eval_pearson",
    "cola": "eval_matthews_correlation",
    "qqp": "eval_f1",
    "sst2": "eval_accuracy",
    "mnli": "eval_accuracy",
    "mrpc": "eval_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
    "snli": "eval_accuracy",
    "allnli": "eval_accuracy",
}

# Official GLUE scoring rules (average across metrics when multiple are reported)
GLUE_SCORE_SPECS = {
    "cola": ("matthews_correlation",),
    "sst2": ("accuracy",),
    "mrpc": ("accuracy", "f1"),
    "stsb": ("pearson", "spearmanr"),
    "qqp": ("accuracy", "f1"),
    "mnli": ("accuracy", "accuracy_mm"),
    "qnli": ("accuracy",),
    "rte": ("accuracy",),
    "wnli": ("accuracy",),
}

TASK_TO_TRANSFER_FROM = {
    "mnli": "snli",
    "qnli": "mnli",
    "wnli": "allnli",
    "stsb": "mnli",
    "mrpc": "mnli",
    "rte": "mnli",
}

_STEP_RE = re.compile(r"(?:^|/)(?:step|checkpoint)_(\d+)(?:$|/)")
_EPOCH_RE = re.compile(r"(?:^|/)(?:epoch)_(\d+)(?:$|/)")


def _get_optimizer_update_step(optimizer: Any) -> Optional[int]:
    """Return the optimizer update counter if available.

    :param Any optimizer: Optimizer or wrapped optimizer.
    :return int | None: Update step counter.
    """
    inner = _unwrap_optimizer(optimizer)
    step = getattr(inner, "_step", None)
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _parse_checkpoint_progress(path: str) -> tuple[Optional[str], Optional[int]]:
    """Parse step/epoch metadata from a checkpoint path.

    :param str path: Checkpoint path or directory name.
    :return tuple[str | None, int | None]: ("step" or "epoch", value) if detected.
    """
    name = Path(path).name
    if name.isdigit():
        return "step", int(name)

    match = _STEP_RE.search(name)
    if match:
        return "step", int(match.group(1))

    match = _EPOCH_RE.search(name)
    if match:
        return "epoch", int(match.group(1))

    return None, None


def _to_serializable(value: Any) -> Any:
    """Convert tensors/NumPy scalars to JSON-serializable values.

    :param Any value: Value to convert.
    :return Any: Serializable representation.
    """
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if torch.is_tensor(value):
        return value.item()
    return value


def _configure_wandb_metrics(accelerator: Accelerator) -> None:
    """Configure W&B metric definitions for GLUE runs.

    :param Accelerator accelerator: Accelerator instance with trackers.
    """
    for tracker in getattr(accelerator, "trackers", []):
        if tracker.__class__.__name__ != "WandBTracker":
            continue
        run = getattr(tracker, "run", None)
        if run is None:
            continue
        try:
            run.define_metric("train/step")
            run.define_metric("train/*", step_metric="train/step")
            run.define_metric("val/epoch")
            run.define_metric("val/*", step_metric="val/epoch")
            run.define_metric("final/step")
            run.define_metric("final/*", step_metric="final/step")
        except Exception as exc:  # pragma: no cover - best-effort safety
            logger.warning(f"Failed to configure W&B metric definitions: {exc}")
        break


def _normalize_metrics_for_scoring(raw_metrics: dict) -> dict[str, float]:
    """Normalize metric keys for GLUE score calculation.

    :param dict raw_metrics: Raw metric mapping from evaluation.
    :return dict[str, float]: Normalized metric mapping.
    """
    normalized: dict[str, float] = {}
    for key, value in (raw_metrics or {}).items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            continue
        metric_key = key[len("eval_") :] if key.startswith("eval_") else key
        normalized[metric_key] = float(value)
    return normalized


def compute_glue_score(task: str, metrics: dict[str, float]) -> float | None:
    """Return official GLUE score (averaged where required) for a task.

    :param str task: GLUE task name.
    :param dict[str, float] metrics: Evaluation metrics for the task.
    :return float | None: Official GLUE score, if available.
    """

    metric_keys = GLUE_SCORE_SPECS.get(task)
    if not metric_keys:
        return None

    normalized = _normalize_metrics_for_scoring(metrics)
    values = []
    for key in metric_keys:
        if key in normalized:
            values.append(normalized[key])
            continue
        if task == "mnli" and key == "accuracy_mm":
            for alias in ("accuracy_mismatched", "mnli_mm", "accuracy-mm"):
                if alias in normalized:
                    values.append(normalized[alias])
                    break

    if not values:
        combined = normalized.get("combined_score")
        return float(combined) if combined is not None else None

    return float(sum(values) / len(values))


def _resolve_glue_task(cfg: Any) -> str:
    """Resolve the active GLUE task name from config.

    :param Any cfg: Runtime config object.
    :return str: Normalized GLUE task name.
    """
    return str(getattr(cfg.glue, "task_name", getattr(cfg, "task", "glue"))).strip()


def _load_glue_metric(glue_task: str, meta_task: str, experiment_id: str) -> Any:
    """Load an evaluate metric object for a GLUE-like task identifier.

    :param str glue_task: Task name (for example ``cola`` or ``mnli``).
    :param str meta_task: Metric namespace (typically ``glue``).
    :param str experiment_id: Evaluate experiment id.
    :return Any: Instantiated evaluate metric object.
    """
    if glue_task in ("multirc", "record"):
        return evaluate.load("accuracy", experiment_id=experiment_id)
    if glue_task == "snli":
        return evaluate.load(meta_task, "mnli", experiment_id=experiment_id)
    if glue_task == "allnli":
        return evaluate.load(meta_task, "wnli", experiment_id=experiment_id)
    return evaluate.load(meta_task, glue_task, experiment_id=experiment_id)


def _load_from_hub_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """Load tokenizer for hub sequence-classification models.

    GLUE/SuperGLUE fine-tuning does not require MLM masking semantics, so hub
    tokenizers without a mask token should be accepted as-is.

    :param Config cfg: Runtime config.
    :return Any: Loaded tokenizer instance.
    """
    return get_tokenizer(
        pretrained_model_name_or_path=cfg.model.name,
        max_length=cfg.glue.max_seq_length,
        trust_remote_code=cfg.tokenizer.trust_remote_code,
        revision=cfg.tokenizer.revision,
        allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
        enforce_mlm_special_tokens=False,
    )


def _update_wandb_config(accelerator: Accelerator, cfg: Config) -> None:
    """Update W&B run config with GLUE metadata.

    :param Accelerator accelerator: Accelerator instance with trackers.
    :param Config cfg: Training configuration.
    """
    metadata = getattr(cfg, "pretraining_metadata", {}) or {}
    glue_task = _resolve_glue_task(cfg)
    glue_max_len = getattr(cfg.glue, "max_seq_length", None)
    glue_labels = getattr(cfg.glue, "num_labels", None)

    to_update = {
        "glue_task": glue_task,
        "glue_max_seq_length": glue_max_len,
        "glue_num_labels": glue_labels,
    }

    for key, value in metadata.items():
        to_update[f"pretraining_{key}"] = value

    for tracker in getattr(accelerator, "trackers", []):
        if tracker.__class__.__name__ != "WandBTracker":
            continue
        run = getattr(tracker, "run", None)
        if run is None:
            continue
        try:
            run.config.update(
                {k: v for k, v in to_update.items() if v is not None},
                allow_val_change=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to update W&B config: {exc}")
        break


def _save_metrics(output_dir: str, split: str, metrics: dict[str, Any]) -> None:
    """Persist evaluation metrics to disk.

    :param str output_dir: Output directory for metrics files.
    :param str split: Dataset split name.
    :param dict[str, Any] metrics: Metrics mapping to write.
    """
    if not metrics:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    serializable = {k: _to_serializable(v) for k, v in metrics.items()}
    with (path / f"{split}_results.json").open("w", encoding="utf-8") as fp:
        json.dump(serializable, fp, indent=2, sort_keys=True)


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits tensor from dict-style or HF output objects.

    :param Any outputs: Model forward outputs.
    :return torch.Tensor: Logits tensor.
    """
    if isinstance(outputs, dict):
        return outputs["logits"]
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise TypeError(
            "Model output does not expose logits as dict['logits'] or .logits."
        )
    return logits


def _forward_classifier_logits(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None = None,
    use_hf_signature: bool,
) -> torch.Tensor:
    """Run classifier forward with explicit kwargs to avoid positional drift.

    HF export models use ``(input_ids, position_ids=None, attention_mask=...)``
    while training models use ``(src, pad_mask)``. Always use explicit keywords so
    attention masks are never accidentally bound to position IDs.

    :param torch.nn.Module model: Model to execute.
    :param torch.Tensor input_ids: Input token IDs.
    :param torch.Tensor attention_mask: Additive attention mask.
    :param torch.Tensor | None token_type_ids: Optional segment IDs for HF models.
    :param bool use_hf_signature: Whether to call HF-style kwargs.
    :return torch.Tensor: Logits tensor.
    """
    if use_hf_signature:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = model(**kwargs)
    else:
        outputs = model(src=input_ids, pad_mask=attention_mask)
    return _extract_logits(outputs)


def get_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    is_regression: bool,
    metric: Any | None = None,
    accelerator: Accelerator | None = None,
    dtype_pad_mask: torch.dtype = torch.float32,
    return_predictions: bool = False,
    compute_metric: bool = True,
    use_hf_signature: bool = False,
) -> dict[str, Any]:
    """Run evaluation over a dataloader and return metrics/predictions.

    :param torch.nn.Module model: Model to evaluate.
    :param DataLoader dataloader: Evaluation dataloader.
    :param bool is_regression: Whether task is regression.
    :param Any | None metric: Optional evaluation metric object.
    :param Accelerator | None accelerator: Accelerator for distributed eval.
    :param torch.dtype dtype_pad_mask: Dtype for attention mask.
    :param bool return_predictions: Whether to return predictions tensor.
    :param bool compute_metric: Whether to compute metric values.
    :param bool use_hf_signature: Whether to call model with HF-style kwargs.
    :return dict[str, Any]: Evaluation outputs (metrics, predictions).
    """
    samples_seen = 0
    # For large GLUE tasks, predictions stay streaming by default; we only
    # accumulate them when explicitly requested (debug/submission workflows).
    predictions_list = [] if return_predictions else None
    eval_metric = None
    show_progress = not (
        accelerator is not None and not accelerator.is_local_main_process
    )
    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Running evaluation...",
        disable=not show_progress,
    )

    # Ensure Flash Attention is disabled when running GLUE evaluations
    sdp_context = (
        sdpa_kernel(SDPBackend.MATH) if torch.cuda.is_available() else nullcontext()
    )
    with sdp_context:
        for step, batch in progress_bar:
            with torch.no_grad(), torch.inference_mode():
                attention_mask = batch["attention_mask"]
                if not use_hf_signature:
                    attention_mask = attention_mask.type(dtype_pad_mask)
                logits = _forward_classifier_logits(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=attention_mask,
                    token_type_ids=batch.get("token_type_ids"),
                    use_hf_signature=use_hf_signature,
                )

            if not is_regression:
                batch_predictions = logits.argmax(dim=-1)
            else:
                batch_predictions = logits.squeeze()

            if compute_metric:
                if accelerator is not None:
                    batch_predictions, references = accelerator.gather(
                        (batch_predictions, batch["labels"])
                    )
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(dataloader) - 1:
                            # ``samples_seen`` intentionally tracks only prior gathered
                            # batches (not the current one), so this slice keeps exactly
                            # the remaining real examples from the final gathered batch.
                            batch_predictions = batch_predictions[
                                : len(dataloader.dataset) - samples_seen
                            ]
                            references = references[
                                : len(dataloader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += references.shape[0]
                else:
                    references = batch["labels"]

                metric.add_batch(
                    predictions=batch_predictions,
                    references=references,
                )

            batch_predictions = batch_predictions.to("cpu")

            if return_predictions:
                # Fix: Append to list instead of concatenating tensors
                predictions_list.append(batch_predictions)

    if compute_metric:
        eval_metric = metric.compute()
        if len(eval_metric) > 1:
            eval_metric["combined_score"] = np.mean(list(eval_metric.values())).item()

    # Fix: Concatenate predictions list once at the end
    predictions = torch.cat(predictions_list) if predictions_list else torch.Tensor()

    return {"predictions": predictions, "eval_metric": eval_metric}


def get_best_checkpoint_path(
    base_dir: str, task: str, num_checkpoints_to_merge: int = 1
) -> tuple[str | None, list[int | None]]:
    """Select the best checkpoint based on saved evaluation metrics.

    :param str base_dir: Base directory containing GLUE runs.
    :param str task: GLUE task name.
    :param int num_checkpoints_to_merge: Number of recent checkpoints to merge.
    :return tuple[str | None, list[int | None]]: Checkpoint dir and ids.
    """
    best_accuracy = -float("inf")
    best_checkpoint_path = None
    best_checkpoint = None

    base_path = Path(base_dir)
    # Explore all directories in the given structure
    for json_path in base_path.rglob("all_results_step_*.json"):
        if task not in json_path.as_posix():
            continue

        # Read the eval accuracy from the JSON file
        with json_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
            eval_accuracy = compute_glue_score(task, results) or results.get(
                TASK_TO_METRIC.get(task, ""), 0
            )

            # Extract step number from the file name (e.g., all_results_step_{i}.json)
            step_number = int(json_path.stem.split("_")[3])

            # Update if a higher eval_accuracy is found
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy

                checkpoint = step_number
                checkpoint_candidates = [
                    json_path.parent / "checkpoints",
                    json_path.parent / "model_checkpoints",
                ]
                for checkpoint_folder in checkpoint_candidates:
                    if (checkpoint_folder / str(checkpoint)).exists():
                        best_checkpoint_path, best_checkpoint = (
                            checkpoint_folder,
                            checkpoint,
                        )
                        break

    if best_checkpoint_path is None or best_checkpoint is None:
        return None, [None]

    checkpoint_list = [best_checkpoint]
    if num_checkpoints_to_merge > 1:
        ckpts = list(Path(best_checkpoint_path).iterdir())
        ckpts = [
            int(ckpt.name) for ckpt in ckpts if int(ckpt.name) <= int(best_checkpoint)
        ]
        ckpts.sort()

        checkpoint_list = (
            ckpts
            if len(ckpts) < num_checkpoints_to_merge
            else ckpts[-num_checkpoints_to_merge:]
        )

    return (
        str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        checkpoint_list,
    )


def _normalize_glue_pretrained_checkpoint_root(checkpoint_root: str | Path) -> Path:
    """Normalize GLUE pretrained-checkpoint root without breaking transfer paths.

    Accepts both modern ``checkpoints/`` and legacy ``model_checkpoints/`` roots.
    If a run root is provided, prefer ``checkpoints/`` when present and fall back
    to ``model_checkpoints/`` for older artifacts.

    :param str | Path checkpoint_root: User-provided checkpoint root.
    :return Path: Normalized checkpoint directory root.
    """
    root = Path(checkpoint_root)
    if root.name in {"checkpoints", "model_checkpoints"}:
        return root

    modern = root / "checkpoints"
    legacy = root / "model_checkpoints"
    if modern.is_dir():
        return modern
    if legacy.is_dir():
        return legacy
    return modern


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_dir: str,
    checkpoint_id: int | str,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Load pretrained weights from a checkpoint directory.

    :param torch.nn.Module model: Model to load weights into.
    :param str checkpoint_dir: Directory containing checkpoints.
    :param int | str checkpoint_id: Checkpoint number or tag to load.
    :param logging.Logger logger: Logger for output.
    :return torch.nn.Module: Model with loaded weights.
    """
    checkpoint_path = Path(checkpoint_dir) / str(checkpoint_id)
    state_dict_path = checkpoint_path / MODEL_WEIGHTS_NAME

    # Portable safetensors payload is preferred when available.
    if state_dict_path.exists():
        logger.info(f"Loading state dict from {state_dict_path}")
        state_dict = load_model_safetensors(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded state dict with {len(state_dict)} keys")
        logger.info(f"✅ Successfully loaded pretrained weights from {state_dict_path}")
    else:
        logger.warning(
            f"No {MODEL_WEIGHTS_NAME} found at {state_dict_path}; "
            "attempting DeepSpeed fp32 shard conversion."
        )
        try:
            state_dict = load_deepspeed_fp32_state_dict(checkpoint_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Unable to load checkpoint {checkpoint_path}: expected either "
                f"{MODEL_WEIGHTS_NAME} or a DeepSpeed ZeRO checkpoint layout."
            ) from exc
        logger.info(
            "Loaded fp32 state dict from DeepSpeed checkpoint shards at "
            f"{checkpoint_path}"
        )

    # For GLUE init we always drop task/LM heads:
    # - ``decoder.*`` is MLM-only.
    # - ``classifier.*`` is task-head specific and often shape-incompatible across
    #   transfer map pairs (for example MNLI:3-way -> QNLI/MRPC/RTE:2-way).
    cleaned_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.startswith("classifier.") or k.startswith("decoder."))
    }
    logger.info(f"After filtering: {len(cleaned_state_dict)} keys to load")

    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict, strict=False
    )

    if missing_keys:
        logger.info(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.info(f"Unexpected keys: {unexpected_keys}")

    return model


def _resolve_checkpoint_retention_limit(cfg: Config) -> int:
    """Resolve effective checkpoint retention limit from trainer config.

    :param Config cfg: Runtime GLUE training configuration.
    :return int: Maximum number of retained checkpoints (0 disables pruning).
    """
    save_total_limit = getattr(cfg.trainer, "save_total_limit", None)
    if save_total_limit is not None:
        return max(0, int(save_total_limit))
    max_ckpt = getattr(cfg.trainer, "max_ckpt", None)
    if max_ckpt is not None:
        return max(0, int(max_ckpt))
    return 0


def _prune_step_checkpoints(checkpoint_dir: Path, retention_limit: int) -> None:
    """Prune old numeric step checkpoint folders in ``checkpoint_dir``.

    :param Path checkpoint_dir: Root directory containing ``<step>/`` folders.
    :param int retention_limit: Number of newest numeric checkpoints to keep.
    """
    if retention_limit <= 0 or not checkpoint_dir.exists():
        return

    checkpoints: list[tuple[int, Path]] = []
    for item_path in checkpoint_dir.iterdir():
        if not item_path.is_dir():
            continue
        try:
            checkpoints.append((int(item_path.name), item_path))
        except ValueError:
            continue

    if len(checkpoints) <= retention_limit:
        return

    checkpoints.sort(key=lambda item: item[0])
    for _, old_path in checkpoints[: len(checkpoints) - retention_limit]:
        shutil.rmtree(old_path)
        logger.info(f"Removed old checkpoint: {old_path} (limit={retention_limit})")


def _save_portable_glue_checkpoint_weights(
    model: torch.nn.Module,
    accelerator: Accelerator,
    checkpoint_path: Path,
) -> bool:
    """Ensure a portable ``model.safetensors`` exists in a GLUE step checkpoint.

    :param torch.nn.Module model: Prepared GLUE model.
    :param Accelerator accelerator: Active accelerator runtime.
    :param Path checkpoint_path: Checkpoint directory ``checkpoints/<step>/``.
    :return bool: True when portable weights exist (written or already present).
    """
    weights_path = checkpoint_path / MODEL_WEIGHTS_NAME
    if weights_path.exists():
        return True

    try:
        # Some distributed backends require all ranks to participate in
        # state-dict collection even if only rank 0 writes the file.
        state_dict = accelerator.get_state_dict(model, unwrap=True)
    except Exception as exc:
        if accelerator.is_main_process:
            logger.warning(
                "Unable to export portable GLUE weights to %s: %s. "
                "Resumable state was still saved.",
                weights_path,
                exc,
            )
        return False

    if not accelerator.is_main_process:
        return False

    try:
        save_state_dict_safetensors(
            state_dict,
            checkpoint_path,
            metadata={"format": "pt", "source": "accelerate.get_state_dict"},
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist portable GLUE weights at %s: %s. "
            "Resumable state was still saved.",
            weights_path,
            exc,
        )
        return False
    return True


def _should_save_glue_checkpoint(
    *,
    save_strategy: str,
    completed_steps: int,
    num_update_steps_per_epoch: int,
    save_steps: int | None,
    eval_ran_this_step: bool,
    metric_improved_this_eval: bool,
) -> bool:
    """Decide whether to save a GLUE checkpoint at the current update step.

    :param str save_strategy: Trainer save strategy.
    :param int completed_steps: Current completed optimizer steps.
    :param int num_update_steps_per_epoch: Steps per epoch.
    :param int | None save_steps: Step interval for ``save_strategy=steps``.
    :param bool eval_ran_this_step: Whether evaluation ran on this update step.
    :param bool metric_improved_this_eval: Whether eval metric improved this step.
    :return bool: True when a checkpoint should be saved.
    """
    normalized = str(save_strategy).strip().lower()
    if normalized == "no":
        return False
    if normalized == "best":
        return eval_ran_this_step and metric_improved_this_eval
    if normalized == "epoch":
        return completed_steps % max(1, num_update_steps_per_epoch) == 0
    if normalized == "steps":
        if save_steps is None or int(save_steps) <= 0:
            return False
        return completed_steps % int(save_steps) == 0
    # Preserve legacy fallback behavior for unknown strategies.
    return eval_ran_this_step


def _resolve_glue_training_schedule(
    cfg: Config,
    *,
    batches_per_process: int,
) -> tuple[int, int, int]:
    """Resolve training step/epoch schedule from prepared dataloader length.

    :param Config cfg: Runtime training config.
    :param int batches_per_process: Prepared per-process train dataloader length.
    :return tuple[int, int, int]: (updates_per_epoch, max_steps, num_train_epochs).
    """
    updates_per_epoch = max(
        1,
        math.ceil(batches_per_process / int(cfg.trainer.gradient_accumulation_steps)),
    )
    if cfg.trainer.max_steps is None or cfg.trainer.max_steps <= 0:
        num_train_epochs = max(1, int(cfg.trainer.num_train_epochs))
        max_steps = num_train_epochs * updates_per_epoch
        return updates_per_epoch, max_steps, num_train_epochs

    max_steps = int(cfg.trainer.max_steps)
    num_train_epochs = math.ceil(max_steps / updates_per_epoch)
    return updates_per_epoch, max_steps, num_train_epochs


def save_training_checkpoint(
    cfg: Config,
    model: torch.nn.Module,
    accelerator: Accelerator,
    completed_steps: int,
) -> None:
    """Save a training checkpoint during fine-tuning.

    :param Config cfg: Configuration object.
    :param torch.nn.Module model: Model to save.
    :param Accelerator accelerator: Accelerator instance.
    :param int completed_steps: Current training step.
    """
    output_dir = Path(cfg.trainer.output_dir)
    resume_checkpoint_dir = output_dir / "checkpoints"
    checkpoint_tag = str(completed_steps)
    checkpoint_path = resume_checkpoint_dir / checkpoint_tag

    # Save resumable Accelerate state for true optimizer/scheduler resume.
    accelerator.save_state(output_dir=str(checkpoint_path))
    accelerator.wait_for_everyone()
    _save_portable_glue_checkpoint_weights(model, accelerator, checkpoint_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        retention_limit = _resolve_checkpoint_retention_limit(cfg)
        if retention_limit > 0:
            _prune_step_checkpoints(resume_checkpoint_dir, retention_limit)
    accelerator.wait_for_everyone()


def _build_glue_attention_mask(
    attention_mask: torch.Tensor,
    *,
    use_hf_signature: bool,
    dtype_pad_mask: torch.dtype,
) -> torch.Tensor:
    """Build task-appropriate attention mask representation.

    :param torch.Tensor attention_mask: Collator-produced 0/1 attention mask.
    :param bool use_hf_signature: Whether model expects HF-style masks.
    :param torch.dtype dtype_pad_mask: Output dtype for additive masks.
    :return torch.Tensor: HF-style 0/1 mask or additive 0/-inf mask.
    """
    if use_hf_signature:
        return attention_mask
    return torch.where(attention_mask == 1, float(0.0), float("-inf")).type(
        dtype_pad_mask
    )


def _create_glue_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> DataCollatorWithPadding:
    """Create the GLUE padding collator from config.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer used for padding.
    :param Config cfg: Training configuration.
    :return DataCollatorWithPadding: Configured collator instance.
    """
    return DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
    )


def trainer(cfg: Config) -> None:
    """Run GLUE/SuperGLUE fine-tuning loop.

    :param Config cfg: Training configuration.
    """
    canonical_cfg = deepcopy(cfg)

    # Extract task and meta_task from config
    glue_task = _resolve_glue_task(canonical_cfg)
    meta_task = "glue"  # Default for GLUE tasks
    experiment_id = getattr(canonical_cfg, "id", "0")

    # Use a mutable runtime copy so canonical config remains task-stable
    cfg = deepcopy(canonical_cfg)

    # Update cfg to have these as direct attributes for compatibility
    cfg.glue.task_name = glue_task
    cfg.meta_task = meta_task
    cfg.id = experiment_id
    cfg.mode = getattr(cfg, "mode", "eval")  # Default to eval mode
    cfg.num_labels = cfg.glue.num_labels if hasattr(cfg, "glue") else 2
    cfg.max_seq_len = cfg.glue.max_seq_length if hasattr(cfg, "glue") else 128
    output_dir = Path(cfg.trainer.output_dir)
    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=False,
    )
    # Handle mixed precision setting (could be bool or string)
    mixed_precision = resolve_mixed_precision(
        cfg.trainer.mixed_precision,
        task="glue",
    )
    cfg.trainer.mixed_precision = mixed_precision

    wandb_enabled = cfg.wandb.enabled and cfg.wandb.mode != "disabled"
    accelerator = create_accelerator(
        use_cpu=bool(getattr(cfg.trainer, "use_cpu", False)),
        log=logger,
        log_with="wandb" if wandb_enabled else None,
        mixed_precision=mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
    )
    validate_muon_distributed_compatibility(
        accelerator=accelerator,
        optimizer_name=cfg.optimizer.name,
        log=logger,
        context="glue",
    )

    tracker_config_dict = prepare_wandb_config(canonical_cfg)
    if accelerator.is_main_process:
        accelerator.print(
            "Resolved task config:\n" + format_resolved_config(tracker_config_dict)
        )

    # Initialise the wandb run and pass wandb parameters
    if wandb_enabled:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            init_kwargs={
                "wandb": {
                    "name": cfg.wandb.name,
                    "entity": cfg.wandb.entity,
                    "config": tracker_config_dict,
                    "tags": cfg.wandb.tags,
                    "dir": cfg.wandb.dir,
                    "mode": cfg.wandb.mode,
                    "resume": cfg.wandb.resume,
                }
            },
        )

        _configure_wandb_metrics(accelerator)
        _update_wandb_config(accelerator, cfg)

    set_seed(int(cfg.seed))

    # Validate configuration after accelerator is initialized (for logger)
    try:
        validate_glue_config(cfg)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Configure TF32 precision for GPUs with compute capability >= 8.0
    configure_tf32(enabled=cfg.trainer.tf32, print_fn=accelerator.print)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir.is_dir():
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Force SDPA for GLUE - variable-length batches are incompatible with packed attention
    if hasattr(cfg.model, "attn_backend") and cfg.model.attn_backend != "sdpa":
        logger.warning(
            "Packed attention is not supported for GLUE evaluation due to "
            "variable-length sequences. Forcing attn_backend='sdpa'."
        )
    # Always use SDPA (eager) attention for GLUE.

    # Check from_hub in raw model dict for GLUE tasks
    from_hub = False
    if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
        from_hub = cfg._raw_model_dict.get("from_hub", False)
    elif hasattr(cfg.model, "from_hub"):
        from_hub = cfg.model.from_hub

    if from_hub:
        tokenizer = _load_from_hub_tokenizer(cfg)
    else:
        # Import our new config system
        from neobert.config import ConfigLoader

        # For GLUE, we MUST have pretrained model info
        # Check if we're allowing random weights for testing
        allow_random_weights = cfg.glue.allow_random_weights
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            allow_random_weights = cfg._raw_model_dict.get(
                "allow_random_weights", allow_random_weights
            )

        if allow_random_weights:
            # Skip pretrained config loading for testing
            pretrained_config_path = None
        elif cfg.glue.pretrained_model_path:
            pretrained_config_path = cfg.glue.pretrained_model_path
        elif (
            hasattr(cfg, "_raw_model_dict")
            and cfg._raw_model_dict
            and "pretrained_config_path" in cfg._raw_model_dict
        ):
            pretrained_config_path = cfg._raw_model_dict["pretrained_config_path"]
        else:
            raise ValueError(
                "GLUE evaluation requires a pretrained model! "
                "Please specify 'glue.pretrained_model_path' in your config, "
                "or set 'allow_random_weights: true' for testing."
            )
        if pretrained_config_path:
            model_pretraining_config = ConfigLoader.load(pretrained_config_path)
            model_pretraining_config.model.attn_backend = "sdpa"
            tokenizer_source = (
                model_pretraining_config.tokenizer.path
                or model_pretraining_config.tokenizer.name
            )
            tokenizer = get_tokenizer(
                pretrained_model_name_or_path=tokenizer_source,
                max_length=model_pretraining_config.tokenizer.max_length,
                trust_remote_code=model_pretraining_config.tokenizer.trust_remote_code,
                revision=model_pretraining_config.tokenizer.revision,
                allow_special_token_rewrite=model_pretraining_config.tokenizer.allow_special_token_rewrite,
            )
        else:
            # Use default tokenizer for random weights test
            tokenizer = get_tokenizer(
                pretrained_model_name_or_path="bert-base-uncased",
                max_length=128,
                trust_remote_code=cfg.tokenizer.trust_remote_code,
                revision=cfg.tokenizer.revision,
                allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
            )

    accelerator.print("Loading metric...")
    # Keep train/eval metric state separate so eval.compute() does not reset
    # the running train metric window and vice versa.
    train_metric_tracker = _load_glue_metric(glue_task, cfg.meta_task, cfg.id)
    eval_metric_tracker = _load_glue_metric(glue_task, cfg.meta_task, cfg.id)

    # Load additional metric for the mismatched validation set of mnli
    if glue_task == "mnli":
        mm_metric = evaluate.load(
            cfg.meta_task, "mnli_mismatched", experiment_id=cfg.id
        )

    # def compute_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    #     result = metric.compute(predictions=preds, references=p.label_ids)
    #     if len(result) > 1:
    #         result["combined_score"] = np.mean(list(result.values())).item()
    #     return result

    # Loading the dataset
    accelerator.print("Loading dataset...")
    if glue_task == "snli":
        raw_datasets = load_dataset("stanfordnlp/snli")
        raw_datasets = raw_datasets.filter(lambda example: example["label"] != -1)
    elif glue_task == "allnli":
        raw_datasets = load_dataset("sentence-transformers/all-nli", name="pair-class")

        def collapse_classes(examples: dict[str, Any]) -> dict[str, Any]:
            """Collapse neutral/contradiction into non-entailment.

            :param dict[str, Any] examples: Batched examples.
            :return dict[str, Any]: Updated examples with collapsed labels.
            """
            examples["label"] = [
                1 if label == 2 else label for label in examples["label"]
            ]
            return examples

        raw_datasets = raw_datasets.map(
            collapse_classes,
            batched=True,
            desc="Collapsing classes into entailment and not entailment.",
        )

    elif cfg.meta_task == "glue":
        raw_datasets = load_dataset("glue", glue_task)
    else:
        raw_datasets = load_dataset(
            "json",
            data_dir=Path(os.environ["HF_DATASETS_CACHE"]) / "super_glue" / glue_task,
        )

    # Preprocessing the datasets
    mapping = partial(process_function, tokenizer=tokenizer, cfg=cfg)
    glue_num_proc = int(getattr(cfg.glue, "preprocessing_num_proc", 0) or 0)
    map_num_proc = glue_num_proc if glue_num_proc > 0 else None
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            mapping,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=map_num_proc,
            desc="Preprocessing the dataset",
        )

    is_regression = glue_task == "stsb"
    if not is_regression:
        processed_datasets = processed_datasets.cast_column(
            "labels", ClassLabel(names=processed_datasets["train"].unique("labels"))
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched"
        if glue_task == "mnli"
        else ("dev" if glue_task == "allnli" else "validation")
    ]

    if glue_task == "mnli":
        mm_eval_dataset = processed_datasets["validation_mismatched"]

    # Labels
    if not is_regression:
        label_list = train_dataset.features["labels"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    if getattr(cfg.glue, "num_labels", None) not in (None, num_labels):
        logger.warning(
            "Config glue.num_labels=%s does not match dataset-inferred value=%s; "
            "using inferred value.",
            cfg.glue.num_labels,
            num_labels,
        )
    cfg.glue.num_labels = num_labels

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # Log a few random samples from the evaluation set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    data_collator = _create_glue_data_collator(tokenizer, cfg)

    # Keep additive pad masks in float32 for numerical stability (match
    # pretraining). This does not force full-fp32 model execution; runtime
    # compute precision is still governed by Accelerator mixed precision.
    dtype_pad_mask = torch.float32

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply padding collator and build additive attention mask.

        :param list[dict[str, Any]] batch: Batch of examples.
        :return dict[str, Any]: Collated batch with attention mask.
        """
        batch = data_collator(batch)
        # NeoBERT path expects additive masks; generic HF baseline path expects
        # standard 0/1 attention masks.
        batch["attention_mask"] = _build_glue_attention_mask(
            batch["attention_mask"],
            use_hf_signature=from_hub,
            dtype_pad_mask=dtype_pad_mask,
        )
        return batch

    # Use per_device batch sizes consistently
    train_batch_size = (
        cfg.trainer.per_device_train_batch_size or cfg.trainer.train_batch_size or 16
    )
    eval_batch_size = (
        cfg.trainer.per_device_eval_batch_size or cfg.trainer.eval_batch_size or 32
    )
    glue_num_workers = max(0, int(getattr(cfg.glue, "num_workers", 0)))
    train_loader_kwargs = {
        "collate_fn": collate_fn,
        "batch_size": train_batch_size,
        "num_workers": glue_num_workers,
    }
    eval_loader_kwargs = {
        "collate_fn": collate_fn,
        "batch_size": eval_batch_size,
        "num_workers": glue_num_workers,
    }
    if glue_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["persistent_workers"] = True

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **train_loader_kwargs,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        **eval_loader_kwargs,
    )
    if glue_task == "mnli":
        mm_eval_dataloader = DataLoader(
            mm_eval_dataset,
            **eval_loader_kwargs,
        )

    # Model
    if from_hub:
        trust_remote_code = bool(getattr(cfg.tokenizer, "trust_remote_code", False))
        config = AutoConfig.from_pretrained(
            cfg.model.name,
            num_labels=num_labels,
            finetuning_task=glue_task,
            revision="main",
            trust_remote_code=trust_remote_code,
        )
        # if "nomic" in cfg.model.name:
        #     base_model = AutoModelForMaskedLM.from_pretrained(
        #         cfg.model.name,
        #         from_tf=False,
        #         config=config,
        #         revision="main",
        #         trust_remote_code=True,
        #         ignore_mismatched_sizes=False,
        #     )
        #     model = NomicBERTForSequenceClassification(
        #         config,
        #         base_model.bert,
        #         num_labels=num_labels,
        #         classifier_dropout=cfg.model.classifier_dropout,
        #         classifier_init_range=cfg.model.classifier_init_range,
        #     )
        # else:
        if True:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model.name,
                from_tf=False,
                config=config,
                revision="main",
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=False,
            )
    else:
        # Convert config objects to dict for unpacking
        if "model_pretraining_config" in locals() and model_pretraining_config:
            model_config_dict = (
                model_pretraining_config.model.__dict__.copy()
                if hasattr(model_pretraining_config.model, "__dict__")
                else {}
            )
        elif hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            # Use raw model dict when allow_random_weights is true
            model_config_dict = cfg._raw_model_dict.copy()
        else:
            # Fallback to cfg.model attributes
            model_config_dict = {
                "hidden_size": getattr(cfg.model, "hidden_size", 768),
                "num_hidden_layers": getattr(cfg.model, "num_hidden_layers", 12),
                "num_attention_heads": getattr(cfg.model, "num_attention_heads", 12),
                "intermediate_size": getattr(cfg.model, "intermediate_size", 3072),
                "vocab_size": getattr(cfg.model, "vocab_size", 30522),
                "hidden_act": getattr(cfg.model, "hidden_act", "gelu"),
                "max_length": getattr(cfg.model, "max_position_embeddings", 512),
                "layer_norm_eps": getattr(cfg.model, "layer_norm_eps", 1e-12),
            }

        # Map dropout_prob to dropout and remove classifier_init_range from model config
        if "dropout_prob" in model_config_dict:
            model_config_dict["dropout"] = model_config_dict.pop("dropout_prob")
        if "max_position_embeddings" in model_config_dict:
            model_config_dict["max_length"] = model_config_dict.pop(
                "max_position_embeddings"
            )
        if "classifier_init_range" in model_config_dict:
            model_config_dict.pop("classifier_init_range")
        if "allow_random_weights" in model_config_dict:
            model_config_dict.pop("allow_random_weights")
        if "pretrained_checkpoint_dir" in model_config_dict:
            model_config_dict.pop("pretrained_checkpoint_dir")
        if "pretrained_checkpoint" in model_config_dict:
            model_config_dict.pop("pretrained_checkpoint")
        if "name_or_path" in model_config_dict:
            model_config_dict.pop("name_or_path")
        if "name" in model_config_dict:
            model_config_dict.pop("name")

        # Use model config directly - don't merge with tokenizer config
        # The tokenizer's vocab_size should match the model's anyway
        combined_config = model_config_dict
        combined_config.pop("xformers_attention", None)
        combined_config.pop("flash_attention", None)
        combined_config["attn_backend"] = "sdpa"

        # If using random weights (for testing), round vocab_size for GPU efficiency
        allow_random_weights = cfg.glue.allow_random_weights
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            allow_random_weights = cfg._raw_model_dict.get(
                "allow_random_weights", allow_random_weights
            )

        if allow_random_weights and "vocab_size" in combined_config:
            from neobert.config import round_up_to_multiple

            # Round vocab_size for GPU efficiency when using random weights
            combined_config["vocab_size"] = round_up_to_multiple(len(tokenizer), 128)

        model = NeoBERTForSequenceClassification(
            NeoBERTConfig(**combined_config),
            num_labels=num_labels,
            classifier_dropout=cfg.glue.classifier_dropout,
            classifier_init_range=cfg.glue.classifier_init_range,
        )

    pretrained_checkpoint_dir: Path | None = None
    pretrained_checkpoint: int | str | None = None
    allow_random_weights = cfg.glue.allow_random_weights

    if cfg.glue.transfer_from_task:
        task_to_transfer_from = TASK_TO_TRANSFER_FROM.get(glue_task, None)
        if not task_to_transfer_from:
            raise ValueError(f"Task to transfer from for {glue_task} is not set.")
        transfer_checkpoint_dir, checkpoint_list = get_best_checkpoint_path(
            str(
                Path(cfg.glue.pretrained_checkpoint_dir)
                / "glue"
                / str(cfg.glue.pretrained_checkpoint)
            ),
            task_to_transfer_from,
        )
        pretrained_checkpoint = checkpoint_list[-1]
        logger.info(
            f"Transfering from: {transfer_checkpoint_dir}, {pretrained_checkpoint}"
        )
        if not transfer_checkpoint_dir or pretrained_checkpoint is None:
            raise ValueError("Unable to retrieve checkpoint to transfer from.")
        pretrained_checkpoint_dir = Path(transfer_checkpoint_dir)

    else:
        # Get checkpoint info from raw model dict for GLUE
        logger.info("Looking for pretrained checkpoint info...")

        # Prefer GLUEConfig for checkpoint info
        pretrained_checkpoint_dir_cfg = cfg.glue.pretrained_checkpoint_dir
        pretrained_checkpoint = cfg.glue.pretrained_checkpoint

        # Fall back to raw model dict for legacy configs
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            pretrained_checkpoint_dir_cfg = cfg._raw_model_dict.get(
                "pretrained_checkpoint_dir", pretrained_checkpoint_dir_cfg
            )
            pretrained_checkpoint = cfg._raw_model_dict.get(
                "pretrained_checkpoint", pretrained_checkpoint
            )
            allow_random_weights = cfg._raw_model_dict.get(
                "allow_random_weights", allow_random_weights
            )

        if pretrained_checkpoint_dir_cfg:
            pretrained_checkpoint_dir = Path(pretrained_checkpoint_dir_cfg)

    if pretrained_checkpoint_dir is None or pretrained_checkpoint is None:
        if allow_random_weights:
            logger.warning(
                "⚠️  Using random weights for testing as allow_random_weights=true"
            )
            pretrained_checkpoint = None
        else:
            raise ValueError(
                "GLUE evaluation requires pretrained weights!\n"
                "Please specify either:\n"
                "  1. 'glue.pretrained_checkpoint_dir' and 'glue.pretrained_checkpoint' in config\n"
                "  2. Set 'glue.allow_random_weights: true' for testing with random weights"
            )
    else:
        pretrained_checkpoint_dir = _normalize_glue_pretrained_checkpoint_root(
            pretrained_checkpoint_dir
        )
        logger.info(
            f"Will load checkpoint {pretrained_checkpoint} from {pretrained_checkpoint_dir}"
        )

    # Load pretrained weights if available
    if not from_hub and pretrained_checkpoint is not None:
        model = load_pretrained_weights(
            model, str(pretrained_checkpoint_dir), pretrained_checkpoint, logger
        )

    model = _maybe_compile_model(model, cfg, accelerator, logger)

    # Optimizer
    optimizer = get_optimizer(
        model,
        accelerator.distributed_type,
        model_config=getattr(model, "config", None),
        name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(getattr(cfg.optimizer, "betas", [0.9, 0.999])),
        eps=getattr(cfg.optimizer, "eps", 1e-8),
        muon_config=getattr(cfg.optimizer, "muon_config", None),
    )

    # Prepare with accelerator before deriving epoch-based max_steps so distributed
    # runs use per-process dataloader length (not global pre-shard length).
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
    )

    if glue_task == "mnli":
        mm_eval_dataloader = accelerator.prepare(mm_eval_dataloader)

    num_update_steps_per_epoch, resolved_max_steps, resolved_num_train_epochs = (
        _resolve_glue_training_schedule(
            cfg,
            batches_per_process=len(train_dataloader),
        )
    )
    cfg.trainer.max_steps = resolved_max_steps
    cfg.trainer.num_train_epochs = resolved_num_train_epochs
    if cfg.trainer.max_steps <= 0:
        raise ValueError(
            "GLUE max_steps resolved to a non-positive value after dataloader "
            "preparation; check num_train_epochs/max_steps config."
        )

    if cfg.scheduler.warmup_percent is not None:
        if cfg.scheduler.warmup_steps is not None:
            warnings.warn(
                "Overriding warmup_steps based on scheduler.warmup_percent.",
                UserWarning,
                stacklevel=2,
            )
        cfg.scheduler.warmup_steps = math.ceil(
            cfg.trainer.max_steps / 100 * cfg.scheduler.warmup_percent
        )
    if cfg.scheduler.decay_percent is not None:
        if cfg.scheduler.decay_steps is not None:
            warnings.warn(
                "Overriding decay_steps based on scheduler.decay_percent.",
                UserWarning,
                stacklevel=2,
            )
        cfg.scheduler.decay_steps = math.ceil(
            cfg.trainer.max_steps / 100 * cfg.scheduler.decay_percent
        )
    elif cfg.scheduler.decay_steps is None:
        cfg.scheduler.decay_steps = cfg.trainer.max_steps

    scheduler_params = (
        cfg.scheduler.__dict__.copy()
        if hasattr(cfg.scheduler, "__dict__")
        else cfg.scheduler.copy()
    )
    if "name" in scheduler_params:
        scheduler_params["decay"] = scheduler_params.pop("name")
    if "final_lr_ratio" in scheduler_params:
        scheduler_params["final_ratio"] = scheduler_params.pop("final_lr_ratio")

    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=cfg.optimizer.lr,
        **scheduler_params,
    )
    scheduler = accelerator.prepare(scheduler)
    lr = cfg.optimizer.lr

    # Handle evaluation strategy - support both 'epoch' and 'steps'
    eval_strategy = getattr(cfg.trainer, "eval_strategy", "steps")
    if eval_strategy == "epoch":
        # Evaluate at the end of each epoch
        cfg.trainer.eval_steps = num_update_steps_per_epoch
        logger.info(
            f"Using epoch-based evaluation: will evaluate every {cfg.trainer.eval_steps} steps (1 epoch)"
        )
    elif eval_strategy == "steps":
        # Use the provided eval_steps or default to min of provided and epoch size
        if hasattr(cfg.trainer, "eval_steps") and cfg.trainer.eval_steps:
            cfg.trainer.eval_steps = min(
                cfg.trainer.eval_steps,
                num_update_steps_per_epoch,
            )
        else:
            cfg.trainer.eval_steps = min(500, num_update_steps_per_epoch)
            logger.info(
                f"No eval_steps provided, defaulting to {cfg.trainer.eval_steps}"
            )
    else:
        raise ValueError(
            f"Invalid eval_strategy: {eval_strategy}. Must be 'epoch' or 'steps'"
        )

    # To keep the last n checkpoints before the best model and do k cycles before early stopping, we save the last k+n models
    early_stopping = getattr(cfg.trainer, "early_stopping", 0)
    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
    if max_ckpt is not None and max_ckpt > 0 and early_stopping > 0:
        cfg.trainer.max_ckpt = max_ckpt + early_stopping
    save_strategy = str(getattr(cfg.trainer, "save_strategy", "steps")).strip().lower()
    save_model = bool(getattr(cfg.trainer, "save_model", True))
    save_steps = getattr(cfg.trainer, "save_steps", None)

    # Get loss function
    if not is_regression:
        loss_fct = CrossEntropyLoss()
    else:
        loss_fct = MSELoss()

    # Train!
    total_steps = cfg.trainer.max_steps
    total_batch_size = (
        train_batch_size
        * accelerator.num_processes
        * cfg.trainer.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {glue_task}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num epochs = {cfg.trainer.num_train_epochs}")
    logger.info(f"  Total training steps = {total_steps}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Learning rate = {lr}")
    logger.info(
        f"  Gradient accumulation steps = {cfg.trainer.gradient_accumulation_steps}"
    )
    logger.info(f"  Evaluation steps = {cfg.trainer.eval_steps}")
    logger.info(f"  Early stopping cycles = {early_stopping}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(total_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    resume_microbatch_in_epoch = 0

    # Optionally resume full Accelerate state (model/optimizer/scheduler/scaler).
    resume_checkpoint_root = output_dir / "checkpoints"
    resume_checkpoint_path, _ = _resolve_resume_checkpoint(
        cfg.trainer.resume_from_checkpoint,
        str(resume_checkpoint_root),
        str(output_dir),
    )
    if cfg.trainer.resume_from_checkpoint and resume_checkpoint_path:
        resume_checkpoint = Path(resume_checkpoint_path)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"resume_from_checkpoint path not found: {resume_checkpoint}"
            )

        accelerator.print(f"Resuming GLUE run from checkpoint: {resume_checkpoint}")
        accelerator.load_state(str(resume_checkpoint))

        step_from_optimizer = _get_optimizer_update_step(optimizer)
        if step_from_optimizer is not None and step_from_optimizer > 0:
            completed_steps = step_from_optimizer
        elif resume_checkpoint.name.isdigit():
            completed_steps = int(resume_checkpoint.name)
        else:
            kind, value = _parse_checkpoint_progress(str(resume_checkpoint))
            if kind == "epoch" and value is not None:
                completed_steps = (value + 1) * num_update_steps_per_epoch
            elif kind == "step" and value is not None:
                completed_steps = value

        starting_epoch = completed_steps // num_update_steps_per_epoch
        resume_update_in_epoch = completed_steps % num_update_steps_per_epoch
        resume_microbatch_in_epoch = (
            resume_update_in_epoch * cfg.trainer.gradient_accumulation_steps
        )
    elif cfg.trainer.resume_from_checkpoint:
        logger.warning(
            "resume_from_checkpoint is set but no valid checkpoints were found in "
            f"{resume_checkpoint_root}"
        )

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Initialize all training loop variables upfront
    results = {}
    total_loss = 0.0
    micro_loss_sum = 0.0
    micro_loss_count = 0
    early_stop = False
    prev_accuracy = 0.0
    early_stopping_counter = 1
    eval_metric = {}
    epoch = starting_epoch
    completed_steps = completed_steps  # Ensure it's in scope
    last_train_metrics = {}
    last_val_metrics = {}
    evaluation_round = 0
    train_predictions_buffer: list[torch.Tensor] = []
    train_references_buffer: list[torch.Tensor] = []

    for epoch in range(starting_epoch, cfg.trainer.num_train_epochs):
        sampler = getattr(train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_dataloader):
            if epoch == starting_epoch and micro_step < resume_microbatch_in_epoch:
                continue

            with accelerator.accumulate(model):
                is_last_microbatch = bool(accelerator.sync_gradients)
                _maybe_prepare_for_forward(
                    optimizer,
                    update_step=completed_steps,
                    is_last_microbatch=is_last_microbatch,
                )

                with accelerator.autocast():
                    logits = _forward_classifier_logits(
                        model,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids"),
                        use_hf_signature=from_hub,
                    )

                    # Debug logging for first few steps
                    if completed_steps < 3 and is_last_microbatch:
                        logger.info(
                            f"Step {completed_steps}: logits shape: {logits.shape}, logits mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}"
                        )
                        logger.info(
                            f"Step {completed_steps}: logits sample: {logits[0].detach().cpu()}"
                        )
                        logger.info(
                            f"Step {completed_steps}: labels: {batch['labels'][:5]}"
                        )

                    if not is_regression:
                        loss = loss_fct(
                            logits.view(-1, num_labels), batch["labels"].view(-1)
                        )
                    else:
                        if num_labels == 1:
                            loss = loss_fct(logits.squeeze(), batch["labels"].squeeze())
                        else:
                            loss = loss_fct(logits, batch["labels"])

                    # Compute train accuracy
                    predictions = (
                        logits.argmax(dim=-1)
                        if not is_regression
                        else (
                            logits.squeeze()
                            if logits.size() != torch.Size([1])
                            else logits
                        )
                    )
                # Keep local train-metric buffers on CPU to avoid GPU-memory
                # growth between eval windows.
                train_predictions_buffer.append(predictions.detach().to("cpu"))
                train_references_buffer.append(batch["labels"].detach().to("cpu"))

                accelerator.backward(loss)
                micro_loss_sum += float(loss.item())
                micro_loss_count += 1

                if is_last_microbatch:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    completed_steps += 1

                    if micro_loss_count > 0:
                        total_loss += micro_loss_sum / micro_loss_count
                    micro_loss_sum = 0.0
                    micro_loss_count = 0

                    ran_evaluation = False
                    metric_improved_this_eval = False

                    # Run evaluation
                    if (
                        cfg.trainer.eval_steps
                        and completed_steps % cfg.trainer.eval_steps == 0
                    ):
                        ran_evaluation = True
                        train_metric: dict[str, Any] = {}
                        if train_predictions_buffer and train_references_buffer:
                            local_train_predictions = torch.cat(
                                train_predictions_buffer,
                                dim=0,
                            ).to(accelerator.device)
                            local_train_references = torch.cat(
                                train_references_buffer,
                                dim=0,
                            ).to(accelerator.device)
                            gathered_train_predictions, gathered_train_references = (
                                accelerator.gather(
                                    (local_train_predictions, local_train_references)
                                )
                            )
                            train_metric_tracker.add_batch(
                                predictions=gathered_train_predictions,
                                references=gathered_train_references,
                            )
                            train_metric = train_metric_tracker.compute()
                            train_predictions_buffer.clear()
                            train_references_buffer.clear()

                        if len(train_metric) > 1:
                            train_metric["combined_score"] = np.mean(
                                list(train_metric.values())
                            ).item()

                        model.eval()
                        eval_metric = get_evaluation(
                            model=model,
                            dataloader=eval_dataloader,
                            accelerator=accelerator,
                            metric=eval_metric_tracker,
                            dtype_pad_mask=dtype_pad_mask,
                            is_regression=is_regression,
                            return_predictions=False,
                            use_hf_signature=from_hub,
                        )["eval_metric"]

                        # Log all metrics properly for STS-B (both Pearson and Spearman)
                        if glue_task == "stsb" and "spearmanr" in eval_metric:
                            logger.info(
                                f"step {completed_steps} eval pearson: {eval_metric.get('pearson', 0):.4f}"
                            )
                            logger.info(
                                f"step {completed_steps} eval spearmanr: {eval_metric.get('spearmanr', 0):.4f}"
                            )
                        else:
                            logger.info(
                                f"step {completed_steps} eval metric: {eval_metric}"
                            )

                        logger.info(
                            f"step {completed_steps} train metric: {train_metric}"
                        )
                        logger.info(
                            f"step {completed_steps} train loss: {total_loss / completed_steps}"
                        )

                        if glue_task == "mnli":
                            # Evaluation on matched MNLI
                            results["accuracy"] = eval_metric["accuracy"]

                            # Evaluation on mismatched MNLI
                            mm_eval_metric = get_evaluation(
                                model=model,
                                dataloader=mm_eval_dataloader,
                                accelerator=accelerator,
                                metric=mm_metric,
                                dtype_pad_mask=dtype_pad_mask,
                                is_regression=is_regression,
                                return_predictions=False,
                                use_hf_signature=from_hub,
                            )["eval_metric"]
                            results["accuracy_mm"] = mm_eval_metric["accuracy"]

                            res_mm = results["accuracy_mm"]
                            logger.info(
                                f"step {completed_steps} eval metric mismatched: {res_mm}"
                            )

                        train_epoch_pos = completed_steps / max(
                            1, num_update_steps_per_epoch
                        )
                        train_avg_loss = (
                            total_loss / completed_steps
                            if completed_steps > 0
                            else loss.item()
                        )

                        log_payload = {
                            "train/step": completed_steps,
                            "train/epoch": train_epoch_pos,
                            "train/loss": train_avg_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }

                        if train_metric:
                            for key, value in train_metric.items():
                                log_payload[f"train/{key}"] = value

                        val_metrics = eval_metric if glue_task != "mnli" else results
                        val_epoch = train_epoch_pos
                        log_payload["val/epoch"] = val_epoch
                        for key, value in val_metrics.items():
                            log_payload[f"val/{key}"] = value

                        score_for_early_stop = compute_glue_score(
                            glue_task, val_metrics
                        )
                        if score_for_early_stop is not None:
                            log_payload["val/score"] = score_for_early_stop

                        log_payload = {
                            k: _to_serializable(v) for k, v in log_payload.items()
                        }

                        evaluation_round += 1
                        accelerator.log(log_payload, step=completed_steps)

                        last_train_metrics = {
                            "step": completed_steps,
                            "epoch": train_epoch_pos,
                            "loss": train_avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        if train_metric:
                            last_train_metrics.update(
                                {
                                    k: _to_serializable(v)
                                    for k, v in train_metric.items()
                                }
                            )

                        last_val_metrics = {
                            "step": completed_steps,
                            "epoch": val_epoch,
                        }
                        last_val_metrics.update(
                            {k: _to_serializable(v) for k, v in val_metrics.items()}
                        )
                        if score_for_early_stop is not None:
                            last_val_metrics["score"] = _to_serializable(
                                score_for_early_stop
                            )

                        all_results = {
                            f"eval_{k}": _to_serializable(v)
                            for k, v in eval_metric.items()
                        }
                        if glue_task == "mnli":
                            all_results = {
                                f"eval_{k}": _to_serializable(v)
                                for k, v in results.items()
                            }

                        if accelerator.is_main_process:
                            _save_metrics(
                                cfg.trainer.output_dir, "train", last_train_metrics
                            )
                            _save_metrics(
                                cfg.trainer.output_dir,
                                "val",
                                last_val_metrics,
                            )
                            result_path = (
                                output_dir / f"all_results_step_{completed_steps}.json"
                            )
                            with result_path.open("w", encoding="utf-8") as f:
                                accelerator.print(
                                    f"Writing eval metrics to {result_path}"
                                )
                                json.dump(all_results, f, indent=2)
                        accelerator.wait_for_everyone()

                        fallback_metric = next(iter(val_metrics.values()), 0.0)
                        curr_accuracy = (
                            score_for_early_stop
                            if score_for_early_stop is not None
                            else fallback_metric
                        )
                        metric_improved_this_eval = curr_accuracy > prev_accuracy

                        # Update early stopping counter
                        if metric_improved_this_eval:
                            prev_accuracy = curr_accuracy
                            early_stopping_counter = 0

                        else:
                            early_stopping_counter += 1

                        if (
                            early_stopping > 0
                            and early_stopping_counter >= early_stopping
                        ):
                            accelerator.print(
                                f"Evaluation accuracy has not improved in {early_stopping} cycles of {cfg.trainer.eval_steps} evaluation steps, stopping the training."
                            )
                            early_stop = True

                        model.train()

                    should_save = _should_save_glue_checkpoint(
                        save_strategy=save_strategy,
                        completed_steps=completed_steps,
                        num_update_steps_per_epoch=num_update_steps_per_epoch,
                        save_steps=save_steps,
                        eval_ran_this_step=ran_evaluation,
                        metric_improved_this_eval=metric_improved_this_eval,
                    )
                    # save_total_limit controls pruning only. save_total_limit=0
                    # means keep all checkpoints while still saving.
                    if should_save and save_model:
                        save_training_checkpoint(
                            cfg, model, accelerator, completed_steps
                        )

            if completed_steps >= cfg.trainer.max_steps or early_stop:
                break

        if completed_steps >= cfg.trainer.max_steps or early_stop:
            break

    # Prepare final metrics for logging and persistence
    if last_val_metrics:
        final_metrics = {
            key: _to_serializable(value)
            for key, value in last_val_metrics.items()
            if key not in {"epoch", "step"}
        }
        final_epoch_value = _to_serializable(last_val_metrics.get("epoch", epoch))
    elif eval_metric:
        source = eval_metric if glue_task != "mnli" else results
        final_metrics = {key: _to_serializable(value) for key, value in source.items()}
        final_epoch_value = epoch
    else:
        final_metrics = {}
        final_epoch_value = epoch

    # Print final metrics to console and log file (main process only).
    if accelerator.is_main_process:
        accelerator.print("=" * 60)
        accelerator.print(f"Training completed for {glue_task.upper()}")
        accelerator.print(f"Final metrics at step {completed_steps}:")
        for key, value in final_metrics.items():
            accelerator.print(f"  {key}: {value:.4f}")
        accelerator.print("=" * 60)

        # Also log for debugging
        logger.info("=" * 60)
        logger.info(f"Training completed for {glue_task.upper()}")
        logger.info(f"Final metrics at step {completed_steps}:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 60)

    # Add final metrics to wandb
    final_train_loss = total_loss / completed_steps if completed_steps > 0 else 0.0
    final_payload = {
        "final/step": completed_steps,
        "final/train_loss": final_train_loss,
        "final/epoch": final_epoch_value,
    }
    for key, value in final_metrics.items():
        final_payload[f"final/{key}"] = value

    accelerator.log(
        {k: _to_serializable(v) for k, v in final_payload.items()}, step=completed_steps
    )

    # Fix: Update W&B summary with final metrics
    if accelerator.is_main_process:
        try:
            # Get wandb tracker and update summary
            for tracker in accelerator.trackers:
                if tracker.__class__.__name__ == "WandBTracker":
                    if hasattr(tracker, "run") and tracker.run:
                        # Update summary with final metrics
                        summary_metrics = {
                            f"summary/final_{k}": v for k, v in final_metrics.items()
                        }
                        summary_metrics["summary/final_train_loss"] = final_train_loss
                        summary_metrics["summary/final_step"] = completed_steps
                        summary_metrics["summary/final_epoch"] = final_epoch_value
                        tracker.run.summary.update(summary_metrics)
                        logger.info("Updated W&B run summary with final metrics")
        except Exception as e:
            logger.warning(f"Failed to update W&B summary: {e}")

    accelerator.end_training()

    # Save final results to disk
    final_eval_dump = {
        f"eval_{k}": _to_serializable(v) for k, v in final_metrics.items()
    }
    if accelerator.is_main_process:
        _save_metrics(
            cfg.trainer.output_dir,
            "final",
            {
                **final_metrics,
                "train_loss": final_train_loss,
                "epoch": final_epoch_value,
            },
        )

        with (output_dir / "all_results.json").open("w", encoding="utf-8") as f:
            json.dump(final_eval_dump, f, indent=2)

        # Also save to timestamped file for clarity
        with (output_dir / f"all_results_step_{completed_steps}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(final_eval_dump, f, indent=2)
            logger.info(
                f"Final results saved to {cfg.trainer.output_dir}/all_results_step_{completed_steps}.json"
            )
    accelerator.wait_for_everyone()
