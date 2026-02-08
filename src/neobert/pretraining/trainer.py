"""Pretraining loop for masked language modeling."""

import inspect
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

# PyTorch
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    DistributedType,
    ProjectConfiguration,
    send_to_device,
    set_seed,
)

# Hugging Face
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase

from neobert.checkpointing import save_model_safetensors
from neobert.config import Config, ConfigLoader, MuonConfig, round_up_to_multiple
from neobert.dataloader import get_dataloader
from neobert.kernels.attention import resolve_runtime_attn_backend
from neobert.kernels.backend import get_cross_entropy_loss, resolve_kernel_backend
from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import get_optimizer
from neobert.pretraining.masked_objective import MaskedPositionsOnlyMLMObjective
from neobert.scheduler import get_scheduler, resolve_scheduler_steps
from neobert.tokenizer import get_tokenizer, resolve_text_column
from neobert.training_utils import (
    _maybe_compile_model,
    _maybe_prepare_for_forward,
    _resolve_resume_checkpoint,
)
from neobert.utils import configure_tf32, model_summary, prepare_wandb_config

# Our metric object and model
from neobert.pretraining.metrics import Metrics, format_metrics

# Set up logger
logger = logging.getLogger(__name__)


def _resolve_masked_logits_only_loss(value: Any) -> bool:
    """Resolve and validate ``trainer.masked_logits_only_loss``.

    :param Any value: Config value to normalize.
    :raises ValueError: If value is not boolean-like.
    :return bool: Normalized boolean selector.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(
        "trainer.masked_logits_only_loss must be a bool or boolean-like string, "
        f"got {value!r} ({type(value).__name__})."
    )


def _move_batch_to_device(batch: BatchEncoding, device: torch.device) -> BatchEncoding:
    """Move batch tensors to device with async H2D copies when possible.

    :param BatchEncoding batch: Batch to move.
    :param torch.device device: Target device.
    :return BatchEncoding: Batch with tensors on device.
    """
    if hasattr(batch, "to") and not torch.is_tensor(batch):
        batch = dict(batch)
    # ``non_blocking`` overlaps copies when DataLoader uses pinned host memory.
    return send_to_device(batch, device, non_blocking=True)


def _ensure_pinned_cpu_batch(batch: BatchEncoding) -> BatchEncoding:
    """Pin CPU tensor values in a batch for async host->device transfer.

    ``torch.cat``/``torch.split`` on pinned tensors produce non-pinned outputs.
    Packed-batch stitching can therefore drop pinned memory guarantees unless we
    re-pin the final CPU tensors before ``send_to_device(..., non_blocking=True)``.

    :param BatchEncoding batch: Batch mapping of tensors/lists/scalars.
    :return BatchEncoding: Batch with CPU tensors pinned when needed.
    """

    def _pin_value(value: Any) -> tuple[Any, bool]:
        if torch.is_tensor(value):
            if value.device.type != "cpu" or value.is_pinned():
                return value, False
            return value.pin_memory(), True

        if isinstance(value, dict):
            updated: dict[Any, Any] = {}
            changed = False
            for key, inner in value.items():
                pinned_inner, inner_changed = _pin_value(inner)
                updated[key] = pinned_inner
                changed = changed or inner_changed
            if not changed:
                return value, False
            return updated, True

        if isinstance(value, list):
            updated_list: list[Any] = []
            changed = False
            for inner in value:
                pinned_inner, inner_changed = _pin_value(inner)
                updated_list.append(pinned_inner)
                changed = changed or inner_changed
            if not changed:
                return value, False
            return updated_list, True

        if isinstance(value, tuple):
            updated_items: list[Any] = []
            changed = False
            for inner in value:
                pinned_inner, inner_changed = _pin_value(inner)
                updated_items.append(pinned_inner)
                changed = changed or inner_changed
            if not changed:
                return value, False
            return tuple(updated_items), True

        return value, False

    pinned_batch, repinned = _pin_value(dict(batch))
    if not repinned:
        return batch
    return pinned_batch


def _promote_tmp_checkpoint_dir(tmp_path: Path, final_path: Path) -> None:
    """Promote ``tmp_path`` to ``final_path`` with crash-safe replacement.

    Keep the previous final checkpoint as ``*.old`` until the tmp->final rename
    succeeds. This avoids deleting the prior checkpoint before the new one is in
    place when running on shared filesystems.

    :param Path tmp_path: Newly written temporary checkpoint directory.
    :param Path final_path: Final checkpoint directory path.
    """
    backup_path = final_path.with_name(f"{final_path.name}.old")
    if backup_path.exists():
        shutil.rmtree(backup_path)

    if final_path.exists():
        final_path.replace(backup_path)

    try:
        tmp_path.replace(final_path)
    except Exception:
        if backup_path.exists() and not final_path.exists():
            backup_path.replace(final_path)
        raise

    if backup_path.exists():
        shutil.rmtree(backup_path)


def _write_deepspeed_latest_file(checkpoint_root: Path, tag: str) -> None:
    """Write DeepSpeed ``latest`` indirection for a checkpoint root.

    DeepSpeed's fp32 conversion helper resolves ``tag=None`` via this file, so
    we keep it updated after tmp->final tag promotion.

    :param Path checkpoint_root: DeepSpeed checkpoint root directory.
    :param str tag: Active checkpoint tag directory name.
    """
    latest_path = checkpoint_root / "latest"
    latest_path.write_text(f"{tag}\n", encoding="utf-8")


def _pad_tokenizer_to_multiple(
    tokenizer: PreTrainedTokenizerBase,
    *,
    multiple: int = 128,
) -> tuple[int, int, int]:
    """Pad tokenizer length to ``multiple`` by adding inert extra tokens.

    The model uses rounded embedding sizes for tensor-core efficiency. Adding
    explicit placeholder tokens keeps tokenizer/model vocab contracts aligned.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer to mutate.
    :param int multiple: Target rounding multiple.
    :return tuple[int, int, int]: ``(original_size, padded_size, added_count)``.
    """
    original_size = len(tokenizer)
    padded_size = round_up_to_multiple(original_size, multiple)
    if padded_size == original_size:
        return original_size, padded_size, 0

    needed = padded_size - original_size
    extra_tokens = [
        f"<|neobert_extra_token_{idx}|>"
        for idx in range(original_size, original_size + needed)
    ]
    added = tokenizer.add_tokens(extra_tokens, special_tokens=False)
    final_size = len(tokenizer)
    if added != needed or final_size != padded_size:
        raise RuntimeError(
            "Failed to pad tokenizer vocabulary to requested multiple: "
            f"needed={needed}, added={added}, final_size={final_size}, "
            f"target={padded_size}."
        )
    return original_size, final_size, added


def _sync_tokenizer_derived_config(
    cfg: Config,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, int, int]:
    """Synchronize tokenizer-derived model config fields.

    :param Config cfg: Runtime configuration to mutate.
    :param PreTrainedTokenizerBase tokenizer: Active tokenizer instance.
    :return tuple[int, int, int]: ``(original_vocab_size, resolved_vocab_size, added)``.
    """
    original_vocab_size, resolved_vocab_size, added_tokens = _pad_tokenizer_to_multiple(
        tokenizer,
        multiple=128,
    )

    cfg.model.vocab_size = resolved_vocab_size
    cfg.tokenizer.vocab_size = resolved_vocab_size
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for pretraining.")
    cfg.model.pad_token_id = int(tokenizer.pad_token_id)

    return original_vocab_size, resolved_vocab_size, added_tokens


def _parse_split_slice(split: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse a split string with optional slice notation.

    Examples
    --------
    - "train[:1%]" -> ("train", "", "1%")
    - "train[99%:]" -> ("train", "99%", "")
    - "train[:1000]" -> ("train", "", "1000")

    :param str split: Split string to parse.
    :return tuple[str, str | None, str | None]: Base split and slice bounds.
    """
    match = re.match(r"^([^\[]+)\[([^\]]+)\]$", split)
    if not match:
        return split, None, None
    base = match.group(1)
    inner = match.group(2)
    if ":" not in inner:
        return split, None, None
    start_str, end_str = inner.split(":", 1)
    start = start_str.strip() or None
    end = end_str.strip() or None
    return base, start, end


def _resolve_slice_index(value: Optional[str], total: Optional[int]) -> Optional[int]:
    """Resolve a slice value (int or percent) to an integer index.

    :param str | None value: Slice token (e.g., "1000" or "1%").
    :param int | None total: Total number of examples (required for %).
    :return int | None: Resolved index or None if not applicable.
    """
    if value is None:
        return None
    if value.endswith("%"):
        if total is None:
            return None
        pct = float(value[:-1])
        return int(total * pct / 100.0)
    return int(value)


def _load_streaming_split(
    dataset_name: str,
    split: str,
    dataset_kwargs: dict[str, object],
) -> Dataset:
    """Load a streaming dataset split with optional slice notation.

    Streaming datasets do not support slice notation in ``load_dataset`` directly,
    so we emulate it using ``skip``/``take`` when possible.

    :param str dataset_name: Dataset identifier.
    :param str split: Split string (e.g., "train[:1%]").
    :param dict[str, object] dataset_kwargs: Additional kwargs for HF datasets.
    :return Dataset: Streaming dataset (IterableDataset).
    """
    base, start_token, end_token = _parse_split_slice(split)
    dataset = load_dataset(
        dataset_name,
        split=base,
        streaming=True,
        **dataset_kwargs,
    )

    if start_token is None and end_token is None:
        return dataset

    needs_total = (start_token is not None and start_token.endswith("%")) or (
        end_token is not None and end_token.endswith("%")
    )
    total_examples: Optional[int] = None
    if needs_total:
        try:
            builder = load_dataset_builder(dataset_name, **dataset_kwargs)
            if base in builder.info.splits:
                total_examples = builder.info.splits[base].num_examples
        except Exception as exc:
            logger.warning(
                "Unable to resolve streaming split size for %s (%s): %s",
                dataset_name,
                split,
                exc,
            )

    if needs_total and total_examples is None:
        raise ValueError(
            f"Streaming split '{split}' uses percent slicing but total size is "
            f"unknown. Use absolute indices (e.g., '{base}[:500]') instead of "
            f"percentages for streaming datasets."
        )

    start_idx = _resolve_slice_index(start_token, total_examples)
    end_idx = _resolve_slice_index(end_token, total_examples)

    if start_idx and start_idx > 0:
        dataset = dataset.skip(start_idx)

    if end_idx is not None:
        take_n = end_idx - (start_idx or 0)
        if take_n <= 0:
            logger.warning("Resolved split %s is empty after slicing.", split)
            return dataset.take(0)
        dataset = dataset.take(take_n)

    return dataset


def _count_masked_correct(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Count correct predictions while ignoring masked labels.

    :param torch.Tensor logits: Logits of shape ``[batch, seq_len, vocab]``.
    :param torch.Tensor labels: Label IDs of shape ``[batch, seq_len]``.
    :param int ignore_index: Label value to ignore (default: -100).
    :return torch.Tensor: Scalar tensor of correct predictions on unmasked tokens.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    # Avoid Python branching on CUDA scalar tensors (implicit sync via .item()).
    return (preds.eq(labels) & mask).sum(dtype=torch.long)


def _set_default_worker_env(num_workers: int) -> None:
    """Set conservative host-thread env defaults for data workers.

    Existing user-provided values take precedence.

    :param int num_workers: DataLoader worker count.
    """
    if num_workers <= 0:
        return
    defaults = {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            os.environ[key] = value
            applied[key] = value
    if applied:
        logger.info(
            "Set default worker env: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
        )


def _resolve_eval_max_batches(
    max_batches: Optional[int], num_processes: int
) -> Optional[int]:
    """Resolve per-rank eval batch cap from a global target.

    :param int | None max_batches: Global eval batch target (all ranks combined).
    :param int num_processes: Number of distributed processes.
    :return int | None: Per-rank eval batch cap.
    """
    if max_batches is None or max_batches <= 0:
        return None
    if num_processes <= 1:
        return max_batches
    return max(1, (max_batches + num_processes - 1) // num_processes)


def _run_eval(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    loss_fn: Optional[torch.nn.Module],
    accelerator: Accelerator,
    model_config: NeoBERTConfig,
    max_batches: Optional[int] = None,
    manual_device_move: bool = True,
    masked_objective: Optional[MaskedPositionsOnlyMLMObjective] = None,
) -> dict[str, float]:
    """Run a lightweight evaluation loop for masked LM perplexity.

    :param torch.nn.Module model: Model to evaluate.
    :param torch.utils.data.DataLoader eval_dataloader: Evaluation dataloader.
    :param torch.nn.Module | None loss_fn: Loss function (sum reduction) for legacy path.
    :param Accelerator accelerator: Accelerator for distributed reductions.
    :param NeoBERTConfig model_config: Model config with vocab size.
    :param int | None max_batches: Optional cap on eval batches.
    :param bool manual_device_move: Whether to manually move each batch to device.
    :param MaskedPositionsOnlyMLMObjective | None masked_objective:
        Optional masked-only objective path.
    :return dict[str, float]: Evaluation metrics for logging.
    """
    was_training = model.training
    model.eval()

    eval_loss_sum = torch.zeros((), device=accelerator.device)
    eval_num_pred = torch.zeros((), device=accelerator.device)
    eval_num_correct = torch.zeros((), device=accelerator.device)
    eval_batches = 0
    max_batches_per_rank = _resolve_eval_max_batches(
        max_batches, accelerator.num_processes
    )

    try:
        with torch.no_grad():
            for batch in eval_dataloader:
                if (
                    max_batches_per_rank is not None
                    and eval_batches >= max_batches_per_rank
                ):
                    break
                if manual_device_move:
                    batch = _move_batch_to_device(batch, accelerator.device)
                packed_seqlens = _packed_seqlens_to_tensor(batch.get("packed_seqlens"))
                pad_mask = (
                    None
                    if packed_seqlens is not None
                    else batch.get("attention_mask", None)
                )
                if masked_objective is not None:
                    hidden = model(
                        src=batch["input_ids"],
                        pad_mask=pad_mask,
                        packed_seqlens=packed_seqlens,
                        return_logits=False,
                    )["hidden_representation"]
                    objective_out = masked_objective(
                        hidden_states=hidden,
                        labels=batch["labels"],
                        lm_weight=model.decoder.weight,
                        compute_accuracy=True,
                    )
                    eval_loss_sum += objective_out.loss_sum_local
                    eval_num_pred += objective_out.num_masked_local
                    if objective_out.num_correct_local is not None:
                        eval_num_correct += objective_out.num_correct_local
                else:
                    if loss_fn is None:
                        raise ValueError(
                            "loss_fn is required when masked_objective is not provided."
                        )
                    logits = model(
                        src=batch["input_ids"],
                        pad_mask=pad_mask,
                        packed_seqlens=packed_seqlens,
                    )["logits"]
                    loss_sum = loss_fn(
                        logits.view(-1, model_config.vocab_size),
                        batch["labels"].view(-1),
                    )
                    num_pred = (batch["labels"] != -100).sum()
                    eval_loss_sum += loss_sum
                    eval_num_pred += num_pred
                    eval_num_correct += _count_masked_correct(logits, batch["labels"])
                eval_batches += 1

        total_loss = accelerator.reduce(eval_loss_sum, reduction="sum")
        total_pred = accelerator.reduce(eval_num_pred, reduction="sum")
        total_correct = accelerator.reduce(eval_num_correct, reduction="sum")
        # Log per-rank batch count to avoid confusion about summed totals.
        metrics: dict[str, float] = {
            "eval/batches": float(eval_batches),
        }
        if total_pred.item() > 0:
            eval_loss = (total_loss / total_pred).item()
            metrics["eval/loss"] = eval_loss
            metrics["eval/perplexity"] = math.exp(eval_loss)
            metrics["eval/accuracy"] = (total_correct / total_pred).item()

        return metrics
    finally:
        if was_training:
            model.train()


def _packed_seqlens_to_tensor(
    packed_seqlens: Any,
) -> Optional[torch.Tensor]:
    """Normalize packed sequence lengths to rank-2 int32 tensors.

    :param Any packed_seqlens: Packed segment lengths tensor or list.
    :return torch.Tensor | None: Packed segment lengths.
    """
    if packed_seqlens is None:
        return None
    if torch.is_tensor(packed_seqlens):
        tensor = packed_seqlens.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 2:
            raise ValueError(
                "packed_seqlens tensor must be rank 1 or 2, got "
                f"shape={tuple(tensor.shape)}"
            )
        return tensor.to(torch.int32)

    normalized_rows: list[list[int]] = []
    max_segments = 0
    for row in packed_seqlens:
        if row is None:
            segs: list[int] = []
        else:
            segs = [int(x) for x in row if int(x) > 0]
        normalized_rows.append(segs)
        max_segments = max(max_segments, len(segs))

    tensor = torch.zeros((len(normalized_rows), max_segments), dtype=torch.int32)
    for idx, segs in enumerate(normalized_rows):
        if segs:
            tensor[idx, : len(segs)] = torch.tensor(segs, dtype=torch.int32)
    return tensor


def _resolve_loader_perf_settings(
    cfg: Config,
    *,
    device: torch.device,
) -> tuple[bool, bool, Optional[int], list[str]]:
    """Resolve effective dataloader performance settings.

    Applies conservative CUDA-friendly defaults when users leave knobs unset:
    - ``pin_memory=True`` on CUDA
    - ``prefetch_factor=4`` when workers are enabled and no value is provided

    :param Config cfg: Training config.
    :param torch.device device: Active accelerator device.
    :return tuple[bool, bool, int | None, list[str]]: Effective
        ``(pin_memory, persistent_workers, prefetch_factor, notes)``.
    """
    num_workers = max(0, int(cfg.dataset.num_workers))
    pin_memory = bool(cfg.dataset.pin_memory)
    persistent_workers = bool(cfg.dataset.persistent_workers and num_workers > 0)
    prefetch_factor = cfg.dataset.prefetch_factor
    if num_workers <= 0:
        prefetch_factor = None
    elif prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)
        if prefetch_factor <= 0:
            raise ValueError(
                f"dataset.prefetch_factor must be > 0 when set, got {prefetch_factor}."
            )

    notes: list[str] = []
    if device.type == "cuda":
        if not pin_memory:
            pin_memory = True
            notes.append(
                "dataset.pin_memory was false; enabling pin_memory=True on CUDA "
                "to improve host->device transfer overlap."
            )
        if num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 4
            notes.append(
                "dataset.prefetch_factor was unset; using prefetch_factor=4 for CUDA throughput."
            )

    return pin_memory, persistent_workers, prefetch_factor, notes


def _scale_gradients(model: torch.nn.Module, scale: torch.Tensor) -> None:
    """Scale gradients in-place using a dtype-safe scalar.

    :param torch.nn.Module model: Model whose gradients should be scaled.
    :param torch.Tensor scale: Scale factor (scalar tensor).
    """
    grads_by_key: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        grads_by_key.setdefault((grad.device, grad.dtype), []).append(grad)

    for (device, dtype), grads in grads_by_key.items():
        scale_value = scale.to(device=device, dtype=dtype)
        try:
            torch._foreach_mul_(grads, scale_value)
        except AttributeError:
            # Older PyTorch fallback; dtype/device mismatches should surface as errors.
            for grad in grads:
                grad.mul_(scale_value)


def _gradient_token_scale(
    tokens_global: torch.Tensor,
    *,
    num_processes: int,
    grad_accumulation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute safe post-accumulation gradient scale.

    For normal updates, this matches standard token-mean scaling exactly:
    ``scale = (num_processes * grad_accumulation_steps) / tokens_global``.
    We only clamp the denominator to a small floor (one masked token per rank
    per accumulation step) to avoid pathological amplification on near-empty
    masked batches.

    :param torch.Tensor tokens_global: Global masked-token count for the update.
    :param int num_processes: Number of distributed processes.
    :param int grad_accumulation_steps: Gradient accumulation steps.
    :return tuple[torch.Tensor, torch.Tensor]: ``(scale, clamped)``.
    """
    token_floor = max(1, int(num_processes) * int(grad_accumulation_steps))
    token_floor_f = float(token_floor)
    clamped_tokens = torch.clamp(tokens_global.float(), min=token_floor_f)
    scale = (token_floor_f / clamped_tokens).to(tokens_global.device)
    clamped = tokens_global < token_floor
    return scale, clamped


def _clear_stored_batch(stored_batch: BatchEncoding) -> None:
    """Drop buffered batch fragments in-place.

    :param BatchEncoding stored_batch: Batch fragment buffer.
    """
    for key in list(stored_batch.keys()):
        stored_batch[key] = None


def _append_to_stored_batch(
    stored_batch: BatchEncoding, batch: BatchEncoding
) -> BatchEncoding:
    """Append a batch fragment into the stored buffer.

    :param BatchEncoding stored_batch: Fragment buffer.
    :param BatchEncoding batch: Batch fragment to append.
    :return BatchEncoding: Updated fragment buffer.
    """
    for key, value in batch.items():
        stored_batch.setdefault(key, None)
        if value is None:
            continue
        existing = stored_batch.get(key)
        if existing is None:
            stored_batch[key] = value
            continue
        if torch.is_tensor(existing) and torch.is_tensor(value):
            if existing.device != value.device:
                value = value.to(existing.device)
            stored_batch[key] = torch.cat([existing, value], dim=0)
        elif isinstance(existing, list) and isinstance(value, list):
            stored_batch[key] = existing + value
        else:
            raise TypeError(
                "Stored batch key '%s' has incompatible types: %s vs %s"
                % (key, type(existing).__name__, type(value).__name__)
            )
    return stored_batch


def _resolve_pack_token_limits(
    tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Tuple[int, Optional[int], Optional[int]]:
    """Compute tokenization limits for packed sequences.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer supplying special tokens.
    :param int max_length: Target packed sequence length.
    :return tuple[int, int | None, int | None]: Trimmed max_length and boundary IDs.
    """
    start_token_id = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    end_token_id = (
        tokenizer.sep_token_id
        if tokenizer.sep_token_id is not None
        else tokenizer.eos_token_id
    )
    reserve = int(start_token_id is not None) + int(end_token_id is not None)
    return max(1, max_length - reserve), start_token_id, end_token_id


def _resolve_tokenize_num_proc(
    requested: Optional[int],
    num_processes: int,
    is_main_process: bool,
) -> int:
    """Resolve per-rank num_proc for dataset tokenization.

    :param int | None requested: Requested num_proc from config (or None).
    :param int num_processes: Number of distributed processes.
    :param bool is_main_process: Whether the caller is the main process.
    :return int: Effective num_proc for this rank.
    """
    if requested is None or requested <= 0:
        if hasattr(os, "sched_getaffinity"):
            try:
                requested = len(os.sched_getaffinity(0))
            except Exception:
                requested = os.cpu_count() or 1
        else:
            requested = os.cpu_count() or 1
    if num_processes > 1:
        requested = max(1, requested // num_processes)
        if not is_main_process:
            requested = 1
    return max(1, requested)


def _select_train_split(
    dataset: Dataset | DatasetDict, train_split: Optional[str]
) -> Dataset:
    """Select a train split from a DatasetDict when needed.

    :param Dataset | DatasetDict dataset: Loaded dataset object.
    :param str | None train_split: Optional split name to select.
    :return Dataset: Resolved training split dataset.
    :raises ValueError: If no suitable split can be resolved.
    """
    if not isinstance(dataset, DatasetDict):
        return dataset

    if train_split:
        if train_split in dataset:
            return dataset[train_split]
        raise ValueError(
            f"train_split='{train_split}' not found in dataset splits: {list(dataset)}"
        )

    if "train" in dataset:
        return dataset["train"]

    raise ValueError(
        "DatasetDict loaded from disk is missing a train split. "
        f"Available splits: {list(dataset)}. Set dataset.train_split in the config."
    )


def _has_stored_batch(stored_batch: BatchEncoding) -> bool:
    """Return whether buffered batch fragments are present.

    :param BatchEncoding stored_batch: Buffered batch fragments.
    :return bool: True when any buffered tensor/value is present.
    """
    for value in stored_batch.values():
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.ndim == 0:
                if value.numel() > 0:
                    return True
            elif value.shape[0] > 0:
                return True
            continue
        if isinstance(value, (list, tuple, dict, str, bytes)):
            if len(value) > 0:
                return True
            continue
        return True
    return False


def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
) -> tuple[BatchEncoding, BatchEncoding]:
    """Adjust batch to a target size by splitting/concatenating.

    :param BatchEncoding batch: Current batch to adjust.
    :param BatchEncoding stored_batch: Buffered batch fragments.
    :param int target_size: Target batch size.
    :return tuple[BatchEncoding, BatchEncoding]: Adjusted batch and buffer.
    """
    tmp = {}
    for key in batch.keys():
        stored_batch.setdefault(key, None)
    buffer_device = None
    for value in stored_batch.values():
        if torch.is_tensor(value):
            buffer_device = value.device
            break
    if buffer_device is None:
        buffer_device = torch.device("cpu")
    if _has_stored_batch(stored_batch):
        merged: BatchEncoding = {}
        for key, value in batch.items():
            stored_value = stored_batch.get(key)
            if stored_value is None:
                merged[key] = value
                continue
            if value is None:
                merged[key] = stored_value
                continue
            if torch.is_tensor(value):
                if (
                    torch.is_tensor(stored_value)
                    and stored_value.device != value.device
                ):
                    stored_value = stored_value.to(value.device)
                merged[key] = torch.cat([stored_value, value], dim=0)
            else:
                merged[key] = (
                    stored_value + value
                )  # list concatenation (non-tensor path)
        for key in merged.keys():
            stored_batch[key] = None
        batch = merged
    batch_size = batch["input_ids"].shape[0]

    # If the batch is too large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            value = batch[key]
            if value is None:
                if key not in stored_batch:
                    stored_batch[key] = None
                continue
            if torch.is_tensor(value):
                tmp[key] = torch.split(
                    value, [target_size, batch_size - target_size], dim=0
                )
                batch[key] = tmp[key][0]
                if stored_batch[key] is None:
                    leftover = tmp[key][1]
                    if torch.is_tensor(leftover) and leftover.device != buffer_device:
                        leftover = leftover.to(buffer_device)
                    stored_batch[key] = leftover
                else:
                    # Keep stored batches on a single device (often CPU) to avoid device mismatches.
                    if stored_batch[key].device != tmp[key][1].device:
                        leftover = tmp[key][1].to(stored_batch[key].device)
                    else:
                        leftover = tmp[key][1]
                    stored_batch[key] = torch.cat([stored_batch[key], leftover], dim=0)
            else:
                batch[key] = value[:target_size]
                leftover = value[target_size:]
                if stored_batch[key] is None:
                    stored_batch[key] = leftover
                else:
                    stored_batch[key] = (
                        stored_batch[key] + leftover
                    )  # list concatenation (non-tensor path)

    # If the batch is too small, we had some stored_batch
    elif batch_size < target_size:
        if stored_batch.get("input_ids") is None:
            return batch, stored_batch
        # We have already enough samples stored
        if stored_batch["input_ids"].shape[0] >= target_size - batch_size:
            for key in batch.keys():
                if stored_batch[key] is None:
                    continue
                if batch[key] is None:
                    continue
                if (
                    torch.is_tensor(stored_batch[key])
                    and stored_batch[key].device != batch[key].device
                ):
                    stored_batch[key] = stored_batch[key].to(batch[key].device)
            for key in batch.keys():
                if stored_batch[key] is None:
                    continue
                if batch[key] is None:
                    continue
                if torch.is_tensor(stored_batch[key]):
                    tmp[key] = torch.split(
                        stored_batch[key],
                        [
                            target_size - batch_size,
                            stored_batch[key].shape[0] - (target_size - batch_size),
                        ],
                        dim=0,
                    )
                    batch[key] = torch.cat([batch[key], tmp[key][0]], dim=0)
                    stored_batch[key] = tmp[key][1]
                    # Save on CPU to prevent full GPU memory; this trades extra H2D copies
                    # for lower peak VRAM during uneven batch packing.
                    # Use blocking transfer so buffered batches are ready when reused.
                    stored_batch[key] = stored_batch[key].to("cpu")
                else:
                    take = target_size - batch_size
                    batch[key] = (
                        batch[key] + stored_batch[key][:take]
                    )  # list concatenation (non-tensor path)
                    stored_batch[key] = stored_batch[key][take:]

        # Concatenate otherwise
        else:
            for key in batch.keys():
                if stored_batch[key] is None:
                    continue
                if batch[key] is None:
                    continue
                if torch.is_tensor(stored_batch[key]):
                    if stored_batch[key].device != batch[key].device:
                        stored_batch[key] = stored_batch[key].to(batch[key].device)
                    batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                else:
                    batch[key] = (
                        batch[key] + stored_batch[key]
                    )  # list concatenation (non-tensor path)
                stored_batch[key] = None

    return batch, stored_batch


def _maybe_shuffle_streaming_dataset(
    dataset: Dataset,
    buffer_size: int,
    seed: int,
    print_fn: Callable[[str], None] | None = None,
) -> Dataset:
    """Shuffle a streaming dataset if a positive buffer size is configured.

    :param Dataset dataset: Dataset to shuffle.
    :param int buffer_size: Shuffle buffer size.
    :param int seed: Random seed for deterministic shuffling.
    :param callable | None print_fn: Optional logging callback.
    :return Dataset: Shuffled dataset (or the original dataset if no shuffle is applied).
    """
    if buffer_size <= 0 or not hasattr(dataset, "shuffle"):
        return dataset

    shuffled = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    if print_fn is not None:
        print_fn(f"Added shuffle buffer with size {buffer_size}")
    return shuffled


def _prepare_resume_dataloader(
    train_dataloader: torch.utils.data.DataLoader,
    metrics: Metrics,
    accelerator: Accelerator,
    is_streaming: bool,
) -> torch.utils.data.DataLoader | None:
    """Prepare a skipped dataloader for resume when possible.

    :param torch.utils.data.DataLoader train_dataloader: Training dataloader.
    :param Metrics metrics: Metrics tracker with resumed counters.
    :param Accelerator accelerator: Accelerator instance.
    :param bool is_streaming: Whether the dataset is streaming.
    :return torch.utils.data.DataLoader | None: Skipped dataloader or ``None``.
    """
    if is_streaming:
        raise ValueError(
            "Cannot resume training with streaming datasets - data position is not "
            "preserved. For resumable long runs, pre-tokenize your dataset:\n"
            "  python scripts/pretraining/tokenize_dataset.py --dataset <name> --output <path>"
        )

    if not hasattr(train_dataloader, "__len__"):
        logger.warning(
            "Resume requested but dataloader has no length; "
            "starting from the current epoch boundary."
        )
        return None

    if hasattr(train_dataloader, "set_epoch"):
        train_dataloader.set_epoch(metrics["train/epochs"])

    resume_step = metrics["train/batches"] % len(train_dataloader)
    if resume_step == 0:
        return None

    return accelerator.skip_first_batches(train_dataloader, resume_step)


def trainer(cfg: Config) -> None:
    """Run the pretraining loop.

    :param Config cfg: Training configuration.
    """
    masked_logits_only_loss = _resolve_masked_logits_only_loss(
        getattr(cfg.trainer, "masked_logits_only_loss", True)
    )

    # Get the last checkpoint id
    output_dir = Path(cfg.trainer.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    model_checkpoint_dir = output_dir / "model_checkpoints"
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path, iteration = _resolve_resume_checkpoint(
        cfg.trainer.resume_from_checkpoint,
        str(checkpoint_dir),
        str(output_dir),
    )

    # Accelerator object - disable automatic checkpointing to avoid duplicate checkpoints/ directory
    project_config = ProjectConfiguration(
        str(output_dir),
        automatic_checkpoint_naming=False,  # We handle checkpointing manually in model_checkpoints/
        iteration=iteration,
    )
    # All parameters participate in the forward graph; keep DDP in fast-path mode.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    wandb_enabled = cfg.wandb.enabled and cfg.wandb.mode != "disabled"
    cfg.model.attn_backend = resolve_runtime_attn_backend(
        cfg.model.attn_backend,
        fallback_to_sdpa=True,
    )
    # Keep manual placement for packed mode only; in non-packed mode we use
    # Accelerate's device placement for better overlap.
    disable_dispatch = bool(cfg.datacollator.pack_sequences)
    dataloader_config = None
    if disable_dispatch:
        dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
        logger.info("Disabling Accelerate dispatch_batches for packed-sequence mode.")

    mixed_precision = cfg.trainer.mixed_precision
    if isinstance(mixed_precision, bool):
        mixed_precision = "bf16" if mixed_precision else "no"
    else:
        mixed_precision = str(mixed_precision).strip().lower()
    if mixed_precision == "fp32":
        mixed_precision = "no"
    if mixed_precision == "fp16":
        raise ValueError(
            "trainer.mixed_precision='fp16' is not supported for pretraining. "
            "Use 'bf16' or 'no'/'fp32'."
        )

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if wandb_enabled else None,
        project_config=project_config,
        kwargs_handlers=[kwargs],
        dataloader_config=dataloader_config,
    )
    if accelerator.distributed_type is DistributedType.MEGATRON_LM:
        raise RuntimeError(
            "Megatron-LM backend is not supported by this training loop. "
            "Use DDP or DeepSpeed instead."
        )
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        is_muon = cfg.optimizer.name.lower() in {"muonclip", "muon-clip", "muon_clip"}
        if is_muon:
            deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
            zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
            if zero_stage is None:
                logger.warning(
                    "MuonClip optimizer enabled with DeepSpeed, but zero stage is unknown. "
                    "Ensure ZeRO stage < 2 to avoid incorrect sharded updates."
                )
            elif zero_stage >= 2:
                raise RuntimeError(
                    "MuonClip is not compatible with DeepSpeed ZeRO stage >= 2 "
                    "(sharded grads/params). Use ZeRO stage 0/1 or disable MuonClip."
                )

    # Initialise the wandb run and pass wandb parameters
    if wandb_enabled:
        Path(cfg.wandb.dir).mkdir(parents=True, exist_ok=True)
        config_dict = prepare_wandb_config(cfg)
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            init_kwargs={
                "wandb": {
                    "name": cfg.wandb.name,
                    "entity": cfg.wandb.entity,
                    "config": config_dict,
                    "tags": cfg.wandb.tags,
                    "dir": cfg.wandb.dir,
                    "mode": cfg.wandb.mode,
                    "resume": cfg.wandb.resume,
                }
            },
        )
        if accelerator.is_main_process and wandb.run is not None:
            wandb.run.config.update(config_dict, allow_val_change=True)
            config_path = getattr(cfg, "config_path", None)
            if config_path:
                abs_config_path = Path(config_path).expanduser().resolve()
                if abs_config_path.is_file():
                    artifact = wandb.Artifact(
                        name=f"{wandb.run.id}-config",
                        type="config",
                        metadata={"source": str(abs_config_path)},
                    )
                    artifact.add_file(str(abs_config_path))
                    wandb.run.log_artifact(artifact)
                else:
                    logger.warning(
                        "Configured config_path '%s' not found; skipping wandb artifact upload",
                        config_path,
                    )

    # Set the seed
    set_seed(cfg.seed)
    _set_default_worker_env(int(cfg.dataset.num_workers))

    # Configure TF32 precision for GPUs with compute capability >= 8.0
    configure_tf32(enabled=cfg.trainer.tf32, print_fn=accelerator.print)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)
    log_interval = max(1, cfg.trainer.logging_steps)
    enforce_full_packed_batches = bool(
        getattr(cfg.trainer, "enforce_full_packed_batches", True)
    )
    log_train_accuracy = bool(getattr(cfg.trainer, "log_train_accuracy", False))
    log_grad_norm = bool(getattr(cfg.trainer, "log_grad_norm", False))
    metrics["train/compute_accuracy"] = int(log_train_accuracy)

    is_streaming = cfg.dataset.streaming
    if cfg.trainer.resume_from_checkpoint and is_streaming:
        raise ValueError(
            "Cannot resume training with streaming datasets - data position is not "
            "preserved. For resumable long runs, pre-tokenize your dataset:\n"
            "  python scripts/pretraining/tokenize_dataset.py --dataset <name> --output <path>"
        )

    if cfg.datacollator.pack_sequences:
        logger.info(
            "Using packed sequences (experimental). "
            "Recommended: attn_backend=flash_attn_varlen with flash-attn installed."
        )
        if cfg.model.attn_backend == "sdpa":
            logger.warning(
                "pack_sequences is enabled with attn_backend=sdpa; "
                "per-segment SDPA fallback will be used (slow on GPU). "
                "Set attn_backend=flash_attn_varlen and install flash-attn for production."
            )
    if not cfg.datacollator.mask_all:
        # Keep BERT-style 80/10/10 masking as a supported mode, but make the
        # methodological difference explicit for NeoBERT-style pretraining runs.
        logger.warning(
            "datacollator.mask_all=false uses standard 80/10/10 MLM corruption. "
            "Set datacollator.mask_all=true to use NeoBERT's 100%% [MASK] strategy."
        )

    # Tokenizer
    with accelerator.main_process_first():
        tokenizer = get_tokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.path or cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
        )

    prior_model_vocab_size = int(cfg.model.vocab_size)
    prior_tokenizer_vocab_size = int(cfg.tokenizer.vocab_size)
    original_vocab_size, resolved_vocab_size, added_tokens = (
        _sync_tokenizer_derived_config(
            cfg,
            tokenizer,
        )
    )
    if accelerator.is_main_process and (
        prior_model_vocab_size != resolved_vocab_size
        or prior_tokenizer_vocab_size != resolved_vocab_size
    ):
        logger.warning(
            "Config vocab_size updated: tokenizer len=%s -> %s (was model=%s).",
            original_vocab_size,
            resolved_vocab_size,
            prior_model_vocab_size,
        )
    if accelerator.is_main_process and added_tokens > 0:
        logger.info(
            "Added %s inert tokenizer tokens to align tokenizer/model vocab_size=%s.",
            added_tokens,
            resolved_vocab_size,
        )

    # Tokenization strategy for packed sequences: strip special tokens and reinsert
    # boundaries in the collator to avoid duplicate BOS/EOS/SEP tokens.
    pack_sequences = cfg.datacollator.pack_sequences
    add_special_tokens = not pack_sequences
    return_special_tokens_mask = True
    tokenize_max_length = cfg.dataset.max_seq_length
    if pack_sequences:
        pack_target_length = cfg.datacollator.max_length or cfg.dataset.max_seq_length
        tokenize_max_length, _, _ = _resolve_pack_token_limits(
            tokenizer, pack_target_length
        )

    # Dataset
    dataset_kwargs: dict[str, object] = {}
    if cfg.dataset.config:
        dataset_kwargs["name"] = cfg.dataset.config
    if cfg.dataset.cache_dir:
        dataset_kwargs["cache_dir"] = cfg.dataset.cache_dir
    if getattr(cfg.dataset, "trust_remote_code", False):
        dataset_kwargs["trust_remote_code"] = True

    train_dataset = None
    if cfg.dataset.path:
        dataset_path = Path(cfg.dataset.path)
        if dataset_path.exists():
            train_dataset = load_from_disk(dataset_path)
            train_dataset = _select_train_split(train_dataset, cfg.dataset.train_split)
        else:
            logger.warning(
                "Dataset path %s not found; falling back to load_dataset().",
                dataset_path,
            )

    if train_dataset is None:
        if cfg.dataset.train_split:
            if cfg.dataset.streaming:
                train_dataset = _load_streaming_split(
                    cfg.dataset.name,
                    cfg.dataset.train_split,
                    dataset_kwargs,
                )
            else:
                train_dataset = load_dataset(
                    cfg.dataset.name,
                    split=cfg.dataset.train_split,
                    streaming=False,
                    **dataset_kwargs,
                )
        else:
            dataset = load_dataset(
                cfg.dataset.name,
                streaming=cfg.dataset.streaming,
                **dataset_kwargs,
            )
            train_dataset = dataset["train"]

    # Check if dataset needs tokenization
    # For streaming datasets, we need to check differently
    needs_tokenization = False

    if train_dataset is not None:
        if is_streaming:
            # For streaming datasets, peek at the first example
            first_example = next(iter(train_dataset))
            needs_tokenization = "input_ids" not in first_example
        else:
            needs_tokenization = "input_ids" not in train_dataset.column_names

    if needs_tokenization:
        accelerator.print("Dataset is not tokenized. Tokenizing now...")
        from neobert.tokenizer import tokenize

        # For non-streaming datasets, check if pre-tokenization is requested
        if not is_streaming and cfg.dataset.pre_tokenize:
            # Create output directory
            if cfg.dataset.pre_tokenize_output:
                output_dir = cfg.dataset.pre_tokenize_output
            else:
                output_dir = f"tokenized_data/{cfg.dataset.name.replace('/', '_')}"

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            success_flag = Path(output_dir) / ".tokenize_complete"
            failure_flag = Path(output_dir) / ".tokenize_failed"

            accelerator.print(f"Pre-tokenizing dataset to: {output_dir}")

            if accelerator.is_main_process and not success_flag.exists():
                if failure_flag.exists():
                    failure_flag.unlink()
                try:
                    text_column = resolve_text_column(
                        train_dataset,
                        is_streaming=False,
                        preferred=cfg.dataset.text_column,
                    )
                    tokenized_dataset = tokenize(
                        train_dataset,
                        tokenizer,
                        column_name=text_column,
                        max_length=tokenize_max_length,
                        remove_columns=True,
                        truncation=True,
                        num_proc=cfg.dataset.num_proc,
                        add_special_tokens=add_special_tokens,
                        return_special_tokens_mask=return_special_tokens_mask,
                    )
                    tokenized_dataset.save_to_disk(output_dir)
                except Exception as exc:
                    failure_flag.write_text(str(exc))
                else:
                    success_flag.write_text("ok")

            accelerator.wait_for_everyone()
            if not success_flag.exists():
                err_msg = (
                    failure_flag.read_text()
                    if failure_flag.exists()
                    else "Pre-tokenization failed; see main process logs."
                )
                raise RuntimeError(f"Tokenization failed: {err_msg}")

            accelerator.print(f"Pre-tokenization complete. Loading from: {output_dir}")
            # Load the pre-tokenized dataset
            train_dataset = load_from_disk(output_dir)
            train_dataset = _select_train_split(train_dataset, cfg.dataset.train_split)
        else:
            # Determine text column
            text_column = resolve_text_column(
                train_dataset,
                is_streaming,
                preferred=cfg.dataset.text_column,
            )
            tokenize_num_proc = (
                None
                if is_streaming
                else _resolve_tokenize_num_proc(
                    cfg.dataset.num_proc,
                    accelerator.num_processes,
                    accelerator.is_main_process,
                )
            )

            # Tokenize dataset
            with accelerator.main_process_first():
                train_dataset = tokenize(
                    train_dataset,
                    tokenizer,
                    column_name=text_column,
                    max_length=tokenize_max_length,
                    remove_columns=True,
                    truncation=True,
                    num_proc=tokenize_num_proc,
                    add_special_tokens=add_special_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                )
        if cfg.dataset.streaming:
            accelerator.print("Tokenization setup complete for streaming dataset.")
        else:
            accelerator.print(
                f"Tokenization complete. Dataset size: {len(train_dataset)}"
            )

    eval_dataset = None
    if cfg.dataset.eval_split:
        if cfg.dataset.path and Path(cfg.dataset.path).exists():
            eval_source = load_from_disk(cfg.dataset.path)
            if isinstance(eval_source, DatasetDict):
                eval_dataset = eval_source[cfg.dataset.eval_split]
            else:
                logger.warning(
                    "eval_split=%s requested but dataset path is not a DatasetDict; "
                    "skipping evaluation.",
                    cfg.dataset.eval_split,
                )
        else:
            if cfg.dataset.streaming:
                eval_dataset = _load_streaming_split(
                    cfg.dataset.name,
                    cfg.dataset.eval_split,
                    dataset_kwargs,
                )
            else:
                eval_dataset = load_dataset(
                    cfg.dataset.name,
                    split=cfg.dataset.eval_split,
                    streaming=False,
                    **dataset_kwargs,
                )
    elif cfg.dataset.validation_split:
        if is_streaming:
            logger.warning(
                "validation_split is not supported for streaming datasets; "
                "provide dataset.eval_split to enable validation."
            )
        else:
            split = train_dataset.train_test_split(
                test_size=cfg.dataset.validation_split, seed=cfg.seed
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]

    if eval_dataset is not None:
        eval_is_streaming = cfg.dataset.streaming
        eval_needs_tokenization = False
        if eval_is_streaming:
            first_example = next(iter(eval_dataset))
            eval_needs_tokenization = "input_ids" not in first_example
        else:
            eval_needs_tokenization = "input_ids" not in eval_dataset.column_names

        if eval_needs_tokenization:
            from neobert.tokenizer import tokenize

            text_column = resolve_text_column(
                eval_dataset,
                eval_is_streaming,
                preferred=cfg.dataset.text_column,
            )
            eval_num_proc = (
                None
                if eval_is_streaming
                else _resolve_tokenize_num_proc(
                    cfg.dataset.num_proc,
                    accelerator.num_processes,
                    accelerator.is_main_process,
                )
            )
            with accelerator.main_process_first():
                eval_dataset = tokenize(
                    eval_dataset,
                    tokenizer,
                    column_name=text_column,
                    max_length=tokenize_max_length,
                    remove_columns=True,
                    truncation=True,
                    num_proc=eval_num_proc,
                    add_special_tokens=add_special_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                )

        if not eval_is_streaming:
            accelerator.print(f"Eval dataset size: {len(eval_dataset)}")

    if cfg.dataset.streaming and hasattr(cfg.dataset, "shuffle_buffer_size"):
        train_dataset = _maybe_shuffle_streaming_dataset(
            train_dataset,
            cfg.dataset.shuffle_buffer_size,
            cfg.seed,
            print_fn=accelerator.print,
        )

    pin_memory, persistent_workers, prefetch_factor, loader_perf_notes = (
        _resolve_loader_perf_settings(cfg, device=accelerator.device)
    )
    for note in loader_perf_notes:
        logger.info(note)

    # Dataloader
    collator_max_length = cfg.datacollator.max_length or cfg.dataset.max_seq_length
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        batch_size=cfg.trainer.per_device_train_batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        mlm_probability=cfg.datacollator.mlm_probability,
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
        mask_all=cfg.datacollator.mask_all,
        pack_sequences=cfg.datacollator.pack_sequences,
        max_length=collator_max_length,
        return_packed_seqlens=cfg.model.attn_backend != "sdpa",
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = get_dataloader(
            eval_dataset,
            tokenizer,
            batch_size=cfg.trainer.per_device_eval_batch_size,
            num_workers=cfg.dataset.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            mlm_probability=cfg.datacollator.mlm_probability,
            pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
            mask_all=cfg.datacollator.mask_all,
            pack_sequences=cfg.datacollator.pack_sequences,
            max_length=collator_max_length,
            shuffle=False,
            return_packed_seqlens=cfg.model.attn_backend != "sdpa",
        )

    # Model
    # vocab_size is now resolved during config preprocessing
    # Debug print
    if cfg.debug:
        print(f"Config model.vocab_size: {cfg.model.vocab_size}")
        print(f"Tokenizer base vocab_size: {tokenizer.vocab_size}")
        print(f"Tokenizer total len(): {len(tokenizer)}")
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")

    # Keep this mapping in sync with ModelConfig fields to avoid config drift.
    model_config = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_length=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,  # Use preprocessed vocab_size
        rope=cfg.model.rope,
        rms_norm=cfg.model.rms_norm,
        hidden_act=cfg.model.hidden_act,
        dropout=cfg.model.dropout_prob,
        norm_eps=cfg.model.norm_eps,
        embedding_init_range=cfg.model.embedding_init_range,
        decoder_init_range=cfg.model.decoder_init_range,
        classifier_init_range=cfg.model.classifier_init_range,
        pad_token_id=cfg.model.pad_token_id,
        attn_backend=cfg.model.attn_backend,
        kernel_backend=cfg.model.kernel_backend,
        ngpt=cfg.model.ngpt,
        base_scale=cfg.model.base_scale,
    )
    model = NeoBERTLMHead(model_config)

    if cfg.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Track flag on config for downstream logging/debug; the forward pass reads
        # the model attribute, so this is purely metadata (HF-compatible).
        setattr(model.config, "gradient_checkpointing", True)

    # Print model summary with hierarchical parameter counts
    if accelerator.is_main_process:
        model_summary(model, max_depth=3, show_param_shapes=True)

    model = _maybe_compile_model(model, cfg, accelerator, logger)

    # Optimizer and Scheduler
    # Log if using MuonClip optimizer
    if cfg.optimizer.name.lower() in ["muonclip", "muon-clip", "muon_clip"]:
        muon_cfg = cfg.optimizer.muon_config or MuonConfig()

        logger.info("=" * 60)
        logger.info("MuonClip Optimizer Configuration")
        logger.info("=" * 60)
        logger.info(f"QK-clipping: {muon_cfg.enable_clipping}")
        logger.info(f"Clipping threshold: {muon_cfg.clipping_threshold}")
        logger.info(f"Newton-Schulz iterations: {muon_cfg.ns_steps}")
        logger.info(f"Orthogonalization: {muon_cfg.orthogonalization}")
        logger.info(f"Clipping warmup steps: {muon_cfg.clipping_warmup_steps}")
        logger.info(f"Clipping interval: {muon_cfg.clipping_interval}")
        logger.info(f"QK chunk size: {muon_cfg.clipping_qk_chunk_size}")
        logger.info(
            f"Capture last microbatch only: {muon_cfg.capture_last_microbatch_only}"
        )
        logger.info("=" * 60)

    optimizer = get_optimizer(
        model,
        accelerator.distributed_type,
        model_config=model_config,  # Pass model config for MuonClip
        name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        muon_config=cfg.optimizer.muon_config,
    )
    _, warmup_steps, decay_steps, constant_steps = resolve_scheduler_steps(
        trainer_max_steps=cfg.trainer.max_steps,
        total_steps=cfg.scheduler.total_steps,
        warmup_steps=cfg.scheduler.warmup_steps,
        warmup_percent=cfg.scheduler.warmup_percent,
        decay_steps=cfg.scheduler.decay_steps,
        decay_percent=cfg.scheduler.decay_percent,
        constant_steps=0,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=cfg.optimizer.lr,
        decay=cfg.scheduler.name,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        final_ratio=cfg.scheduler.final_lr_ratio,
        constant_steps=constant_steps,
    )

    if hasattr(accelerator, "prepare_data_loader"):

        def _prepare_loader(
            dataloader: torch.utils.data.DataLoader,
        ) -> torch.utils.data.DataLoader:
            """Prepare dataloaders with tuned dispatch/device placement.

            :param torch.utils.data.DataLoader dataloader: Dataloader to prepare.
            :return torch.utils.data.DataLoader: Prepared dataloader.
            """
            kwargs = {"device_placement": not disable_dispatch}
            try:
                supports_dispatch = (
                    "dispatch_batches"
                    in inspect.signature(accelerator.prepare_data_loader).parameters
                )
                if supports_dispatch:
                    kwargs["dispatch_batches"] = not disable_dispatch
                return accelerator.prepare_data_loader(dataloader, **kwargs)
            except TypeError:
                return accelerator.prepare_data_loader(
                    dataloader, device_placement=not disable_dispatch
                )

        train_dataloader = _prepare_loader(train_dataloader)
        if eval_dataloader is not None:
            eval_dataloader = _prepare_loader(eval_dataloader)
        model, optimizer, scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )
    else:
        if disable_dispatch:
            raise RuntimeError(
                "Accelerate version too old to configure dispatch_batches. "
                "Upgrade accelerate when using packed_seqlens."
            )
        if accelerator.distributed_type is DistributedType.DEEPSPEED:
            logger.warning(
                "Accelerate backend does not support per-object device placement; "
                "falling back to default dataloader placement."
            )
            if eval_dataloader is not None:
                (
                    train_dataloader,
                    eval_dataloader,
                    model,
                    optimizer,
                    scheduler,
                ) = accelerator.prepare(
                    train_dataloader,
                    eval_dataloader,
                    model,
                    optimizer,
                    scheduler,
                )
            else:
                train_dataloader, model, optimizer, scheduler = accelerator.prepare(
                    train_dataloader,
                    model,
                    optimizer,
                    scheduler,
                )
        else:
            if eval_dataloader is not None:
                (
                    train_dataloader,
                    eval_dataloader,
                    model,
                    optimizer,
                    scheduler,
                ) = accelerator.prepare(
                    train_dataloader,
                    eval_dataloader,
                    model,
                    optimizer,
                    scheduler,
                    device_placement=[False, False, True, True, True],
                )
            else:
                train_dataloader, model, optimizer, scheduler = accelerator.prepare(
                    train_dataloader,
                    model,
                    optimizer,
                    scheduler,
                    device_placement=[False, True, True, True],
                )

    # Packed mode keeps manual batch transfers; non-packed mode uses Accelerate
    # device placement for lower Python overhead and better overlap.
    manual_device_move = disable_dispatch or not hasattr(
        accelerator, "prepare_data_loader"
    )

    if wandb_enabled and accelerator.is_main_process:
        wandb_watch = os.environ.get("WANDB_WATCH")
        if wandb_watch is not None:
            watch_mode = wandb_watch.strip().lower()
            if watch_mode in {"", "false", "0", "none", "off"}:
                watch_mode = None
            elif watch_mode == "weights":
                watch_mode = "parameters"
            elif watch_mode not in {"gradients", "parameters", "all"}:
                logger.warning(
                    "Unrecognized WANDB_WATCH value '%s'; skipping wandb.watch()",
                    wandb_watch,
                )
                watch_mode = None

            if watch_mode:
                wandb.watch(
                    accelerator.unwrap_model(model),
                    log=watch_mode,
                    log_freq=getattr(cfg.wandb, "log_interval", 100),
                )

    train_loss_fn: Optional[torch.nn.Module] = None
    masked_objective: Optional[MaskedPositionsOnlyMLMObjective] = None
    if masked_logits_only_loss:
        masked_objective = MaskedPositionsOnlyMLMObjective(
            ignore_index=-100,
        )
        logger.info("Using masked-logits-only MLM objective.")
    else:
        resolved_kb = resolve_kernel_backend(cfg.model.kernel_backend)
        train_loss_fn = get_cross_entropy_loss(
            reduction="sum",
            ignore_index=-100,
            backend=resolved_kb,
        )
        logger.warning(
            "Using legacy original full-logits MLM loss path "
            "(trainer.masked_logits_only_loss=false). This path is intended for "
            "ablation/debug and has higher memory use."
        )
    eval_max_batches = getattr(cfg.trainer, "eval_max_batches", None)
    if isinstance(eval_max_batches, int) and eval_max_batches <= 0:
        eval_max_batches = None
    if eval_max_batches is None and eval_dataset is not None and cfg.dataset.streaming:
        eval_max_batches = 100
        logger.info(
            "Streaming eval detected; defaulting eval_max_batches to %s. "
            "Set trainer.eval_max_batches to override.",
            eval_max_batches,
        )

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume_from_checkpoint and resume_checkpoint_path:
        resume_checkpoint = Path(resume_checkpoint_path)
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"resume_from_checkpoint path not found: {resume_checkpoint_path}"
            )
        accelerator.load_state(str(resume_checkpoint))
        skipped_train_dataloader = _prepare_resume_dataloader(
            train_dataloader, metrics, accelerator, is_streaming
        )
    elif cfg.trainer.resume_from_checkpoint:
        logger.warning(
            "resume_from_checkpoint is set but no valid checkpoints were found in %s",
            checkpoint_dir,
        )

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(not accelerator.is_main_process),
    )

    accum_tokens = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_samples = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_tokens = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_num_pred = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_sum_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    local_num_correct = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_loss_path_liger_flce = torch.zeros(
        (), device=accelerator.device, dtype=torch.long
    )
    local_loss_path_checkpointed = torch.zeros(
        (), device=accelerator.device, dtype=torch.long
    )
    local_loss_path_zero_masked = torch.zeros(
        (), device=accelerator.device, dtype=torch.long
    )
    local_loss_path_other = torch.zeros((), device=accelerator.device, dtype=torch.long)
    logged_masked_loss_path = False
    stored_batch = {
        "input_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seqlens": None,
    }
    warned_low_token_scale = False
    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = (
            train_dataloader
            if skipped_train_dataloader is None
            else skipped_train_dataloader
        )
        for batch in dataloader:
            # Pack or truncate to target per-step batch size. Packed mode can emit
            # variable batch dimensions, so we buffer/merge there too now that
            # packed_seqlens uses fixed-width tensor metadata.
            is_packed = batch.get("packed_seqlens") is not None
            stored_is_packed = stored_batch.get("packed_seqlens") is not None
            if _has_stored_batch(stored_batch) and is_packed != stored_is_packed:
                # Mixed packed/non-packed batches are not expected; avoid cross-mode
                # concatenation if stale buffered fragments remain.
                _clear_stored_batch(stored_batch)
            if batch["input_ids"].shape[
                0
            ] != cfg.trainer.per_device_train_batch_size or _has_stored_batch(
                stored_batch
            ):
                # ``to_target_batch_size`` may emit variable-size microbatches
                # at epoch tails; update correctness is preserved by token-based
                # gradient rescaling in ``_gradient_token_scale``.
                batch, stored_batch = to_target_batch_size(
                    batch, stored_batch, cfg.trainer.per_device_train_batch_size
                )
            if (
                enforce_full_packed_batches
                and cfg.datacollator.pack_sequences
                and batch["input_ids"].shape[0]
                < cfg.trainer.per_device_train_batch_size
            ):
                # Packed collation can emit undersized batches when source text is
                # short. Buffer and combine these fragments to keep full-size
                # microbatches for better kernel efficiency and compile stability.
                _append_to_stored_batch(stored_batch, batch)
                continue

            if manual_device_move:
                if pin_memory and accelerator.device.type == "cuda":
                    batch = _ensure_pinned_cpu_batch(batch)
                batch = _move_batch_to_device(batch, accelerator.device)

            # Update number of batches only when we will execute a backward pass.
            metrics["train/batches"] += 1

            num_pred = (batch["labels"] != -100).sum()
            num_tokens = (batch["input_ids"] != model_config.pad_token_id).sum()
            packed_seqlens = _packed_seqlens_to_tensor(batch.get("packed_seqlens"))
            pad_mask = (
                None
                if packed_seqlens is not None
                else batch.get("attention_mask", None)
            )

            # Keep all accumulation semantics backend-agnostic (DDP/FSDP/DeepSpeed).
            with accelerator.accumulate(model):
                sync_gradients = bool(accelerator.sync_gradients)
                _maybe_prepare_for_forward(
                    optimizer,
                    update_step=metrics["train/steps"],
                    is_last_microbatch=sync_gradients,
                )
                with accelerator.autocast():
                    if masked_objective is not None:
                        hidden = model(
                            src=batch["input_ids"],
                            pad_mask=pad_mask,
                            packed_seqlens=packed_seqlens,
                            return_logits=False,
                        )["hidden_representation"]
                        objective_out = masked_objective(
                            hidden_states=hidden,
                            labels=batch["labels"],
                            lm_weight=model.decoder.weight,
                            compute_accuracy=log_train_accuracy,
                        )
                        loss_sum = objective_out.loss_sum_local
                        num_pred = objective_out.num_masked_local
                        if objective_out.used_path == "liger_flce":
                            local_loss_path_liger_flce += 1
                        elif objective_out.used_path == "train_checkpointed_masked_ce":
                            local_loss_path_checkpointed += 1
                        elif objective_out.used_path == "zero_masked":
                            local_loss_path_zero_masked += 1
                        else:
                            local_loss_path_other += 1
                        if (
                            not logged_masked_loss_path
                            and accelerator.is_main_process
                            and objective_out.used_path != "zero_masked"
                        ):
                            accelerator.print(
                                "Masked-logits loss path active (first non-empty microbatch): "
                                f"{objective_out.used_path}"
                            )
                            logger.info(
                                "Masked-logits loss path active (first non-empty microbatch): %s",
                                objective_out.used_path,
                            )
                            logged_masked_loss_path = True
                    else:
                        if train_loss_fn is None:
                            raise RuntimeError(
                                "Legacy loss path selected but train_loss_fn is undefined."
                            )
                        logits = model(
                            src=batch["input_ids"],
                            pad_mask=pad_mask,
                            packed_seqlens=packed_seqlens,
                        )["logits"]
                        loss_sum = train_loss_fn(
                            logits.view(-1, model_config.vocab_size),
                            batch["labels"].view(-1),
                        )

                # Compute gradient
                accelerator.backward(loss_sum)
                accum_tokens += num_pred.to(accum_tokens.dtype)

                # Accumulate metrics on device to avoid per-batch syncs.
                local_samples += batch["input_ids"].shape[0]
                local_tokens += num_tokens
                local_num_pred += num_pred
                local_sum_loss += loss_sum.detach().float()
                if log_train_accuracy:
                    if masked_objective is not None:
                        if objective_out.num_correct_local is not None:
                            local_num_correct += objective_out.num_correct_local
                    else:
                        local_num_correct += _count_masked_correct(
                            logits, batch["labels"]
                        )

            if sync_gradients:
                should_log = (metrics["train/steps"] + 1) % log_interval == 0
                # Reduce to global token count to handle uneven sharding across ranks.
                tokens_global = accelerator.reduce(accum_tokens, reduction="sum")
                scale, clamped = _gradient_token_scale(
                    tokens_global,
                    num_processes=accelerator.num_processes,
                    grad_accumulation_steps=accelerator.gradient_accumulation_steps,
                )
                # Match full-batch normalization across variable-length microbatches.
                # accelerator.backward() already divides by grad_accumulation_steps, and DDP averages
                # across processes on the sync step, so we rescale by
                # (num_processes * grad_accum_steps) / tokens_global.
                # For near-empty masked updates, we floor tokens_global to one token
                # per rank per accumulation step to avoid gradient amplification.
                # This post-accumulation rescale is equivalent to scaling each microbatch loss
                # because gradients are linear in the loss scalar.
                # Ref: Unsloth blog (archived) https://archive.ph/RmO0U
                _scale_gradients(model, scale)
                if (
                    should_log
                    and not warned_low_token_scale
                    and bool(clamped.detach().cpu().item())
                ):
                    warned_low_token_scale = True
                    logger.warning(
                        "Masked-token count was below the safe minimum for an update "
                        "(tokens_global=%s, min=%s); clamped gradient scale to avoid "
                        "pathological amplification.",
                        int(tokens_global.item()),
                        accelerator.num_processes
                        * accelerator.gradient_accumulation_steps,
                    )

                grad_norm_value = None

                # Optional gradient clipping for stability on deep/long-context runs.
                max_grad_norm = cfg.trainer.gradient_clipping

                if max_grad_norm is not None and max_grad_norm > 0:
                    grad_norm_pre_clip = accelerator.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    if should_log and grad_norm_pre_clip is not None:
                        grad_norm_value = float(
                            grad_norm_pre_clip.item()
                            if isinstance(grad_norm_pre_clip, torch.Tensor)
                            else grad_norm_pre_clip
                        )
                elif should_log and log_grad_norm:
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        get_global_grad = getattr(model, "get_global_grad_norm", None)
                        if callable(get_global_grad):
                            grad_norm = get_global_grad()
                            if isinstance(grad_norm, torch.Tensor):
                                grad_norm_value = float(grad_norm.item())
                            elif grad_norm is not None:
                                grad_norm_value = float(grad_norm)
                    else:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), float("inf")
                        )
                        if grad_norm is not None:
                            grad_norm_value = float(
                                grad_norm.item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            )

                # Log metrics
                pbar.update(1)
                metrics["train/steps"] += 1

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()
                accum_tokens.zero_()

                if metrics["train/steps"] % log_interval == 0:
                    if grad_norm_value is not None:
                        metrics["train/grad_norm"] = grad_norm_value

                    if cfg.trainer.log_weight_norms and accelerator.is_main_process:
                        if accelerator.distributed_type is DistributedType.DEEPSPEED:
                            metrics["train/weight_norm"] = (
                                sum(
                                    [
                                        safe_get_full_fp32_param(p).norm(2) ** 2
                                        for p in model.parameters()
                                    ]
                                )
                                ** 0.5
                            ).item()
                        else:
                            metrics["train/weight_norm"] = (
                                sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5
                            ).item()

                    # Add MuonClip metrics if available
                    if hasattr(optimizer, "get_metrics"):
                        muonclip_metrics = optimizer.get_metrics()
                        for key, value in muonclip_metrics.items():
                            metrics[key] = value

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    metrics["train/local_samples"] = int(local_samples.item())
                    metrics["train/local_tokens"] = int(local_tokens.item())
                    metrics["train/local_num_pred"] = int(local_num_pred.item())
                    metrics["train/local_sum_loss"] = float(local_sum_loss.item())
                    metrics["train/local_num_correct"] = int(local_num_correct.item())
                    if masked_objective is not None:
                        loss_path_counts = torch.stack(
                            [
                                local_loss_path_liger_flce,
                                local_loss_path_checkpointed,
                                local_loss_path_zero_masked,
                                local_loss_path_other,
                            ]
                        )
                        loss_path_counts = accelerator.reduce(
                            loss_path_counts, reduction="sum"
                        )
                        path_total = int(loss_path_counts.sum().item())
                        metrics["train/loss_path_steps_liger_flce"] = int(
                            loss_path_counts[0].item()
                        )
                        metrics["train/loss_path_steps_checkpointed"] = int(
                            loss_path_counts[1].item()
                        )
                        metrics["train/loss_path_steps_zero_masked"] = int(
                            loss_path_counts[2].item()
                        )
                        metrics["train/loss_path_steps_other"] = int(
                            loss_path_counts[3].item()
                        )
                        if path_total > 0:
                            metrics["train/loss_path_ratio_liger_flce"] = (
                                metrics["train/loss_path_steps_liger_flce"] / path_total
                            )
                            metrics["train/loss_path_ratio_checkpointed"] = (
                                metrics["train/loss_path_steps_checkpointed"]
                                / path_total
                            )
                            metrics["train/loss_path_ratio_zero_masked"] = (
                                metrics["train/loss_path_steps_zero_masked"]
                                / path_total
                            )
                            metrics["train/loss_path_ratio_other"] = (
                                metrics["train/loss_path_steps_other"] / path_total
                            )
                    metrics.log(
                        accelerator,
                        emit_console=(
                            (
                                (not wandb_enabled)
                                or str(cfg.wandb.mode).lower() == "offline"
                            )
                            and accelerator.is_main_process
                        ),
                        console_fn=accelerator.print,
                    )
                    local_samples.zero_()
                    local_tokens.zero_()
                    local_num_pred.zero_()
                    local_sum_loss.zero_()
                    local_num_correct.zero_()
                    local_loss_path_liger_flce.zero_()
                    local_loss_path_checkpointed.zero_()
                    local_loss_path_zero_masked.zero_()
                    local_loss_path_other.zero_()

                # Save accelerator state for resumable training
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    state_checkpoint_path = checkpoint_dir / str(metrics["train/steps"])
                    accelerator.save_state(output_dir=str(state_checkpoint_path))
                    accelerator.wait_for_everyone()

                    # Accelerator checkpoints are the source of truth for resuming.
                    save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                    limit = max(save_total_limit, max_ckpt)
                    if (
                        limit > 0
                        and checkpoint_dir.exists()
                        and accelerator.is_main_process
                    ):
                        # Prune accelerator checkpoints to the same retention policy
                        # as model_checkpoints to avoid unbounded disk growth.
                        accel_checkpoints = []
                        for item_path in checkpoint_dir.iterdir():
                            if item_path.is_dir() and item_path.name.isdigit():
                                accel_checkpoints.append(int(item_path.name))
                        if len(accel_checkpoints) > limit:
                            accel_checkpoints.sort()
                            for old_ckpt in accel_checkpoints[
                                : len(accel_checkpoints) - limit
                            ]:
                                old_path = checkpoint_dir / str(old_ckpt)
                                if old_path.exists():
                                    shutil.rmtree(old_path)
                                    logger.info(
                                        "Removed old accelerator checkpoint: %s (limit=%s)",
                                        old_path,
                                        limit,
                                    )

                # Save model weights checkpoint
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    # Model checkpoints are used for inference/export and can be pruned independently.
                    # Save the checkpoint
                    step_tag = str(metrics["train/steps"])
                    tmp_tag = f"{step_tag}.tmp"
                    tmp_path = model_checkpoint_dir / tmp_tag

                    # Clean up any stale tmp directory from a previous failed save.
                    # Use exist_ok pattern to avoid TOCTOU race conditions in distributed setting.
                    if accelerator.is_main_process:
                        if tmp_path.exists():
                            shutil.rmtree(tmp_path)
                        tmp_path.mkdir(parents=True, exist_ok=True)
                    accelerator.wait_for_everyone()

                    checkpoint_path = tmp_path
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        # DeepSpeed writes into model_checkpoint_dir/<tag>. We save to a
                        # temporary tag and atomically promote it to the final numeric tag,
                        # then refresh root/latest so zero_to_fp32 helpers can still resolve
                        # canonical (root, tag) checkpoint layout.
                        model.save_checkpoint(model_checkpoint_dir, tag=tmp_tag)
                    else:
                        save_model_safetensors(
                            accelerator.unwrap_model(model),
                            checkpoint_path,
                        )

                    # Save config and tokenizer info (only from main process)
                    if accelerator.is_main_process:
                        # Save config as YAML
                        config_path = checkpoint_path / "config.yaml"
                        ConfigLoader.save(cfg, str(config_path))

                        # Save tokenizer info as JSON
                        tokenizer_info = {
                            "tokenizer_name": cfg.tokenizer.path or cfg.tokenizer.name,
                            "vocab_size": cfg.model.vocab_size,
                            "base_vocab_size": tokenizer.vocab_size,
                            "total_vocab_size": len(tokenizer),
                            "pad_token_id": tokenizer.pad_token_id,
                        }
                        tokenizer_info_path = checkpoint_path / "tokenizer_info.json"
                        with tokenizer_info_path.open("w", encoding="utf-8") as f:
                            json.dump(tokenizer_info, f, indent=2)

                        # Save full tokenizer with save_pretrained
                        tokenizer_dir = checkpoint_path / "tokenizer"
                        tokenizer_dir.mkdir(parents=True, exist_ok=True)

                        # Ensure tokenizer.model_max_length matches model's max_position_embeddings
                        tokenizer.model_max_length = cfg.model.max_position_embeddings
                        tokenizer.save_pretrained(tokenizer_dir)

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        final_path = model_checkpoint_dir / step_tag
                        _promote_tmp_checkpoint_dir(checkpoint_path, final_path)
                        if accelerator.distributed_type is DistributedType.DEEPSPEED:
                            # Keep DeepSpeed root semantics valid even though we
                            # atomically rename tmp tags to final numeric tags.
                            _write_deepspeed_latest_file(model_checkpoint_dir, step_tag)
                        checkpoint_path = final_path
                        # Use logger instead of accelerator.print to avoid progress bar interference
                        logger.info(
                            "Saved checkpoint with config, tokenizer info, and full tokenizer to %s",
                            checkpoint_path,
                        )

                    accelerator.wait_for_everyone()

                    # Clean up old checkpoints if limit is set (after saving).
                    save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                    limit = max(save_total_limit, max_ckpt)  # Use whichever is set

                    if (
                        limit > 0
                        and model_checkpoint_dir.exists()
                        and accelerator.is_main_process
                    ):
                        # Get all checkpoint directories
                        checkpoints = []
                        for item_path in model_checkpoint_dir.iterdir():
                            if item_path.is_dir() and item_path.name.isdigit():
                                checkpoints.append(int(item_path.name))

                        # Sort and remove oldest checkpoints if over limit
                        if len(checkpoints) > limit:
                            checkpoints.sort()
                            # Remove oldest checkpoints
                            for old_ckpt in checkpoints[: len(checkpoints) - limit]:
                                old_path = model_checkpoint_dir / str(old_ckpt)
                                if old_path.exists():
                                    shutil.rmtree(old_path)
                                    # Use logger instead of accelerator.print to avoid progress bar interference
                                    logger.info(
                                        f"Removed old checkpoint: {old_path} (limit={limit})"
                                    )

                if (
                    eval_dataloader is not None
                    and getattr(cfg.trainer, "eval_strategy", "steps") == "steps"
                    and cfg.trainer.eval_steps > 0
                    and metrics["train/steps"] % cfg.trainer.eval_steps == 0
                ):
                    eval_metrics = _run_eval(
                        model,
                        eval_dataloader,
                        train_loss_fn,
                        accelerator,
                        model_config,
                        max_batches=eval_max_batches,
                        manual_device_move=manual_device_move,
                        masked_objective=masked_objective,
                    )
                    accelerator.log(
                        format_metrics(eval_metrics), step=metrics["train/steps"]
                    )

                # Zero out the optimizer
                optimizer.zero_grad()

                # Log metrics
                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    pbar.close()
                    return

        if (
            eval_dataloader is not None
            and getattr(cfg.trainer, "eval_strategy", "steps") == "epoch"
        ):
            eval_metrics = _run_eval(
                model,
                eval_dataloader,
                train_loss_fn,
                accelerator,
                model_config,
                max_batches=eval_max_batches,
                manual_device_move=manual_device_move,
                masked_objective=masked_objective,
            )
            accelerator.log(format_metrics(eval_metrics), step=metrics["train/steps"])

        # Update the number of epochs
        metrics["train/epochs"] += 1
        skipped_train_dataloader = None

        # For streaming datasets, update the epoch to ensure different shuffling
        if cfg.dataset.streaming and hasattr(train_dataset, "set_epoch"):
            # Called after epoch increment so the next epoch uses the new seed.
            train_dataset.set_epoch(metrics["train/epochs"])
            accelerator.print(
                f"Set streaming dataset epoch to {metrics['train/epochs']}"
            )

    pbar.close()
