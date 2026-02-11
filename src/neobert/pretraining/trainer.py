"""Pretraining loop for masked language modeling."""

import inspect
import json
import logging
import math
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

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

from neobert.checkpointing import MODEL_WEIGHTS_NAME, save_state_dict_safetensors
from neobert.config import (
    Config,
    ConfigLoader,
    MuonConfig,
    resolve_mixed_precision,
    round_up_to_multiple,
)
from neobert.dataloader import get_dataloader
from neobert.kernels.attention import resolve_runtime_attn_backend
from neobert.kernels.backend import get_cross_entropy_loss, resolve_kernel_backend
from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.optimizer import get_optimizer
from neobert.pretraining.masked_objective import (
    MaskedObjectiveOut,
    MaskedPositionsOnlyMLMObjective,
)
from neobert.scheduler import get_scheduler, resolve_scheduler_steps
from neobert.tokenizer import get_tokenizer, resolve_text_column
from neobert.training_utils import (
    _maybe_compile_model,
    _maybe_prepare_for_forward,
    _resolve_resume_checkpoint,
    create_accelerator,
    resolve_wandb_watch_mode,
)
from neobert.utils import (
    configure_tf32,
    format_resolved_config,
    model_summary,
    prepare_wandb_config,
)

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


def _resolve_eval_samples(value: Any) -> Optional[int]:
    """Resolve and validate ``dataset.eval_samples``.

    :param Any value: Config value to normalize.
    :raises ValueError: If value is not integer-like or is boolean.
    :return int | None: Positive sample count or ``None`` when unset/disabled.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"dataset.eval_samples must be an integer sample count, got bool {value!r}."
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "dataset.eval_samples must be an integer sample count, "
            f"got {value!r} ({type(value).__name__})."
        ) from exc
    if resolved <= 0:
        return None
    return resolved


def _format_percent(value: float) -> str:
    """Format a percentage value for log output.

    :param float value: Percentage in the range ``[0, 100]``.
    :return str: Human-readable percentage string.
    """
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _log_masking_strategy(cfg: Config) -> None:
    """Log the effective MLM corruption policy from runtime config.

    :param Config cfg: Runtime configuration.
    :raises ValueError: If ``datacollator.mlm_probability`` is out of range.
    """
    mlm_probability = float(cfg.datacollator.mlm_probability)
    if mlm_probability < 0.0 or mlm_probability > 1.0:
        raise ValueError(
            f"datacollator.mlm_probability must be in [0, 1], got {mlm_probability}."
        )

    selected_pct = mlm_probability * 100.0
    untouched_pct = max(0.0, 100.0 - selected_pct)
    selected_pct_str = _format_percent(selected_pct)
    untouched_pct_str = _format_percent(untouched_pct)

    if cfg.datacollator.mask_all:
        logger.info(
            "datacollator.mask_all=true with mlm_probability="
            f"{selected_pct_str}%: {selected_pct_str}% tokens replaced with [MASK], "
            f"{untouched_pct_str}% untouched."
        )
        return

    mask_pct = selected_pct * 0.8
    random_pct = selected_pct * 0.1
    unchanged_labeled_pct = selected_pct * 0.1
    mask_pct_str = _format_percent(mask_pct)
    random_pct_str = _format_percent(random_pct)
    unchanged_labeled_pct_str = _format_percent(unchanged_labeled_pct)
    logger.warning(
        "datacollator.mask_all=false with mlm_probability="
        f"{selected_pct_str}%: {selected_pct_str}% of tokens are sampled; sampled "
        "tokens use BERT 80/10/10 ([MASK]/random/original). Global token mix is "
        f"{untouched_pct_str}% unsampled (untouched), {mask_pct_str}% [MASK], "
        f"{random_pct_str}% random-token, {unchanged_labeled_pct_str}% sampled-but-"
        "unchanged original-token (still labeled). Set datacollator.mask_all=true "
        "for 100% [MASK] replacement on sampled tokens."
    )


def _resolve_fsdp_version(accelerator: Accelerator) -> int:
    """Resolve FSDP version from Accelerate state.

    Defaults to ``1`` when plugin metadata is unavailable.

    :param Accelerator accelerator: Active accelerator runtime.
    :return int: FSDP plugin version.
    """
    state = getattr(accelerator, "state", None)
    fsdp_plugin = getattr(state, "fsdp_plugin", None) if state is not None else None
    raw_version = getattr(fsdp_plugin, "fsdp_version", None)
    try:
        return int(raw_version) if raw_version is not None else 1
    except (TypeError, ValueError):
        return 1


@contextmanager
def _gather_decoder_weight_for_masked_objective(
    model: torch.nn.Module,
    accelerator: Accelerator,
) -> Iterator[torch.Tensor]:
    """Yield decoder weights, gathering sharded parameters when required.

    Masked-only loss computes logits from ``decoder.weight`` outside the decoder
    module forward. Under FSDP2 and DeepSpeed ZeRO-3 this parameter can be
    partitioned, so we must materialize the full tensor for the objective path.

    :param torch.nn.Module model: Prepared training model (possibly wrapped).
    :param Accelerator accelerator: Active accelerator runtime.
    :raises RuntimeError: If FSDP v1 is active (unsupported policy).
    :yield torch.Tensor: Decoder projection weight tensor.
    :return Iterator[torch.Tensor]: Context manager yielding decoder weight.
    """

    def _resolve_decoder_weight() -> torch.Tensor:
        """Resolve decoder projection weight from wrapped or unwrapped model.

        :return torch.Tensor: Decoder projection weight parameter.
        :raises AttributeError: If ``decoder.weight`` cannot be resolved.
        """
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "weight"):
            return decoder.weight
        unwrap_model = getattr(accelerator, "unwrap_model", None)
        if callable(unwrap_model):
            unwrapped = unwrap_model(model)
            decoder = getattr(unwrapped, "decoder", None)
            if decoder is not None and hasattr(decoder, "weight"):
                return decoder.weight
        raise AttributeError("Could not resolve decoder.weight for masked objective.")

    if accelerator.distributed_type is DistributedType.FSDP:
        fsdp_version = _resolve_fsdp_version(accelerator)
        if fsdp_version >= 2:
            base_model = model
            unwrapped_model: Optional[torch.nn.Module] = None
            unwrap_model = getattr(accelerator, "unwrap_model", None)
            if callable(unwrap_model):
                try:
                    unwrapped_model = unwrap_model(model)
                except Exception:
                    unwrapped_model = None
            if unwrapped_model is not None:
                base_model = unwrapped_model

            decoder_module = getattr(base_model, "decoder", None)
            if decoder_module is None:
                decoder_module = getattr(model, "decoder", None)
            if decoder_module is None or not hasattr(decoder_module, "weight"):
                raise AttributeError(
                    "Could not resolve decoder module for FSDP2 masked objective."
                )
            decoder_weight = decoder_module.weight

            def _is_fsdp2_module(module: torch.nn.Module) -> bool:
                """Check whether a module exposes FSDP2 unshard/reshard hooks.

                :param torch.nn.Module module: Module to inspect.
                :return bool: ``True`` when the module looks FSDP2-wrapped.
                """
                return hasattr(module, "unshard") and hasattr(module, "reshard")

            def _find_fsdp_owner(
                search_root: torch.nn.Module,
            ) -> Optional[torch.nn.Module]:
                """Find FSDP2 module that should unshard ``decoder.weight``.

                :param torch.nn.Module search_root: Candidate module tree root.
                :return torch.nn.Module | None: Owning FSDP2 module or ``None``.
                """
                owner: Optional[torch.nn.Module] = None
                for module in search_root.modules():
                    if not _is_fsdp2_module(module):
                        continue
                    for param in module.parameters(recurse=False):
                        if param is decoder_weight:
                            owner = module
                            break
                    if owner is not None:
                        break
                if owner is not None:
                    return owner

                named_modules = dict(search_root.named_modules())
                decoder_path = next(
                    (
                        path
                        for path, module in named_modules.items()
                        if (
                            module is decoder_module
                            or (
                                hasattr(module, "weight")
                                and getattr(module, "weight", None) is decoder_weight
                            )
                        )
                    ),
                    None,
                )
                if decoder_path is not None:
                    path_parts = decoder_path.split(".")
                    for depth in range(len(path_parts), -1, -1):
                        ancestor_path = ".".join(path_parts[:depth])
                        ancestor = (
                            search_root
                            if ancestor_path == ""
                            else named_modules.get(ancestor_path)
                        )
                        if ancestor is not None and _is_fsdp2_module(ancestor):
                            return ancestor

                if _is_fsdp2_module(search_root):
                    return search_root
                return None

            # Search wrapped model first; unwrapped views can hide FSDP2 hooks.
            fsdp_owner: Optional[torch.nn.Module] = _find_fsdp_owner(model)
            if fsdp_owner is None and base_model is not model:
                fsdp_owner = _find_fsdp_owner(base_model)

            if fsdp_owner is None:
                raise RuntimeError(
                    "FSDP2 is active but no unshard-capable module owns decoder.weight."
                )

            handle = fsdp_owner.unshard(async_op=True)
            if handle is not None:
                handle.wait()
            try:
                lm_weight = _resolve_decoder_weight()
                yield lm_weight
            finally:
                fsdp_owner.reshard()
        else:
            raise RuntimeError(
                "FSDP v1 is not supported for NeoBERT pretraining. "
                "Use FSDP2 (Accelerate fsdp_version=2)."
            )
        return

    lm_weight = _resolve_decoder_weight()
    if accelerator.distributed_type is not DistributedType.DEEPSPEED:
        yield lm_weight
        return
    if getattr(lm_weight, "ds_id", None) is None:
        yield lm_weight
        return

    import deepspeed

    fwd_module = None
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        try:
            fwd_module = unwrap_model(model)
        except Exception:
            fwd_module = None

    with deepspeed.zero.GatheredParameters(
        [lm_weight], modifier_rank=None, fwd_module=fwd_module
    ):
        yield lm_weight


def _should_backward_inside_gathered_decoder_weight(
    accelerator: Accelerator,
    lm_weight: torch.Tensor,
) -> bool:
    """Return whether backward must run while decoder weight is gathered.

    In FSDP2 and DeepSpeed ZeRO-3, masked-objective fallback paths can touch
    decoder weights during backward recomputation. Keep backward under gather
    when ``decoder.weight`` may otherwise be partitioned.

    :param Accelerator accelerator: Active accelerator runtime.
    :param torch.Tensor lm_weight: Decoder projection weight.
    :return bool: ``True`` when backward should run inside gather context.
    """
    if accelerator.distributed_type is DistributedType.FSDP:
        return _resolve_fsdp_version(accelerator) >= 2
    if accelerator.distributed_type is not DistributedType.DEEPSPEED:
        return False
    return getattr(lm_weight, "ds_id", None) is not None


def _run_masked_objective_step(
    model: torch.nn.Module,
    batch: BatchEncoding,
    pad_mask: Optional[torch.Tensor],
    packed_seqlens: Optional[torch.Tensor],
    masked_objective: MaskedPositionsOnlyMLMObjective,
    accelerator: Accelerator,
    *,
    log_train_accuracy: bool,
) -> tuple[MaskedObjectiveOut, torch.Tensor, bool]:
    """Compute masked-objective loss and optionally backprop inside gather.

    :param torch.nn.Module model: Training model.
    :param BatchEncoding batch: Prepared input batch.
    :param torch.Tensor | None pad_mask: Additive attention mask.
    :param torch.Tensor | None packed_seqlens: Packed sequence lengths metadata.
    :param MaskedPositionsOnlyMLMObjective masked_objective: Objective module.
    :param Accelerator accelerator: Active accelerator runtime.
    :param bool log_train_accuracy: Whether to compute masked accuracy.
    :return tuple[MaskedObjectiveOut, torch.Tensor, bool]:
        Objective output, local loss sum, and whether backward already ran.
    """
    hidden = model(
        src=batch["input_ids"],
        pad_mask=pad_mask,
        packed_seqlens=packed_seqlens,
        return_logits=False,
    )["hidden_representation"]
    with _gather_decoder_weight_for_masked_objective(model, accelerator) as lm_weight:
        objective_out = masked_objective(
            hidden_states=hidden,
            labels=batch["labels"],
            lm_weight=lm_weight,
            compute_accuracy=log_train_accuracy,
        )
        loss_sum = objective_out.loss_sum_local
        backward_done = False
        if _should_backward_inside_gathered_decoder_weight(accelerator, lm_weight):
            accelerator.backward(loss_sum)
            backward_done = True
    return objective_out, loss_sum, backward_done


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
        """Pin tensors in nested structures and report whether anything changed.

        :param Any value: Candidate tensor/container/scalar value.
        :return tuple[Any, bool]: Possibly pinned value and change flag.
        """
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
                "Unable to resolve streaming split size for "
                f"{dataset_name} ({split}): {exc}"
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
            logger.warning(f"Resolved split {split} is empty after slicing.")
            return dataset.take(0)
        dataset = dataset.take(take_n)

    return dataset


def _infer_eval_split_name(
    dataset_name: str,
    dataset_kwargs: dict[str, object],
    *,
    train_split: Optional[str],
) -> Optional[str]:
    """Infer a reasonable eval split name from dataset metadata.

    :param str dataset_name: Dataset identifier.
    :param dict[str, object] dataset_kwargs: Additional kwargs for HF datasets.
    :param str | None train_split: Train split selector (possibly sliced).
    :return str | None: Preferred eval split name or ``None``.
    """
    train_base = _parse_split_slice(train_split or "train")[0].lower()
    try:
        builder = load_dataset_builder(dataset_name, **dataset_kwargs)
        split_names = list(getattr(builder.info, "splits", {}).keys())
    except Exception as exc:
        logger.warning(
            f"Unable to infer eval split for {dataset_name} from dataset metadata: {exc}"
        )
        return None

    if not split_names:
        return None

    preferred = ("validation", "eval", "test", "dev")
    by_lower = {name.lower(): name for name in split_names}
    for candidate in preferred:
        resolved = by_lower.get(candidate)
        if resolved is not None and candidate != train_base:
            return resolved
    return None


def _split_train_dataset_for_eval_samples(
    train_dataset: Dataset,
    eval_samples: int,
    *,
    is_streaming: bool,
) -> tuple[Dataset, Dataset]:
    """Create eval data from training data and avoid train/eval overlap.

    :param Dataset train_dataset: Training dataset.
    :param int eval_samples: Number of eval samples to reserve.
    :param bool is_streaming: Whether the dataset is streaming.
    :raises ValueError: If eval_samples is invalid or non-streaming data is too small.
    :return tuple[Dataset, Dataset]: ``(remaining_train, eval_dataset)``.
    """
    if eval_samples <= 0:
        raise ValueError(f"eval_samples must be > 0, got {eval_samples}.")

    can_stream_split = (
        is_streaming
        and hasattr(train_dataset, "take")
        and hasattr(train_dataset, "skip")
    )
    if can_stream_split:
        eval_dataset = train_dataset.take(eval_samples)
        remaining_train = train_dataset.skip(eval_samples)
        return remaining_train, eval_dataset

    total_samples = len(train_dataset)
    if total_samples <= 1:
        raise ValueError(
            "dataset.eval_samples requires at least 2 non-streaming samples to keep "
            "both train and eval non-empty."
        )
    eval_count = min(eval_samples, total_samples - 1)
    if eval_count < eval_samples:
        logger.warning(
            "Requested dataset.eval_samples="
            f"{eval_samples} but dataset has {total_samples} rows; using "
            f"eval_samples={eval_count} to keep at least one training sample."
        )
    eval_dataset = train_dataset.select(range(eval_count))
    remaining_train = train_dataset.select(range(eval_count, total_samples))
    return remaining_train, eval_dataset


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
            "Set default worker env: "
            + ", ".join(f"{k}={v}" for k, v in sorted(applied.items()))
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


def _resolve_streaming_eval_budget(
    eval_max_batches: Any,
    eval_samples: Optional[int],
    per_device_eval_batch_size: int,
) -> tuple[int, str]:
    """Resolve explicit streaming eval budget for comparable metrics.

    :param Any eval_max_batches: Optional ``trainer.eval_max_batches`` value.
    :param int | None eval_samples: Optional ``dataset.eval_samples`` value.
    :param int per_device_eval_batch_size: Eval batch size for sample->batch conversion.
    :raises ValueError: If no explicit budget is available for streaming eval.
    :return tuple[int, str]: Resolved batch cap and source key.
    """
    if eval_max_batches is not None:
        try:
            resolved_eval_max_batches = int(eval_max_batches)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trainer.eval_max_batches must be an integer when set, got "
                f"{eval_max_batches!r}."
            ) from exc
        if resolved_eval_max_batches > 0:
            return resolved_eval_max_batches, "trainer.eval_max_batches"

    if eval_samples is None:
        raise ValueError(
            "Streaming eval requires an explicit evaluation budget for reproducible "
            "metrics. Set trainer.eval_max_batches or dataset.eval_samples."
        )

    eval_batch_size = max(1, int(per_device_eval_batch_size))
    return max(1, math.ceil(eval_samples / eval_batch_size)), "dataset.eval_samples"


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
                    with _gather_decoder_weight_for_masked_objective(
                        model, accelerator
                    ) as lm_weight:
                        objective_out = masked_objective(
                            hidden_states=hidden,
                            labels=batch["labels"],
                            lm_weight=lm_weight,
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


def _compute_weight_norm_for_logging(
    model: torch.nn.Module,
    accelerator: Accelerator,
) -> Optional[float]:
    """Compute model weight norm for logging across distributed backends.

    DeepSpeed helper ``safe_get_full_fp32_param`` can return ``None`` (for
    example on params without ZeRO/FP32 mappings), so we skip those tensors
    rather than failing during logging.

    :param torch.nn.Module model: Training model (possibly wrapped).
    :param Accelerator accelerator: Active accelerator runtime.
    :return float | None: L2 weight norm or ``None`` when unavailable.
    """
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        squared_norms: list[torch.Tensor] = []
        for param in model.parameters():
            full_param = safe_get_full_fp32_param(param)
            if full_param is None:
                continue
            squared_norms.append(full_param.norm(2) ** 2)
        if not squared_norms:
            return None
        return (sum(squared_norms) ** 0.5).item()

    return (sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5).item()


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
                f"Stored batch key '{key}' has incompatible types: "
                f"{type(existing).__name__} vs {type(value).__name__}"
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
    """Prepare a skipped dataloader for resume.

    :param torch.utils.data.DataLoader train_dataloader: Training dataloader.
    :param Metrics metrics: Metrics tracker with resumed counters.
    :param Accelerator accelerator: Accelerator instance.
    :param bool is_streaming: Whether the dataset is streaming.
    :return torch.utils.data.DataLoader | None: Skipped dataloader or ``None``.
    """
    completed_batches = int(metrics.get("train/batches", 0))
    if completed_batches <= 0:
        return None

    loader_len: Optional[int] = None
    if hasattr(train_dataloader, "__len__"):
        try:
            loader_len = len(train_dataloader)
        except TypeError:
            loader_len = None

    if hasattr(train_dataloader, "set_epoch"):
        train_dataloader.set_epoch(int(metrics.get("train/epochs", 0)))

    if loader_len is not None:
        if loader_len <= 0:
            logger.warning(
                "Resume requested but dataloader length is non-positive; "
                "starting from the current epoch boundary."
            )
            return None
        resume_step = completed_batches % loader_len
        if is_streaming:
            logger.info(
                "Streaming resume: skipping "
                f"{resume_step} batch(es) within current epoch "
                f"(consumed={completed_batches}, loader_len={loader_len})."
            )
    else:
        if not is_streaming:
            logger.warning(
                "Resume requested but dataloader has no length; "
                "starting from the current epoch boundary."
            )
            return None
        # Streaming fallback: we cannot derive a per-epoch modulo without a
        # dataloader length, so skip the consumed batch count from stream start.
        resume_step = completed_batches
        logger.warning(
            "Streaming resume with unknown dataloader length: skipping "
            f"{resume_step} consumed batch(es) from stream start "
            "(best-effort position recovery)."
        )

    if resume_step == 0:
        return None

    return accelerator.skip_first_batches(train_dataloader, resume_step)


def _safe_len(dataloader: torch.utils.data.DataLoader) -> Optional[int]:
    """Return dataloader length when available, else ``None``.

    :param torch.utils.data.DataLoader dataloader: Dataloader object.
    :return int | None: Length, or ``None`` when not defined.
    """
    if not hasattr(dataloader, "__len__"):
        return None
    try:
        return len(dataloader)
    except TypeError:
        return None


def _resolve_checkpoint_retention_limit(cfg: Config) -> int:
    """Resolve effective checkpoint retention limit from trainer config.

    ``trainer.save_total_limit`` is the canonical knob. Deprecated
    ``trainer.max_ckpt`` is only used as a fallback when save_total_limit is unset.

    :param Config cfg: Runtime training configuration.
    :return int: Maximum number of retained checkpoints (0 disables pruning).
    """
    save_total_limit = getattr(cfg.trainer, "save_total_limit", None)
    if save_total_limit is not None:
        return max(0, int(save_total_limit))
    max_ckpt = getattr(cfg.trainer, "max_ckpt", None)
    if max_ckpt is not None:
        return max(0, int(max_ckpt))
    return 0


def _save_portable_checkpoint_weights(
    model: torch.nn.Module,
    accelerator: Accelerator,
    checkpoint_path: Path,
) -> bool:
    """Save backend-agnostic ``model.safetensors`` into a step checkpoint.

    The resumable Accelerate state remains the source of truth for restart; this
    helper adds a portable inference/export payload for downstream consumers.

    :param torch.nn.Module model: Prepared training model.
    :param Accelerator accelerator: Active accelerator runtime.
    :param Path checkpoint_path: Step checkpoint directory path.
    :return bool: True when portable weights were saved.
    """
    if not accelerator.is_main_process:
        return False

    try:
        state_dict = accelerator.get_state_dict(model, unwrap=True)
    except Exception as exc:
        logger.warning(
            "Unable to export portable checkpoint weights to "
            f"{checkpoint_path / MODEL_WEIGHTS_NAME}: {exc}. "
            "Resumable state was still saved. For DeepSpeed ZeRO-3, enable "
            "`stage3_gather_16bit_weights_on_model_save`/`zero3_save_16bit_model` "
            "to emit consolidated portable weights automatically."
        )
        return False

    try:
        weights_path = save_state_dict_safetensors(
            state_dict,
            checkpoint_path,
            metadata={"format": "pt", "source": "accelerate.get_state_dict"},
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist portable checkpoint weights at "
            f"{checkpoint_path / MODEL_WEIGHTS_NAME}: {exc}. "
            "Resumable state was still saved."
        )
        return False

    logger.info(f"Saved portable model weights to {weights_path}.")
    return True


def trainer(cfg: Config) -> None:
    """Run the pretraining loop.

    :param Config cfg: Training configuration.
    """
    masked_logits_only_loss = _resolve_masked_logits_only_loss(
        getattr(cfg.trainer, "masked_logits_only_loss", True)
    )
    eval_samples = _resolve_eval_samples(getattr(cfg.dataset, "eval_samples", None))

    # Checkpoint layout (BREAKING): all resumable/exportable artifacts are written
    # under output_dir/checkpoints/<step>/.
    output_dir = Path(cfg.trainer.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_retention_limit = _resolve_checkpoint_retention_limit(cfg)
    resume_checkpoint_path, iteration = _resolve_resume_checkpoint(
        cfg.trainer.resume_from_checkpoint,
        str(checkpoint_dir),
        str(output_dir),
    )

    # Accelerator object - disable automatic checkpoint naming so the trainer can
    # control a single checkpoint tree (checkpoints/<step>).
    project_config = ProjectConfiguration(
        str(output_dir),
        automatic_checkpoint_naming=False,
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

    mixed_precision = resolve_mixed_precision(
        cfg.trainer.mixed_precision,
        task="pretraining",
    )
    cfg.trainer.mixed_precision = mixed_precision

    accelerator = create_accelerator(
        use_cpu=bool(getattr(cfg.trainer, "use_cpu", False)),
        log=logger,
        accelerator_factory=Accelerator,
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
    if getattr(accelerator, "is_main_process", True):
        logger.info(
            "Pretraining checkpoint layout: writing resumable + export artifacts to "
            f"{checkpoint_dir}/<step>/"
        )
        if checkpoint_retention_limit > 0:
            logger.info(
                "Checkpoint retention policy: keep "
                f"{checkpoint_retention_limit} latest checkpoint(s)."
            )
        else:
            logger.info("Checkpoint retention policy: disabled (keep all checkpoints).")
    if accelerator.distributed_type is DistributedType.FSDP:
        fsdp_version = _resolve_fsdp_version(accelerator)
        if fsdp_version < 2:
            raise RuntimeError(
                "NeoBERT pretraining is FSDP2-first. "
                "FSDP v1 is unsupported; set Accelerate fsdp_version=2."
            )
        logger.info(f"FSDP2 runtime detected (fsdp_version={fsdp_version}).")
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
    tracker_config_dict = prepare_wandb_config(cfg)
    if wandb_enabled:
        Path(cfg.wandb.dir).mkdir(parents=True, exist_ok=True)
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
        if accelerator.is_main_process and wandb.run is not None:
            wandb.run.config.update(tracker_config_dict, allow_val_change=True)
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
                        f"Configured config_path '{config_path}' not found; "
                        "skipping wandb artifact upload"
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
    _log_masking_strategy(cfg)

    # Tokenizer
    with accelerator.main_process_first():
        tokenizer = get_tokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.path or cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            trust_remote_code=cfg.tokenizer.trust_remote_code,
            revision=cfg.tokenizer.revision,
            allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
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
            "Config vocab_size updated: tokenizer len="
            f"{original_vocab_size} -> {resolved_vocab_size} "
            f"(was model={prior_model_vocab_size})."
        )
    if accelerator.is_main_process and added_tokens > 0:
        logger.info(
            f"Added {added_tokens} inert tokenizer tokens to align "
            f"tokenizer/model vocab_size={resolved_vocab_size}."
        )

    resolved_config_dict = prepare_wandb_config(cfg)
    if accelerator.is_main_process:
        accelerator.print(
            "Resolved task config:\n" + format_resolved_config(resolved_config_dict)
        )
    if accelerator.is_main_process and wandb_enabled and wandb.run is not None:
        wandb.run.config.update(resolved_config_dict, allow_val_change=True)

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
                f"Dataset path {dataset_path} not found; falling back to load_dataset()."
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
                        truncation=cfg.tokenizer.truncation,
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
                    truncation=cfg.tokenizer.truncation,
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

    eval_split = cfg.dataset.eval_split
    if isinstance(eval_split, str):
        eval_split = eval_split.strip() or None
    if (
        eval_split is None
        and cfg.dataset.streaming
        and not (cfg.dataset.path and Path(cfg.dataset.path).exists())
    ):
        inferred_eval_split = _infer_eval_split_name(
            cfg.dataset.name,
            dataset_kwargs,
            train_split=cfg.dataset.train_split,
        )
        if inferred_eval_split is not None:
            eval_split = inferred_eval_split
            logger.info(
                "Auto-detected streaming eval split "
                f"'{eval_split}'. Override with dataset.eval_split when needed."
            )

    eval_dataset = None
    if eval_split:
        if cfg.dataset.path and Path(cfg.dataset.path).exists():
            eval_source = load_from_disk(cfg.dataset.path)
            if isinstance(eval_source, DatasetDict):
                eval_dataset = eval_source[eval_split]
            else:
                logger.warning(
                    f"eval_split={eval_split} requested but dataset path is not a "
                    "DatasetDict; skipping evaluation."
                )
        else:
            if cfg.dataset.streaming:
                eval_dataset = _load_streaming_split(
                    cfg.dataset.name,
                    eval_split,
                    dataset_kwargs,
                )
            else:
                eval_dataset = load_dataset(
                    cfg.dataset.name,
                    split=eval_split,
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

    if eval_dataset is None and eval_samples is not None:
        train_dataset, eval_dataset = _split_train_dataset_for_eval_samples(
            train_dataset,
            eval_samples,
            is_streaming=is_streaming,
        )
        logger.info(
            "dataset.eval_samples="
            f"{eval_samples} with no eval split configured; reserving head samples "
            "for eval and excluding them from training."
        )

    if eval_dataset is not None and eval_samples is not None:
        eval_dataset_is_streaming = (
            cfg.dataset.streaming
            and hasattr(eval_dataset, "take")
            and hasattr(eval_dataset, "skip")
        )
        if eval_dataset_is_streaming:
            eval_dataset = eval_dataset.take(eval_samples)
        else:
            eval_size = len(eval_dataset)
            eval_dataset = eval_dataset.select(range(min(eval_samples, eval_size)))

    if eval_dataset is not None:
        eval_is_streaming = (
            cfg.dataset.streaming
            and hasattr(eval_dataset, "take")
            and hasattr(eval_dataset, "skip")
        )
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
                    truncation=cfg.tokenizer.truncation,
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

    train_loader_len = _safe_len(train_dataloader)
    if train_loader_len is not None and train_loader_len == 0:
        raise ValueError(
            "Training dataloader resolved to zero batches. Check dataset filtering "
            "(for example dataset.min_length and split settings) and ensure "
            "tokenization produced non-empty inputs."
        )
    if eval_dataloader is not None:
        eval_loader_len = _safe_len(eval_dataloader)
        if eval_loader_len is not None and eval_loader_len == 0:
            logger.warning(
                "Eval dataloader resolved to zero batches; disabling evaluation for "
                "this run."
            )
            eval_dataloader = None

    if wandb_enabled and accelerator.is_main_process:
        watch_mode, watch_warning = resolve_wandb_watch_mode(
            wandb_mode=cfg.wandb.mode,
            env_value=os.environ.get("WANDB_WATCH"),
        )
        if watch_warning:
            logger.warning(watch_warning)
        if watch_mode:
            watch_log_freq = max(1, int(getattr(cfg.trainer, "logging_steps", 100)))
            logger.info(
                f"Enabling wandb.watch(log={watch_mode}, log_freq={watch_log_freq})."
            )
            wandb.watch(
                accelerator.unwrap_model(model),
                log=watch_mode,
                log_freq=watch_log_freq,
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
    eval_dataset_is_streaming = (
        eval_dataset is not None
        and cfg.dataset.streaming
        and hasattr(eval_dataset, "take")
        and hasattr(eval_dataset, "skip")
    )
    if eval_dataset_is_streaming:
        eval_max_batches, eval_budget_source = _resolve_streaming_eval_budget(
            eval_max_batches=eval_max_batches,
            eval_samples=eval_samples,
            per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        )
        logger.info(
            "Streaming eval budget resolved from "
            f"{eval_budget_source}: eval_max_batches={eval_max_batches} "
            "(global batches across all ranks)."
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
        if is_streaming and hasattr(train_dataset, "set_epoch"):
            resume_epoch = int(metrics.get("train/epochs", 0))
            train_dataset.set_epoch(resume_epoch)
            logger.info(
                "Restored streaming dataset epoch to "
                f"{resume_epoch} before resume skipping."
            )
        skipped_train_dataloader = _prepare_resume_dataloader(
            train_dataloader, metrics, accelerator, is_streaming
        )
    elif cfg.trainer.resume_from_checkpoint:
        logger.warning(
            "resume_from_checkpoint is set but no valid checkpoints were found in "
            f"{checkpoint_dir}"
        )

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
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
        saw_batch_this_epoch = False
        for batch in dataloader:
            saw_batch_this_epoch = True
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
                backward_called = False
                with accelerator.autocast():
                    if masked_objective is not None:
                        objective_out, loss_sum, backward_called = (
                            _run_masked_objective_step(
                                model,
                                batch,
                                pad_mask,
                                packed_seqlens,
                                masked_objective,
                                accelerator,
                                log_train_accuracy=log_train_accuracy,
                            )
                        )
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
                            logger.debug(
                                "Masked-logits loss path active (first non-empty "
                                f"microbatch): {objective_out.used_path}"
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
                if not backward_called:
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
                    min_safe_tokens = (
                        accelerator.num_processes
                        * accelerator.gradient_accumulation_steps
                    )
                    logger.warning(
                        "Masked-token count was below the safe minimum for an update "
                        f"(tokens_global={int(tokens_global.item())}, "
                        f"min={min_safe_tokens}); clamped gradient scale to avoid "
                        "pathological amplification."
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
                        weight_norm_value = _compute_weight_norm_for_logging(
                            model, accelerator
                        )
                        if weight_norm_value is not None:
                            metrics["train/weight_norm"] = weight_norm_value
                        else:
                            metrics.pop("train/weight_norm", None)

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
                    if log_train_accuracy:
                        metrics["train/local_num_correct"] = int(
                            local_num_correct.item()
                        )
                    else:
                        metrics.pop("train/local_num_correct", None)
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
                        if accelerator.is_main_process:
                            steps_liger = int(loss_path_counts[0].item())
                            steps_checkpointed = int(loss_path_counts[1].item())
                            steps_zero = int(loss_path_counts[2].item())
                            steps_other = int(loss_path_counts[3].item())
                            if path_total > 0:
                                logger.debug(
                                    "Masked-loss path window: "
                                    f"liger_flce={steps_liger} "
                                    f"({steps_liger / path_total:.3f}), "
                                    f"checkpointed={steps_checkpointed} "
                                    f"({steps_checkpointed / path_total:.3f}), "
                                    f"zero_masked={steps_zero} "
                                    f"({steps_zero / path_total:.3f}), "
                                    f"other={steps_other} "
                                    f"({steps_other / path_total:.3f})"
                                )
                            else:
                                logger.debug(
                                    "Masked-loss path window: no non-empty microbatches."
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

                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    step_tag = str(metrics["train/steps"])
                    checkpoint_path = checkpoint_dir / step_tag
                    accelerator.save_state(output_dir=str(checkpoint_path))
                    accelerator.wait_for_everyone()
                    _save_portable_checkpoint_weights(
                        model, accelerator, checkpoint_path
                    )
                    accelerator.wait_for_everyone()

                    # Save export metadata alongside resumable state.
                    if accelerator.is_main_process:
                        config_path = checkpoint_path / "config.yaml"
                        ConfigLoader.save(cfg, str(config_path))

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

                        tokenizer_dir = checkpoint_path / "tokenizer"
                        tokenizer_dir.mkdir(parents=True, exist_ok=True)
                        tokenizer.model_max_length = cfg.model.max_position_embeddings
                        tokenizer.save_pretrained(tokenizer_dir)
                        logger.info(
                            "Saved checkpoint to "
                            f"{checkpoint_path} (includes Accelerate state, model "
                            "weights, config, and tokenizer artifacts)."
                        )

                    accelerator.wait_for_everyone()

                    if (
                        checkpoint_retention_limit > 0
                        and checkpoint_dir.exists()
                        and accelerator.is_main_process
                    ):
                        checkpoints = []
                        for item_path in checkpoint_dir.iterdir():
                            if item_path.is_dir() and item_path.name.isdigit():
                                checkpoints.append(int(item_path.name))

                        if len(checkpoints) > checkpoint_retention_limit:
                            checkpoints.sort()
                            for old_ckpt in checkpoints[
                                : len(checkpoints) - checkpoint_retention_limit
                            ]:
                                old_path = checkpoint_dir / str(old_ckpt)
                                if old_path.exists():
                                    shutil.rmtree(old_path)
                                    logger.info(
                                        "Removed old checkpoint: "
                                        f"{old_path} (limit={checkpoint_retention_limit})"
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

        if not saw_batch_this_epoch:
            raise RuntimeError(
                "Training dataloader yielded zero batches for an epoch. This usually "
                "means the active split is empty after preprocessing/filtering."
            )

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
