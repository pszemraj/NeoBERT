"""Pretraining loop for masked language modeling."""

import json
import math
import logging
import os
import re
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

# PyTorch
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    ProjectConfiguration,
    set_seed,
)

# Hugging Face
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase

from ..config import Config, ConfigLoader, MuonConfig, round_up_to_multiple
from ..dataloader import get_dataloader
from ..model import NeoBERTConfig, NeoBERTLMHead
from ..model.model import XFORMERS_AVAILABLE, XFORMERS_ERROR
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler, resolve_scheduler_steps
from ..tokenizer import get_tokenizer, resolve_text_column
from ..utils import configure_tf32, model_summary, prepare_wandb_config

# Our metric object and model
from .metrics import Metrics

# Set up logger
logger = logging.getLogger(__name__)


def _count_masked_correct(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Count correct predictions while ignoring masked labels.

    :param torch.Tensor logits: Logits of shape ``[batch, seq_len, vocab]``.
    :param torch.Tensor labels: Label IDs of shape ``[batch, seq_len]``.
    :param int ignore_index: Label value to ignore (default: -100).
    :return torch.Tensor: Scalar tensor of correct predictions on unmasked tokens.
    """
    mask = labels != ignore_index
    if not mask.any():
        return torch.zeros((), device=logits.device, dtype=torch.long)
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).sum()


def _run_eval(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accelerator: Accelerator,
    model_config: NeoBERTConfig,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Run a lightweight evaluation loop for masked LM perplexity.

    :param torch.nn.Module model: Model to evaluate.
    :param torch.utils.data.DataLoader eval_dataloader: Evaluation dataloader.
    :param torch.nn.Module loss_fn: Loss function (sum reduction).
    :param Accelerator accelerator: Accelerator for distributed reductions.
    :param NeoBERTConfig model_config: Model config with vocab size.
    :param int | None max_batches: Optional cap on eval batches.
    :return dict[str, float]: Evaluation metrics for logging.
    """
    was_training = model.training
    model.eval()

    eval_loss_sum = torch.zeros((), device=accelerator.device)
    eval_num_pred = torch.zeros((), device=accelerator.device)
    eval_num_correct = torch.zeros((), device=accelerator.device)
    eval_batches = 0

    try:
        with torch.no_grad():
            for batch in eval_dataloader:
                if max_batches is not None and eval_batches >= max_batches:
                    break
                packed_seqlens = batch.get("packed_seqlens")
                pad_mask = (
                    None
                    if packed_seqlens is not None
                    else batch.get("attention_mask", None)
                )
                logits = model(
                    batch["input_ids"],
                    pad_mask,
                    packed_seqlens=packed_seqlens,
                )["logits"]
                loss_sum = loss_fn(
                    logits.view(-1, model_config.vocab_size), batch["labels"].view(-1)
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
        except (AttributeError, RuntimeError):
            for grad in grads:
                grad.mul_(scale_value)


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


def _resolve_resume_checkpoint(
    resume_from_checkpoint: Optional[str],
    checkpoint_dir: str,
    output_dir: str,
) -> Tuple[Optional[str], int]:
    """Resolve an explicit or latest checkpoint path for resuming.

    :param str | None resume_from_checkpoint: Configured resume value.
    :param str checkpoint_dir: Default checkpoint directory to scan for latest.
    :param str output_dir: Output directory for relative path resolution.
    :return tuple[str | None, int]: Resolved checkpoint path and iteration.
    """
    if not resume_from_checkpoint:
        return None, 0

    if isinstance(resume_from_checkpoint, str):
        resume_value = resume_from_checkpoint.strip()
        if resume_value.lower() not in {"true", "latest", "auto"}:
            resume_path = resume_value
            if not os.path.isabs(resume_path):
                candidate = os.path.join(output_dir, resume_path)
                if os.path.exists(candidate):
                    resume_path = candidate
            base = os.path.basename(os.path.normpath(resume_path))
            iteration = int(base) + 1 if base.isdigit() else 0
            return resume_path, iteration

    if not (os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir)):
        return None, 0

    folders = [
        folder
        for folder in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, folder)) and folder.isdigit()
    ]
    if not folders:
        return None, 0

    latest_step = max(
        int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0]) for folder in folders
    )
    return os.path.join(checkpoint_dir, str(latest_step)), latest_step + 1


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
        requested = len(os.sched_getaffinity(0))
    if num_processes > 1:
        requested = max(1, requested // num_processes)
        if not is_main_process:
            requested = 1
    return max(1, requested)


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
    batch_size = batch["input_ids"].shape[0]

    # If the batch is too large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            value = batch[key]
            if torch.is_tensor(value):
                tmp[key] = torch.split(
                    value, [target_size, batch_size - target_size], dim=0
                )
                batch[key] = tmp[key][0]
                if stored_batch[key] is None:
                    stored_batch[key] = tmp[key][1]
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
                    stored_batch[key] = stored_batch[key] + leftover

    # If the batch is too small, we had some stored_batch
    elif batch_size < target_size:
        if stored_batch["input_ids"] is None:
            return batch, stored_batch
        # We have already enough samples stored
        if stored_batch["input_ids"].shape[0] >= target_size - batch_size:
            for key in batch.keys():
                if stored_batch[key] is None:
                    continue
                if (
                    torch.is_tensor(stored_batch[key])
                    and stored_batch[key].device != batch[key].device
                ):
                    stored_batch[key] = stored_batch[key].to(batch[key].device)
            for key in batch.keys():
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
                    batch[key] = batch[key] + stored_batch[key][:take]
                    stored_batch[key] = stored_batch[key][take:]

        # Concatenate otherwise
        else:
            for key in batch.keys():
                if stored_batch[key] is None:
                    continue
                if torch.is_tensor(stored_batch[key]):
                    if stored_batch[key].device != batch[key].device:
                        stored_batch[key] = stored_batch[key].to(batch[key].device)
                    batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                else:
                    batch[key] = batch[key] + stored_batch[key]
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
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.output_dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.output_dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    resume_checkpoint_path, iteration = _resolve_resume_checkpoint(
        cfg.trainer.resume_from_checkpoint,
        checkpoint_dir,
        cfg.trainer.output_dir,
    )

    # Accelerator object - disable automatic checkpointing to avoid duplicate checkpoints/ directory
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=False,  # We handle checkpointing manually in model_checkpoints/
        iteration=iteration,
    )
    # All parameters participate in the forward graph; keep DDP in fast-path mode.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    wandb_enabled = cfg.wandb.enabled and cfg.wandb.mode != "disabled"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if wandb_enabled else None,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    if wandb_enabled:
        os.makedirs(cfg.wandb.dir, exist_ok=True)
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
                abs_config_path = os.path.abspath(config_path)
                if os.path.isfile(abs_config_path):
                    artifact = wandb.Artifact(
                        name=f"{wandb.run.id}-config",
                        type="config",
                        metadata={"source": abs_config_path},
                    )
                    artifact.add_file(abs_config_path)
                    wandb.run.log_artifact(artifact)
                else:
                    logger.warning(
                        "Configured config_path '%s' not found; skipping wandb artifact upload",
                        config_path,
                    )

    # Set the seed
    set_seed(cfg.seed)

    # Configure TF32 precision for GPUs with compute capability >= 8.0
    configure_tf32(enabled=cfg.trainer.tf32, print_fn=accelerator.print)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)
    log_interval = max(1, cfg.trainer.logging_steps)

    is_streaming = cfg.dataset.streaming
    if cfg.trainer.resume_from_checkpoint and is_streaming:
        raise ValueError(
            "Cannot resume training with streaming datasets - data position is not "
            "preserved. For resumable long runs, pre-tokenize your dataset:\n"
            "  python scripts/pretraining/tokenize_dataset.py --dataset <name> --output <path>"
        )

    if cfg.datacollator.pack_sequences:
        logger.info(
            "Using packed sequences with xFormers block-diagonal attention (experimental)."
        )
        if not cfg.model.xformers_attention:
            raise ValueError(
                "Packed sequences require model.xformers_attention=true (xFormers)."
            )
        if not XFORMERS_AVAILABLE:
            raise ImportError(
                "Packed sequences require xformers. Install with: pip install xformers. "
                f"Import error: {XFORMERS_ERROR}"
            )

    # Tokenizer
    with accelerator.main_process_first():
        tokenizer = get_tokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.path or cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
        )

    actual_vocab_size = len(tokenizer)
    rounded_vocab_size = round_up_to_multiple(actual_vocab_size, 128)
    if (
        cfg.model.vocab_size != rounded_vocab_size
        or cfg.tokenizer.vocab_size != rounded_vocab_size
    ):
        if accelerator.is_main_process:
            logger.warning(
                "Config vocab_size updated: tokenizer=%s rounded to %s (was model=%s).",
                actual_vocab_size,
                rounded_vocab_size,
                cfg.model.vocab_size,
            )
    cfg.model.vocab_size = rounded_vocab_size
    cfg.tokenizer.vocab_size = rounded_vocab_size

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
    dataset_kwargs = {"name": cfg.dataset.config} if cfg.dataset.config else {}

    if cfg.dataset.path:
        train_dataset = load_from_disk(cfg.dataset.path)
    else:
        # Parse split if it contains slice notation (e.g., "train[:1000]")
        if cfg.dataset.train_split and "[" in cfg.dataset.train_split:
            dataset = load_dataset(
                cfg.dataset.name,
                split=cfg.dataset.train_split,
                streaming=cfg.dataset.streaming,
                **dataset_kwargs,
            )
            train_dataset = dataset
        else:
            dataset = load_dataset(
                cfg.dataset.name,
                streaming=cfg.dataset.streaming,
                **dataset_kwargs,
            )
            train_dataset = (
                dataset[cfg.dataset.train_split]
                if cfg.dataset.train_split
                else dataset["train"]
            )

    # Check if dataset needs tokenization
    # For streaming datasets, we need to check differently
    needs_tokenization = False

    if train_dataset:
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
            import subprocess
            from pathlib import Path

            # Create output directory
            if cfg.dataset.pre_tokenize_output:
                output_dir = cfg.dataset.pre_tokenize_output
            else:
                output_dir = f"tokenized_data/{cfg.dataset.name.replace('/', '_')}"

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            success_flag = Path(output_dir) / ".tokenize_complete"
            failure_flag = Path(output_dir) / ".tokenize_failed"

            # Run tokenization script
            accelerator.print(f"Pre-tokenizing dataset to: {output_dir}")

            # Get absolute path to script
            repo_root = Path(__file__).resolve().parents[3]
            script_path = repo_root / "scripts" / "pretraining" / "tokenize_dataset.py"

            if accelerator.is_main_process and not success_flag.exists():
                cmd = [
                    "python",
                    str(script_path),
                    "--dataset",
                    cfg.dataset.name,
                    "--tokenizer",
                    cfg.tokenizer.path or cfg.tokenizer.name,
                    "--output",
                    output_dir,
                    "--max-length",
                    str(tokenize_max_length),
                ]

                if cfg.dataset.config:
                    cmd.extend(["--dataset-config", cfg.dataset.config])

                if cfg.dataset.train_split:
                    cmd.extend(["--split", cfg.dataset.train_split])

                if cfg.dataset.text_column:
                    cmd.extend(["--text-column", cfg.dataset.text_column])

                if cfg.dataset.num_proc:
                    cmd.extend(["--num-proc", str(cfg.dataset.num_proc)])
                if not add_special_tokens:
                    cmd.append("--no-special-tokens")
                if return_special_tokens_mask:
                    cmd.append("--return-special-tokens-mask")

                # Run the tokenization on the main process only.
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    failure_flag.write_text(result.stderr)
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
        if cfg.dataset.path:
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
            eval_dataset = load_dataset(
                cfg.dataset.name,
                split=cfg.dataset.eval_split,
                streaming=cfg.dataset.streaming,
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

    # Dataloader
    collator_max_length = cfg.datacollator.max_length or cfg.dataset.max_seq_length
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        batch_size=cfg.trainer.per_device_train_batch_size,
        num_workers=cfg.dataset.num_workers,
        mlm_probability=cfg.datacollator.mlm_probability,
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
        mask_all=cfg.datacollator.mask_all,
        pack_sequences=cfg.datacollator.pack_sequences,
        max_length=collator_max_length,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = get_dataloader(
            eval_dataset,
            tokenizer,
            batch_size=cfg.trainer.per_device_eval_batch_size,
            num_workers=cfg.dataset.num_workers,
            mlm_probability=cfg.datacollator.mlm_probability,
            pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
            mask_all=cfg.datacollator.mask_all,
            pack_sequences=cfg.datacollator.pack_sequences,
            max_length=collator_max_length,
            shuffle=False,
        )

    # Model
    # vocab_size is now resolved during config preprocessing
    # Debug print
    if cfg.debug:
        print(f"Config model.vocab_size: {cfg.model.vocab_size}")
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
        print(f"Tokenizer len(): {len(tokenizer)}")
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")

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
        pad_token_id=tokenizer.pad_token_id,
        flash_attention=cfg.model.xformers_attention,
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

    # Prepare with accelerate
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

    # Loss function
    # Note: logits are fully materialized; consider fused/chunked CE for very long contexts.
    train_loss_fn = CrossEntropyLoss(reduction="sum")
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
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(
                f"resume_from_checkpoint path not found: {resume_checkpoint_path}"
            )
        accelerator.load_state(resume_checkpoint_path)
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

    accum_tokens = torch.zeros((), device=accelerator.device)
    local_samples = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_tokens = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_num_pred = torch.zeros((), device=accelerator.device, dtype=torch.long)
    local_sum_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    local_num_correct = torch.zeros((), device=accelerator.device, dtype=torch.long)
    stored_batch = {
        "input_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seqlens": None,
    }
    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = (
            train_dataloader
            if skipped_train_dataloader is None
            else skipped_train_dataloader
        )
        for batch in dataloader:
            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            # Skip batch buffering when using packed sequences, as packed_seqlens metadata would be lost.
            is_packed = batch.get("packed_seqlens") is not None
            if batch["input_ids"].shape[0] != cfg.trainer.per_device_train_batch_size:
                if is_packed:
                    # Packed batches can't be split/merged; just process as-is (may be smaller at end of epoch)
                    pass
                else:
                    batch, stored_batch = to_target_batch_size(
                        batch, stored_batch, cfg.trainer.per_device_train_batch_size
                    )

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            # For packed sequences, we allow smaller final batches rather than buffering
            if (
                not is_packed
                and batch["input_ids"].shape[0]
                < cfg.trainer.per_device_train_batch_size
            ):
                stored_batch = batch
                continue

            # Update number of batches only when we will execute a backward pass.
            metrics["train/batches"] += 1

            num_pred = (batch["labels"] != -100).sum()
            num_tokens = (batch["input_ids"] != model_config.pad_token_id).sum()
            packed_seqlens = batch.get("packed_seqlens")
            pad_mask = (
                None
                if packed_seqlens is not None
                else batch.get("attention_mask", None)
            )

            sync_gradients = (
                metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps == 0
            )
            context = nullcontext() if sync_gradients else accelerator.no_sync(model)
            with context:
                # Forward pass
                logits = model(
                    batch["input_ids"],
                    pad_mask,
                    packed_seqlens=packed_seqlens,
                )["logits"]
                loss_sum = train_loss_fn(
                    logits.view(-1, model_config.vocab_size), batch["labels"].view(-1)
                )

                # Compute gradient
                accelerator.backward(loss_sum)
                accum_tokens += num_pred.to(accum_tokens.dtype)

                # Accumulate metrics on device to avoid per-batch syncs.
                local_samples += batch["input_ids"].shape[0]
                local_tokens += num_tokens
                local_num_pred += num_pred
                local_sum_loss += loss_sum.detach().float()
                local_num_correct += _count_masked_correct(logits, batch["labels"])

            if sync_gradients:
                # Reduce to global token count to handle uneven sharding across ranks.
                tokens_global = accelerator.reduce(accum_tokens, reduction="sum")
                if tokens_global.item() > 0:
                    # Match full-batch normalization across variable-length microbatches.
                    # accelerator.backward() already divides by grad_accumulation_steps, and DDP averages
                    # across processes on the sync step, so we rescale by
                    # (num_processes * grad_accum_steps) / tokens_global
                    # to recover per-token mean gradients for the global batch size.
                    # This post-accumulation rescale is equivalent to scaling each microbatch loss
                    # because gradients are linear in the loss scalar.
                    # Ref: Unsloth blog (archived) https://archive.ph/RmO0U
                    scale = (
                        accelerator.num_processes
                        * accelerator.gradient_accumulation_steps
                    ) / tokens_global.float()
                    _scale_gradients(model, scale)
                else:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()

                # Measure gradient norm prior to optional clipping so we always log it
                grad_norm_value = None
                if accelerator.distributed_type is DistributedType.DEEPSPEED:
                    get_global_grad = getattr(model, "get_global_grad_norm", None)
                    if callable(get_global_grad):
                        grad_norm = get_global_grad()
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm_value = float(grad_norm.item())
                        elif grad_norm is not None:
                            grad_norm_value = float(grad_norm)
                else:
                    grad_norm_sq = None
                    for param in model.parameters():
                        if param.grad is None:
                            continue
                        param_norm = param.grad.norm(2)
                        grad_norm_sq = (
                            param_norm**2
                            if grad_norm_sq is None
                            else grad_norm_sq + param_norm**2
                        )
                    if grad_norm_sq is not None:
                        grad_norm_value = float(torch.sqrt(grad_norm_sq).item())

                max_grad_norm = cfg.trainer.gradient_clipping

                if max_grad_norm is not None and max_grad_norm > 0:
                    grad_norm_pre_clip = accelerator.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    if grad_norm_value is None and grad_norm_pre_clip is not None:
                        grad_norm_value = float(
                            grad_norm_pre_clip.item()
                            if isinstance(grad_norm_pre_clip, torch.Tensor)
                            else grad_norm_pre_clip
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
                    metrics.log(accelerator)
                    local_samples.zero_()
                    local_tokens.zero_()
                    local_num_pred.zero_()
                    local_sum_loss.zero_()
                    local_num_correct.zero_()

                # Save accelerator state for resumable training
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    state_checkpoint_path = os.path.join(
                        checkpoint_dir, str(metrics["train/steps"])
                    )
                    accelerator.save_state(output_dir=state_checkpoint_path)
                    accelerator.wait_for_everyone()

                    # Accelerator checkpoints are the source of truth for resuming.
                    save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                    limit = max(save_total_limit, max_ckpt)
                    if (
                        limit > 0
                        and os.path.exists(checkpoint_dir)
                        and accelerator.is_main_process
                    ):
                        # Prune accelerator checkpoints to the same retention policy
                        # as model_checkpoints to avoid unbounded disk growth.
                        accel_checkpoints = []
                        for item in os.listdir(checkpoint_dir):
                            item_path = os.path.join(checkpoint_dir, item)
                            if os.path.isdir(item_path) and item.isdigit():
                                accel_checkpoints.append(int(item))
                        if len(accel_checkpoints) > limit:
                            accel_checkpoints.sort()
                            for old_ckpt in accel_checkpoints[
                                : len(accel_checkpoints) - limit
                            ]:
                                old_path = os.path.join(checkpoint_dir, str(old_ckpt))
                                if os.path.exists(old_path):
                                    import shutil

                                    shutil.rmtree(old_path)
                                    logger.info(
                                        "Removed old accelerator checkpoint: %s (limit=%s)",
                                        old_path,
                                        limit,
                                    )

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    # Model checkpoints are used for inference/export and can be pruned independently.
                    # Save the checkpoint
                    step_tag = str(metrics["train/steps"])
                    tmp_tag = f"{step_tag}.tmp"
                    tmp_path = os.path.join(model_checkpoint_dir, tmp_tag)

                    # Clean up any stale tmp directory from a previous failed save.
                    # Use exist_ok pattern to avoid TOCTOU race conditions in distributed setting.
                    if accelerator.is_main_process:
                        import shutil

                        if os.path.exists(tmp_path):
                            shutil.rmtree(tmp_path)
                        os.makedirs(tmp_path, exist_ok=True)
                    accelerator.wait_for_everyone()

                    checkpoint_path = tmp_path
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        # DeepSpeed checkpoints are not directly portable; use zero_to_fp32 to export.
                        # DeepSpeed writes into model_checkpoint_dir/tmp_tag, which we later rename
                        # atomically alongside config/tokenizer for a single cohesive checkpoint.
                        model.save_checkpoint(model_checkpoint_dir, tag=tmp_tag)
                    else:
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            os.path.join(checkpoint_path, "state_dict.pt"),
                        )

                    # Save config and tokenizer info (only from main process)
                    if accelerator.is_main_process:
                        # Save config as YAML
                        config_path = os.path.join(checkpoint_path, "config.yaml")
                        ConfigLoader.save(cfg, config_path)

                        # Save tokenizer info as JSON
                        tokenizer_info = {
                            "tokenizer_name": cfg.tokenizer.path or cfg.tokenizer.name,
                            "vocab_size": cfg.model.vocab_size,
                            "pad_token_id": tokenizer.pad_token_id,
                        }
                        tokenizer_info_path = os.path.join(
                            checkpoint_path, "tokenizer_info.json"
                        )
                        with open(tokenizer_info_path, "w") as f:
                            json.dump(tokenizer_info, f, indent=2)

                        # Save full tokenizer with save_pretrained
                        tokenizer_dir = os.path.join(checkpoint_path, "tokenizer")
                        os.makedirs(tokenizer_dir, exist_ok=True)

                        # Ensure tokenizer.model_max_length matches model's max_position_embeddings
                        tokenizer.model_max_length = cfg.model.max_position_embeddings
                        tokenizer.save_pretrained(tokenizer_dir)

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        final_path = os.path.join(model_checkpoint_dir, step_tag)
                        if os.path.exists(final_path):
                            import shutil

                            shutil.rmtree(final_path)
                        os.replace(checkpoint_path, final_path)
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
                        and os.path.exists(model_checkpoint_dir)
                        and accelerator.is_main_process
                    ):
                        # Get all checkpoint directories
                        checkpoints = []
                        for item in os.listdir(model_checkpoint_dir):
                            item_path = os.path.join(model_checkpoint_dir, item)
                            if os.path.isdir(item_path) and item.isdigit():
                                checkpoints.append(int(item))

                        # Sort and remove oldest checkpoints if over limit
                        if len(checkpoints) > limit:
                            checkpoints.sort()
                            # Remove oldest checkpoints
                            for old_ckpt in checkpoints[: len(checkpoints) - limit]:
                                old_path = os.path.join(
                                    model_checkpoint_dir, str(old_ckpt)
                                )
                                if os.path.exists(old_path):
                                    import shutil

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
                    )
                    accelerator.log(eval_metrics, step=metrics["train/steps"])

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
            )
            accelerator.log(eval_metrics, step=metrics["train/steps"])

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
