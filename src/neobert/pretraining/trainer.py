"""Pretraining loop for masked language modeling."""

import json
import logging
import os
import re

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
from datasets import load_dataset, load_from_disk

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BatchEncoding

from ..config import Config, ConfigLoader, MuonConfig
from ..dataloader import get_dataloader
from ..model import NeoBERTConfig, NeoBERTLMHead
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tokenizer import get_tokenizer
from ..utils import configure_tf32, model_summary, prepare_wandb_config

# Our metric object and model
from .metrics import Metrics

# Set up logger
logger = logging.getLogger(__name__)


def _count_masked_correct(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> int:
    """Count correct predictions while ignoring masked labels.

    :param torch.Tensor logits: Logits of shape ``[batch, seq_len, vocab]``.
    :param torch.Tensor labels: Label IDs of shape ``[batch, seq_len]``.
    :param int ignore_index: Label value to ignore (default: -100).
    :return int: Number of correct predictions on unmasked tokens.
    """
    mask = labels != ignore_index
    if not mask.any():
        return 0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).sum().item()


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

    # If the batch is to large, we store samples
    if batch_size > target_size:
        for key in batch.keys():
            tmp[key] = torch.split(
                batch[key], [target_size, batch_size - target_size], dim=0
            )
            batch[key] = tmp[key][0]
            stored_batch[key] = (
                tmp[key][1]
                if stored_batch[key] is None
                else torch.cat([stored_batch[key], tmp[key][1]], dim=0)
            )

    # If the batch is too small, we had some stored_batch
    elif batch_size < target_size:
        if stored_batch["input_ids"] is None:
            return batch, stored_batch
        # We have already enough samples stored
        if stored_batch["input_ids"].shape[0] >= target_size - batch_size:
            for key in batch.keys():
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
                # Save on CPU to prevent full GPU memory
                stored_batch[key] = stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch


def trainer(cfg: Config) -> None:
    """Run the pretraining loop.

    :param Config cfg: Training configuration.
    """
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.output_dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.output_dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    iteration = 0
    resume_checkpoint_path = None

    if (
        cfg.trainer.resume_from_checkpoint
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        # This regular expression was taken from accelerator.load_state()
        folders = [
            folder
            for folder in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, folder)) and folder.isdigit()
        ]
        if folders:
            latest_step = max(
                int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0])
                for folder in folders
            )
            iteration = latest_step + 1
            resume_checkpoint_path = os.path.join(checkpoint_dir, str(latest_step))

    # Accelerator object - disable automatic checkpointing to avoid duplicate checkpoints/ directory
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=False,  # We handle checkpointing manually in model_checkpoints/
        iteration=iteration,
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.mode != "disabled" else None,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    # Initialise the wandb run and pass wandb parameters
    if cfg.wandb.mode != "disabled":
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
    configure_tf32(print_fn=accelerator.print)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    # Always use bf16 for mixed precision
    if accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    if cfg.datacollator.pack_sequences:
        logger.info(
            "Using packed sequences with block-diagonal attention masks (experimental)."
        )

    # Tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        vocab_size=cfg.tokenizer.vocab_size or cfg.model.vocab_size,
    )

    # Dataset
    if cfg.dataset.path:
        train_dataset = load_from_disk(cfg.dataset.path)
    else:
        # Parse split if it contains slice notation (e.g., "train[:1000]")
        if cfg.dataset.train_split and "[" in cfg.dataset.train_split:
            dataset = load_dataset(
                cfg.dataset.name,
                split=cfg.dataset.train_split,
                streaming=cfg.dataset.streaming,
            )
            train_dataset = dataset
        else:
            dataset = load_dataset(cfg.dataset.name, streaming=cfg.dataset.streaming)
            train_dataset = (
                dataset[cfg.dataset.train_split]
                if cfg.dataset.train_split
                else dataset["train"]
            )

    # Check if dataset needs tokenization
    # For streaming datasets, we need to check differently
    is_streaming = cfg.dataset.streaming
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

            # Run tokenization script
            accelerator.print(f"Pre-tokenizing dataset to: {output_dir}")

            # Get absolute path to script
            script_path = (
                Path(__file__).parent.parent.parent
                / "scripts"
                / "pretraining"
                / "tokenize_dataset.py"
            )

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
                str(cfg.dataset.max_seq_length),
            ]

            if cfg.dataset.train_split:
                cmd.extend(["--split", cfg.dataset.train_split])

            if cfg.dataset.num_proc:
                cmd.extend(["--num-proc", str(cfg.dataset.num_proc)])

            # Run the tokenization
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Tokenization failed: {result.stderr}")

            accelerator.print(f"Pre-tokenization complete. Loading from: {output_dir}")
            # Load the pre-tokenized dataset
            train_dataset = load_from_disk(output_dir)
        else:
            # Determine text column
            text_column = None
            if is_streaming:
                # For streaming, check the first example
                first_example = next(iter(train_dataset))
                for col in ["text", "sentence", "content"]:
                    if col in first_example:
                        text_column = col
                        break
                if text_column is None:
                    raise ValueError(
                        f"Could not find text column in dataset. "
                        f"Available columns: {list(first_example.keys())}"
                    )
            else:
                for col in ["text", "sentence", "content"]:
                    if col in train_dataset.column_names:
                        text_column = col
                        break
                if text_column is None:
                    raise ValueError(
                        f"Could not find text column in dataset. "
                        f"Available columns: {train_dataset.column_names}"
                    )

            # Tokenize dataset
            train_dataset = tokenize(
                train_dataset,
                tokenizer,
                column_name=text_column,
                max_length=cfg.dataset.max_seq_length,
                remove_columns=True,
                truncation=True,
                num_proc=cfg.dataset.num_proc if not cfg.dataset.streaming else None,
            )
        if cfg.dataset.streaming:
            accelerator.print("Tokenization setup complete for streaming dataset.")
            # Add shuffle buffer for streaming datasets
            if (
                hasattr(cfg.dataset, "shuffle_buffer_size")
                and cfg.dataset.shuffle_buffer_size > 0
            ):
                train_dataset = train_dataset.shuffle(
                    buffer_size=cfg.dataset.shuffle_buffer_size, seed=cfg.trainer.seed
                )
                accelerator.print(
                    f"Added shuffle buffer with size {cfg.dataset.shuffle_buffer_size}"
                )
        else:
            accelerator.print(
                f"Tokenization complete. Dataset size: {len(train_dataset)}"
            )

    # Dataloader
    collator_max_length = cfg.datacollator.max_length or cfg.dataset.max_seq_length
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        batch_size=cfg.trainer.per_device_train_batch_size,
        num_workers=cfg.dataset.num_workers,
        mlm_probability=cfg.datacollator.mlm_probability,
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
        mask_all=cfg.datacollator.mask_all,
        pack_sequences=cfg.datacollator.pack_sequences,
        max_length=collator_max_length,
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
        max_position_embeddings=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,  # Use preprocessed vocab_size
        rope=cfg.model.rope,
        rms_norm=cfg.model.rms_norm,
        hidden_act=cfg.model.hidden_act,
        dropout_prob=cfg.model.dropout_prob,
        norm_eps=cfg.model.norm_eps,
        embedding_init_range=cfg.model.embedding_init_range,
        decoder_init_range=cfg.model.decoder_init_range,
        classifier_init_range=cfg.model.classifier_init_range,
        pad_token_id=tokenizer.pad_token_id,
        flash_attention=cfg.model.flash_attention,
    )
    model = NeoBERTLMHead(model_config)

    if cfg.trainer.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Track flag on config for downstream logging/debug, mirroring HF behaviour.
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
    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=cfg.optimizer.lr,
        decay=cfg.scheduler.name,
        warmup_steps=min(cfg.scheduler.warmup_steps, cfg.trainer.max_steps),
        decay_steps=max(
            cfg.trainer.max_steps, cfg.scheduler.warmup_steps + 1
        ),  # Ensure decay_steps > warmup_steps
        constant_steps=0,  # No constant phase for simplicity
    )

    # Prepare with accelerate
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
        scheduler,
    )

    if cfg.wandb.mode != "disabled" and accelerator.is_main_process:
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
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if cfg.trainer.resume_from_checkpoint and resume_checkpoint_path:
        accelerator.load_state(resume_checkpoint_path)
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(
            train_dataloader, metrics["train/batches"] % len(train_dataloader)
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

    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Use skipped_train_dataloader the first epoch after resuming
        dataloader = (
            train_dataloader
            if skipped_train_dataloader is None
            else skipped_train_dataloader
        )

        stored_batch = {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }
        i = 0
        for batch in dataloader:
            # Update number of batches
            metrics["train/batches"] += 1
            i += 1

            # Pack or truncate the batch to target batch size (batch size might be variable due to sequence packing).
            if batch["input_ids"].shape[0] != cfg.trainer.per_device_train_batch_size:
                batch, stored_batch = to_target_batch_size(
                    batch, stored_batch, cfg.trainer.per_device_train_batch_size
                )

            # If it is still smaller, stored batches were not enough and we skip to the next iteration to fill the batch
            if batch["input_ids"].shape[0] < cfg.trainer.per_device_train_batch_size:
                stored_batch = batch
                continue

            # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
            # called, and the first call to .backward() outside this context manager will trigger the synchronization.
            # Accumulating manually gives more flexibility and is compatible with TPUs.
            if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    # Forward pass
                    logits = model(
                        batch["input_ids"], batch.get("attention_mask", None)
                    )["logits"]
                    train_loss = train_loss_fn(
                        logits.view(-1, model_config.vocab_size),
                        batch["labels"].view(-1),
                    )

                    # Compute gradient
                    accelerator.backward(train_loss)

                    # Log metrics
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    if (
                        "attention_mask" in batch.keys()
                        and batch["attention_mask"] is not None
                    ):
                        if batch["attention_mask"].dim() == 2:
                            metrics["train/local_tokens"] += (
                                (batch["attention_mask"] == 0).sum().item()
                            )
                        else:
                            metrics["train/local_tokens"] += batch["input_ids"].numel()
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].numel()
                    metrics["train/local_num_pred"] += (
                        (batch["labels"] != -100).sum().item()
                    )
                    metrics["train/local_sum_loss"] += (
                        train_loss.item() * (batch["labels"] != -100).sum().item()
                    )
                    metrics["train/local_num_correct"] += _count_masked_correct(
                        logits, batch["labels"]
                    )

            else:
                # Forward pass
                logits = model(batch["input_ids"], batch.get("attention_mask", None))[
                    "logits"
                ]
                train_loss = train_loss_fn(
                    logits.view(-1, model_config.vocab_size), batch["labels"].view(-1)
                )

                # Compute gradient
                accelerator.backward(train_loss)

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

                max_grad_norm = (
                    cfg.trainer.gradient_clipping
                    if cfg.trainer.gradient_clipping is not None
                    else (1.0 if cfg.trainer.gradient_checkpointing else None)
                )

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
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                if (
                    "attention_mask" in batch.keys()
                    and batch["attention_mask"] is not None
                ):
                    if batch["attention_mask"].dim() == 2:
                        metrics["train/local_tokens"] += (
                            (batch["attention_mask"] == 0).sum().item()
                        )
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].numel()
                else:
                    metrics["train/local_tokens"] += batch["input_ids"].numel()
                metrics["train/local_num_pred"] += (
                    (batch["labels"] != -100).sum().item()
                )
                metrics["train/local_sum_loss"] += (
                    train_loss.item() * (batch["labels"] != -100).sum().item()
                )
                metrics["train/local_num_correct"] += _count_masked_correct(
                    logits, batch["labels"]
                )

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                    if grad_norm_value is not None:
                        metrics["train/grad_norm"] = grad_norm_value

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
                    metrics.log(accelerator)

                # Save accelerator state for resumable training
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                    limit = max(save_total_limit, max_ckpt)
                    if limit > 0 and os.path.exists(checkpoint_dir):
                        accel_checkpoints = []
                        for item in os.listdir(checkpoint_dir):
                            item_path = os.path.join(checkpoint_dir, item)
                            if os.path.isdir(item_path) and item.isdigit():
                                accel_checkpoints.append(int(item))
                        if len(accel_checkpoints) >= limit:
                            accel_checkpoints.sort()
                            for old_ckpt in accel_checkpoints[
                                : len(accel_checkpoints) - limit + 1
                            ]:
                                old_path = os.path.join(checkpoint_dir, str(old_ckpt))
                                if os.path.exists(old_path):
                                    import shutil

                                    shutil.rmtree(old_path)
                                    if accelerator.is_main_process:
                                        logger.info(
                                            "Removed old accelerator checkpoint: %s (limit=%s)",
                                            old_path,
                                            limit,
                                        )

                    state_checkpoint_path = os.path.join(
                        checkpoint_dir, str(metrics["train/steps"])
                    )
                    accelerator.save_state(output_dir=state_checkpoint_path)

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    # Clean up old checkpoints if limit is set
                    save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                    limit = max(save_total_limit, max_ckpt)  # Use whichever is set

                    if limit > 0 and os.path.exists(model_checkpoint_dir):
                        # Get all checkpoint directories
                        checkpoints = []
                        for item in os.listdir(model_checkpoint_dir):
                            item_path = os.path.join(model_checkpoint_dir, item)
                            if os.path.isdir(item_path) and item.isdigit():
                                checkpoints.append(int(item))

                        # Sort and remove oldest checkpoints if over limit
                        if len(checkpoints) >= limit:
                            checkpoints.sort()
                            # Remove oldest checkpoints
                            for old_ckpt in checkpoints[: len(checkpoints) - limit + 1]:
                                old_path = os.path.join(
                                    model_checkpoint_dir, str(old_ckpt)
                                )
                                if os.path.exists(old_path):
                                    import shutil

                                    shutil.rmtree(old_path)
                                    # Use logger instead of accelerator.print to avoid progress bar interference
                                    if accelerator.is_main_process:
                                        logger.info(
                                            f"Removed old checkpoint: {old_path} (limit={limit})"
                                        )

                    # Save the checkpoint
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        model.save_checkpoint(
                            model_checkpoint_dir, tag=metrics["train/steps"]
                        )
                        checkpoint_path = os.path.join(
                            model_checkpoint_dir, str(metrics["train/steps"])
                        )
                    else:
                        path = os.path.join(
                            model_checkpoint_dir, str(metrics["train/steps"])
                        )
                        os.makedirs(path, exist_ok=True)
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            os.path.join(path, "state_dict.pt"),
                        )
                        checkpoint_path = path

                    # Save config and tokenizer info (only from main process)
                    if accelerator.is_main_process:
                        # Save config as YAML
                        config_path = os.path.join(checkpoint_path, "config.yaml")
                        ConfigLoader.save(cfg, config_path)

                        # Save tokenizer info as JSON
                        tokenizer_info = {
                            "tokenizer_name": cfg.tokenizer.path or cfg.tokenizer.name,
                            "vocab_size": tokenizer.vocab_size,
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

                        # Use logger instead of accelerator.print to avoid progress bar interference
                        logger.info(
                            f"Saved checkpoint with config, tokenizer info, and full tokenizer to {checkpoint_path}"
                        )

                # Zero out the optimizer
                optimizer.zero_grad()

                # Log metrics
                if metrics["train/steps"] >= cfg.trainer.max_steps:
                    pbar.close()
                    return

        # Update the number of epochs
        metrics["train/epochs"] += 1
        skipped_train_dataloader = None

        # For streaming datasets, update the epoch to ensure different shuffling
        if cfg.dataset.streaming and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(metrics["train/epochs"])
            accelerator.print(
                f"Set streaming dataset epoch to {metrics['train/epochs']}"
            )

    pbar.close()
