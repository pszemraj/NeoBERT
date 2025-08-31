import json
import os
import re

# PyTorch
import torch
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

from ..config import Config, ConfigLoader
from ..dataloader import get_dataloader
from ..model import NeoBERTConfig, NeoBERTLMHead
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tokenizer import get_tokenizer

# Our metric object and model
from .metrics import Metrics


def to_target_batch_size(
    batch: BatchEncoding,
    stored_batch: BatchEncoding,
    target_size: int = 8,
):
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
                stored_batch[key].to("cpu", non_blocking=True)

        # Concatenate otherwise
        else:
            for key in batch.keys():
                batch[key] = torch.cat([batch[key], stored_batch[key]], dim=0)
                stored_batch[key] = None

    return batch, stored_batch


def trainer(cfg: Config):
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.output_dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.output_dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    iteration = 0

    if (
        cfg.trainer.resume_from_checkpoint
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        # This regular expression was taken from accelerator.load_state()
        folders = os.listdir(checkpoint_dir)
        iteration = (
            max(
                int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0])
                for folder in folders
            )
            + 1
        )

    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=2,  # Keep only 2 accelerate checkpoints
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
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            init_kwargs={
                "wandb": {
                    "name": cfg.wandb.name,
                    "entity": cfg.wandb.entity,
                    "config": cfg.__dict__,
                    "tags": cfg.wandb.tags,
                    "dir": cfg.wandb.dir,
                    "mode": cfg.wandb.mode,
                    "resume": cfg.wandb.resume,
                }
            },
        )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN (if available)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    # Always use bf16 for mixed precision
    if accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

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
    train_dataloader = get_dataloader(
        train_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        batch_size=cfg.trainer.per_device_train_batch_size,
        num_workers=cfg.dataset.num_workers,
        mlm_probability=cfg.datacollator.mlm_probability,
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
    )

    # Model
    # Debug print
    if cfg.debug:
        print(f"Model vocab_size: {cfg.model.vocab_size}")
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")

    model_config = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,
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

    # Log model parameters to console instead of wandb
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model parameters: {model_params:,}")

    # Optimizer and Scheduler
    optimizer = get_optimizer(
        model,
        accelerator.distributed_type,
        name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
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

    # Loss function
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_train_dataloader = None
    if (
        cfg.trainer.resume_from_checkpoint
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        accelerator.load_state()
        train_dataloader.set_epoch(metrics["train/epochs"])
        skipped_train_dataloader = accelerator.skip_first_batches(
            train_dataloader, metrics["train/batches"] % len(train_dataloader)
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
                        logits.view(-1, cfg.model.vocab_size),
                        batch["labels"].view(-1),
                    )

                    # Compute gradient
                    accelerator.backward(train_loss)

                    # Log metrics
                    metrics["train/local_samples"] += batch["input_ids"].shape[0]
                    if "attention_mask" in batch.keys():
                        metrics["train/local_tokens"] += (
                            (batch["attention_mask"] == 0).sum().item()
                        )
                    else:
                        metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                    metrics["train/local_num_pred"] += (
                        (batch["labels"] != -100).sum().item()
                    )
                    metrics["train/local_sum_loss"] += (
                        train_loss.item() * (batch["labels"] != -100).sum().item()
                    )
                    metrics["train/local_num_correct"] += (
                        (logits.argmax(dim=-1) == batch["labels"]).sum().item()
                    )

            else:
                # Forward pass
                logits = model(batch["input_ids"], batch.get("attention_mask", None))[
                    "logits"
                ]
                train_loss = train_loss_fn(
                    logits.view(-1, cfg.model.vocab_size), batch["labels"].view(-1)
                )

                # Compute gradient and apply clipping
                accelerator.backward(train_loss)
                if cfg.trainer.gradient_checkpointing:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                # Log metrics
                pbar.update(1)
                metrics["train/steps"] += 1
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                if "attention_mask" in batch.keys():
                    metrics["train/local_tokens"] += (
                        (batch["attention_mask"] == 0).sum().item()
                    )
                else:
                    metrics["train/local_tokens"] += batch["input_ids"].shape[1]
                metrics["train/local_num_pred"] += (
                    (batch["labels"] != -100).sum().item()
                )
                metrics["train/local_sum_loss"] += (
                    train_loss.item() * (batch["labels"] != -100).sum().item()
                )
                metrics["train/local_num_correct"] += (
                    (logits.argmax(dim=-1) == batch["labels"]).sum().item()
                )

                # Update the parameters and the scheduler
                optimizer.step()
                scheduler.step()

                if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                    # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                    if accelerator.distributed_type is DistributedType.DEEPSPEED:
                        metrics["train/grad_norm"] = model.get_global_grad_norm()
                        metrics["train/weight_norm"] = (
                            sum(
                                [
                                    safe_get_full_fp32_param(p).norm(2) ** 2
                                    for p in model.parameters()
                                ]
                            )
                            ** 0.5
                        ).item()
                    # DDP
                    else:
                        metrics["train/grad_norm"] = (
                            sum([p.grad.norm(2) ** 2 for p in model.parameters()])
                            ** 0.5
                        ).item()
                        metrics["train/weight_norm"] = (
                            sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5
                        ).item()

                    metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    metrics.log(accelerator)

                # Save the accelerator state from the main process
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                    accelerator.save_state()

                # Save the pytorch model
                if metrics["train/steps"] % cfg.trainer.save_steps == 0:
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
                            model.state_dict(),
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
                        tokenizer.save_pretrained(tokenizer_dir)

                        accelerator.print(
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
