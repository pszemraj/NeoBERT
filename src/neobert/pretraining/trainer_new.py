import os
import re

# PyTorch
import torch
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs, DistributedType,
                              ProjectConfiguration, set_seed)
# Hugging Face
from datasets import load_from_disk
# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import BatchEncoding

from ..config import Config
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
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    # Tokenizer
    tokenizer = get_tokenizer(
        name=cfg.tokenizer.name,
        path=cfg.tokenizer.path,
        max_length=cfg.tokenizer.max_length,
        padding=cfg.tokenizer.padding,
        truncation=cfg.tokenizer.truncation,
    )

    # Dataset
    train_dataset = load_from_disk(cfg.dataset.path)

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
    )
    model = NeoBERTLMHead(model_config)

    accelerator.log(
        {
            "model_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
    )

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
        name=cfg.scheduler.name,
        warmup_steps=cfg.scheduler.warmup_steps,
        total_steps=cfg.trainer.max_steps,
        num_cycles=cfg.scheduler.num_cycles,
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
                    else:
                        path = os.path.join(
                            model_checkpoint_dir, str(metrics["train/steps"])
                        )
                        os.makedirs(path, exist_ok=True)
                        torch.save(
                            model.state_dict(),
                            os.path.join(path, "state_dict.pt"),
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

    pbar.close()
