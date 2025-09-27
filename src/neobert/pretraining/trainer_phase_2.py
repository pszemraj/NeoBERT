import os
import re
import shutil
import signal
import sys
from dataclasses import asdict

import numpy as np

# PyTorch
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

# Hugging Face
from datasets import load_from_disk

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from ..model import NeoBERTConfig, NeoBERTLMHead
from ..tokenizer import get_tokenizer

# Our metric object and model
from .metrics import Metrics


def trainer(cfg):
    # Get the last checkpoint id
    checkpoint_dir = os.path.join(cfg.trainer.dir, "checkpoints")
    model_checkpoint_dir = os.path.join(cfg.trainer.dir, "model_checkpoints")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    iteration = 0
    if (
        cfg.trainer.resume
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
        cfg.trainer.dir,
        automatic_checkpoint_naming=True,
        total_limit=cfg.trainer.accelerate.max_ckpt,
        iteration=iteration,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb",
        project_config=project_config,
    )

    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": asdict(cfg)
                | {"distributed_type": accelerator.distributed_type},
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.trainer.resume,
            }
        },
    )

    # Set the seed
    set_seed(cfg.seed)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    tokenizer = get_tokenizer(**cfg.tokenizer)

    # Dataset
    train_dataset = load_from_disk(cfg.dataset.path_to_disk)
    train_dataset_p = load_from_disk(
        cfg.dataset.path_to_disk + f"+{cfg.dataset.min_length}"
    )
    train_dataset_pp = load_from_disk(
        cfg.dataset.path_to_disk + f"+{2 * cfg.dataset.min_length}"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, return_tensors="pt", **cfg.datacollator
    )
    dataloaders = [
        DataLoader(dataset, collate_fn=data_collator, **cfg.dataloader.train)
        for dataset in [train_dataset, train_dataset_p, train_dataset_pp]
    ]

    # Model
    # Calculate optimal vocab_size for GPU efficiency when creating from scratch
    from ..config import round_up_to_multiple

    actual_vocab_size = len(tokenizer)
    rounded_vocab_size = round_up_to_multiple(actual_vocab_size, 128)

    # Update all config sources with the actual rounded vocab_size BEFORE anything uses them
    cfg.model.vocab_size = rounded_vocab_size
    if hasattr(cfg.tokenizer, "vocab_size"):
        cfg.tokenizer.vocab_size = rounded_vocab_size

    tokenizer_config = {**cfg.tokenizer.__dict__}
    tokenizer_config["vocab_size"] = rounded_vocab_size

    model = NeoBERTLMHead(
        NeoBERTConfig(
            **cfg.model.__dict__,
            **tokenizer_config,
            pad_token_id=tokenizer.pad_token_id,
        )
    )

    # Log the number of parameters
    # Log model parameters to console instead of wandb
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model parameters: {model_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), **cfg.optimizer.hparams)

    # Scheduler
    scheduler1 = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=cfg.scheduler.warmup_steps,
    )
    if cfg.scheduler.decay == "cosine":
        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.decay_steps,
            eta_min=cfg.optimizer.hparams.lr * 0.1,
        )
    else:
        scheduler2 = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=cfg.scheduler.decay_steps,
        )

    def _constant_min_lr(_):
        """LambdaLR multiplies the optimizer's lr with lr_lambda(epoch)"""
        return 0.1

    scheduler3 = LambdaLR(optimizer, lr_lambda=_constant_min_lr)
    scheduler = SequentialLR(
        optimizer,
        [scheduler1, scheduler2, scheduler3],
        [cfg.scheduler.warmup_steps, cfg.scheduler.decay_steps],
    )

    # Accelerate
    model, optimizer, scheduler, *dataloaders = accelerator.prepare(
        model,
        optimizer,
        scheduler,
        *dataloaders,
    )

    # Loss function
    train_loss_fn = CrossEntropyLoss()

    # Resume from the latest checkpoint
    skipped_dataloaders = [None, None, None]
    if (
        cfg.trainer.resume
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        accelerator.load_state()
        for i, dataloader in enumerate(dataloaders):
            dataloader.set_epoch(metrics["train/epochs"])
            skipped_dataloaders[i] = accelerator.skip_first_batches(
                dataloader, metrics[f"train/batches_{i}"] % len(dataloader)
            )

    # Signal handler that save the accelerate state
    def handler(signum, frame):
        print(
            f"Signal {signum} received on rank {accelerator.process_index}, checkpointing..."
        )
        accelerator.save_state()
        accelerator.wait_for_everyone()
        print(f"Done on rank {accelerator.process_index}")
        sys.exit(0)

    # Register handler to the signal SIGTERM
    signal.signal(signal.SIGTERM, handler)

    # Progress bar
    pbar = tqdm(
        desc="Train",
        unit="step",
        initial=metrics["train/steps"],
        total=cfg.trainer.max_steps,
        disable=(cfg.trainer.disable_tqdm or not accelerator.is_main_process),
    )

    if cfg.trainer.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16
    else:
        dtype_pad_mask = torch.float32

    iterators = [iter(dataloader) for dataloader in dataloaders]
    skipped_iterators = [
        iter(skipped_dataloader) if skipped_dataloader is not None else None
        for skipped_dataloader in skipped_dataloaders
    ]
    while cfg.trainer.max_steps > metrics["train/steps"]:
        # Sample which dataloader to use
        i = np.random.choice(range(len(iterators)), p=[0.2, 0.4, 0.4])

        # Update number of batches
        metrics["train/batches"] += 1
        metrics[f"train/batches_{i}"] += 1

        # Retrieve batch and "remove" skipped_iterator when exhausted
        try:
            batch = (
                next(iterators[i])
                if skipped_iterators[i] is None
                else next(skipped_iterators[i])
            )
        except StopIteration:
            iterators[i] = iter(dataloaders[i])
            skipped_iterators[i] = None
            metrics[f"train/epochs_dataset_{i}"] += 1
            batch = next(iterators[i])

        # Convert Hugging Face multiplicative mask to xformers additive mask
        pad_mask = torch.where(
            batch["attention_mask"] == 1, float(0.0), float("-inf")
        ).type(dtype_pad_mask)

        # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
        # called, and the first call to .backward() outside this context manager will trigger the synchronization.
        # Accumulating manually gives more flexibility and is compatible with TPUs.
        if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
            with accelerator.no_sync(model):
                # Forward pass
                logits = model(batch["input_ids"], pad_mask)["logits"]
                train_loss = train_loss_fn(
                    logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1)
                )

                # Compute gradient
                accelerator.backward(train_loss)

                # Log metrics
                metrics["train/local_samples"] += batch["input_ids"].shape[0]
                metrics["train/local_tokens"] += (pad_mask == 0).sum().item()
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
            logits = model(batch["input_ids"], pad_mask)["logits"]
            train_loss = train_loss_fn(
                logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1)
            )

            # Compute gradient and apply clipping
            accelerator.backward(train_loss)
            if (
                cfg.trainer.gradient_clipping is not None
                and cfg.trainer.gradient_clipping > 0
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), cfg.trainer.gradient_clipping
                )

            # Update the parameters and the scheduler
            optimizer.step()
            scheduler.step()

            # Log metrics

            pbar.update(1)
            metrics["train/steps"] += 1
            metrics["train/local_samples"] += batch["input_ids"].shape[0]
            metrics["train/local_tokens"] += (pad_mask == 0).sum().item()
            metrics["train/local_num_pred"] += (batch["labels"] != -100).sum().item()
            metrics["train/local_sum_loss"] += (
                train_loss.item() * (batch["labels"] != -100).sum().item()
            )
            metrics["train/local_num_correct"] += (
                (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            )

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
                        sum([p.grad.norm(2) ** 2 for p in model.parameters()]) ** 0.5
                    ).item()
                    metrics["train/weight_norm"] = (
                        sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5
                    ).item()

                metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics.log(accelerator)

            # Save the accelerator state from the main process
            if metrics["train/steps"] % cfg.trainer.accelerate.save_steps == 0:
                accelerator.save_state()

            # Save the pytorch model
            if metrics["train/steps"] % cfg.trainer.model.save_steps == 0:
                if cfg.trainer.model.max_ckpt is not None:
                    # Delete checkpoints if there are too many
                    files = os.listdir(model_checkpoint_dir)
                    iterations = [int(f) for f in files if f.isdigit()]
                    iterations.sort()

                    # Remove files with the smallest iterations until the limit is met
                    while (
                        iterations is not None
                        and len(iterations) >= cfg.trainer.model.max_ckpt
                    ):
                        file_to_remove = iterations.pop(0)
                        shutil.rmtree(
                            os.path.join(model_checkpoint_dir, str(file_to_remove))
                        )
                        print(
                            f"Deleted old model checkpoint {file_to_remove} due to limit "
                            f"(max_ckpt = {cfg.trainer.model.max_ckpt})"
                        )
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
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(path, "state_dict.pt"),
                    )

            if metrics["train/steps"] >= cfg.trainer.max_steps:
                break

            # Reset the gradient
            optimizer.zero_grad()

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
