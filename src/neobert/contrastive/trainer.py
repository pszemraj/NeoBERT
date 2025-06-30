import os
import re
import shutil
import signal
import sys

import numpy

# PyTorch
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_from_disk

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hugging Face
from transformers import DataCollatorWithPadding

# Configuration
from ..config import Config
from ..model import NeoBERT, NeoBERTConfig
from ..tokenizer import get_tokenizer
from .datasets import get_bsz
from .loss import SupConLoss

# Our metric object and model
from .metrics import Metrics


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        features_queries = [
            {
                "input_ids": f["input_ids_query"],
                "attention_mask": f["attention_mask_query"],
            }
            for f in features
        ]
        features_corpus = [
            {
                "input_ids": f["input_ids_corpus"],
                "attention_mask": f["attention_mask_corpus"],
            }
            for f in features
        ]

        batch_queries = super().__call__(features_queries)
        batch_corpus = super().__call__(features_corpus)

        batch = {f"{k}_queries": v for k, v in batch_queries.items()} | {
            f"{k}_corpus": v for k, v in batch_corpus.items()
        }

        if "input_ids_negative" in features[0].keys():
            if isinstance(features[0]["input_ids_negative"][0], list):
                features_negatives = [
                    {
                        "input_ids": f["input_ids_negative"][i],
                        "attention_mask": f["attention_mask_negative"][i],
                    }
                    for f in features
                    for i in range(len(f["input_ids_negative"]))
                ]
            else:
                features_negatives = [
                    {
                        "input_ids": f["input_ids_negative"],
                        "attention_mask": f["attention_mask_negative"],
                    }
                    for f in features
                ]

            batch_negatives = super().__call__(features_negatives)

            batch |= {f"{k}_negative": v for k, v in batch_negatives.items()}

        return batch


def trainer(cfg: Config):
    # Check if dropout is non zero
    if cfg.model.dropout_prob <= 0:
        raise ValueError(
            "Dropout needs to be positive in order to perform steps of SimCSE."
        )

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
        total_limit=2,  # Keep only 2 checkpoints
        iteration=iteration,
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.mode != "disabled" else None,
        project_config=project_config,
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

    # Tokenizer
    tokenizer = get_tokenizer(
        name=cfg.tokenizer.name,
        path=cfg.tokenizer.path,
        max_length=cfg.tokenizer.max_length,
        padding=cfg.tokenizer.padding,
        truncation=cfg.tokenizer.truncation,
    )

    # Dataset
    dataset = load_from_disk(os.path.join(cfg.dataset.path, "all"))
    pretraining_dataset = load_from_disk(
        cfg.dataset.path
    )  # Base dataset for pretraining SimCSE

    data_collator = CustomDataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors="pt", **cfg.datacollator
    )
    dataloaders = {
        key: DataLoader(
            dataset[key],
            collate_fn=data_collator,
            batch_size=get_bsz(key, cfg.dataloader.target_bsz),
            **cfg.dataloader.train,
        )
        for key in dataset.keys()
    } | {
        "pretraining": DataLoader(
            pretraining_dataset,
            collate_fn=data_collator,
            batch_size=cfg.dataloader.target_bsz,
            **cfg.dataloader.train,
        )
    }

    total = sum(x**cfg.datasets.alpha for x in dataset.num_rows.values())
    sample_probs = {
        key: num_rows**cfg.datasets.alpha / total
        for key, num_rows in dataset.num_rows.items()
    }

    # Model
    model = NeoBERT(config=NeoBERTConfig(**cfg.model, **cfg.tokenizer))

    # Get path of desired checkpoint
    if "ckpt" in cfg.model.keys() and cfg.model.ckpt != "latest":
        tag = cfg.model.ckpt
    else:
        latest_path = os.path.join(cfg.model.ckpt_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    # Load weights
    if cfg.model.deepspeed:
        model = load_state_dict_from_zero_checkpoint(
            model, cfg.model.ckpt_dir, tag=str(tag)
        )
    else:
        raise NotImplementedError

    # Log the number of parameters
    accelerator.log(
        {
            "model_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), **cfg.optimizer.hparams)

    # Scheduler
    scheduler1 = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=cfg.scheduler.warmup_steps,
    )
    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.decay_steps,
        eta_min=cfg.optimizer.hparams.lr * 0.1,
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
    keys = list(dataset.keys())

    dataloaders[keys[0]], model, optimizer, scheduler = accelerator.prepare(
        dataloaders[
            keys[0]
        ],  # Accelerate with deepspeed requires at least one dataloader to be passed along with other objects to prepare.
        model,
        optimizer,
        scheduler,
    )

    for key in keys[1:]:
        dataloaders[key] = accelerator.prepare(dataloaders[key])

    # Loss function
    train_loss_fn = SupConLoss()

    # # Resume from the latest checkpoint
    # skipped_train_dataloader = None
    # if cfg.trainer.resume and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    #     accelerator.load_state()
    #     skipped_train_dataloader = accelerator.skip_first_batches(train_dataloader, metrics["train/batches"] % len(train_dataloader))

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

    while cfg.trainer.max_steps > metrics["train/steps"]:
        coin_flip = numpy.random.random()

        # Choose from one of the finetuning datasets
        if coin_flip > cfg.datasets.pretraining_prob:
            # Randomly select which task to draw a batch from
            task_name = numpy.random.choice(
                list(sample_probs.keys()), p=list(sample_probs.values())
            )
            dataloader = dataloaders[task_name]
            batch = next(iter(dataloader))

            # Convert Hugging Face multiplicative mask to xformers additive mask
            pad_mask_queries = torch.where(
                batch["attention_mask_queries"] == 1, float(0.0), float("-inf")
            ).type(dtype_pad_mask)
            pad_mask_corpus = torch.where(
                batch["attention_mask_corpus"] == 1, float(0.0), float("-inf")
            ).type(dtype_pad_mask)
            if "input_ids_negative" in batch.keys():
                pad_mask_negatives = torch.where(
                    batch["attention_mask_negative"] == 1, float(0.0), float("-inf")
                ).type(dtype_pad_mask)

            # Update specific number of batches
            metrics[f"train/{task_name}_batches"] += 1

        # Else, we do a step of SimCSE with the original pretraing dataset in order to avoid catastrophic forgetting. Warning: dropout needs to be greater than zero!
        else:
            batch = next(iter(dataloaders["pretraining"]))

            # Here, queries and corpus are identical
            batch["input_ids_queries"] = batch["input_ids"]
            batch["input_ids_corpus"] = batch["input_ids"]

            # Convert Hugging Face multiplicative mask to xformers additive mask
            pad_mask_queries = torch.where(
                batch["attention_mask"] == 1, float(0.0), float("-inf")
            ).type(dtype_pad_mask)
            pad_mask_corpus = pad_mask_queries

            # Update specific number of batches
            metrics["train/pretraining_batches"] += 1

        # Update global number of batches
        metrics["train/batches"] += 1

        # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
        # called, and the first call to .backward() outside this context manager will trigger the synchronization.
        # Accumulating manually gives more flexibility and is compatible with TPUs.
        if metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps != 0:
            with accelerator.no_sync(model):
                # Forward pass
                queries = model(batch["input_ids_queries"], pad_mask_queries)
                corpus = model(batch["input_ids_corpus"], pad_mask_corpus)
                if "input_ids_negative" in batch.keys():
                    negatives = model(batch["input_ids_negative"], pad_mask_negatives)

                # Pool representations
                pooled_queries = (
                    queries * batch["attention_mask_queries"].unsqueeze(-1)
                ).sum(dim=1) / batch["attention_mask_queries"].sum(dim=1, keepdim=True)
                pooled_corpus = (
                    corpus * batch["attention_mask_corpus"].unsqueeze(-1)
                ).sum(dim=1) / batch["attention_mask_corpus"].sum(dim=1, keepdim=True)

                # Pool each negative's representation
                if "input_ids_negative" in batch.keys():
                    pooled_negatives = (
                        negatives * batch["attention_mask_negative"].unsqueeze(-1)
                    ).sum(dim=1) / batch["attention_mask_negative"].sum(
                        dim=1, keepdim=True
                    )
                else:
                    pooled_negatives = None

                # Loss
                train_loss = train_loss_fn(
                    pooled_queries, pooled_corpus, pooled_negatives
                )

                # Compute gradient
                accelerator.backward(train_loss)

                # Log metrics
                metrics["train/local_samples"] += batch["input_ids_queries"].shape[0]
                metrics["train/local_sum_loss"] += train_loss

        else:
            # Forward pass
            queries = model(batch["input_ids_queries"], pad_mask_queries)
            corpus = model(batch["input_ids_corpus"], pad_mask_corpus)
            if "input_ids_negative" in batch.keys():
                negatives = model(batch["input_ids_negative"], pad_mask_negatives)

            # Pool representations
            pooled_queries = (
                queries * batch["attention_mask_queries"].unsqueeze(-1)
            ).sum(dim=1) / batch["attention_mask_queries"].sum(dim=1, keepdim=True)
            pooled_corpus = (corpus * batch["attention_mask_corpus"].unsqueeze(-1)).sum(
                dim=1
            ) / batch["attention_mask_corpus"].sum(dim=1, keepdim=True)

            # Pool each negative's representation
            if "input_ids_negative" in batch.keys():
                pooled_negatives = (
                    negatives * batch["attention_mask_negative"].unsqueeze(-1)
                ).sum(dim=1) / batch["attention_mask_negative"].sum(dim=1, keepdim=True)
            else:
                pooled_negatives = None

            # Loss
            train_loss = train_loss_fn(pooled_queries, pooled_corpus, pooled_negatives)

            # Compute gradient and apply clipping
            accelerator.backward(train_loss)
            if (
                cfg.trainer.gradient_clipping is not None
                and cfg.trainer.gradient_clipping > 0
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), cfg.trainer.gradient_clipping
                )

            # Update the parameters
            optimizer.step()
            scheduler.step()

            # Log metrics
            pbar.update(1)
            metrics["train/steps"] += 1
            metrics["train/local_samples"] += batch["input_ids_queries"].shape[0]
            metrics["train/local_sum_loss"] += train_loss

            if metrics["train/steps"] % cfg.wandb.log_interval == 0:
                # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                if accelerator.distributed_type is DistributedType.DEEPSPEED:
                    metrics["train/grad_norm"] = model.get_global_grad_norm()  # .item()
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
                        model.state_dict(),
                        os.path.join(path, "state_dict.pt"),
                    )

            if metrics["train/steps"] >= cfg.trainer.max_steps:
                break

            # Reset the gradient
            optimizer.zero_grad()

    # # Log metrics
    # metrics["train/epochs"] += 1

    # Make sure that the wandb tracker finishes correctly and close the progress bar
    pbar.close()
    accelerator.end_training()
