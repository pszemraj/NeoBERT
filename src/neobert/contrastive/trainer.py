"""Training loop for contrastive and SimCSE-style finetuning."""

import logging
import os
import re
import shutil
import signal
import sys
from types import FrameType

import numpy

# PyTorch
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_from_disk

# Deepspeed
from deepspeed.utils import safe_get_full_fp32_param
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hugging Face
from transformers import DataCollatorWithPadding

# Configuration
from ..config import Config
from ..model import NeoBERT, NeoBERTConfig
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler
from ..tokenizer import get_tokenizer
from ..utils import prepare_wandb_config
from .datasets import get_bsz
from .loss import SupConLoss
from .metrics import Metrics

logger = logging.getLogger(__name__)


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """Collator that pads query/corpus/negative fields separately."""

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


def trainer(cfg: Config) -> None:
    """Run contrastive training loop.

    :param Config cfg: Training configuration.
    """
    # Check if dropout is non zero
    if cfg.model.dropout_prob <= 0:
        raise ValueError(
            "Dropout needs to be positive in order to perform steps of SimCSE."
        )
    if not cfg.dataset.path:
        raise ValueError(
            "Contrastive training requires dataset.path to point to a preprocessed dataset. "
            "Run scripts/contrastive/preprocess.py to build it first."
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
        folders = [
            folder
            for folder in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, folder))
            and re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)
        ]
        if folders:
            iteration = (
                max(
                    int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)[0])
                    for folder in folders
                )
                + 1
            )

    save_total_limit = max(
        getattr(cfg.trainer, "save_total_limit", 0), getattr(cfg.trainer, "max_ckpt", 0)
    )
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=save_total_limit or None,
        iteration=iteration,
    )
    wandb_enabled = cfg.wandb.enabled and cfg.wandb.mode != "disabled"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if wandb_enabled else None,
        project_config=project_config,
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

    # Enable TF32 on matmul and on cuDNN (if available)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)

    # Tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.path or cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        vocab_size=cfg.tokenizer.vocab_size or cfg.model.vocab_size,
    )

    # Dataset
    dataset_path = str(cfg.dataset.path)
    dataset = load_from_disk(os.path.join(dataset_path, "all"))
    pretraining_dataset = load_from_disk(
        dataset_path
    )  # Base dataset for pretraining SimCSE

    data_collator = CustomDataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
        pad_to_multiple_of=cfg.datacollator.pad_to_multiple_of,
    )
    target_bsz = (
        cfg.trainer.per_device_train_batch_size or cfg.trainer.train_batch_size or 16
    )
    dataloader_kwargs = {
        "collate_fn": data_collator,
        "num_workers": cfg.trainer.dataloader_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": True,
    }
    dataloaders = {
        key: DataLoader(
            dataset[key],
            batch_size=max(1, get_bsz(key, target_bsz)),
            **dataloader_kwargs,
        )
        for key in dataset.keys()
    }
    dataloaders["pretraining"] = DataLoader(
        pretraining_dataset,
        batch_size=target_bsz,
        **dataloader_kwargs,
    )

    alpha = getattr(cfg.dataset, "alpha", 1.0)
    total = sum(x**alpha for x in dataset.num_rows.values())
    sample_probs = {
        key: num_rows**alpha / total for key, num_rows in dataset.num_rows.items()
    }

    # Model
    max_length = cfg.tokenizer.max_length or cfg.model.max_position_embeddings
    model_config = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_length=max_length,
        vocab_size=cfg.model.vocab_size,
        rope=cfg.model.rope,
        rms_norm=cfg.model.rms_norm,
        hidden_act=cfg.model.hidden_act,
        dropout=cfg.model.dropout_prob,
        norm_eps=cfg.model.norm_eps,
        embedding_init_range=cfg.model.embedding_init_range,
        decoder_init_range=cfg.model.decoder_init_range,
        flash_attention=cfg.model.flash_attention,
        ngpt=cfg.model.ngpt,
        base_scale=cfg.model.base_scale,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = NeoBERT(config=model_config)

    def _resolve_checkpoint_tag(
        checkpoint_dir: str, checkpoint: str | int | None
    ) -> str:
        """Resolve a checkpoint tag to a concrete step directory name.

        :param str checkpoint_dir: Base directory that holds checkpoint step folders.
        :param str | int | None checkpoint: Tag or step to resolve, or ``None``/"latest".
        :return str: Resolved checkpoint step tag.
        """
        if checkpoint is None or str(checkpoint).lower() == "latest":
            latest_path = os.path.join(checkpoint_dir, "latest")
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    return fd.read().strip()
            steps = [
                int(entry)
                for entry in os.listdir(checkpoint_dir)
                if entry.isdigit()
                and os.path.isdir(os.path.join(checkpoint_dir, entry))
            ]
            if not steps:
                raise ValueError(
                    f"No checkpoint steps found in {checkpoint_dir} to resolve 'latest'."
                )
            return str(max(steps))
        return str(checkpoint)

    # Load weights if provided
    pretrained_checkpoint_dir = None
    pretrained_checkpoint = None
    allow_random_weights = False
    use_deepspeed = getattr(cfg, "use_deepspeed", False)
    if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
        pretrained_checkpoint_dir = cfg._raw_model_dict.get("pretrained_checkpoint_dir")
        pretrained_checkpoint = cfg._raw_model_dict.get("pretrained_checkpoint")
        allow_random_weights = cfg._raw_model_dict.get("allow_random_weights", False)
        if "deepspeed" in cfg._raw_model_dict:
            use_deepspeed = cfg._raw_model_dict.get("deepspeed")

    if pretrained_checkpoint_dir:
        if not pretrained_checkpoint_dir.endswith("model_checkpoints"):
            pretrained_checkpoint_dir = os.path.join(
                pretrained_checkpoint_dir, "model_checkpoints"
            )
        tag = _resolve_checkpoint_tag(
            pretrained_checkpoint_dir,
            pretrained_checkpoint or cfg.pretrained_checkpoint,
        )
        if use_deepspeed:
            model = load_state_dict_from_zero_checkpoint(
                model, pretrained_checkpoint_dir, tag=str(tag)
            )
        else:
            state_dict_path = os.path.join(
                pretrained_checkpoint_dir, str(tag), "state_dict.pt"
            )
            if not os.path.exists(state_dict_path):
                raise ValueError(
                    f"Expected state_dict.pt at {state_dict_path}. "
                    "Set pretrained_checkpoint_dir or enable DeepSpeed loading."
                )
            state_dict = torch.load(state_dict_path, map_location="cpu")
            # NOTE: We allow partial loads for flexibility; checkpoint/config mismatches
            # are not validated beyond this strict=False load.
            model.load_state_dict(state_dict, strict=False)
    elif allow_random_weights:
        logger.warning(
            "allow_random_weights=true: contrastive training will start from random initialization."
        )
    else:
        logger.warning(
            "No pretrained checkpoint provided. Contrastive training will start from random initialization."
        )

    # Log the number of parameters
    # Log model parameters to console instead of wandb
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model parameters: {model_params:,}")

    # Optimizer
    optimizer = get_optimizer(
        model,
        accelerator.distributed_type,
        model_config=model_config,
        name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        muon_config=cfg.optimizer.muon_config,
    )

    # Scheduler
    total_steps = cfg.scheduler.total_steps or cfg.trainer.max_steps
    decay_steps = max(total_steps, cfg.scheduler.warmup_steps + 1)
    scheduler = get_scheduler(
        optimizer=optimizer,
        lr=cfg.optimizer.lr,
        decay=cfg.scheduler.name,
        warmup_steps=min(cfg.scheduler.warmup_steps, cfg.trainer.max_steps),
        decay_steps=decay_steps,
        constant_steps=0,
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

    if wandb_enabled and accelerator.is_main_process:
        wandb_watch = os.environ.get("WANDB_WATCH")
        if wandb_watch is not None:
            watch_mode = wandb_watch.strip().lower()
            if watch_mode in {"", "false", "0", "none", "off"}:
                watch_mode = None
            elif watch_mode == "weights":
                watch_mode = "parameters"
            elif watch_mode not in {"gradients", "parameters", "all"}:
                accelerator.print(
                    f"Unrecognized WANDB_WATCH value '{wandb_watch}'; skipping wandb.watch()"
                )
                watch_mode = None

            if watch_mode:
                wandb.watch(
                    accelerator.unwrap_model(model),
                    log=watch_mode,
                    log_freq=getattr(cfg.wandb, "log_interval", 100),
                )

    # Loss function
    train_loss_fn = SupConLoss(temperature=cfg.contrastive.temperature)

    # Resume from the latest checkpoint if available
    if (
        cfg.trainer.resume_from_checkpoint
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        accelerator.load_state()

    # Signal handler that save the accelerate state
    def handler(signum: int, frame: FrameType | None) -> None:
        """Handle termination signals by checkpointing state.

        :param int signum: Signal number received.
        :param FrameType | None frame: Current stack frame.
        """
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
        if coin_flip > cfg.dataset.pretraining_prob:
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
                metrics["train/local_sum_loss"] += train_loss.item()

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
            metrics["train/local_sum_loss"] += train_loss.item()

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

            # Save accelerator state
            if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                accelerator.save_state()

            # Save the pytorch model
            if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
                limit = max(save_total_limit, max_ckpt)
                if limit > 0:
                    # Delete checkpoints if there are too many
                    files = os.listdir(model_checkpoint_dir)
                    iterations = [int(f) for f in files if f.isdigit()]
                    iterations.sort()

                    # Remove files with the smallest iterations until the limit is met
                    while iterations and len(iterations) >= limit:
                        file_to_remove = iterations.pop(0)
                        shutil.rmtree(
                            os.path.join(model_checkpoint_dir, str(file_to_remove))
                        )
                        print(
                            f"Deleted old model checkpoint {file_to_remove} due to limit "
                            f"(limit = {limit})"
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
