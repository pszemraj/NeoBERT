"""Training loop for contrastive and SimCSE-style finetuning."""

import logging
import os
import shutil
import signal
import sys
from pathlib import Path
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
from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_model_safetensors,
    save_model_safetensors,
)
from neobert.collator.collator import (
    _is_right_padded_mask,
    attention_mask_to_packed_seqlens,
)
from neobert.config import Config
from neobert.kernels.attention import resolve_runtime_attn_backend
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import get_optimizer
from neobert.scheduler import get_scheduler, resolve_scheduler_steps
from neobert.tokenizer import get_tokenizer
from neobert.training_utils import (
    _maybe_compile_model,
    _maybe_prepare_for_forward,
    _resolve_resume_checkpoint,
)
from neobert.contrastive.datasets import get_bsz
from neobert.contrastive.loss import SupConLoss
from neobert.contrastive.metrics import Metrics
from neobert.utils import configure_tf32, format_resolved_config, prepare_wandb_config

logger = logging.getLogger(__name__)


def _build_packed_seqlens(attention_mask: torch.Tensor, *, name: str) -> torch.Tensor:
    """Build packed sequence lengths from a right-padded attention mask.

    :param torch.Tensor attention_mask: 0/1 attention mask of shape [B, S].
    :param str name: Context label used for error messages.
    :return torch.Tensor: Packed sequence lengths tensor on CPU.
    :raises ValueError: If the mask is not right-padded.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"{name} attention_mask must be rank-2 [B, S], got {tuple(attention_mask.shape)}"
        )
    mask_cpu = attention_mask.detach()
    if mask_cpu.is_cuda:
        mask_cpu = mask_cpu.cpu()
    if not _is_right_padded_mask(mask_cpu):
        raise ValueError(
            f"Packed attention requires right-padded attention_mask; '{name}' is not "
            "right-padded. Set tokenizer.padding_side='right' or use "
            "model.attn_backend='sdpa'."
        )
    return attention_mask_to_packed_seqlens(mask_cpu)


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
    cfg.model.attn_backend = resolve_runtime_attn_backend(
        cfg.model.attn_backend,
        fallback_to_sdpa=True,
    )
    pretraining_mix_prob = float(cfg.contrastive.pretraining_prob)
    if pretraining_mix_prob < 0.0 or pretraining_mix_prob > 1.0:
        raise ValueError(
            "contrastive.pretraining_prob must be in [0, 1], got "
            f"{pretraining_mix_prob}."
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

    raw_save_total_limit = getattr(cfg.trainer, "save_total_limit", None)
    raw_max_ckpt = getattr(cfg.trainer, "max_ckpt", None)
    save_total_limit = max(
        int(raw_save_total_limit or 0),
        int(raw_max_ckpt or 0),
    )
    project_config = ProjectConfiguration(
        str(output_dir),
        automatic_checkpoint_naming=True,
        total_limit=save_total_limit or None,
        iteration=iteration,
    )
    wandb_enabled = cfg.wandb.enabled and cfg.wandb.mode != "disabled"
    accelerator = Accelerator(
        cpu=bool(getattr(cfg.trainer, "use_cpu", False)),
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg.trainer.mixed_precision,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with="wandb" if wandb_enabled else None,
        project_config=project_config,
    )
    tracker_config_dict = prepare_wandb_config(cfg)
    if accelerator.is_main_process:
        accelerator.print(
            "Resolved task config:\n" + format_resolved_config(tracker_config_dict)
        )
        logger.info(
            f"contrastive.pretraining_prob={pretraining_mix_prob}: "
            f"{pretraining_mix_prob * 100.0:.1f}% of steps sample the pretraining "
            "dataset branch (SimCSE anti-forgetting path)."
        )

    # Initialise the wandb run and pass wandb parameters
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

    # Configure TF32 for supported GPUs
    configure_tf32(enabled=cfg.trainer.tf32, print_fn=accelerator.print)

    # Local and global counters
    metrics = Metrics()
    accelerator.register_for_checkpointing(metrics)
    log_interval = max(1, cfg.trainer.logging_steps)

    # Tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.path or cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        trust_remote_code=cfg.tokenizer.trust_remote_code,
        revision=cfg.tokenizer.revision,
        allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
    )
    use_packed = cfg.model.attn_backend != "sdpa"
    if use_packed and tokenizer.padding_side != "right":
        logger.warning(
            f"tokenizer.padding_side={tokenizer.padding_side} is incompatible with "
            "packed attention; falling back to attn_backend='sdpa'."
        )
        use_packed = False

    # Dataset
    dataset_path = Path(cfg.dataset.path)
    dataset = load_from_disk(os.fspath(dataset_path / "all"))
    pretraining_dataset = load_from_disk(
        os.fspath(dataset_path)
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
        attn_backend=cfg.model.attn_backend if use_packed else "sdpa",
        kernel_backend=cfg.model.kernel_backend,
        ngpt=cfg.model.ngpt,
        base_scale=cfg.model.base_scale,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = NeoBERT(config=model_config)

    def _resolve_checkpoint_tag(
        checkpoint_dir: Path, checkpoint: str | int | None
    ) -> str:
        """Resolve a checkpoint tag to a concrete step directory name.

        :param Path checkpoint_dir: Base directory that holds checkpoint step folders.
        :param str | int | None checkpoint: Tag or step to resolve, or ``None``/"latest".
        :return str: Resolved checkpoint step tag.
        """
        if checkpoint is None or str(checkpoint).lower() == "latest":
            latest_path = checkpoint_dir / "latest"
            if latest_path.is_file():
                return latest_path.read_text(encoding="utf-8").strip()
            steps = [
                int(entry.name)
                for entry in checkpoint_dir.iterdir()
                if entry.is_dir() and entry.name.isdigit()
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
        pretrained_checkpoint_dir = Path(pretrained_checkpoint_dir)
        if pretrained_checkpoint_dir.name != "model_checkpoints":
            pretrained_checkpoint_dir = pretrained_checkpoint_dir / "model_checkpoints"
        tag = _resolve_checkpoint_tag(
            pretrained_checkpoint_dir,
            pretrained_checkpoint or cfg.pretrained_checkpoint,
        )
        if use_deepspeed:
            model = load_state_dict_from_zero_checkpoint(
                model, pretrained_checkpoint_dir, tag=str(tag)
            )
        else:
            state_dict_path = pretrained_checkpoint_dir / str(tag) / MODEL_WEIGHTS_NAME
            if not state_dict_path.exists():
                raise ValueError(
                    f"Expected {MODEL_WEIGHTS_NAME} at {state_dict_path}. "
                    "Set pretrained_checkpoint_dir or enable DeepSpeed loading."
                )
            state_dict = load_model_safetensors(
                pretrained_checkpoint_dir / str(tag),
                map_location="cpu",
            )
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

    model = _maybe_compile_model(model, cfg, accelerator, logger)

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
    if cfg.trainer.resume_from_checkpoint:
        if resume_checkpoint_path is not None:
            resume_checkpoint = Path(resume_checkpoint_path)
            if not resume_checkpoint.exists():
                raise FileNotFoundError(
                    f"resume_from_checkpoint path not found: {resume_checkpoint_path}"
                )
            accelerator.load_state(str(resume_checkpoint))
        elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            accelerator.load_state()
        else:
            logger.warning(
                "resume_from_checkpoint is set but no checkpoints were found in "
                f"{checkpoint_dir}"
            )

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
        pad_mask_negatives = None
        packed_negatives = None

        # Choose from one of the finetuning datasets
        def _prepare_attention(
            mask: torch.Tensor, *, name: str
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            """Build additive pad mask or packed seqlens from a 0/1 attention mask.

            :param torch.Tensor mask: Binary attention mask.
            :param str name: Batch name for validation errors.
            :return tuple[torch.Tensor | None, torch.Tensor | None]: Pad mask and packed metadata.
            """
            if use_packed:
                return None, _build_packed_seqlens(mask, name=name)
            pad_mask = torch.where(mask == 1, float(0.0), float("-inf")).type(
                dtype_pad_mask
            )
            return pad_mask, None

        if coin_flip > pretraining_mix_prob:
            # Randomly select which task to draw a batch from
            task_name = numpy.random.choice(
                list(sample_probs.keys()), p=list(sample_probs.values())
            )
            dataloader = dataloaders[task_name]
            batch = next(iter(dataloader))

            pad_mask_queries, packed_queries = _prepare_attention(
                batch["attention_mask_queries"], name="queries"
            )
            pad_mask_corpus, packed_corpus = _prepare_attention(
                batch["attention_mask_corpus"], name="corpus"
            )
            pad_mask_negatives = None
            packed_negatives = None
            if "input_ids_negative" in batch.keys():
                pad_mask_negatives, packed_negatives = _prepare_attention(
                    batch["attention_mask_negative"], name="negative"
                )

            # Update specific number of batches
            metrics[f"train/{task_name}_batches"] += 1

        # Else, we do a step of SimCSE with the original pretraing dataset in order to avoid catastrophic forgetting. Warning: dropout needs to be greater than zero!
        else:
            batch = next(iter(dataloaders["pretraining"]))

            # Here, queries and corpus are identical
            batch["input_ids_queries"] = batch["input_ids"]
            batch["input_ids_corpus"] = batch["input_ids"]

            pad_mask_queries, packed_queries = _prepare_attention(
                batch["attention_mask"], name="pretraining"
            )
            pad_mask_corpus = pad_mask_queries
            packed_corpus = packed_queries

            # Update specific number of batches
            metrics["train/pretraining_batches"] += 1

        # Update global number of batches
        metrics["train/batches"] += 1

        # Under the no_sync context manager, PyTorch will skip synchronizing the gradients when .backward() is
        # called, and the first call to .backward() outside this context manager will trigger the synchronization.
        # Accumulating manually gives more flexibility and is compatible with TPUs.
        is_last_microbatch = (
            metrics["train/batches"] % cfg.trainer.gradient_accumulation_steps == 0
        )
        _maybe_prepare_for_forward(
            optimizer,
            update_step=metrics["train/steps"],
            is_last_microbatch=is_last_microbatch,
        )

        if not is_last_microbatch:
            with accelerator.no_sync(model):
                with accelerator.autocast():
                    # Forward pass
                    queries = model(
                        batch["input_ids_queries"],
                        pad_mask_queries,
                        packed_seqlens=packed_queries,
                    )
                    corpus = model(
                        batch["input_ids_corpus"],
                        pad_mask_corpus,
                        packed_seqlens=packed_corpus,
                    )
                    if "input_ids_negative" in batch.keys():
                        negatives = model(
                            batch["input_ids_negative"],
                            pad_mask_negatives,
                            packed_seqlens=packed_negatives,
                        )

                    # Pool representations
                    pooled_queries = (
                        queries * batch["attention_mask_queries"].unsqueeze(-1)
                    ).sum(dim=1) / batch["attention_mask_queries"].sum(
                        dim=1, keepdim=True
                    )
                    pooled_corpus = (
                        corpus * batch["attention_mask_corpus"].unsqueeze(-1)
                    ).sum(dim=1) / batch["attention_mask_corpus"].sum(
                        dim=1, keepdim=True
                    )

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
            with accelerator.autocast():
                # Forward pass
                queries = model(
                    batch["input_ids_queries"],
                    pad_mask_queries,
                    packed_seqlens=packed_queries,
                )
                corpus = model(
                    batch["input_ids_corpus"],
                    pad_mask_corpus,
                    packed_seqlens=packed_corpus,
                )
                if "input_ids_negative" in batch.keys():
                    negatives = model(
                        batch["input_ids_negative"],
                        pad_mask_negatives,
                        packed_seqlens=packed_negatives,
                    )

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

            if metrics["train/steps"] % log_interval == 0:
                # https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.utils.safe_get_full_grad
                if accelerator.distributed_type is DistributedType.DEEPSPEED:
                    metrics["train/grad_norm"] = model.get_global_grad_norm()  # .item()
                    if cfg.trainer.log_weight_norms and accelerator.is_main_process:
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
                    if cfg.trainer.log_weight_norms and accelerator.is_main_process:
                        metrics["train/weight_norm"] = (
                            sum([p.norm(2) ** 2 for p in model.parameters()]) ** 0.5
                        ).item()

                metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics.log(accelerator)

            # Save accelerator state
            if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                accelerator.save_state()

            # Save model weights checkpoint
            if metrics["train/steps"] % cfg.trainer.save_steps == 0:
                save_total_limit = getattr(cfg.trainer, "save_total_limit", 0)
                max_ckpt = int(getattr(cfg.trainer, "max_ckpt", 0) or 0)
                limit = max(save_total_limit, max_ckpt)
                if limit > 0:
                    # Delete checkpoints if there are too many
                    files = list(model_checkpoint_dir.iterdir())
                    iterations = [int(f.name) for f in files if f.name.isdigit()]
                    iterations.sort()

                    # Remove files with the smallest iterations until the limit is met
                    while iterations and len(iterations) >= limit:
                        file_to_remove = iterations.pop(0)
                        shutil.rmtree(model_checkpoint_dir / str(file_to_remove))
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
                    path = model_checkpoint_dir / str(metrics["train/steps"])
                    path.mkdir(parents=True, exist_ok=True)
                    save_model_safetensors(
                        accelerator.unwrap_model(model),
                        path,
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
