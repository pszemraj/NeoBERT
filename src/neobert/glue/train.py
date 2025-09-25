"""Finetuning a NeoBERT model for sequence classification on GLUE or Super GLUE."""

import json
import math
import os
import random
import shutil
from functools import partial

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import ClassLabel, load_dataset
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from neobert.model import NeoBERTConfig, NeoBERTForSequenceClassification
from neobert.tokenizer import get_tokenizer

from ..config import Config
from ..scheduler import get_scheduler
from ..validation import ValidationError, validate_glue_config
from .process import process_function

logger = get_logger(__name__)

TASK_TO_METRIC = {
    "stsb": "eval_pearson",
    "cola": "eval_matthews_correlation",
    "qqp": "eval_f1",
    "sst2": "eval_accuracy",
    "mnli": "eval_accuracy",
    "mrpc": "eval_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
    "snli": "eval_accuracy",
    "allnli": "eval_accuracy",
}

TASK_TO_TRANSFER_FROM = {
    "mnli": "snli",
    "qnli": "mnli",
    "wnli": "allnli",
    "stsb": "mnli",
    "mrpc": "mnli",
    "rte": "mnli",
}


def get_evaluation(
    model,
    dataloader,
    is_regression,
    metric=None,
    accelerator=None,
    dtype_pad_mask=torch.float32,
    return_predictions=False,
    compute_metric=True,
    flash_attention=False,
):
    samples_seen = 0
    # Fix: Use list for efficient accumulation instead of repeated torch.cat
    predictions_list = [] if return_predictions else None
    eval_metric = None
    progress_bar = tqdm(range(len(dataloader)), desc="Running evaluation...")

    # Ensure Flash Attention is properly disabled for GLUE
    with (
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
        if torch.cuda.is_available()
        else torch.no_grad()
    ):
        for step, batch in tqdm(enumerate(dataloader)):
            progress_bar.update(1)
            with torch.no_grad(), torch.inference_mode():
                if flash_attention:
                    pad_mask = torch.where(
                        batch["attention_mask"] == 1, float(0.0), float("-inf")
                    ).type(dtype_pad_mask)
                else:
                    pad_mask = batch["attention_mask"].type(dtype_pad_mask)
                logits = model(batch["input_ids"], pad_mask)["logits"]

            if not is_regression:
                batch_predictions = logits.argmax(dim=-1)
            else:
                batch_predictions = logits.squeeze()

            if compute_metric:
                if accelerator is not None:
                    batch_predictions, references = accelerator.gather(
                        (batch_predictions, batch["labels"])
                    )
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(dataloader) - 1:
                            batch_predictions = batch_predictions[
                                : len(dataloader.dataset) - samples_seen
                            ]
                            references = references[
                                : len(dataloader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += references.shape[0]
                else:
                    references = batch["labels"]

                metric.add_batch(
                    predictions=batch_predictions,
                    references=references,
                )

            batch_predictions = batch_predictions.to("cpu")

            if return_predictions:
                # Fix: Append to list instead of concatenating tensors
                predictions_list.append(batch_predictions)

    if compute_metric:
        eval_metric = metric.compute()
        if len(eval_metric) > 1:
            eval_metric["combined_score"] = np.mean(list(eval_metric.values())).item()

    # Fix: Concatenate predictions list once at the end
    predictions = torch.cat(predictions_list) if predictions_list else torch.Tensor()

    return {"predictions": predictions, "eval_metric": eval_metric}


def run_evaluation_and_save(
    model,
    eval_dataloader,
    metric,
    cfg,
    accelerator,
    dtype_pad_mask,
    is_regression,
    flash_attention,
    completed_steps,
    epoch,
    train_metric,
    total_loss,
    logger,
    mm_eval_dataloader=None,
    mm_metric=None,
):
    """Run evaluation, log metrics, and save results.

    Returns:
        tuple: (eval_metric dict, current_accuracy float, should_stop bool)
    """
    model.eval()
    eval_result = get_evaluation(
        model=model,
        dataloader=eval_dataloader,
        accelerator=accelerator,
        metric=metric,
        dtype_pad_mask=dtype_pad_mask,
        is_regression=is_regression,
        return_predictions=False,
        flash_attention=flash_attention,
    )
    eval_metric = eval_result["eval_metric"]

    # Log metrics
    if cfg.task == "stsb" and "spearmanr" in eval_metric:
        logger.info(
            f"step {completed_steps} eval pearson: {eval_metric.get('pearson', 0):.4f}"
        )
        logger.info(
            f"step {completed_steps} eval spearmanr: {eval_metric.get('spearmanr', 0):.4f}"
        )
    else:
        logger.info(f"step {completed_steps} eval metric: {eval_metric}")

    logger.info(f"step {completed_steps} train metric: {train_metric}")
    logger.info(
        f"step {completed_steps} train loss: {total_loss / completed_steps if completed_steps > 0 else 0}"
    )

    # Handle MNLI mismatched set
    results = {}
    if cfg.task == "mnli":
        results["accuracy"] = eval_metric["accuracy"]

        if mm_eval_dataloader is not None and mm_metric is not None:
            mm_eval_result = get_evaluation(
                model=model,
                dataloader=mm_eval_dataloader,
                accelerator=accelerator,
                metric=mm_metric,
                dtype_pad_mask=dtype_pad_mask,
                is_regression=is_regression,
                return_predictions=False,
                flash_attention=flash_attention,
            )
            mm_eval_metric = mm_eval_result["eval_metric"]
            results["accuracy_mm"] = mm_eval_metric["accuracy"]
            logger.info(
                f"step {completed_steps} eval metric mismatched: {results['accuracy_mm']}"
            )

    # Prepare metrics for logging
    metrics_to_log = {
        "train_loss": total_loss / completed_steps if completed_steps > 0 else 0,
        "epoch": epoch,
        "step": completed_steps,
        "learning_rate": model.module.config
        if hasattr(model, "module")
        else 0.0001,  # Placeholder
    }

    # Add evaluation metrics
    if cfg.task != "mnli":
        for key, value in eval_metric.items():
            metrics_to_log[f"eval_{key}"] = value
    else:
        for key, value in results.items():
            metrics_to_log[f"eval_{key}"] = value

    # Add training metrics
    if train_metric:
        for key, value in train_metric.items():
            metrics_to_log[f"train_{key}"] = value

    # Log to wandb
    accelerator.log(metrics_to_log, step=completed_steps)

    # Save results to JSON
    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
    if cfg.task == "mnli":
        all_results = {f"eval_{k}": v for k, v in results.items()}

    output_file = os.path.join(
        cfg.trainer.output_dir, f"all_results_step_{completed_steps}.json"
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f)
    logger.info(f"Saved evaluation results to {output_file}")

    # Return current accuracy for early stopping
    curr_accuracy = list(eval_metric.values())[0]

    model.train()
    return eval_metric, curr_accuracy, False  # Last value is early_stop flag


def get_best_checkpoint_path(base_dir, task, num_checkpoints_to_merge=1):
    best_accuracy = -float("inf")
    best_checkpoint_path = None
    best_checkpoint = None

    # Explore all directories in the given structure
    for root, _, files in os.walk(base_dir):
        if task in root:
            # Filter out the JSON files following the naming convention
            json_files = [
                f
                for f in files
                if f.startswith("all_results_step_") and f.endswith(".json")
            ]

            for json_file in json_files:
                json_path = os.path.join(root, json_file)

                # Read the eval accuracy from the JSON file
                with open(json_path, "r") as f:
                    results = json.load(f)
                    eval_accuracy = results.get(TASK_TO_METRIC[task], 0)

                    # Extract step number from the file name (e.g., all_results_step_{i}.json)
                    step_number = int(json_file.split("_")[3].split(".")[0])

                    # Update if a higher eval_accuracy is found
                    if eval_accuracy > best_accuracy:
                        best_accuracy = eval_accuracy

                        # Construct the corresponding checkpoint folder path
                        checkpoint_folder = os.path.join(root, "model_checkpoints")
                        checkpoint = step_number
                        if os.path.exists(
                            os.path.join(checkpoint_folder, str(checkpoint))
                        ):
                            best_checkpoint_path, best_checkpoint = (
                                checkpoint_folder,
                                checkpoint,
                            )

    checkpoint_list = [best_checkpoint]
    if num_checkpoints_to_merge > 1:
        ckpts = list(os.listdir(best_checkpoint_path))
        ckpts = [int(ckpt) for ckpt in ckpts if int(ckpt) <= int(best_checkpoint)]
        ckpts.sort()

        checkpoint_list = (
            ckpts
            if len(ckpts) < num_checkpoints_to_merge
            else ckpts[-num_checkpoints_to_merge:]
        )

    return best_checkpoint_path, checkpoint_list


def load_pretrained_weights(model, checkpoint_dir, checkpoint_id, logger):
    """Load pretrained weights from a checkpoint directory.

    Args:
        model: The model to load weights into
        checkpoint_dir: Directory containing checkpoint folders
        checkpoint_id: Specific checkpoint number/name to load
        logger: Logger for output

    Returns:
        model with loaded weights
    """
    checkpoint_path = os.path.join(checkpoint_dir, str(checkpoint_id))

    # Check if it's a DeepSpeed checkpoint
    is_deepspeed = os.path.exists(os.path.join(checkpoint_path, "zero_to_fp32.py"))

    if is_deepspeed:
        logger.info(f"Loading DeepSpeed checkpoint from {checkpoint_path}")
        try:
            model = load_state_dict_from_zero_checkpoint(
                model,
                checkpoint_path,
                tag="",  # Empty tag since path includes checkpoint number
            )
            logger.info("Successfully loaded DeepSpeed checkpoint")
        except Exception as e:
            logger.error(f"Failed to load DeepSpeed checkpoint: {e}")
            raise
    else:
        # Load state_dict directly
        state_dict_path = os.path.join(checkpoint_path, "state_dict.pt")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"No state_dict.pt found at {state_dict_path}")

        logger.info(f"Loading state dict from {state_dict_path}")
        state_dict = torch.load(state_dict_path)

        # Log state dict info
        logger.info(f"Loaded state dict with {len(state_dict)} keys")

        # Remove classifier and decoder keys for fine-tuning
        cleaned_state_dict = {
            k: v
            for k, v in state_dict.items()
            if "classifier" not in k and "decoder" not in k
        }
        logger.info(f"After filtering: {len(cleaned_state_dict)} keys to load")

        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_state_dict, strict=False
        )

        if missing_keys:
            logger.info(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys: {unexpected_keys}")

        logger.info(f"✅ Successfully loaded pretrained weights from {state_dict_path}")

    return model


def save_training_checkpoint(cfg, model, accelerator, completed_steps):
    """Save a training checkpoint during fine-tuning.

    Args:
        cfg: Configuration object
        model: Model to save
        accelerator: Accelerator object
        completed_steps: Current training step
    """
    model_checkpoint_dir = os.path.join(cfg.trainer.output_dir, "model_checkpoints")

    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
    save_total_limit = getattr(cfg.trainer, "save_total_limit", None)

    # Determine effective limit from save_total_limit (preferred) or max_ckpt
    effective_limit = None
    if save_total_limit is not None and save_total_limit > 0:
        effective_limit = save_total_limit
    elif max_ckpt is not None and max_ckpt > 0:
        effective_limit = max_ckpt

    if effective_limit is not None and os.path.isdir(model_checkpoint_dir):
        files = os.listdir(model_checkpoint_dir)
        iterations = sorted([int(f) for f in files if f.isdigit()])

        # Remove oldest checkpoints until under limit
        while iterations and len(iterations) >= effective_limit:
            file_to_remove = iterations.pop(0)
            shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
            print(
                f"Deleted old model checkpoint {file_to_remove} due to limit "
                f"(limit = {effective_limit})"
            )

    # Save the checkpoint
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        model.save_checkpoint(model_checkpoint_dir, tag=completed_steps)
    else:
        path = os.path.join(model_checkpoint_dir, str(completed_steps))
        os.makedirs(path, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(path, "state_dict.pt"),
        )


def trainer(cfg: Config):
    # Extract task and meta_task from config
    task = cfg.glue.task_name if hasattr(cfg, "glue") else cfg.task
    meta_task = "glue"  # Default for GLUE tasks
    experiment_id = getattr(cfg, "id", "0")

    # Update cfg to have these as direct attributes for compatibility
    cfg.task = task
    cfg.meta_task = meta_task
    cfg.id = experiment_id
    cfg.mode = getattr(cfg, "mode", "eval")  # Default to eval mode
    cfg.num_labels = cfg.glue.num_labels if hasattr(cfg, "glue") else 2
    cfg.max_seq_len = cfg.glue.max_seq_length if hasattr(cfg, "glue") else 128
    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=False,
    )
    # Handle mixed precision setting (could be bool or string)
    mixed_precision = cfg.trainer.mixed_precision
    if isinstance(mixed_precision, bool):
        mixed_precision = "no" if not mixed_precision else "bf16"
    elif mixed_precision == "fp32":
        mixed_precision = "no"

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=mixed_precision,
        project_config=project_config,
    )

    # Initialise the wandb run and pass wandb parameters
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

    set_seed(int(cfg.seed))

    # Validate configuration after accelerator is initialized (for logger)
    try:
        validate_glue_config(cfg)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Handle the repository creation
    if accelerator.is_main_process:
        if os.path.isdir(cfg.trainer.output_dir):
            for file in os.listdir(cfg.trainer.output_dir):
                if os.path.isfile(os.path.join(cfg.trainer.output_dir, str(file))):
                    os.remove(os.path.join(cfg.trainer.output_dir, str(file)))
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Override flash attention setting for GLUE - always use eager attention
    # Flash attention has memory alignment issues with variable-length sequences in GLUE tasks
    # xformers requires sequences to be aligned to multiples of 8, which is incompatible
    # with GLUE's dynamic batching and variable sequence lengths
    if hasattr(cfg.model, "flash_attention") and cfg.model.flash_attention:
        logger.warning(
            "Flash attention is not supported for GLUE evaluation due to memory alignment issues "
            "with variable-length sequences. Using eager attention instead."
        )
    flash_attention = False  # Always use eager attention for GLUE

    # Check from_hub in raw model dict for GLUE tasks
    from_hub = False
    if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
        from_hub = cfg._raw_model_dict.get("from_hub", False)
    elif hasattr(cfg.model, "from_hub"):
        from_hub = cfg.model.from_hub

    if from_hub:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            use_fast=True,
            revision="main",
            trust_remote_code=True,
        )
    else:
        # Import our new config system
        from neobert.config import ConfigLoader

        # For GLUE, we MUST have pretrained model info
        # Check if we're allowing random weights for testing
        allow_random_weights = (
            hasattr(cfg, "_raw_model_dict")
            and cfg._raw_model_dict
            and cfg._raw_model_dict.get("allow_random_weights", False)
        )

        if allow_random_weights:
            # Skip pretrained config loading for testing
            pretrained_config_path = None
        elif (
            hasattr(cfg, "_raw_model_dict")
            and cfg._raw_model_dict
            and "pretrained_config_path" in cfg._raw_model_dict
        ):
            pretrained_config_path = cfg._raw_model_dict["pretrained_config_path"]
        else:
            raise ValueError(
                "GLUE evaluation requires a pretrained model! "
                "Please specify 'pretrained_config_path' in the model section of your config, "
                "or set 'allow_random_weights: true' for testing."
            )
        if pretrained_config_path:
            model_pretraining_config = ConfigLoader.load(pretrained_config_path)
            model_pretraining_config.model.flash_attention = flash_attention
            tokenizer = get_tokenizer(
                pretrained_model_name_or_path=model_pretraining_config.tokenizer.name,
                max_length=model_pretraining_config.tokenizer.max_length,
            )
        else:
            # Use default tokenizer for random weights test
            tokenizer = get_tokenizer(
                pretrained_model_name_or_path="bert-base-uncased",
                max_length=128,
            )

    print("Loading metric...")
    # Get the metric function
    if cfg.task in ("multirc", "record"):
        metric = evaluate.load("accuracy", experiment_id=cfg.id)
    elif cfg.task == "snli":
        metric = evaluate.load(cfg.meta_task, "mnli", experiment_id=cfg.id)
    elif cfg.task == "allnli":
        metric = evaluate.load(cfg.meta_task, "wnli", experiment_id=cfg.id)
    else:
        metric = evaluate.load(cfg.meta_task, cfg.task, experiment_id=cfg.id)

    # Load additional metric for the mismatched validation set of mnli
    if cfg.task == "mnli":
        mm_metric = evaluate.load(
            cfg.meta_task, "mnli_mismatched", experiment_id=cfg.id
        )

    # def compute_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    #     result = metric.compute(predictions=preds, references=p.label_ids)
    #     if len(result) > 1:
    #         result["combined_score"] = np.mean(list(result.values())).item()
    #     return result

    # Loading the dataset
    print("Loading dataset...")
    if cfg.task == "snli":
        raw_datasets = load_dataset("stanfordnlp/snli")
        raw_datasets = raw_datasets.filter(lambda example: example["label"] != -1)
    elif cfg.task == "allnli":
        raw_datasets = load_dataset("sentence-transformers/all-nli", name="pair-class")

        def collapse_classes(examples):
            examples["label"] = [
                1 if label == 2 else label for label in examples["label"]
            ]
            return examples

        raw_datasets.map(
            collapse_classes,
            batched=True,
            desc="Collapsing classes into entailment and not entailment.",
        )

    elif cfg.meta_task == "glue":
        raw_datasets = load_dataset("glue", cfg.task)
    else:
        raw_datasets = load_dataset(
            "json",
            data_dir=os.path.join(
                os.environ["HF_DATASETS_CACHE"], "super_glue", cfg.task
            ),
        )

    # Split between train and validation for datasets that don't have it natively
    if cfg.task in ("axb", "axg"):
        tmp = raw_datasets["train"].train_test_split(test_size=0.1)
        raw_datasets["train"] = tmp["train"]
        raw_datasets["validation"] = tmp["test"]

    # Preprocessing the datasets
    mapping = partial(process_function, tokenizer=tokenizer, cfg=cfg)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            mapping,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Preprocessing the dataset",
        )

    is_regression = cfg.task == "stsb"
    if not is_regression:
        processed_datasets = processed_datasets.cast_column(
            "labels", ClassLabel(names=processed_datasets["train"].unique("labels"))
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched"
        if cfg.task == "mnli"
        else ("dev" if cfg.task == "allnli" else "validation")
    ]

    if cfg.task == "mnli":
        mm_eval_dataset = processed_datasets["validation_mismatched"]

    # Labels
    if not is_regression:
        label_list = train_dataset.features["labels"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # Log a few random samples from the evaluation set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    def collate_fn(batch):
        batch = data_collator(batch)
        batch["attention_mask"] = torch.where(
            batch["attention_mask"] == 1, float(0.0), float("-inf")
        ).type(dtype_pad_mask)
        return batch

    # Use per_device batch sizes consistently
    train_batch_size = (
        cfg.trainer.per_device_train_batch_size or cfg.trainer.train_batch_size or 16
    )
    eval_batch_size = (
        cfg.trainer.per_device_eval_batch_size or cfg.trainer.eval_batch_size or 32
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=eval_batch_size
    )
    if cfg.task == "mnli":
        mm_eval_dataloader = DataLoader(
            mm_eval_dataset,
            collate_fn=collate_fn,
            batch_size=eval_batch_size,
        )

    # Model
    if from_hub:
        config = AutoConfig.from_pretrained(
            cfg.model.name,
            num_labels=num_labels,
            finetuning_task=cfg.task,
            revision="main",
            trust_remote_code=True,
        )
        # if "nomic" in cfg.model.name:
        #     base_model = AutoModelForMaskedLM.from_pretrained(
        #         cfg.model.name,
        #         from_tf=False,
        #         config=config,
        #         revision="main",
        #         trust_remote_code=True,
        #         ignore_mismatched_sizes=False,
        #     )
        #     model = NomicBERTForSequenceClassification(
        #         config,
        #         base_model.bert,
        #         num_labels=num_labels,
        #         classifier_dropout=cfg.model.classifier_dropout,
        #         classifier_init_range=cfg.model.classifier_init_range,
        #     )
        # else:
        if True:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model.name,
                from_tf=False,
                config=config,
                revision="main",
                trust_remote_code=True,
                ignore_mismatched_sizes=False,
            )
    else:
        # Convert config objects to dict for unpacking
        if "model_pretraining_config" in locals() and model_pretraining_config:
            model_config_dict = (
                model_pretraining_config.model.__dict__.copy()
                if hasattr(model_pretraining_config.model, "__dict__")
                else {}
            )
        elif hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            # Use raw model dict when allow_random_weights is true
            model_config_dict = cfg._raw_model_dict.copy()
        else:
            # Fallback to cfg.model attributes
            model_config_dict = {
                "hidden_size": getattr(cfg.model, "hidden_size", 768),
                "num_hidden_layers": getattr(cfg.model, "num_hidden_layers", 12),
                "num_attention_heads": getattr(cfg.model, "num_attention_heads", 12),
                "intermediate_size": getattr(cfg.model, "intermediate_size", 3072),
                "vocab_size": getattr(cfg.model, "vocab_size", 30522),
                "hidden_act": getattr(cfg.model, "hidden_act", "gelu"),
                "max_position_embeddings": getattr(
                    cfg.model, "max_position_embeddings", 512
                ),
                "layer_norm_eps": getattr(cfg.model, "layer_norm_eps", 1e-12),
            }

        # Map dropout_prob to dropout and remove classifier_init_range from model config
        if "dropout_prob" in model_config_dict:
            model_config_dict["dropout"] = model_config_dict.pop("dropout_prob")
        if "classifier_init_range" in model_config_dict:
            model_config_dict.pop("classifier_init_range")
        if "allow_random_weights" in model_config_dict:
            model_config_dict.pop("allow_random_weights")
        if "pretrained_checkpoint_dir" in model_config_dict:
            model_config_dict.pop("pretrained_checkpoint_dir")
        if "pretrained_checkpoint" in model_config_dict:
            model_config_dict.pop("pretrained_checkpoint")

        # Use model config directly - don't merge with tokenizer config
        # The tokenizer's vocab_size should match the model's anyway
        combined_config = model_config_dict

        model = NeoBERTForSequenceClassification(
            NeoBERTConfig(**combined_config),
            num_labels=num_labels,
            classifier_dropout=getattr(cfg.model, "classifier_dropout", 0.1),
            classifier_init_range=getattr(cfg.model, "classifier_init_range", 0.02),
        )

    if hasattr(cfg.model, "transfer_from_task") and cfg.model.transfer_from_task:
        task_to_transfer_from = TASK_TO_TRANSFER_FROM.get(cfg.task, None)
        if not task_to_transfer_from:
            raise ValueError(f"Task to transfer from for {cfg.task} is not set.")
        cfg.model.pretrained_checkpoint_dir, checkpoint_list = get_best_checkpoint_path(
            os.path.join(
                cfg.model.pretrained_checkpoint_dir,
                "glue",
                str(cfg.model.pretrained_checkpoint),
            ),
            task_to_transfer_from,
        )
        cfg.model.pretrained_checkpoint = checkpoint_list[-1]
        logger.info(
            f"Transfering from: {cfg.model.pretrained_checkpoint_dir}, {cfg.model.pretrained_checkpoint}"
        )
        if (
            not cfg.model.pretrained_checkpoint_dir
            or not cfg.model.pretrained_checkpoint
        ):
            raise ValueError("Unable to retrieve checkpoint to transfer from.")

    else:
        # Get checkpoint info from raw model dict for GLUE
        logger.info("Looking for pretrained checkpoint info...")

        # Check for checkpoint info in raw model dict
        if hasattr(cfg, "_raw_model_dict") and cfg._raw_model_dict:
            pretrained_checkpoint_dir = cfg._raw_model_dict.get(
                "pretrained_checkpoint_dir", None
            )
            pretrained_checkpoint = cfg._raw_model_dict.get(
                "pretrained_checkpoint", None
            )
            allow_random_weights = cfg._raw_model_dict.get(
                "allow_random_weights", False
            )
        # Also check GLUEConfig if available
        elif hasattr(cfg, "glue"):
            pretrained_checkpoint_dir = cfg.glue.pretrained_checkpoint_dir
            pretrained_checkpoint = cfg.glue.pretrained_checkpoint
            allow_random_weights = cfg.glue.allow_random_weights
        else:
            pretrained_checkpoint_dir = None
            pretrained_checkpoint = None
            allow_random_weights = False

        # Validate checkpoint configuration
        if not pretrained_checkpoint_dir or not pretrained_checkpoint:
            if allow_random_weights:
                logger.warning(
                    "⚠️  Using random weights for testing as allow_random_weights=true"
                )
                pretrained_checkpoint = None
            else:
                raise ValueError(
                    "GLUE evaluation requires pretrained weights!\n"
                    "Please specify either:\n"
                    "  1. 'pretrained_checkpoint_dir' and 'pretrained_checkpoint' in model config\n"
                    "  2. Set 'allow_random_weights: true' for testing with random weights"
                )
        else:
            # Ensure we have the full path to model_checkpoints
            if not pretrained_checkpoint_dir.endswith("model_checkpoints"):
                pretrained_checkpoint_dir = os.path.join(
                    pretrained_checkpoint_dir, "model_checkpoints"
                )
            logger.info(
                f"Will load checkpoint {pretrained_checkpoint} from {pretrained_checkpoint_dir}"
            )

    # Load pretrained weights if available
    if (
        not from_hub
        and "pretrained_checkpoint" in locals()
        and pretrained_checkpoint is not None
    ):
        model = load_pretrained_weights(
            model, pretrained_checkpoint_dir, pretrained_checkpoint, logger
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]

    # Handle both config styles (with and without hparams)
    if hasattr(cfg.optimizer, "hparams"):
        weight_decay = cfg.optimizer.hparams.weight_decay
        optimizer_params = cfg.optimizer.hparams
    else:
        weight_decay = cfg.optimizer.weight_decay
        optimizer_params = {
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
            "betas": getattr(cfg.optimizer, "betas", [0.9, 0.999]),
            "eps": getattr(cfg.optimizer, "eps", 1e-8),
        }

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)

    # Calculate training steps consistently
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.trainer.gradient_accumulation_steps
    )

    # Determine max_steps: explicit value or calculate from epochs
    if cfg.trainer.max_steps is None or cfg.trainer.max_steps <= 0:
        cfg.trainer.max_steps = (
            cfg.trainer.num_train_epochs * num_update_steps_per_epoch
        )
        logger.info(f"Calculated max_steps from epochs: {cfg.trainer.max_steps}")
    else:
        logger.info(f"Using explicit max_steps: {cfg.trainer.max_steps}")

    if cfg.scheduler.warmup_percent is not None:
        if cfg.scheduler.warmup_steps is not None:
            UserWarning(
                "Overrinding number of warmup steps based on warmup percentage."
            )
        cfg.scheduler.warmup_steps = math.ceil(
            cfg.trainer.max_steps / 100 * cfg.scheduler.warmup_percent
        )
    if cfg.scheduler.decay_percent is not None:
        if cfg.scheduler.decay_steps is not None:
            UserWarning("Overrinding number of decay steps based on decay percentage.")
        cfg.scheduler.decay_steps = math.ceil(
            cfg.trainer.max_steps / 100 * cfg.scheduler.decay_percent
        )
    elif cfg.scheduler.decay_steps is None:
        # For linear scheduler without decay_percent, set decay_steps to total steps
        cfg.scheduler.decay_steps = cfg.trainer.max_steps

    # Get learning rate from optimizer config
    lr = optimizer_params.get(
        "lr",
        optimizer_params["lr"]
        if isinstance(optimizer_params, dict)
        else cfg.optimizer.lr,
    )

    # Convert scheduler config to dict if needed
    scheduler_params = (
        cfg.scheduler.__dict__.copy()
        if hasattr(cfg.scheduler, "__dict__")
        else cfg.scheduler.copy()
    )

    # Map 'name' to 'decay' if present
    if "name" in scheduler_params:
        scheduler_params["decay"] = scheduler_params.pop("name")

    # Debug logging
    logger.info(f"Scheduler params before calling get_scheduler: {scheduler_params}")
    logger.info(
        f"warmup_steps: {scheduler_params.get('warmup_steps')}, decay_steps: {scheduler_params.get('decay_steps')}"
    )

    scheduler = get_scheduler(optimizer=optimizer, lr=lr, **scheduler_params)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
    )

    if cfg.task == "mnli":
        mm_eval_dataloader = accelerator.prepare(mm_eval_dataloader)

    # Recalculate steps after accelerator preparation
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.trainer.gradient_accumulation_steps
    )
    # Recalculate epochs based on max_steps
    cfg.trainer.num_train_epochs = math.ceil(
        cfg.trainer.max_steps / num_update_steps_per_epoch
    )

    # Handle evaluation strategy - support both 'epoch' and 'steps'
    eval_strategy = getattr(cfg.trainer, "eval_strategy", "steps")
    if eval_strategy == "epoch":
        # Evaluate at the end of each epoch
        cfg.trainer.eval_steps = num_update_steps_per_epoch
        logger.info(
            f"Using epoch-based evaluation: will evaluate every {cfg.trainer.eval_steps} steps (1 epoch)"
        )
    elif eval_strategy == "steps":
        # Use the provided eval_steps or default to min of provided and epoch size
        if hasattr(cfg.trainer, "eval_steps"):
            cfg.trainer.eval_steps = min(
                cfg.trainer.eval_steps,
                len(train_dataset) // train_batch_size,
            )
        else:
            cfg.trainer.eval_steps = min(500, num_update_steps_per_epoch)
            logger.info(
                f"No eval_steps provided, defaulting to {cfg.trainer.eval_steps}"
            )
    else:
        raise ValueError(
            f"Invalid eval_strategy: {eval_strategy}. Must be 'epoch' or 'steps'"
        )

    # To keep the last n checkpoints before the best model and do k cycles before early stopping, we save the last k+n models
    early_stopping = getattr(cfg.trainer, "early_stopping", 0)
    max_ckpt = getattr(cfg.trainer, "max_ckpt", 0)
    if max_ckpt is not None and max_ckpt > 0 and early_stopping > 0:
        cfg.trainer.max_ckpt = max_ckpt + early_stopping

    # Get loss function
    if not is_regression:
        loss_fct = CrossEntropyLoss()
    else:
        loss_fct = MSELoss()

    # Train!
    total_steps = cfg.trainer.max_steps
    total_batch_size = (
        train_batch_size
        * accelerator.num_processes
        * cfg.trainer.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {cfg.task}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num epochs = {cfg.trainer.num_train_epochs}")
    logger.info(f"  Total training steps = {total_steps}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Learning rate = {lr}")
    logger.info(
        f"  Gradient accumulation steps = {cfg.trainer.gradient_accumulation_steps}"
    )
    logger.info(f"  Evaluation steps = {cfg.trainer.eval_steps}")
    logger.info(f"  Early stopping cycles = {early_stopping}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(total_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if cfg.trainer.resume_from_checkpoint:
        if (
            cfg.trainer.resume_from_checkpoint is not None
            or cfg.trainer.resume_from_checkpoint != ""
        ):
            accelerator.print(f"Resumed from checkpoint: {cfg.trainer.checkpoint_dir}")
            accelerator.load_state(cfg.trainer.checkpoint_dir)
            path = os.path.basename(cfg.trainer.checkpoint_dir)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * cfg.trainer.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // cfg.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Initialize all training loop variables upfront
    results = {}
    total_loss = 0.0
    early_stop = False
    prev_accuracy = 0.0
    early_stopping_counter = 1
    eval_metric = {}
    epoch = starting_epoch
    completed_steps = completed_steps  # Ensure it's in scope

    for epoch in range(starting_epoch, cfg.trainer.num_train_epochs):
        for batch in train_dataloader:
            logits = model(batch["input_ids"], batch["attention_mask"])["logits"]

            # Debug logging for first few steps
            if completed_steps < 3:
                logger.info(
                    f"Step {completed_steps}: logits shape: {logits.shape}, logits mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}"
                )
                logger.info(
                    f"Step {completed_steps}: logits sample: {logits[0].detach().cpu()}"
                )
                logger.info(f"Step {completed_steps}: labels: {batch['labels'][:5]}")

            if not is_regression:
                loss = loss_fct(logits.view(-1, num_labels), batch["labels"].view(-1))
            else:
                if num_labels == 1:
                    loss = loss_fct(logits.squeeze(), batch["labels"].squeeze())
                else:
                    loss = loss_fct(logits, batch["labels"])

            # Compute train accuracy
            predictions = (
                logits.argmax(dim=-1)
                if not is_regression
                else (logits.squeeze() if logits.size() != torch.Size([1]) else logits)
            )
            predictions, references = accelerator.gather((predictions, batch["labels"]))

            # print(logits, loss)

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)
            completed_steps += 1

            # We keep track of the loss at each epoch
            total_loss += loss.item()

            # Run evaluation
            if (
                completed_steps
                % min(
                    cfg.trainer.eval_steps,
                    len(train_dataloader) * train_batch_size // 10,
                )
                == 0
            ):
                train_metric = metric.compute()
                if len(train_metric) > 1:
                    train_metric["combined_score"] = np.mean(
                        list(train_metric.values())
                    ).item()

                model.eval()
                eval_metric = get_evaluation(
                    model=model,
                    dataloader=eval_dataloader,
                    accelerator=accelerator,
                    metric=metric,
                    dtype_pad_mask=dtype_pad_mask,
                    is_regression=is_regression,
                    return_predictions=False,
                    flash_attention=flash_attention,
                )["eval_metric"]

                # Log all metrics properly for STS-B (both Pearson and Spearman)
                if cfg.task == "stsb" and "spearmanr" in eval_metric:
                    logger.info(
                        f"step {completed_steps} eval pearson: {eval_metric.get('pearson', 0):.4f}"
                    )
                    logger.info(
                        f"step {completed_steps} eval spearmanr: {eval_metric.get('spearmanr', 0):.4f}"
                    )
                else:
                    logger.info(f"step {completed_steps} eval metric: {eval_metric}")

                logger.info(f"step {completed_steps} train metric: {train_metric}")
                logger.info(
                    f"step {completed_steps} train loss: {total_loss / completed_steps}"
                )

                if cfg.task == "mnli":
                    # Evaluation on matched MNLI
                    results["accuracy"] = eval_metric["accuracy"]

                    # Evaluation on mismatched MNLI
                    mm_eval_metric = get_evaluation(
                        model=model,
                        dataloader=mm_eval_dataloader,
                        accelerator=accelerator,
                        metric=mm_metric,
                        dtype_pad_mask=dtype_pad_mask,
                        is_regression=is_regression,
                        return_predictions=False,
                        flash_attention=flash_attention,
                    )["eval_metric"]
                    results["accuracy_mm"] = mm_eval_metric["accuracy"]

                    res_mm = results["accuracy_mm"]
                    logger.info(
                        f"step {completed_steps} eval metric mismatched: {res_mm}"
                    )

                # Flatten eval metrics for proper wandb logging
                metrics_to_log = {
                    "train_loss": total_loss / completed_steps,
                    "epoch": epoch,
                    "step": completed_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                # Add evaluation metrics with proper names
                if cfg.task != "mnli":
                    for key, value in eval_metric.items():
                        metrics_to_log[f"eval_{key}"] = value
                else:
                    for key, value in results.items():
                        metrics_to_log[f"eval_{key}"] = value

                # Add training metrics
                if train_metric:
                    for key, value in train_metric.items():
                        metrics_to_log[f"train_{key}"] = value

                accelerator.log(metrics_to_log, step=completed_steps)

                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                if cfg.task == "mnli":
                    all_results = {f"eval_{k}": v for k, v in results.items()}

                with open(
                    os.path.join(
                        cfg.trainer.output_dir,
                        f"all_results_step_{completed_steps}.json",
                    ),
                    "w",
                ) as f:
                    print(
                        "dumping in",
                        os.path.join(
                            cfg.trainer.output_dir,
                            f"all_results_step_{completed_steps}.json",
                        ),
                    )
                    json.dump(all_results, f)

                curr_accuracy = list(eval_metric.values())[0]

                # Update early stopping counter
                if curr_accuracy > prev_accuracy:
                    prev_accuracy = curr_accuracy
                    early_stopping_counter = 0

                else:
                    early_stopping_counter += 1

                if early_stopping > 0 and early_stopping_counter >= early_stopping:
                    print(
                        f"Evaluation accuracy has not improved in {early_stopping} cycles of {cfg.trainer.eval_steps} evaluation steps, stopping the training."
                    )
                    early_stop = True

                # Save model checkpoint based on save_strategy
                save_strategy = getattr(cfg.trainer, "save_strategy", "steps")
                should_save = False

                if (
                    save_strategy == "epoch"
                    and completed_steps % num_update_steps_per_epoch == 0
                ):
                    should_save = True
                elif save_strategy == "steps" and hasattr(cfg.trainer, "save_steps"):
                    if completed_steps % cfg.trainer.save_steps == 0:
                        should_save = True
                elif save_strategy == "best":
                    # Save only if this is the best model so far
                    if curr_accuracy > prev_accuracy:
                        should_save = True
                elif save_strategy != "no":
                    # Default to saving at eval steps if strategy is not 'no'
                    should_save = True

                # Only save checkpoint if explicitly enabled
                save_model = getattr(cfg.trainer, "save_model", True)
                save_total_limit = getattr(cfg.trainer, "save_total_limit", None)

                # Save if either save_total_limit>0 or max_ckpt>0 is configured
                limit_enabled = (
                    save_total_limit is not None and save_total_limit > 0
                ) or (max_ckpt is not None and max_ckpt > 0)

                if should_save and save_model and limit_enabled:
                    save_training_checkpoint(cfg, model, accelerator, completed_steps)

                model.train()

            if completed_steps >= cfg.trainer.max_steps or early_stop:
                break

            # Reset the gradient
            optimizer.zero_grad()

        if completed_steps >= cfg.trainer.max_steps or early_stop:
            break

    # Log final metrics to wandb before ending training
    if eval_metric:  # Only create final metrics if we have evaluation results
        final_metrics = {f"eval_{k}": v for k, v in eval_metric.items()}
    else:
        final_metrics = {}

    if cfg.task == "mnli":
        final_metrics = {f"eval_{k}": v for k, v in results.items()}

    # Print final metrics to console (both logger and print for visibility)
    if accelerator.is_main_process:
        print("=" * 60)
        print(f"Training completed for {cfg.task.upper()}")
        print(f"Final metrics at step {completed_steps}:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 60)

        # Also log for debugging
        logger.info("=" * 60)
        logger.info(f"Training completed for {cfg.task.upper()}")
        logger.info(f"Final metrics at step {completed_steps}:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 60)

    # Add final metrics to wandb
    accelerator.log(
        {
            **final_metrics,
            "final_train_loss": total_loss / completed_steps
            if completed_steps > 0
            else 0,
            "final_epoch": epoch,
            "final_step": completed_steps,
        },
        step=completed_steps,
    )

    # Fix: Update W&B summary with final metrics
    if accelerator.is_main_process:
        try:
            # Get wandb tracker and update summary
            for tracker in accelerator.trackers:
                if tracker.__class__.__name__ == "WandBTracker":
                    if hasattr(tracker, "run") and tracker.run:
                        # Update summary with final metrics
                        summary_metrics = {
                            f"summary/{k}": v for k, v in final_metrics.items()
                        }
                        summary_metrics["summary/final_train_loss"] = (
                            total_loss / completed_steps if completed_steps > 0 else 0
                        )
                        summary_metrics["summary/final_step"] = completed_steps
                        tracker.run.summary.update(summary_metrics)
                        logger.info("Updated W&B run summary with final metrics")
        except Exception as e:
            logger.warning(f"Failed to update W&B summary: {e}")

    accelerator.end_training()

    # Save final results to JSON
    with open(os.path.join(cfg.trainer.output_dir, "all_results.json"), "w") as f:
        json.dump(final_metrics, f)

    # Also save to timestamped file for clarity
    with open(
        os.path.join(
            cfg.trainer.output_dir, f"all_results_step_{completed_steps}.json"
        ),
        "w",
    ) as f:
        json.dump(final_metrics, f)
        logger.info(
            f"Final results saved to {cfg.trainer.output_dir}/all_results_step_{completed_steps}.json"
        )
