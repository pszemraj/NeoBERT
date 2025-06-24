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
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from neobert.model import (
    NeoBERTConfig,
    NeoBERTForSequenceClassification,
    NomicBERTForSequenceClassification,
)
from neobert.tokenizer import get_tokenizer

from ..config import Config
from ..scheduler import get_scheduler
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
    predictions = torch.Tensor()
    eval_metric = None
    progress_bar = tqdm(range(len(dataloader)), desc="Running evaluation...")
    for step, batch in tqdm(enumerate(dataloader)):
        progress_bar.update(1)
        with torch.no_grad():
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
            predictions = torch.cat([predictions, batch_predictions])

    if compute_metric:
        eval_metric = metric.compute()
        if len(eval_metric) > 1:
            eval_metric["combined_score"] = np.mean(list(eval_metric.values())).item()

    return {"predictions": predictions, "eval_metric": eval_metric}


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


def save_checkpoint(cfg, model, accelerator, completed_steps):
    model_checkpoint_dir = os.path.join(cfg.trainer.dir, "model_checkpoints")

    # Delete checkpoints with the lesser evaluation accuracy if there are too many
    if (
        cfg.trainer.max_ckpt is not None
        and cfg.trainer.max_ckpt > 0
        and os.path.isdir(model_checkpoint_dir)
    ):
        files = os.listdir(model_checkpoint_dir)
        iterations = [f for f in files if f.isdigit()]
        iterations.sort()

        # Remove files with the smallest iterations until the limit is met
        while iterations is not None and len(iterations) >= cfg.trainer.max_ckpt:
            file_to_remove = iterations.pop(0)
            shutil.rmtree(os.path.join(model_checkpoint_dir, str(file_to_remove)))
            print(
                f"Deleted old model checkpoint {file_to_remove} due to limit "
                f"(max_ckpt = {cfg.trainer.max_ckpt})"
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
    # Accelerator object
    project_config = ProjectConfiguration(
        cfg.trainer.output_dir,
        automatic_checkpoint_naming=False,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="no"
        if cfg.trainer.mixed_precision == "fp32"
        else cfg.trainer.mixed_precision,
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

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg.trainer.tf32

    # Handle the repository creation
    if accelerator.is_main_process:
        if os.path.isdir(cfg.trainer.dir):
            for file in os.listdir(cfg.trainer.dir):
                if os.path.isfile(os.path.join(cfg.trainer.dir, str(file))):
                    os.remove(os.path.join(cfg.trainer.dir, str(file)))
        os.makedirs(cfg.trainer.dir, exist_ok=True)
    accelerator.wait_for_everyone()

    flash_attention = cfg.flash_attention if "flash_attention" in cfg.keys() else True
    if cfg.model.from_hub:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            use_fast=True,
            revision="main",
            trust_remote_code=True,
        )
    else:
        # Import our new config system
        from neobert.config import ConfigLoader

        model_pretraining_config = ConfigLoader.load(cfg.model.pretrained_config_path)
        model_pretraining_config.model.flash_attention = flash_attention
        tokenizer = get_tokenizer(
            model_pretraining_config.tokenizer.name,
            model_pretraining_config.tokenizer.path,
            model_pretraining_config.tokenizer.max_length,
            model_pretraining_config.tokenizer.padding,
            model_pretraining_config.tokenizer.truncation,
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
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    def collate_fn(batch):
        batch = data_collator(batch)
        batch["attention_mask"] = torch.where(
            batch["attention_mask"] == 1, float(0.0), float("-inf")
        ).type(dtype_pad_mask)
        return batch

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.trainer.train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=cfg.trainer.eval_batch_size
    )
    if cfg.task == "mnli":
        mm_eval_dataloader = DataLoader(
            mm_eval_dataset,
            collate_fn=collate_fn,
            batch_size=cfg.trainer.eval_batch_size,
        )

    # Model
    if cfg.model.from_hub:
        config = AutoConfig.from_pretrained(
            cfg.model.name,
            num_labels=num_labels,
            finetuning_task=cfg.task,
            revision="main",
            trust_remote_code=True,
        )
        if "nomic" in cfg.model.name:
            base_model = AutoModelForMaskedLM.from_pretrained(
                cfg.model.name,
                from_tf=False,
                config=config,
                revision="main",
                trust_remote_code=True,
                ignore_mismatched_sizes=False,
            )
            model = NomicBERTForSequenceClassification(
                config,
                base_model.bert,
                num_labels=num_labels,
                classifier_dropout=cfg.model.classifier_dropout,
                classifier_init_range=cfg.model.classifier_init_range,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model.name,
                from_tf=False,
                config=config,
                revision="main",
                trust_remote_code=True,
                ignore_mismatched_sizes=False,
            )
    else:
        model = NeoBERTForSequenceClassification(
            NeoBERTConfig(
                **model_pretraining_config.model, **model_pretraining_config.tokenizer
            ),
            num_labels=num_labels,
            classifier_dropout=cfg.model.classifier_dropout,
            classifier_init_range=cfg.model.classifier_init_range,
        )

    if cfg.model.transfer_from_task:
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
        cfg.model.pretrained_checkpoint_dir = os.path.join(
            cfg.model.pretrained_checkpoint_dir, "model_checkpoints"
        )

    if not cfg.model.from_hub:
        try:
            model = load_state_dict_from_zero_checkpoint(
                model,
                cfg.model.pretrained_checkpoint_dir,
                tag=str(cfg.model.pretrained_checkpoint),
            )
        except FileNotFoundError:
            state_dict = torch.load(
                os.path.join(
                    cfg.model.pretrained_checkpoint_dir,
                    str(cfg.model.pretrained_checkpoint),
                    "state_dict.pt",
                )
            )
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            model.load_state_dict(state_dict, strict=False)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.optimizer.hparams.weight_decay,
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **cfg.optimizer.hparams)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.trainer.gradient_accumulation_steps
    )
    if cfg.trainer.max_steps is None:
        cfg.trainer.max_steps = (
            cfg.trainer.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

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

    scheduler = get_scheduler(
        optimizer=optimizer, lr=cfg.optimizer.hparams.lr, **cfg.scheduler
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
    )

    if cfg.task == "mnli":
        mm_eval_dataloader = accelerator.prepare(mm_eval_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.trainer.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.trainer.max_steps = (
            cfg.trainer.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    cfg.trainer.num_train_epochs = math.ceil(
        cfg.trainer.max_steps / num_update_steps_per_epoch
    )

    # Overwrite the number of steps performed between evaluation based on the dataset size.
    cfg.trainer.eval_steps = min(
        cfg.trainer.eval_steps, len(train_dataset) // cfg.trainer.train_batch_size
    )

    # To keep the last n checkpoints before the best model and do k cycles before early stopping, we save the last k+n models
    if (
        cfg.trainer.max_ckpt is not None
        and cfg.trainer.max_ckpt > 0
        and cfg.trainer.early_stopping > 0
    ):
        cfg.trainer.max_ckpt += cfg.trainer.early_stopping

    # Get loss function
    if not is_regression:
        loss_fct = CrossEntropyLoss()
    else:
        loss_fct = MSELoss()

    # Train!
    total_batch_size = (
        cfg.trainer.train_batch_size
        * accelerator.num_processes
        * cfg.trainer.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Task = {cfg.task}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num epochs = {cfg.trainer.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.trainer.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Learning rate = {cfg.optimizer.hparams.lr}")
    logger.info(
        f"  Gradient accumulation steps = {cfg.trainer.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.trainer.max_steps}")
    logger.info(f"  Evaluation steps = {cfg.trainer.eval_steps}")
    logger.info(f"  Early stopping cycles = {cfg.trainer.early_stopping}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(cfg.trainer.max_steps), disable=not accelerator.is_local_main_process
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

    results = {}

    total_loss = 0

    early_stop = False
    prev_accuracy = 0
    early_stopping_counter = 1

    for epoch in range(starting_epoch, cfg.trainer.num_train_epochs):
        for batch in train_dataloader:
            logits = model(batch["input_ids"], batch["attention_mask"])["logits"]

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
                    len(train_dataloader) * cfg.trainer.train_batch_size // 10,
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

                accelerator.log(
                    {
                        "eval_metric": eval_metric if cfg.task != "mnli" else results,
                        "train_metric": train_metric,
                        "train_loss": total_loss / completed_steps,
                        "epoch": epoch,
                        "step": completed_steps,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=completed_steps,
                )

                all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                if cfg.task == "mnli":
                    all_results = {f"eval_{k}": v for k, v in results.items()}

                with open(
                    os.path.join(
                        cfg.trainer.dir, f"all_results_step_{completed_steps}.json"
                    ),
                    "w",
                ) as f:
                    print(
                        "dumping in",
                        os.path.join(
                            cfg.trainer.dir, f"all_results_step_{completed_steps}.json"
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

                if (
                    cfg.trainer.early_stopping > 0
                    and early_stopping_counter >= cfg.trainer.early_stopping
                ):
                    print(
                        f"Evaluation accuracy has not improved in {cfg.trainer.early_stopping} cycles of {cfg.trainer.eval_steps} evaluation steps, stopping the training."
                    )
                    early_stop = True

                # Save model checkpoint
                if cfg.trainer.max_ckpt != 0:
                    save_checkpoint(cfg, model, accelerator, completed_steps)

                model.train()

            if completed_steps >= cfg.trainer.max_steps or early_stop:
                break

            # Reset the gradient
            optimizer.zero_grad()

        if completed_steps >= cfg.trainer.max_steps or early_stop:
            break

    accelerator.end_training()

    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
    if cfg.task == "mnli":
        all_results = {f"eval_{k}": v for k, v in results.items()}

    with open(os.path.join(cfg.trainer.dir, "all_results.json"), "w") as f:
        json.dump(all_results, f)
