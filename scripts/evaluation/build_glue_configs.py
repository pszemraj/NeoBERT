#!/usr/bin/env python3
"""Utility for generating GLUE config directories from a pretrained checkpoint."""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "configs" / "glue" / "generated"
DEFAULT_RESULTS_ROOT = Path("outputs/glue")
DEFAULT_WANDB_PROJECT = "neobert-glue"


# Task-specific overrides relative to the shared GLUE configuration
TASK_SETTINGS: Dict[str, Dict[str, Dict[str, object]]] = {
    "cola": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_matthews_correlation",
            "eval_steps": 200,
            "logging_steps": 50,
        },
    },
    "sst2": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_accuracy",
            "eval_steps": 500,
        },
    },
    "mrpc": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_f1",
            "eval_steps": 100,
        },
    },
    "stsb": {
        "glue": {"num_labels": 1, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_pearson",
            "eval_steps": 150,
        },
    },
    "qqp": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_f1",
            "eval_steps": 1000,
        },
    },
    "mnli": {
        "glue": {"num_labels": 3, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_accuracy",
            "eval_steps": 1000,
        },
    },
    "qnli": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_accuracy",
            "eval_steps": 500,
        },
    },
    "rte": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_accuracy",
            "eval_steps": 50,
        },
    },
    "wnli": {
        "glue": {"num_labels": 2, "max_seq_length": 512},
        "trainer": {
            "metric_for_best_model": "eval_accuracy",
            "eval_steps": 20,
        },
    },
}


BASE_TRAINER = {
    "num_train_epochs": 3,
    "max_steps": -1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "eval_strategy": "epoch",
    "save_model": False,
    "save_strategy": "no",
    "save_steps": -1,
    "save_total_limit": 0,
    "logging_steps": 100,
    "early_stopping": 5,
    "greater_is_better": True,
    "load_best_model_at_end": True,
    "mixed_precision": "bf16",
    "tf32": True,
    "seed": 42,
    "report_to": ["wandb"],
}


BASE_OPTIMIZER = {
    "name": "adamw",
    "lr": 2e-5,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
}


BASE_SCHEDULER = {
    "name": "linear",
    "warmup_percent": 10,
}


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""

    value = value.strip()
    if not value:
        return "run"
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-_") or "run"


def relpath(path: Path, base: Path) -> str:
    """Convert path to POSIX relative path when possible."""

    try:
        return Path(".") / Path(path).resolve().relative_to(base)
    except ValueError:
        return path.resolve()


def find_checkpoint_step(checkpoint_dir: Path, requested_step: Optional[str]) -> str:
    """Resolve the checkpoint step. If "latest", pick the highest numeric step."""

    if requested_step and requested_step != "latest":
        return requested_step

    candidates = [
        p.name
        for p in (checkpoint_dir / "model_checkpoints").glob("*")
        if p.is_dir() and p.name.isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No numbered checkpoints found under: {checkpoint_dir / 'model_checkpoints'}"
        )
    return max(candidates, key=lambda name: int(name))


def load_pretraining_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def build_trainer_section(
    base_output_dir: Path, task_name: str, overrides: Dict[str, object]
) -> Dict[str, object]:
    trainer_cfg = deepcopy(BASE_TRAINER)
    trainer_cfg.update(overrides)
    trainer_cfg.setdefault("metric_for_best_model", "eval_accuracy")
    trainer_cfg["output_dir"] = str(base_output_dir / task_name)
    return trainer_cfg


def build_wandb_section(
    project: str, run_prefix: str, task: str, checkpoint_step: str
) -> Dict[str, object]:
    return {
        "project": project,
        "name": f"{run_prefix}-{task}-{checkpoint_step}",
        "mode": "online",
    }


def build_pretraining_metadata(
    checkpoint_dir: Path,
    checkpoint_step: str,
    config_path: Path,
    pretrain_config: Dict[str, object],
) -> Dict[str, object]:
    trainer_cfg = pretrain_config.get("trainer", {}) or {}
    wandb_cfg = pretrain_config.get("wandb", {}) or {}

    metadata = {
        "checkpoint_dir": str(relpath(checkpoint_dir, REPO_ROOT)),
        "checkpoint_step": checkpoint_step,
        "config_path": str(relpath(config_path, REPO_ROOT)),
    }

    if trainer_cfg.get("run_name"):
        metadata["trainer_run_name"] = trainer_cfg["run_name"]
    if trainer_cfg.get("output_dir"):
        metadata["trainer_output_dir"] = trainer_cfg["output_dir"]

    if wandb_cfg.get("project"):
        metadata["wandb_project"] = wandb_cfg["project"]
    if wandb_cfg.get("name"):
        metadata["wandb_run_name"] = wandb_cfg["name"]
    if wandb_cfg.get("entity"):
        metadata["wandb_entity"] = wandb_cfg["entity"]

    return metadata


@dataclass
class BuildArgs:
    checkpoint_dir: Path
    checkpoint_step: str
    pretrain_config_path: Path
    output_root: Path
    results_root: Path
    wandb_project: str
    tasks: Iterable[str]
    run_prefix: str
    model_name: Optional[str]


def build_configs(args: BuildArgs) -> Dict[str, Dict[str, object]]:
    pretrain_cfg = load_pretraining_config(args.pretrain_config_path)
    tokenizer_cfg = pretrain_cfg.get("tokenizer", {}) or {}

    tokenizer_block = {}
    if tokenizer_cfg.get("name"):
        tokenizer_block["name"] = tokenizer_cfg["name"]
    if tokenizer_cfg.get("max_length"):
        tokenizer_block["max_length"] = tokenizer_cfg["max_length"]

    # Model identifiers
    checkpoint_value: object
    if args.checkpoint_step.isdigit():
        checkpoint_value = int(args.checkpoint_step)
    else:
        checkpoint_value = args.checkpoint_step

    metadata_block = build_pretraining_metadata(
        args.checkpoint_dir,
        str(checkpoint_value),
        args.pretrain_config_path,
        pretrain_cfg,
    )

    model_section = {
        "pretrained_checkpoint_dir": str(
            relpath(args.checkpoint_dir, REPO_ROOT)
        ),
        "pretrained_checkpoint": checkpoint_value,
        "pretrained_config_path": str(relpath(args.pretrain_config_path, REPO_ROOT)),
    }
    if args.model_name:
        model_section["name_or_path"] = args.model_name

    output_dir_root = args.results_root / f"{args.run_prefix}-ckpt{args.checkpoint_step}"

    configs: Dict[str, Dict[str, object]] = {}
    for task in args.tasks:
        settings = TASK_SETTINGS[task]

        trainer_cfg = build_trainer_section(
            output_dir_root, task, settings.get("trainer", {})
        )

        glue_cfg = {"task_name": task}
        glue_cfg.update(settings.get("glue", {}))

        config_dict: Dict[str, object] = {
            "task": "glue",
            "model": model_section,
            "glue": glue_cfg,
            "trainer": trainer_cfg,
            "optimizer": deepcopy(BASE_OPTIMIZER),
            "scheduler": deepcopy(BASE_SCHEDULER),
            "datacollator": {"pad_to_multiple_of": 8},
            "wandb": build_wandb_section(
                args.wandb_project, args.run_prefix, task, args.checkpoint_step
            ),
            "pretraining_metadata": metadata_block,
        }

        if tokenizer_block:
            config_dict["tokenizer"] = tokenizer_block

        configs[task] = config_dict

    return configs


def write_configs(configs: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, cfg in configs.items():
        path = output_dir / f"{task}.yaml"
        with path.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GLUE config set for a pretrained checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing model_checkpoints/ for the pretrained run",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=str,
        default="latest",
        help="Checkpoint step to evaluate (default: latest numbered step)",
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=None,
        help="Path to the pretraining config.yaml (default: derived from checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write generated configs (default: configs/glue/generated/<run>-ckpt<step>)"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for GLUE outputs (default: outputs/glue)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model identifier to log in configs (defaults to run slug)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name for GLUE runs",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=sorted(TASK_SETTINGS.keys()),
        choices=sorted(TASK_SETTINGS.keys()),
        help="Subset of GLUE tasks to generate (default: all)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_step = find_checkpoint_step(checkpoint_dir, args.checkpoint_step)

    pretrain_config_path = (
        args.pretrain_config.resolve()
        if args.pretrain_config
        else checkpoint_dir
        / "model_checkpoints"
        / checkpoint_step
        / "config.yaml"
    )
    if not pretrain_config_path.exists():
        raise FileNotFoundError(
            f"Pretraining config not found at: {pretrain_config_path}."
            " Use --pretrain-config to override."
        )

    pretrain_cfg = load_pretraining_config(pretrain_config_path)
    run_source = pretrain_cfg.get("wandb", {}).get("name") or checkpoint_dir.name
    run_prefix = slugify(run_source)

    output_root = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT
    results_root = args.results_root.expanduser()
    final_output_dir = output_root / f"{run_prefix}-ckpt{checkpoint_step}"

    build_args = BuildArgs(
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=checkpoint_step,
        pretrain_config_path=pretrain_config_path,
        output_root=output_root,
        results_root=results_root,
        wandb_project=args.wandb_project,
        tasks=args.tasks,
        run_prefix=run_prefix,
        model_name=args.model_name or run_prefix,
    )

    configs = build_configs(build_args)
    write_configs(configs, final_output_dir)

    print("Generated GLUE configs:")
    for task in sorted(configs):
        print(f"  - {task}: {final_output_dir / f'{task}.yaml'}")

    print(
        "\nExample command:"
        f"\n  bash scripts/evaluation/run_all_glue.sh {final_output_dir}"
    )


if __name__ == "__main__":
    main()
