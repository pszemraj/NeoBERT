#!/usr/bin/env python3
"""Validate pretraining configuration files."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml


class PretrainConfigValidator:
    """Validator for pretraining configuration files."""

    REQUIRED_FIELDS = {
        "task": str,
        "model": dict,
        "dataset": dict,
        "tokenizer": dict,
        "trainer": dict,
        "optimizer": dict,
        "scheduler": dict,
    }

    MODEL_REQUIRED = {
        "name": str,
        "hidden_size": int,
        "num_hidden_layers": int,
        "num_attention_heads": int,
        "intermediate_size": int,
        "vocab_size": int,
        "max_position_embeddings": int,
    }

    DATASET_REQUIRED = {
        "name": str,
        "max_seq_length": int,
    }

    TRAINER_REQUIRED = {
        "output_dir": str,
        "num_train_epochs": int,
        "per_device_train_batch_size": int,
        "gradient_accumulation_steps": int,
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config_path: Path) -> bool:
        """Validate a pretraining configuration file."""
        self.errors = []
        self.warnings = []

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return False

        # Check required top-level fields
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
            elif not isinstance(config[field], expected_type):
                self.errors.append(
                    f"Field {field} should be {expected_type.__name__}, got {type(config[field]).__name__}"
                )

        # Validate task
        if config.get("task") not in ["pretrain", "pretraining"]:
            self.errors.append(
                f"Task should be 'pretrain' or 'pretraining', got '{config.get('task')}'"
            )

        # Validate model section
        if "model" in config:
            self._validate_model_section(config["model"])

        # Validate dataset section
        if "dataset" in config:
            self._validate_dataset_section(config["dataset"])

        # Validate trainer section
        if "trainer" in config:
            self._validate_trainer_section(config["trainer"])

        # Validate optimizer section
        if "optimizer" in config:
            self._validate_optimizer_section(config["optimizer"])

        # Validate scheduler section
        if "scheduler" in config:
            self._validate_scheduler_section(config["scheduler"])

        # Validate tokenizer section
        if "tokenizer" in config:
            self._validate_tokenizer_section(config["tokenizer"])

        return len(self.errors) == 0

    def _validate_model_section(self, model_config: Dict):
        """Validate model configuration."""
        for field, expected_type in self.MODEL_REQUIRED.items():
            if field not in model_config:
                self.errors.append(f"Missing required model field: model.{field}")
            elif not isinstance(model_config[field], expected_type):
                self.errors.append(f"model.{field} should be {expected_type.__name__}")

        # Validate model dimensions
        if all(k in model_config for k in ["hidden_size", "num_attention_heads"]):
            if model_config["hidden_size"] % model_config["num_attention_heads"] != 0:
                self.errors.append(
                    f"hidden_size ({model_config['hidden_size']}) must be divisible by num_attention_heads ({model_config['num_attention_heads']})"
                )

        # Check activation function
        if "hidden_act" in model_config:
            valid_acts = ["gelu", "relu", "swiglu", "silu", "gelu_new"]
            if model_config["hidden_act"] not in valid_acts:
                self.warnings.append(
                    f"Unusual activation function: {model_config['hidden_act']}. Common choices: {valid_acts}"
                )

    def _validate_dataset_section(self, dataset_config: Dict):
        """Validate dataset configuration."""
        for field, expected_type in self.DATASET_REQUIRED.items():
            if field not in dataset_config:
                self.errors.append(f"Missing required dataset field: dataset.{field}")
            elif not isinstance(dataset_config[field], expected_type):
                self.errors.append(
                    f"dataset.{field} should be {expected_type.__name__}"
                )

        # Check dataset path or name
        if "name" in dataset_config:
            dataset_name = dataset_config["name"]
            if dataset_name.startswith("/") or dataset_name.startswith("./"):
                # Local dataset path
                dataset_path = Path(dataset_name)
                if not dataset_path.exists():
                    self.warnings.append(f"Dataset path does not exist: {dataset_path}")
            elif "/" not in dataset_name and dataset_name not in [
                "wikipedia",
                "bookcorpus",
                "openwebtext",
            ]:
                self.warnings.append(
                    f"Unusual dataset name: {dataset_name}. Ensure it's available on HuggingFace or locally."
                )

        # Check sequence length
        if "max_seq_length" in dataset_config:
            max_seq = dataset_config["max_seq_length"]
            if max_seq > 2048:
                self.warnings.append(
                    f"max_seq_length={max_seq} is quite large. Ensure your model supports this."
                )
            elif max_seq < 128:
                self.warnings.append(
                    f"max_seq_length={max_seq} is quite small for pretraining."
                )

    def _validate_trainer_section(self, trainer_config: Dict):
        """Validate trainer configuration."""
        for field, expected_type in self.TRAINER_REQUIRED.items():
            if field not in trainer_config:
                self.errors.append(f"Missing required trainer field: trainer.{field}")
            elif not isinstance(trainer_config[field], expected_type):
                self.errors.append(
                    f"trainer.{field} should be {expected_type.__name__}"
                )

        # Check batch size configuration
        if all(
            k in trainer_config
            for k in ["per_device_train_batch_size", "gradient_accumulation_steps"]
        ):
            effective_batch = (
                trainer_config["per_device_train_batch_size"]
                * trainer_config["gradient_accumulation_steps"]
            )
            if effective_batch < 32:
                self.warnings.append(
                    f"Effective batch size ({effective_batch}) is small for pretraining. Consider increasing."
                )

        # Check mixed precision
        if "mixed_precision" not in trainer_config:
            self.warnings.append(
                "No mixed_precision specified. Consider using 'bf16' for faster training."
            )
        elif trainer_config.get("mixed_precision") == "fp16":
            self.warnings.append(
                "Using fp16. Consider bf16 for better stability with modern GPUs."
            )

        # Check output directory
        if "output_dir" in trainer_config:
            output_dir = trainer_config["output_dir"]
            if not output_dir.startswith("./outputs/"):
                self.warnings.append(
                    f"Output directory should be under './outputs/', got '{output_dir}'"
                )

    def _validate_optimizer_section(self, optimizer_config: Dict):
        """Validate optimizer configuration."""
        if "name" not in optimizer_config:
            self.errors.append("Missing optimizer.name")

        if "lr" not in optimizer_config:
            self.errors.append("Missing optimizer.lr")
        else:
            # Handle scientific notation which may be parsed as string
            lr = optimizer_config["lr"]
            if isinstance(lr, str):
                try:
                    lr = float(lr)
                except ValueError:
                    self.errors.append("optimizer.lr must be a number")
                    return
            elif not isinstance(lr, (int, float)):
                self.errors.append("optimizer.lr must be a number")
                return

            # Check learning rate range
            if lr > 1e-2:
                self.warnings.append(
                    f"Learning rate {lr} seems very high for pretraining"
                )
            elif lr < 1e-5:
                self.warnings.append(
                    f"Learning rate {lr} seems very low for pretraining"
                )

        # Check weight decay
        if "weight_decay" in optimizer_config:
            if optimizer_config["weight_decay"] > 0.5:
                self.warnings.append(
                    f"weight_decay={optimizer_config['weight_decay']} seems high"
                )

    def _validate_scheduler_section(self, scheduler_config: Dict):
        """Validate scheduler configuration."""
        if "name" not in scheduler_config:
            self.errors.append("Missing scheduler.name")

        valid_schedulers = ["linear", "cosine", "constant", "constant_with_warmup"]
        if scheduler_config.get("name") not in valid_schedulers:
            self.warnings.append(
                f"Unusual scheduler: {scheduler_config.get('name')}. Common choices: {valid_schedulers}"
            )

        # Check warmup configuration
        if "warmup_steps" in scheduler_config and "warmup_percent" in scheduler_config:
            self.warnings.append(
                "Both warmup_steps and warmup_percent specified. warmup_percent will take precedence."
            )

    def _validate_tokenizer_section(self, tokenizer_config: Dict):
        """Validate tokenizer configuration."""
        if "name" not in tokenizer_config:
            self.errors.append("Missing tokenizer.name")

        if "vocab_size" in tokenizer_config:
            vocab_size = tokenizer_config["vocab_size"]
            # Check if vocab size matches model
            if (
                "model" in self.current_config
                and "vocab_size" in self.current_config["model"]
            ):
                model_vocab = self.current_config["model"]["vocab_size"]
                if vocab_size != model_vocab:
                    self.errors.append(
                        f"Tokenizer vocab_size ({vocab_size}) doesn't match model vocab_size ({model_vocab})"
                    )

    def print_report(self, config_path: Path):
        """Print validation report."""
        print(f"\nValidation Report for {config_path}")
        print("=" * 60)

        if not self.errors and not self.warnings:
            print("✅ Configuration is valid!")
            return

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate pretraining configuration files"
    )
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    validator = PretrainConfigValidator(verbose=args.verbose)

    # Load config for cross-validation
    with open(config_path, "r") as f:
        validator.current_config = yaml.safe_load(f)

    is_valid = validator.validate(config_path)
    validator.print_report(config_path)

    if not is_valid or (args.strict and validator.warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
