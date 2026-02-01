#!/usr/bin/env python3
"""Validate GLUE evaluation configuration files."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml


class GLUEConfigValidator:
    """Validator for GLUE configuration files."""

    REQUIRED_FIELDS = {
        "task": str,
        "model": dict,
        "glue": dict,
        "tokenizer": dict,
        "trainer": dict,
        "optimizer": dict,
        "scheduler": dict,
    }

    GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]

    MODEL_REQUIRED = {
        "name_or_path": str,
        "hidden_size": int,
        "num_hidden_layers": int,
        "num_attention_heads": int,
        "intermediate_size": int,
        "vocab_size": int,
        "max_position_embeddings": int,
    }

    TRAINER_REQUIRED = {
        "output_dir": str,
        "num_train_epochs": int,
        "per_device_train_batch_size": int,
        "per_device_eval_batch_size": int,
    }

    TRAINER_WARNINGS = {
        "save_model": (
            False,
            "save_model should be False for GLUE evaluation to avoid unnecessary checkpoints",
        ),
        "save_strategy": (
            "no",
            "save_strategy should be 'no' for GLUE evaluation",
        ),
        "save_total_limit": (
            0,
            "save_total_limit should be 0 for GLUE evaluation",
        ),
    }

    def __init__(self, verbose: bool = False):
        """Initialize the validator.

        :param bool verbose: Enable verbose output.
        """
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config_path: Path) -> bool:
        """Validate a GLUE configuration file.

        :param Path config_path: Path to the config file.
        :return bool: True if validation passes.
        """
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
        if config.get("task") != "glue":
            self.errors.append(f"Task should be 'glue', got '{config.get('task')}'")

        # Validate GLUE section
        if "glue" in config:
            self._validate_glue_section(config["glue"])

        # Validate model section
        if "model" in config:
            self._validate_model_section(config["model"])

        # Validate trainer section
        if "trainer" in config:
            self._validate_trainer_section(config["trainer"])

        # Validate optimizer section
        if "optimizer" in config:
            self._validate_optimizer_section(config["optimizer"])

        # Validate tokenizer section
        if "tokenizer" in config:
            self._validate_tokenizer_section(config["tokenizer"])

        return len(self.errors) == 0

    def _validate_glue_section(self, glue_config: Dict) -> None:
        """Validate GLUE-specific configuration.

        :param dict glue_config: GLUE section mapping.
        """
        if "task_name" not in glue_config:
            self.errors.append("Missing glue.task_name")
        elif glue_config["task_name"] not in self.GLUE_TASKS:
            self.errors.append(
                f"Invalid GLUE task: {glue_config['task_name']}. Must be one of {self.GLUE_TASKS}"
            )

        if "num_labels" not in glue_config:
            self.errors.append("Missing glue.num_labels")
        elif not isinstance(glue_config["num_labels"], int):
            self.errors.append("glue.num_labels must be an integer")

        if "max_seq_length" not in glue_config:
            self.warnings.append(
                "Missing glue.max_seq_length (will use tokenizer max_length)"
            )

    def _validate_model_section(self, model_config: Dict) -> None:
        """Validate model configuration.

        :param dict model_config: Model section mapping.
        """
        for field, expected_type in self.MODEL_REQUIRED.items():
            if field not in model_config:
                self.errors.append(f"Missing required model field: model.{field}")
            elif not isinstance(model_config[field], expected_type):
                self.errors.append(
                    f"model.{field} should be {expected_type.__name__}, got {type(model_config[field]).__name__}"
                )

        # Check for pretrained checkpoint if specified
        if "pretrained_checkpoint_dir" in model_config:
            checkpoint_dir = Path(model_config["pretrained_checkpoint_dir"])
            if not checkpoint_dir.exists():
                self.warnings.append(
                    f"Pretrained checkpoint directory does not exist: {checkpoint_dir}"
                )

    def _validate_trainer_section(self, trainer_config: Dict) -> None:
        """Validate trainer configuration.

        :param dict trainer_config: Trainer section mapping.
        """
        for field, expected_type in self.TRAINER_REQUIRED.items():
            if field not in trainer_config:
                self.errors.append(f"Missing required trainer field: trainer.{field}")
            elif not isinstance(trainer_config[field], expected_type):
                self.errors.append(
                    f"trainer.{field} should be {expected_type.__name__}"
                )

        # Check for settings that should be disabled for GLUE evaluation
        for field, (expected_value, message) in self.TRAINER_WARNINGS.items():
            if field in trainer_config:
                if trainer_config[field] != expected_value:
                    self.warnings.append(f"trainer.{field}: {message}")
            else:
                self.warnings.append(f"Missing trainer.{field}: {message}")

        # Validate output directory structure
        if "output_dir" in trainer_config:
            output_dir = trainer_config["output_dir"]
            if not output_dir.startswith("./outputs/glue/"):
                self.warnings.append(
                    f"Output directory should follow pattern './outputs/glue/{{model_name}}/{{task}}', got '{output_dir}'"
                )

    def _validate_optimizer_section(self, optimizer_config: Dict) -> None:
        """Validate optimizer configuration.

        :param dict optimizer_config: Optimizer section mapping.
        """
        if "name" not in optimizer_config:
            self.errors.append("Missing optimizer.name")
        elif optimizer_config["name"] not in ["adamw", "adam", "sgd"]:
            self.warnings.append(
                f"Unusual optimizer: {optimizer_config['name']}. Common choices are 'adamw', 'adam'"
            )

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
            if lr > 1e-3:
                self.warnings.append(
                    f"Learning rate {lr} seems high for GLUE fine-tuning. Typical range is 1e-5 to 5e-5"
                )

    def _validate_tokenizer_section(self, tokenizer_config: Dict) -> None:
        """Validate tokenizer configuration.

        :param dict tokenizer_config: Tokenizer section mapping.
        """
        if "name" not in tokenizer_config:
            self.errors.append("Missing tokenizer.name")

        if "max_length" not in tokenizer_config:
            self.warnings.append("Missing tokenizer.max_length")
        elif tokenizer_config["max_length"] > 512:
            self.warnings.append(
                f"tokenizer.max_length={tokenizer_config['max_length']} is larger than typical BERT max (512)"
            )

    def print_report(self, config_path: Path) -> None:
        """Print a validation report.

        :param Path config_path: Path to the config file.
        """
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


def main() -> None:
    """Run the GLUE config validation CLI."""
    parser = argparse.ArgumentParser(description="Validate GLUE configuration files")
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

    validator = GLUEConfigValidator(verbose=args.verbose)
    is_valid = validator.validate(config_path)
    validator.print_report(config_path)

    if not is_valid or (args.strict and validator.warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
