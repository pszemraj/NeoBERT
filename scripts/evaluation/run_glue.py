#!/usr/bin/env python3
"""Run GLUE evaluation."""

import argparse
import warnings

from neobert.glue import trainer
from neobert.glue.validation import load_validated_glue_config


def main() -> None:
    """Run GLUE evaluation from a config file."""
    parser = argparse.ArgumentParser(description="Run GLUE evaluation")
    parser.add_argument("config", type=str, nargs="?", help="Path to config file")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Path to config file (legacy flag; positional still supported)",
    )
    parser.add_argument("--task_name", type=str, default=None, help="GLUE task name")
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="Model path"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    # Parse args
    args = parser.parse_args()

    if args.config is None:
        parser.error("config path is required (positional or --config)")

    config, validation_warnings = load_validated_glue_config(
        args.config,
        task_name=args.task_name,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )
    for message in validation_warnings:
        warnings.warn(message, UserWarning, stacklevel=2)

    # Run the GLUE trainer
    trainer(config)


if __name__ == "__main__":
    main()
