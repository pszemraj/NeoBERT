#!/usr/bin/env python3
"""Run GLUE evaluation."""

import argparse

from neobert.config import ConfigLoader
from neobert.glue import trainer


def main():
    parser = argparse.ArgumentParser(description="Run GLUE evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task_name", type=str, default=None, help="GLUE task name")
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="Model path"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    # Parse args
    args = parser.parse_args()

    # Load base config
    config = ConfigLoader.load(args.config)

    # Override specific fields if provided
    if args.task_name:
        config.glue.task_name = args.task_name
    if args.model_name_or_path:
        config.model.name_or_path = args.model_name_or_path
    if args.output_dir:
        config.trainer.output_dir = args.output_dir

    # Run the GLUE trainer
    trainer(config)


if __name__ == "__main__":
    main()
