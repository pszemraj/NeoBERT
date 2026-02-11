"""Filter datasets to longer sequence lengths."""

import os
from typing import Any

from datasets import load_from_disk

from neobert.config import DatasetConfig, load_config_from_args


def longer_seq(cfg: Any) -> None:
    """Filter a dataset into longer-sequence variants.

    :param Any cfg: Configuration object with dataset settings.
    """
    # Get the number of cpu cores available to the process
    num_proc = len(os.sched_getaffinity(0))

    dataset = load_from_disk(cfg.dataset.path)

    # Keep this utility focused on long-sequence filtering. Global config defaults
    # are short-text-oriented, so only honor cfg.dataset.min_length when explicitly
    # changed away from the dataclass default.
    configured_min_length = getattr(
        cfg.dataset, "min_length", DatasetConfig().min_length
    )
    min_length = (
        512
        if configured_min_length == DatasetConfig().min_length
        else int(configured_min_length)
    )

    dataset = dataset.filter(
        lambda example: len(example["input_ids"]) >= min_length,
        num_proc=num_proc,
    )
    print(f"Dataset with min_length {min_length}: {len(dataset)} samples")
    dataset.save_to_disk(
        cfg.dataset.path + f"+{min_length}",
        max_shard_size="1GB",
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        lambda example: len(example["input_ids"]) >= 2 * min_length,
        num_proc=num_proc,
    )
    print(f"Dataset with min_length {2 * min_length}: {len(dataset)} samples")
    dataset.save_to_disk(
        cfg.dataset.path + f"+{2 * min_length}",
        max_shard_size="1GB",
        num_proc=num_proc,
    )


def main() -> None:
    """Run the longer sequence filtering CLI."""
    # Load configuration from command line arguments
    config = load_config_from_args(require_config=True)

    # Run sequence length filtering
    longer_seq(config)


if __name__ == "__main__":
    main()
