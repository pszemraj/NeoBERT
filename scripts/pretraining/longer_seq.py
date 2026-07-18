"""Filter datasets to longer sequence lengths."""

import argparse
import os
from typing import Any

from datasets import load_from_disk

from neobert.config import ConfigLoader


def longer_seq(cfg: Any, *, min_length: int) -> None:
    """Filter a dataset into longer-sequence variants.

    :param Any cfg: Configuration object with dataset settings.
    :param int min_length: Explicit minimum length for the first output dataset.
    :raises ValueError: If ``min_length`` is not positive.
    """
    if min_length <= 0:
        raise ValueError("min_length must be positive.")

    # Get the number of cpu cores available to the process
    num_proc = len(os.sched_getaffinity(0))

    dataset = load_from_disk(cfg.dataset.path)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the NeoBERT YAML configuration")
    parser.add_argument(
        "--min-length",
        type=int,
        required=True,
        help="Minimum token length for the first output dataset",
    )
    args, overrides = parser.parse_known_args()
    config = ConfigLoader.load(args.config, overrides=overrides)

    longer_seq(config, min_length=args.min_length)


if __name__ == "__main__":
    main()
