import os

from datasets import load_from_disk

from neobert.config import load_config_from_args


def longer_seq(cfg):
    # Get the number of cpu cores available to the process
    num_proc = len(os.sched_getaffinity(0))

    dataset = load_from_disk(cfg.dataset.path)

    # Default min_length if not specified
    min_length = getattr(cfg.dataset, "min_length", 512)

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


def main():
    # Load configuration from command line arguments
    config = load_config_from_args()

    # Run sequence length filtering
    longer_seq(config)


if __name__ == "__main__":
    main()
