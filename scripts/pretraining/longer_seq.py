import os

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="pretraining")
def longer_seq(cfg: DictConfig):
    # Get the number of cpu cores available to the process
    num_proc = len(os.sched_getaffinity(0))

    # tokenizer = get_tokenizer(**cfg.tokenizer)
    # dataset = load_dataset(**cfg.dataset.train)

    # dataset = tokenize(dataset, tokenizer, column_name=cfg.dataset.column, **cfg.tokenizer)
    # dataset.save_to_disk(cfg.dataset.path_to_disk, max_shard_size="1GB")
    dataset = load_from_disk(cfg.dataset.path_to_disk)

    dataset = dataset.filter(
        lambda example: len(example["input_ids"]) >= cfg.dataset.min_length,
        num_proc=num_proc,
    )
    print(len(dataset))
    dataset.save_to_disk(
        cfg.dataset.path_to_disk + f"+{cfg.dataset.min_length}",
        max_shard_size="1GB",
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        lambda example: len(example["input_ids"]) >= 2 * cfg.dataset.min_length,
        num_proc=num_proc,
    )
    print(len(dataset))
    dataset.save_to_disk(
        cfg.dataset.path_to_disk + f"+{2 * cfg.dataset.min_length}",
        max_shard_size="1GB",
        num_proc=num_proc,
    )


if __name__ == "__main__":
    longer_seq()
