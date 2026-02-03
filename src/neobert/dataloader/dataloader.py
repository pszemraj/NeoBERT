"""Dataloader helpers for masked language model pretraining."""

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

# HuggingFace
from transformers import PreTrainedTokenizer

# Ours
from ..collator import get_collator


def get_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    mlm_probability: float = 0.15,
    mask_all: bool = False,
    pad_to_multiple_of: int = 8,
    num_workers: int = 4,
    batch_size: int = 64,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    pack_sequences: bool = False,
    max_length: int = 512,
) -> torch.utils.data.DataLoader:
    """Build a ``torch`` dataloader with an MLM collator and pad mask.

    :param Dataset dataset: Dataset to iterate over.
    :param PreTrainedTokenizer tokenizer: Tokenizer used by the collator.
    :param float mlm_probability: Ratio of tokens to mask.
    :param bool mask_all: Whether to mask all sampled tokens.
    :param int pad_to_multiple_of: Pad length to a multiple of this value.
    :param int num_workers: Number of dataloader workers.
    :param int batch_size: Batch size per device.
    :param bool shuffle: Whether to shuffle the dataset each epoch.
    :param bool pin_memory: Whether to pin memory in the dataloader.
    :param bool persistent_workers: Keep workers alive across epochs.
    :param bool pack_sequences: Whether to pack sequences before collation.
    :param int max_length: Maximum sequence length for packing.
    :return torch.utils.data.DataLoader: Configured dataloader instance.
    """

    collate_fn = get_collator(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of,
        mask_all=mask_all,
        max_length=max_length,
        pack_sequences=pack_sequences,
    )

    # Check if this is a streaming dataset
    is_streaming = hasattr(dataset, "_iter") or "IterableDataset" in str(type(dataset))

    # Streaming datasets can't use shuffle in DataLoader
    dataloader_kwargs = {
        "dataset": dataset,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        # Keep tail batches (important for unbiased eval); training logic tolerates
        # smaller final batches when packing is enabled.
        "drop_last": False,
    }

    # Only add shuffle for non-streaming datasets
    if not is_streaming:
        dataloader_kwargs["shuffle"] = shuffle

    dataloader = DataLoader(**dataloader_kwargs)

    return dataloader
