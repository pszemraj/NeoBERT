#!/usr/bin/env python3
"""Script to pre-tokenize datasets for NeoBERT training."""

import argparse
from pathlib import Path

from datasets import load_dataset
from neobert.tokenizer import get_tokenizer, resolve_text_column, tokenize


def main() -> None:
    """Run the dataset tokenization CLI."""
    parser = argparse.ArgumentParser(
        description="Pre-tokenize datasets for NeoBERT training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or path to local dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to tokenize (default: train)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config name (e.g., wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Tokenizer model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized dataset",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Name of text column in dataset (default: auto-detect)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for tokenization (default: all CPUs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to tokenize (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dataset_kwargs = {"name": args.dataset_config} if args.dataset_config else {}
    if args.max_samples:
        dataset = load_dataset(
            args.dataset,
            split=f"{args.split}[:{args.max_samples}]",
            **dataset_kwargs,
        )
    else:
        dataset = load_dataset(args.dataset, split=args.split, **dataset_kwargs)

    print(f"Dataset size: {len(dataset)} samples")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=args.tokenizer,
        max_length=args.max_length,
    )

    text_column = resolve_text_column(
        dataset, is_streaming=False, preferred=args.text_column
    )

    print(f"Tokenizing with max_length={args.max_length}...")
    tokenized_dataset = tokenize(
        dataset,
        tokenizer,
        column_name=text_column,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    print(f"Saving tokenized dataset to: {args.output}")
    tokenized_dataset.save_to_disk(args.output)

    # Save tokenizer info for reference
    info_file = output_path / "tokenizer_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Tokenizer: {args.tokenizer}\n")
        f.write(f"Max length: {args.max_length}\n")
        f.write(f"Text column: {text_column}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Samples: {len(tokenized_dataset)}\n")

    print(f"Done! Tokenized {len(tokenized_dataset)} samples")


if __name__ == "__main__":
    main()
