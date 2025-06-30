#!/usr/bin/env python3
"""Tokenize a dataset for pretraining."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import load_dataset
from transformers import AutoTokenizer

from neobert.tokenizer import tokenize


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset for pretraining")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pszemraj/simple_wikipedia_LM",
        help="Dataset name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--output", type=str, default="./tokenized_data", help="Output directory"
    )
    parser.add_argument(
        "--train_samples", type=int, default=5000, help="Number of training samples"
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Name of text column"
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=f"train[:{args.train_samples}]")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")

    # Tokenize the dataset
    print(f"\nTokenizing with max_length={args.max_length}...")
    tokenized_dataset = tokenize(
        dataset,
        tokenizer,
        column_name=args.text_column,
        max_length=args.max_length,
        remove_columns=True,
        truncation=True,
    )

    print(f"\nTokenized dataset columns: {tokenized_dataset.column_names}")
    print(f"Saving to: {args.output}")
    tokenized_dataset.save_to_disk(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
