"""Preprocess and tokenize pretraining datasets.

Important:
- ``neobert.tokenizer.tokenize`` intentionally does NO padding (padding is done
  in the collator), so this script must not forward a ``padding=`` kwarg.
"""

from __future__ import annotations

from typing import Any

from datasets import concatenate_datasets, load_dataset

from neobert.config import load_config_from_args
from neobert.tokenizer import get_tokenizer, resolve_text_column, tokenize


def preprocess(cfg: Any) -> None:
    """Tokenize and save the pretraining dataset."""
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
    )
    print(tokenizer)

    print("Loading dataset")
    if cfg.dataset.name == "wikibook":
        bookcorpus = load_dataset("bookcorpus", split="train")
        wiki = load_dataset("wikipedia", "20220301.en", split="train")
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

        if bookcorpus.features.type != wiki.features.type:
            raise ValueError(
                "wikibook sources have mismatched schema types: "
                f"bookcorpus={bookcorpus.features.type} vs wiki={wiki.features.type}"
            )
        dataset = concatenate_datasets([bookcorpus, wiki]).shuffle(seed=0)
    else:
        dataset = load_dataset(cfg.dataset.name, split="train")

    text_column = resolve_text_column(
        dataset,
        is_streaming=False,
        preferred=getattr(cfg.dataset, "text_column", None),
    )

    print(f"Tokenizing dataset (column={text_column})")
    dataset = tokenize(
        dataset,
        tokenizer,
        column_name=text_column,
        max_length=cfg.tokenizer.max_length,
        truncation=cfg.tokenizer.truncation,
        remove_columns=True,
        return_special_tokens_mask=True,
    )

    print("Saving tokenized dataset")
    dataset.save_to_disk(cfg.dataset.path, max_shard_size="1GB")


def main() -> None:
    """Run the preprocessing CLI."""
    config = load_config_from_args(require_config=True)
    preprocess(config)


if __name__ == "__main__":
    main()
