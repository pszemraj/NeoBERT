"""Preprocess and tokenize pretraining datasets.

Important:
- ``neobert.tokenizer.tokenize`` intentionally does NO padding (padding is done
  in the collator), so this script must not forward a ``padding=`` kwarg.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from neobert.config import load_config_from_args
from neobert.tokenizer import (
    get_tokenizer,
    resolve_text_column,
    tokenize_pretraining_dataset,
)


def preprocess(cfg: Any) -> None:
    """Tokenize and save the pretraining dataset."""
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
        trust_remote_code=cfg.tokenizer.trust_remote_code,
        revision=cfg.tokenizer.revision,
        allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
    )
    print(tokenizer)

    print("Loading dataset")
    dataset = load_dataset(cfg.dataset.name, split="train")

    text_column = resolve_text_column(
        dataset,
        is_streaming=False,
        preferred=getattr(cfg.dataset, "text_column", None),
    )

    pack_sequences = bool(cfg.datacollator.pack_sequences)
    max_length = cfg.dataset.max_seq_length
    if pack_sequences:
        max_length = cfg.datacollator.max_length or max_length

    print(f"Tokenizing dataset (column={text_column})")
    dataset = tokenize_pretraining_dataset(
        dataset,
        tokenizer,
        column_name=text_column,
        max_length=max_length,
        truncation=cfg.tokenizer.truncation,
        pack_sequences=pack_sequences,
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
