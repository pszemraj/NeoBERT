"""Tokenizer utilities and dataset tokenization helpers."""

import logging
import os
from functools import partial
from typing import Any, Optional, Tuple

from datasets import Dataset, Features, Sequence, Value
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger("neobert.tokenizer")


def get_tokenizer(
    pretrained_model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    max_length: int = 4096,
    token: Optional[str] = None,
    vocab_size: Optional[int] = None,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    allow_special_token_rewrite: bool = False,
    **kwargs: Any,
) -> PreTrainedTokenizer:
    """Load and configure a tokenizer for NeoBERT usage.

    :param str pretrained_model_name_or_path: Tokenizer model name or path.
    :param int max_length: Maximum sequence length.
    :param str | None token: Optional auth token for gated models.
    :param int | None vocab_size: Deprecated; tokenizer vocab size is derived from the model.
    :param bool use_fast: Whether to require a fast tokenizer backend.
    :param bool trust_remote_code: Allow remote tokenizer code execution.
    :param str | None revision: Optional tokenizer revision/commit.
    :param bool allow_special_token_rewrite: Allow fallback special-token mutation.
    :param Any kwargs: Additional kwargs forwarded to ``from_pretrained``.
    :return PreTrainedTokenizer: Configured tokenizer instance.
    """
    if vocab_size is not None:
        logger.warning(
            "get_tokenizer(): 'vocab_size' is deprecated and ignored; "
            "resize model embeddings instead."
        )
    kwargs.pop("vocab_size", None)
    kwargs.setdefault("use_fast", use_fast)
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    if revision is not None:
        kwargs.setdefault("revision", revision)

    # Load Tokenizer and replace/add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        token=token,
        **kwargs,
    )

    # Set model_max_length (not max_length which is deprecated)
    tokenizer.model_max_length = max_length

    # Store original special tokens for comparison
    original_special_tokens = tokenizer.special_tokens_map.copy()
    original_mask = tokenizer.mask_token if hasattr(tokenizer, "mask_token") else None

    # Check if tokenizer already has mask token defined
    # If it does, keep the existing special tokens
    if hasattr(tokenizer, "mask_token") and tokenizer.mask_token is not None:
        # Tokenizer already has special tokens configured, keep them
        logger.info(
            f"Keeping existing special tokens for {pretrained_model_name_or_path}"
        )
        logger.info(f"  Special tokens map: {tokenizer.special_tokens_map}")
        should_keep_special_tokens = True
    else:
        # No mask token defined, this is likely a standard LLM tokenizer
        logger.info(
            f"No mask token found for {pretrained_model_name_or_path}, adding RoBERTa-style special tokens"
        )
        should_keep_special_tokens = False

    if not should_keep_special_tokens:
        if not allow_special_token_rewrite:
            raise ValueError(
                "Tokenizer is missing a mask token and implicit special-token rewrite "
                "is disabled. Set tokenizer.allow_special_token_rewrite=true to allow "
                "NeoBERT fallback special-token injection."
            )
        if not isinstance(tokenizer, PreTrainedTokenizerFast) or not tokenizer.is_fast:
            raise ValueError(
                "Tokenizer does not provide a fast backend; cannot override post-processing "
                "for MLM-style special tokens. Use a fast tokenizer or a model that already "
                "defines a mask token."
            )
        if not hasattr(tokenizer, "_tokenizer") or tokenizer._tokenizer is None:
            raise ValueError(
                "Fast tokenizer backend is missing; cannot set post_processor for special tokens."
            )

        # Define special tokens to be consistent with RoBERTa
        special_tokens = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
            "cls_token": "<s>",
            "mask_token": "<mask>",
        }

        tokenizer.add_special_tokens(special_tokens)

        # Update the processor to add <eos> and <bos> tokens
        if tokenizer._tokenizer.post_processor is not None:
            logger.warning(
                "Overriding existing tokenizer post_processor to enforce MLM-style "
                "special tokens; custom post-processing will be replaced."
            )
        # Pair template intentionally uses a single BOS/CLS token:
        # "<s> A </s> B </s>". Do not insert an extra BOS before $B.
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
            pair=tokenizer.bos_token
            + " $A "
            + tokenizer.sep_token
            + " $B "
            + tokenizer.eos_token,
            special_tokens=[
                (tokenizer.eos_token, tokenizer.eos_token_id),
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.sep_token, tokenizer.sep_token_id),
            ],
        )

    if tokenizer.pad_token is None:
        logger.warning(
            "Tokenizer is missing a pad token; adding '<pad>' for batching support."
        )
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batching.")

    # Check if special tokens were modified and warn the user prominently
    final_special_tokens = tokenizer.special_tokens_map
    final_mask = tokenizer.mask_token if hasattr(tokenizer, "mask_token") else None

    # Check for modifications
    tokens_modified = False
    modified_tokens = []

    if original_mask and final_mask and original_mask != final_mask:
        tokens_modified = True
        modified_tokens.append(f"mask_token: {original_mask} -> {final_mask}")

    for key in original_special_tokens:
        if key in final_special_tokens:
            if original_special_tokens[key] != final_special_tokens[key]:
                tokens_modified = True
                modified_tokens.append(
                    f"{key}: {original_special_tokens[key]} -> {final_special_tokens[key]}"
                )

    if tokens_modified:
        # Print clear warning to console
        print("\n" + "=" * 60, flush=True)
        print(
            f"⚠️  WARNING: Special tokens modified for {pretrained_model_name_or_path}",
            flush=True,
        )
        print("=" * 60, flush=True)
        for change in modified_tokens:
            print(f"  {change}", flush=True)
        print("=" * 60 + "\n", flush=True)

        # Also log it
        logger.warning(
            f"Special tokens modified for {pretrained_model_name_or_path}: {modified_tokens}"
        )

    return tokenizer


def resolve_text_column(
    dataset: Dataset, is_streaming: bool, preferred: Optional[str] = None
) -> str:
    """Resolve the text column for tokenization.

    :param Dataset dataset: Dataset to inspect.
    :param bool is_streaming: Whether the dataset is streaming.
    :param str | None preferred: Optional preferred column name to validate.
    :return str: Name of the text column.
    """
    if is_streaming:
        first_example = next(iter(dataset))
        columns = list(first_example.keys())
    else:
        columns = dataset.column_names

    if preferred is not None:
        if preferred in columns:
            return preferred
        raise ValueError(
            f"Requested text column '{preferred}' not found. Available columns: "
            + ", ".join(columns)
        )

    for col in ["text", "sentence", "content"]:
        if col in columns:
            return col
    raise ValueError(
        "Could not find text column in dataset. Available columns: "
        + ", ".join(columns)
    )


def single_column_mapping(
    x: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    column_name: str,
    max_length: int,
    truncation: bool,
    add_special_tokens: bool,
    return_special_tokens_mask: bool,
) -> dict[str, Any]:
    """Tokenize a single text column in a batched mapping call.

    :param dict[str, Any] x: Batch of examples from the dataset.
    :param PreTrainedTokenizer tokenizer: Tokenizer to apply.
    :param str column_name: Column containing text inputs.
    :param int max_length: Maximum sequence length.
    :param bool truncation: Whether to truncate sequences.
    :param bool add_special_tokens: Whether to add tokenizer special tokens.
    :param bool return_special_tokens_mask: Whether to return special token masks.
    :return dict[str, Any]: Tokenized outputs for the batch.
    """
    return tokenizer(
        x[column_name],
        truncation=truncation,
        max_length=max_length,
        padding=False,  # no padding saves time and memory
        add_special_tokens=add_special_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
    )


def multi_column_mapping(
    x: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    column_name: tuple[str, ...],
    max_length: int,
    truncation: bool,
    add_special_tokens: bool,
    return_special_tokens_mask: bool,
) -> dict[str, Any]:
    """Tokenize multiple text columns in a batched mapping call.

    :param dict[str, Any] x: Batch of examples from the dataset.
    :param PreTrainedTokenizer tokenizer: Tokenizer to apply.
    :param tuple[str, ...] column_name: Columns containing text inputs.
    :param int max_length: Maximum sequence length.
    :param bool truncation: Whether to truncate sequences.
    :param bool add_special_tokens: Whether to add tokenizer special tokens.
    :param bool return_special_tokens_mask: Whether to return special token masks.
    :return dict[str, Any]: Tokenized outputs for the batch.
    """
    output = {}
    for col in column_name:
        if isinstance(x[col][0], list):
            tokenized_list = [
                tokenizer(
                    item,
                    truncation=truncation,
                    max_length=max_length,
                    padding=False,
                    return_token_type_ids=False,
                    is_split_into_words=False,
                    add_special_tokens=add_special_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                )
                for item in x[col]
            ]
            output[f"input_ids_{col}"] = [
                tokenized["input_ids"] for tokenized in tokenized_list
            ]
            output[f"attention_mask_{col}"] = [
                tokenized["attention_mask"] for tokenized in tokenized_list
            ]
            if return_special_tokens_mask:
                output[f"special_tokens_mask_{col}"] = [
                    tokenized["special_tokens_mask"] for tokenized in tokenized_list
                ]
        else:
            tokenized = tokenizer(
                x[col],
                truncation=truncation,
                max_length=max_length,
                padding=False,
                return_token_type_ids=False,
                is_split_into_words=False,
                add_special_tokens=add_special_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
            )
            output[f"input_ids_{col}"] = tokenized["input_ids"]
            output[f"attention_mask_{col}"] = tokenized["attention_mask"]
            if return_special_tokens_mask:
                output[f"special_tokens_mask_{col}"] = tokenized["special_tokens_mask"]
    return output


def tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str | Tuple[str, ...],
    max_length: int = 4096,
    remove_columns: bool = True,
    truncation: bool = True,
    add_special_tokens: bool = True,
    return_special_tokens_mask: bool = False,
    **kwargs: Any,
) -> Dataset:
    """Tokenize a dataset with a single- or multi-column schema.

    :param Dataset dataset: Dataset to tokenize.
    :param PreTrainedTokenizer tokenizer: Tokenizer to apply.
    :param str | tuple[str, ...] column_name: Column(s) to tokenize.
    :param int max_length: Maximum sequence length.
    :param bool remove_columns: Whether to remove non-token columns.
    :param bool truncation: Whether to truncate sequences.
    :param bool add_special_tokens: Whether to add tokenizer special tokens.
    :param bool return_special_tokens_mask: Whether to return special token masks.
    :param Any kwargs: Extra arguments passed to ``Dataset.map``.
    :return Dataset: Tokenized dataset.
    """
    # Check if this is a streaming dataset (IterableDataset)
    is_streaming = hasattr(dataset, "_iter") or "IterableDataset" in str(type(dataset))

    # Get the number of cpu cores available to the process
    # Override with kwargs if provided (e.g., from trainer)
    if "num_proc" in kwargs:
        num_proc = kwargs.pop("num_proc")
    else:
        num_proc = None if is_streaming else len(os.sched_getaffinity(0))

    # Remove all columns except for the `input_ids` and `attention_mask`
    if remove_columns:
        if is_streaming:
            # For streaming datasets, we need to get column names from first example
            try:
                first_example = next(iter(dataset))
                all_columns = list(first_example.keys())
                # Remove all columns except the ones we're creating
                columns_to_remove = [
                    col
                    for col in all_columns
                    if col not in ["input_ids", "attention_mask", "token_type_ids"]
                ]
            except Exception:
                # Fallback to just removing the text column(s) we're tokenizing
                if isinstance(column_name, str):
                    columns_to_remove = [column_name]
                else:
                    columns_to_remove = list(column_name)
        else:
            columns_to_remove = dataset.column_names
    else:
        columns_to_remove = None

    # Single column tokenization
    if isinstance(column_name, str):
        # Check if tokenizer returns token_type_ids
        test_output = tokenizer("test", truncation=True, max_length=10)
        has_token_type_ids = "token_type_ids" in test_output

        # We deliberately avoid emitting labels here; MLM labels are created
        # in the data collator to prevent duplicating input_ids on disk.
        feature_dict = {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("bool")),
        }
        if has_token_type_ids:
            feature_dict["token_type_ids"] = Sequence(Value("int32"))
        if return_special_tokens_mask:
            feature_dict["special_tokens_mask"] = Sequence(Value("bool"))

        features = Features(feature_dict)
        mapping = partial(
            single_column_mapping,
            tokenizer=tokenizer,
            column_name=column_name,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    # Multi column tokenization
    else:
        features = None
        if not is_streaming:
            feat = {}
            for col in column_name:
                sample = dataset[col][0] if len(dataset) > 0 else ""
                if isinstance(sample, list):
                    feat[f"input_ids_{col}"] = Sequence(Sequence(Value("int32")))
                    feat[f"attention_mask_{col}"] = Sequence(Sequence(Value("bool")))
                    if return_special_tokens_mask:
                        feat[f"special_tokens_mask_{col}"] = Sequence(
                            Sequence(Value("bool"))
                        )
                else:
                    feat[f"input_ids_{col}"] = Sequence(Value("int32"))
                    feat[f"attention_mask_{col}"] = Sequence(Value("bool"))
                    if return_special_tokens_mask:
                        feat[f"special_tokens_mask_{col}"] = Sequence(Value("bool"))
            features = Features(feat)

        mapping = partial(
            multi_column_mapping,
            tokenizer=tokenizer,
            column_name=column_name,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    # Tokenize the dataset
    map_kwargs = {
        "function": mapping,
        "batched": True,
        "remove_columns": columns_to_remove,
    }

    # Only add num_proc and features for non-streaming datasets
    if not is_streaming:
        map_kwargs["num_proc"] = num_proc
        map_kwargs["features"] = features

    dataset = dataset.map(**map_kwargs)

    return dataset
