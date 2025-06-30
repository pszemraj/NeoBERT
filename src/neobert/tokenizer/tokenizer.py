import os
from functools import partial
from typing import Tuple

from datasets import Dataset, Features, Sequence, Value
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(
    pretrained_model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
    vocab_size: int = 32064,
    max_length: int = 4096,
    token: str = None,
    **kwargs,
):
    # Load Tokenizer and replace/add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        max_length=max_length,
        vocab_size=vocab_size,
        token=token,
        trust_remote_code=True,
    )

    # List of tokenizers that already have the correct special tokens
    tokenizers_with_special_tokens = [
        "bert-base-uncased",
        "google-bert/bert-base-uncased",
        "BEE-spoke-data/wordpiece-tokenizer-32k-en_code-msp",
    ]

    if pretrained_model_name_or_path not in tokenizers_with_special_tokens:
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
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
            pair=tokenizer.bos_token
            + " $A "
            + tokenizer.sep_token
            + " "
            + tokenizer.bos_token
            + " $B "
            + tokenizer.eos_token,
            special_tokens=[
                (tokenizer.eos_token, tokenizer.eos_token_id),
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.sep_token, tokenizer.sep_token_id),
            ],
        )

    return tokenizer


def single_column_mapping(x, tokenizer, column_name, max_length, truncation):
    return tokenizer(
        x[column_name],
        truncation=truncation,
        max_length=max_length,
        padding=False,  # no padding saves time and memory
    )


def multi_column_mapping(x, tokenizer, column_name, max_length, truncation):
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
                )
                for item in x[col]
            ]
            output[f"input_ids_{col}"] = [
                tokenized["input_ids"] for tokenized in tokenized_list
            ]
            output[f"attention_mask_{col}"] = [
                tokenized["attention_mask"] for tokenized in tokenized_list
            ]
        else:
            tokenized = tokenizer(
                x[col],
                truncation=truncation,
                max_length=max_length,
                padding=False,
                return_token_type_ids=False,
                is_split_into_words=False,
            )
            output[f"input_ids_{col}"] = tokenized["input_ids"]
            output[f"attention_mask_{col}"] = tokenized["attention_mask"]
    return output


def tokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    column_name: str | Tuple[str],
    max_length: int = 4096,
    remove_columns: bool = True,
    truncation: bool = True,
    **kwargs,
):
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
                columns_to_remove = [col for col in all_columns if col not in ['input_ids', 'attention_mask', 'token_type_ids']]
            except:
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
        features = Features(
            {
                "input_ids": Sequence(Value("int32")),
                "attention_mask": Sequence(Value("bool")),
            }
        )
        mapping = partial(
            single_column_mapping,
            tokenizer=tokenizer,
            column_name=column_name,
            max_length=max_length,
            truncation=truncation,
        )

    # Multi column tokenization
    else:
        feat = {}
        for col in column_name:
            if isinstance(dataset[col][0], list):
                feat[f"input_ids_{col}"] = Sequence(Sequence(Value("int32")))
                feat[f"attention_mask_{col}"] = Sequence(Sequence(Value("bool")))
            else:
                feat[f"input_ids_{col}"] = Sequence(Value("int32"))
                feat[f"attention_mask_{col}"] = Sequence(Value("bool"))
        features = Features(feat)

        mapping = partial(
            multi_column_mapping,
            tokenizer=tokenizer,
            column_name=column_name,
            max_length=max_length,
            truncation=truncation,
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
