import torch

from typing import Any, Optional, Tuple
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator


# Adapted from https://github.com/huggingface/transformers/blob/125de4164364420854d7fe537a9bd2fdaf7369d4/src/transformers/data/data_collator.py#L828
class CustomCollatorForMLM(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


class DataCollatorWithPacking(DefaultDataCollator):
    """
    Data collator used for padding free approach, with sequence packing.
    """

    def __init__(self, sep_token_id, max_length, default_data_collator, **kwargs):
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.default_data_collator = default_data_collator

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        packed_sequences = []
        current_sequence = []

        i = 0
        while i < len(features) or current_sequence:
            current_length = len(current_sequence)
            while current_length < self.max_length and i < len(features):
                seq = features[i]["input_ids"]
                i += 1

                current_sequence.extend(seq)
                current_length = len(current_sequence)

            # Truncate sequence and add to packed sequences
            if current_length >= self.max_length:
                packed_sequences.append({"input_ids": current_sequence[: self.max_length]})

            # Keep truncated end of sequence for the next packing
            current_sequence = current_sequence[self.max_length :] if current_length > self.max_length + 1 else []

        return self.default_data_collator(packed_sequences, return_tensors)


def get_collator(
    tokenizer,
    dtype: torch.dtype = torch.float32,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: int = 8,
    mask_all: bool = False,
    pack_sequences: bool = False,
    max_length: int = 512,
):
    # No need to apply any padding if sequences are packed
    if pack_sequences:
        pad_to_multiple_of = None

    mlm_collator = (
        CustomCollatorForMLM(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        if mask_all
        else DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            return_tensors="pt",
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    )

    if pack_sequences:
        collator = DataCollatorWithPacking(
            sep_token_id=tokenizer.sep_token_id,
            max_length=max_length,
            default_data_collator=mlm_collator,
        )

        def collate_fn(batch):
            batch = collator(batch)
            batch["attention_mask"] = None
            return batch

    else:

        def collate_fn(batch):
            batch = mlm_collator(batch)
            batch["attention_mask"] = torch.where(batch["attention_mask"] == 1, float(0.0), float("-inf")).type(dtype)
            return batch

    return collate_fn
