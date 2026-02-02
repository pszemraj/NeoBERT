"""Data collators used for pretraining and packing."""

from typing import Any, Callable, Optional, Tuple

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    PreTrainedTokenizerBase,
)


# Adapted from https://github.com/huggingface/transformers/blob/125de4164364420854d7fe537a9bd2fdaf7369d4/src/transformers/data/data_collator.py#L828
class CustomCollatorForMLM(DataCollatorForLanguageModeling):
    """Language modeling collator that masks all sampled tokens."""

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Prepare masked tokens/labels for masked language modeling (100% mask).

        :param Any inputs: Input token IDs.
        :param Any | None special_tokens_mask: Optional mask of special tokens to ignore.
        :return tuple[Any, Any]: Masked inputs and labels.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


class DataCollatorWithPacking(DefaultDataCollator):
    """Data collator used for padding-free sequence packing."""

    def __init__(
        self,
        sep_token_id: int,
        max_length: int,
        default_data_collator: DefaultDataCollator,
        **kwargs: Any,
    ) -> None:
        """Initialize a sequence-packing collator.

        :param int sep_token_id: Token ID used to separate packed sequences.
        :param int max_length: Maximum packed sequence length.
        :param DefaultDataCollator default_data_collator: Base collator to apply.
        :param Any kwargs: Forwarded to ``DefaultDataCollator``.
        """
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.default_data_collator = default_data_collator

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        packed_sequences = []
        packed_segments = []
        current_sequence: list[int] = []
        current_segments: list[int] = []
        current_segment_id = 0

        for feature in features:
            seq = feature["input_ids"]
            seq_len = len(seq)

            if current_sequence:
                sep_len = 1 if self.sep_token_id is not None else 0
                if len(current_sequence) + sep_len + seq_len > self.max_length:
                    packed_sequences.append({"input_ids": current_sequence})
                    packed_segments.append(current_segments)
                    current_sequence = []
                    current_segments = []
                    current_segment_id = 0

            if current_sequence and self.sep_token_id is not None:
                current_sequence.append(self.sep_token_id)
                current_segments.append(current_segment_id)
                if len(current_sequence) == self.max_length:
                    packed_sequences.append({"input_ids": current_sequence})
                    packed_segments.append(current_segments)
                    current_sequence = []
                    current_segments = []
                    current_segment_id = 0

            for token in seq:
                if len(current_sequence) == self.max_length:
                    packed_sequences.append({"input_ids": current_sequence})
                    packed_segments.append(current_segments)
                    current_sequence = []
                    current_segments = []
                    current_segment_id = 0
                current_sequence.append(token)
                current_segments.append(current_segment_id)
                if len(current_sequence) == self.max_length:
                    packed_sequences.append({"input_ids": current_sequence})
                    packed_segments.append(current_segments)
                    current_sequence = []
                    current_segments = []
                    current_segment_id = 0

            current_segment_id += 1

        if current_sequence:
            packed_sequences.append({"input_ids": current_sequence})
            packed_segments.append(current_segments)

        batch = self.default_data_collator(packed_sequences, return_tensors)

        if not packed_segments:
            return batch

        max_len = batch["input_ids"].shape[1]
        masks = []
        for seg in packed_segments:
            seg_tensor = torch.tensor(seg, dtype=torch.long)
            if seg_tensor.numel() < max_len:
                pad = torch.full(
                    (max_len - seg_tensor.numel(),), -1, dtype=seg_tensor.dtype
                )
                seg_tensor = torch.cat([seg_tensor, pad], dim=0)
            valid = seg_tensor != -1
            # NOTE: This builds a dense block mask (O(seq^2)) for SDPA/xFormers.
            # For very long sequences, prefer packed cu_seqlens with flash-attn.
            mask = (
                (seg_tensor[:, None] == seg_tensor[None, :])
                & valid[:, None]
                & valid[None, :]
            )
            if not valid.all():
                # Avoid all-masked query rows (all False) that would become all -inf
                # after additive conversion and can yield NaNs in SDPA.
                pad_idx = torch.where(~valid)[0]
                mask[pad_idx, pad_idx] = True
            masks.append(mask)

        batch["attention_mask"] = torch.stack(masks, dim=0)
        return batch


def get_collator(
    tokenizer: PreTrainedTokenizerBase,
    dtype: torch.dtype = torch.float32,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: int = 8,
    mask_all: bool = False,
    pack_sequences: bool = False,
    max_length: int = 512,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for masked language modeling.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer used by the collator.
    :param torch.dtype dtype: Dtype for the attention mask when not packing sequences.
    :param float mlm_probability: Probability of masking tokens.
    :param int pad_to_multiple_of: Pad sequence length to a multiple of this value.
    :param bool mask_all: If True, mask all sampled tokens.
    :param bool pack_sequences: If True, pack sequences into fixed-length chunks.
    :param int max_length: Maximum sequence length for packing.
    :return Callable[[list[dict[str, Any]]], dict[str, Any]]: Collate function.
    """
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

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            """Collate packed sequences and build block attention mask.

            :param list[dict[str, Any]] batch: List of dataset examples.
            :return dict[str, Any]: Batch dictionary with packed input IDs.
            """
            batch = collator(batch)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                batch["attention_mask"] = torch.where(
                    batch["attention_mask"], float(0.0), float("-inf")
                ).type(dtype)
            return batch

    else:

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            """Collate and build attention mask for non-packed batches.

            :param list[dict[str, Any]] batch: List of dataset examples.
            :return dict[str, Any]: Batch dictionary with attention mask applied.
            """
            batch = mlm_collator(batch)
            batch["attention_mask"] = torch.where(
                batch["attention_mask"] == 1, float(0.0), float("-inf")
            ).type(dtype)
            return batch

    return collate_fn
