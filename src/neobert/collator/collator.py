"""Data collators used for pretraining and packing."""

from typing import Any, Callable, Optional, Tuple

import warnings

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
        """Prepare masked tokens/labels for MLM (100% mask).

        :param Any inputs: Input token IDs.
        :param Any | None special_tokens_mask: Optional mask of special tokens to ignore.
        :return tuple[Any, Any]: Masked inputs and labels.
        """
        labels = inputs.clone()
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
        labels[~masked_indices] = -100  # loss only on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


# Training-only collator for packed-sequence pretraining (not used in HF export).
class DataCollatorWithPacking(DefaultDataCollator):
    """Collator that packs segments into fixed-length sequences.

    Adds ``packed_seqlens`` (B, max_segments) with per-segment lengths.
    """

    def __init__(
        self,
        start_token_id: Optional[int],
        end_token_id: Optional[int],
        max_length: int,
        default_data_collator: DefaultDataCollator,
        **kwargs: Any,
    ) -> None:
        """Initialize a sequence-packing collator.

        :param int | None start_token_id: Optional BOS/CLS token to prepend per segment.
        :param int | None end_token_id: Optional EOS/SEP token to append per segment.
        :param int max_length: Maximum packed sequence length.
        :param DefaultDataCollator default_data_collator: Base collator to apply.
        :param Any kwargs: Forwarded to ``DefaultDataCollator``.
        """
        super().__init__(**kwargs)
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_length = max_length
        self.default_data_collator = default_data_collator
        reserve = int(start_token_id is not None) + int(end_token_id is not None)
        min_segment_len = max(1, reserve)
        # Fixed-width packed_seqlens avoids shape mismatches in dispatch/concatenate.
        self.max_segments = max(1, max_length // min_segment_len)

    def __call__(self, features, return_tensors=None):
        """Pack segments into fixed-length sequences and build attention masks."""
        if return_tensors is None:
            return_tensors = self.return_tensors

        packed_sequences = []
        packed_segments = []
        current_sequence: list[int] = []
        current_special_mask: list[int] = []
        current_attention_mask: list[int] = []
        current_segments: list[int] = []
        current_segment_id = 0

        for feature in features:
            seq = feature["input_ids"]
            if torch.is_tensor(seq):
                seq = seq.tolist()

            special_mask = feature.get("special_tokens_mask")
            if torch.is_tensor(special_mask):
                special_mask = special_mask.tolist()
            if special_mask is None:
                tokenizer = getattr(self.default_data_collator, "tokenizer", None)
                if tokenizer is not None:
                    special_mask = tokenizer.get_special_tokens_mask(
                        seq, already_has_special_tokens=True
                    )
                else:
                    special_mask = [0] * len(seq)

            segment_tokens: list[int] = []
            segment_special_mask: list[int] = []
            if self.start_token_id is not None:
                segment_tokens.append(self.start_token_id)
                segment_special_mask.append(1)
            segment_tokens.extend(seq)
            segment_special_mask.extend(int(val) for val in special_mask)
            if self.end_token_id is not None:
                segment_tokens.append(self.end_token_id)
                segment_special_mask.append(1)

            if len(segment_tokens) > self.max_length:
                raise ValueError(
                    "Packed segment length "
                    f"{len(segment_tokens)} exceeds max_length={self.max_length}. "
                    "This includes special tokens (CLS/SEP). If you tokenized without "
                    "accounting for packing, re-tokenize with a smaller max_length "
                    "or increase datacollator.max_length."
                )

            if current_sequence and (
                len(current_sequence) + len(segment_tokens) > self.max_length
            ):
                packed_sequences.append(
                    {
                        "input_ids": current_sequence,
                        "attention_mask": current_attention_mask,
                        "special_tokens_mask": current_special_mask,
                    }
                )
                packed_segments.append(current_segments)
                current_sequence = []
                current_special_mask = []
                current_attention_mask = []
                current_segments = []
                current_segment_id = 0

            current_sequence.extend(segment_tokens)
            current_special_mask.extend(segment_special_mask)
            current_attention_mask.extend([1] * len(segment_tokens))
            current_segments.extend([current_segment_id] * len(segment_tokens))
            current_segment_id += 1

            if len(current_sequence) == self.max_length:
                packed_sequences.append(
                    {
                        "input_ids": current_sequence,
                        "attention_mask": current_attention_mask,
                        "special_tokens_mask": current_special_mask,
                    }
                )
                packed_segments.append(current_segments)
                current_sequence = []
                current_special_mask = []
                current_attention_mask = []
                current_segments = []
                current_segment_id = 0

        if current_sequence:
            packed_sequences.append(
                {
                    "input_ids": current_sequence,
                    "attention_mask": current_attention_mask,
                    "special_tokens_mask": current_special_mask,
                }
            )
            packed_segments.append(current_segments)

        pad_token_id = getattr(
            getattr(self.default_data_collator, "tokenizer", None), "pad_token_id", None
        )
        if pad_token_id is None:
            pad_token_id = 0

        for seq in packed_sequences:
            if len(seq["input_ids"]) > self.max_length:
                raise ValueError(
                    f"Packed sequence length {len(seq['input_ids'])} exceeds max_length={self.max_length}."
                )
            if len(seq["input_ids"]) < self.max_length:
                pad_len = self.max_length - len(seq["input_ids"])
                seq["input_ids"].extend([pad_token_id] * pad_len)
                seq["attention_mask"].extend([0] * pad_len)
                seq["special_tokens_mask"].extend([1] * pad_len)

        batch = self.default_data_collator(packed_sequences, return_tensors)

        if not packed_segments:
            return batch

        packed_seqlens: list[list[int]] = []
        for seg in packed_segments:
            if not seg:
                packed_seqlens.append([])
                continue
            lengths: list[int] = []
            current = seg[0]
            count = 0
            for seg_id in seg:
                if seg_id != current:
                    lengths.append(count)
                    current = seg_id
                    count = 1
                else:
                    count += 1
            if count > 0:
                lengths.append(count)
            packed_seqlens.append(lengths)

        max_segments = self.max_segments if packed_seqlens else 0
        if max_segments and any(
            len(lengths) > max_segments for lengths in packed_seqlens
        ):
            raise ValueError(
                "Packed segment count exceeds fixed packed_seqlens width. "
                "Increase datacollator.max_length or reduce segment sizes."
            )

        if max_segments == 0:
            packed_tensor = torch.zeros((len(packed_seqlens), 0), dtype=torch.int32)
        else:
            packed_tensor = torch.zeros(
                (len(packed_seqlens), max_segments), dtype=torch.int32
            )
            for idx, lengths in enumerate(packed_seqlens):
                if lengths:
                    packed_tensor[idx, : len(lengths)] = torch.tensor(
                        lengths, dtype=torch.int32
                    )

        batch["packed_seqlens"] = packed_tensor
        return batch


def _ensure_attention_mask(
    batch: dict[str, Any], pad_token_id: Optional[int]
) -> torch.Tensor:
    """Ensure batch contains a 0/1 attention mask on CPU.

    :param dict[str, Any] batch: Collated batch.
    :param int | None pad_token_id: Token used for padding.
    :return torch.Tensor: 0/1 attention mask.
    """
    if "attention_mask" in batch and torch.is_tensor(batch["attention_mask"]):
        return batch["attention_mask"]

    input_ids = batch.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise ValueError(
            "Batch missing tensor 'input_ids'; cannot infer attention_mask."
        )

    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)

    return (input_ids != int(pad_token_id)).long()


def _is_right_padded_mask(attention_mask: torch.Tensor) -> bool:
    """Return True if a 0/1 attention mask uses right padding only.

    :param torch.Tensor attention_mask: 0/1 mask (CPU).
    :return bool: True if tokens are a prefix (right padding), else False.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            "Expected attention_mask rank-2 [B,S] for padding check, got "
            f"shape={tuple(attention_mask.shape)}"
        )
    mask = attention_mask
    if mask.is_cuda:
        mask = mask.cpu()
    mask = (mask != 0).to(torch.int)
    # For right padding, the mask must be non-increasing along the sequence.
    return bool(torch.all(mask.cummin(dim=-1).values == mask))


def attention_mask_to_packed_seqlens(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert right-padded 0/1 attention mask [B,S] into packed_seqlens [B,1].

    :param torch.Tensor attention_mask: 0/1 mask (CPU).
    :return torch.Tensor: Per-sample segment lengths tensor.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention_mask rank-2 [B,S], got shape={tuple(attention_mask.shape)}"
        )
    if attention_mask.is_cuda:
        raise ValueError(
            "attention_mask_to_packed_seqlens must run on CPU. "
            "Packed seqlens should be derived before moving batches to CUDA."
        )

    # packed_seqlens only encodes lengths, valid only for right padding.
    lengths = attention_mask.sum(dim=1, keepdim=True).to(torch.int32)
    return lengths


def get_collator(
    tokenizer: PreTrainedTokenizerBase,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: int = 8,
    mask_all: bool = False,
    pack_sequences: bool = False,
    max_length: int = 512,
    return_packed_seqlens: bool = False,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for masked language modeling.

    :param PreTrainedTokenizerBase tokenizer: Tokenizer used by the collator.
    :param float mlm_probability: Probability of masking tokens.
    :param int pad_to_multiple_of: Pad sequence length to a multiple of this value.
    :param bool mask_all: If True, mask all sampled tokens.
    :param bool pack_sequences: If True, pack sequences into fixed-length chunks.
    :param int max_length: Maximum sequence length for packing.
    :param bool return_packed_seqlens: Emit packed_seqlens for right-padded non-packed batches.
    :return Callable[[list[dict[str, Any]]], dict[str, Any]]: Collate function.
    """
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
        start_token_id = (
            tokenizer.cls_token_id
            if tokenizer.cls_token_id is not None
            else tokenizer.bos_token_id
        )
        end_token_id = (
            tokenizer.sep_token_id
            if tokenizer.sep_token_id is not None
            else tokenizer.eos_token_id
        )
        collator = DataCollatorWithPacking(
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_length=max_length,
            default_data_collator=mlm_collator,
        )

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return collator(batch)

    else:

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            batch = mlm_collator(batch)
            attention_mask = _ensure_attention_mask(
                batch, getattr(tokenizer, "pad_token_id", None)
            )
            if return_packed_seqlens:
                if _is_right_padded_mask(attention_mask):
                    # Keep packed_seqlens on CPU to avoid GPU syncs downstream.
                    batch["packed_seqlens"] = attention_mask_to_packed_seqlens(
                        attention_mask
                    )
                else:
                    warnings.warn(
                        "Skipping packed_seqlens because attention_mask is not right-padded. "
                        "Use tokenizer.padding_side='right' or disable return_packed_seqlens "
                        "to avoid corrupting packed attention.",
                        stacklevel=2,
                    )
            # Use float32 masks for softmax stability (bf16 can propagate NaNs).
            batch["attention_mask"] = torch.where(
                attention_mask == 1, 0.0, float("-inf")
            ).to(torch.float32)
            return batch

    return collate_fn
