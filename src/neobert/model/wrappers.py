"""Training-time LM and embedding wrappers built on the NeoBERT backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from neobert.collator import attention_mask_to_packed_seqlens
from neobert.training_utils import _pin_cpu_tensors
from neobert.utils import additive_attention_mask

from .model import NeoBERT, NeoBERTConfig, NeoBERTPreTrainedModel, PackedSeqLens

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import BatchedInput, PromptType
    from torch.utils.data import DataLoader

MTEB_POOLING_ALIASES = {"avg": "avg", "mean": "avg", "cls": "cls"}


def normalize_mteb_pooling(pooling: str) -> str:
    """Normalize and validate an MTEB pooling strategy.

    :param str pooling: Pooling name (``avg``/``mean`` or ``cls``).
    :raises ValueError: If the pooling strategy is unsupported.
    :return str: Canonical pooling name (``avg`` or ``cls``).
    """
    normalized = str(pooling).strip().lower()
    try:
        return MTEB_POOLING_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MTEB pooling {pooling!r}; expected one of "
            f"{sorted(MTEB_POOLING_ALIASES)}."
        ) from exc


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    """NeoBERT with a language modeling head."""

    config_class = NeoBERTConfig
    _tied_weights_keys = {"decoder.weight": "model.encoder.weight"}

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the language modeling head.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.model = NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        should_tie = bool(getattr(self.config, "tie_word_embeddings", False))

        # ``post_init()`` applies HF-style init; explicit ``tie_weights()`` keeps
        # decoder/input embedding aliasing deterministic in this training module.
        self.post_init()
        if should_tie:
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return input token embeddings for weight tying.

        :return nn.Embedding: Input embedding module.
        """
        return self.model.encoder

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """Set input token embeddings (used by HF APIs)."""
        self.model.encoder = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        """Return output embeddings for weight tying.

        :return nn.Linear: Output projection module.
        """
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set output embeddings (used by HF APIs)."""
        self.decoder = new_embeddings

    def forward(
        self,
        src: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        packed_seqlens: Optional[PackedSeqLens] = None,
        *,
        return_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run the LM head forward pass.

        :param torch.Tensor src: Input token IDs.
        :param torch.Tensor | None pad_mask: Additive attention mask.
        :param torch.Tensor | list[list[int]] | None packed_seqlens: Packed segment lengths.
        :param bool return_logits: Whether to materialize logits.
        :return dict[str, torch.Tensor]: Hidden states and optional logits.
        """
        hidden_representation = self.model.forward(src, pad_mask, packed_seqlens)
        output: Dict[str, torch.Tensor] = {
            "hidden_representation": hidden_representation
        }
        if return_logits:
            output["logits"] = self.decoder(hidden_representation)
        return output


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    """NeoBERT wrapper for MTEB-style encoding."""

    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs: Any,
    ) -> None:
        """Initialize the MTEB encoder wrapper.

        :param NeoBERTConfig config: Model configuration.
        :param PreTrainedTokenizerFast tokenizer: Tokenizer for text inputs.
        :param int max_length: Maximum sequence length.
        :param int batch_size: Encoding batch size.
        :param str pooling: Pooling strategy (avg/mean or cls).
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        del kwargs
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = normalize_mteb_pooling(pooling)

    @staticmethod
    def similarity(
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Compute all-pairs cosine similarity between two embedding collections.

        :param np.ndarray | torch.Tensor embeddings1: First embedding collection.
        :param np.ndarray | torch.Tensor embeddings2: Second embedding collection.
        :raises ValueError: If the inputs are not vectors or matrices with matching dimensions.
        :return torch.Tensor: Pairwise cosine-similarity matrix.
        """
        left = torch.as_tensor(embeddings1, dtype=torch.float32)
        right = torch.as_tensor(embeddings2, dtype=torch.float32, device=left.device)
        if left.ndim == 1:
            left = left.unsqueeze(0)
        if right.ndim == 1:
            right = right.unsqueeze(0)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError(
                "Cosine similarity expects vectors or matrices with matching embedding dimensions."
            )
        left = torch.nn.functional.normalize(left, p=2, dim=-1)
        right = torch.nn.functional.normalize(right, p=2, dim=-1)
        return left @ right.transpose(0, 1)

    @staticmethod
    def similarity_pairwise(
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between corresponding embedding pairs.

        :param np.ndarray | torch.Tensor embeddings1: First embedding collection.
        :param np.ndarray | torch.Tensor embeddings2: Second embedding collection.
        :raises ValueError: If the input shapes do not match.
        :return torch.Tensor: Pairwise cosine-similarity scores.
        """
        left = torch.as_tensor(embeddings1, dtype=torch.float32)
        right = torch.as_tensor(embeddings2, dtype=torch.float32, device=left.device)
        if left.ndim == 1:
            left = left.unsqueeze(0)
        if right.ndim == 1:
            right = right.unsqueeze(0)
        if left.ndim != 2 or left.shape != right.shape:
            raise ValueError(
                "Pairwise cosine similarity expects identically shaped vectors or matrices."
            )
        return torch.nn.functional.cosine_similarity(left, right, dim=-1)

    @torch.no_grad()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text batches supplied by MTEB.

        :param DataLoader inputs: MTEB dataloader whose batches contain standardized ``text`` inputs.
        :param TaskMetadata task_metadata: Metadata for the active MTEB task.
        :param str hf_split: Active Hugging Face dataset split.
        :param str hf_subset: Active Hugging Face dataset subset.
        :param PromptType | None prompt_type: Whether inputs are queries or documents.
        :param Any kwargs: Additional MTEB encoding arguments.
        :return np.ndarray: Encoded sentence embeddings.
        """
        from tqdm import tqdm

        del task_metadata, hf_split, hf_subset, prompt_type
        # Respect the model's current device to avoid CPU/GPU mismatches.
        param = next(self.parameters())
        device = param.device
        # Keep additive masks in float32 for numerical stability (match training).
        mask_dtype = torch.float32
        # MTEB uses batch_size to build inputs, then forwards it to encode again.
        kwargs.pop("batch_size", None)
        show_progress_bar = bool(kwargs.pop("show_progress_bar", True))
        pin_memory = bool(kwargs.pop("pin_memory", device.type == "cuda"))
        non_blocking = bool(pin_memory and device.type == "cuda")
        encodings = []

        for batch in tqdm(
            inputs,
            desc="encoding",
            mininterval=10,
            disable=not show_progress_bar,
        ):
            if "text" not in batch:
                raise TypeError("NeoBERTForMTEB only supports text inputs.")
            tokenized = self.tokenizer(
                list(batch["text"]),
                truncation=True,
                max_length=self.max_length,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            if non_blocking:
                tokenized = _pin_cpu_tensors(tokenized)
            input_ids = tokenized["input_ids"].to(device, non_blocking=non_blocking)
            int_mask = tokenized["attention_mask"]

            if self.config.attn_backend != "sdpa":
                # Packed path: compute packed_seqlens on CPU to avoid CUDA sync,
                # then pass pad_mask=None so the model uses packed attention.
                packed_seqlens = attention_mask_to_packed_seqlens(int_mask)
                outputs = self.model(input_ids, None, packed_seqlens=packed_seqlens)
                pool_mask = int_mask.to(
                    device=device,
                    dtype=mask_dtype,
                    non_blocking=non_blocking,
                )
            else:
                pool_mask = int_mask.to(
                    device=device,
                    dtype=mask_dtype,
                    non_blocking=non_blocking,
                )
                additive_mask = additive_attention_mask(pool_mask, dtype=mask_dtype)
                outputs = self.model(input_ids, additive_mask)

            if self.pooling == "avg":
                outputs = outputs * pool_mask.unsqueeze(-1).expand(
                    -1, -1, outputs.shape[-1]
                )
                denominator = pool_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                outputs = outputs.sum(dim=1) / denominator
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.detach().float().cpu().numpy())

        return np.concatenate(encodings, axis=0)
