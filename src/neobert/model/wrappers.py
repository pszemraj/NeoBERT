"""Training-time LM and embedding wrappers built on the NeoBERT backbone."""

import logging
import warnings
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

from neobert.training_utils import _pin_cpu_tensors

from .model import (
    NeoBERT,
    NeoBERTConfig,
    NeoBERTPreTrainedModel,
    NormNeoBERT,
    PackedSeqLens,
)

logger = logging.getLogger(__name__)


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    """NeoBERT with a language modeling head."""

    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig) -> None:
        """Initialize the language modeling head.

        :param NeoBERTConfig config: Model configuration.
        """
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        should_tie = bool(getattr(self.config, "tie_word_embeddings", False))
        if self.config.ngpt and should_tie:
            logger.warning(
                "Disabling tie_word_embeddings for ngpt=True. "
                "NormNeoBERT emits unit-normalized hidden states, so tying decoder "
                "weights to raw token embeddings is not a stable parameterization."
            )
            self.config.tie_word_embeddings = False
            should_tie = False

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
        :param str pooling: Pooling strategy (avg/cls).
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        del kwargs
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        """Encode a list of queries.

        :param list[str] queries: Query strings to encode.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded query embeddings.
        """
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [
                    (query + " " + kwargs["instructions"][query]).strip()
                    for query in queries
                ]
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]] | Dict[str, List[str]],
        batch_size: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a corpus of documents.

        :param list[dict[str, str]] | dict[str, list[str]] corpus: Corpus inputs.
        :param int batch_size: Encoding batch size.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded corpus embeddings.
        """
        del batch_size
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                    for doc in corpus
                ]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Optional overrides for the DataLoader.
                - num_workers (int): DataLoader worker processes (default: 0).
                - pin_memory (bool | None): Pin CPU memory for CUDA transfer. Defaults
                  to ``True`` on CUDA devices, otherwise ``False``.

        Returns:
            The encoded sentences.
        """
        from datasets import Dataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        # Respect the model's current device to avoid CPU/GPU mismatches.
        param = next(self.parameters())
        device = param.device
        # Keep additive masks in float32 for numerical stability (match training).
        mask_dtype = torch.float32
        num_workers = int(kwargs.pop("num_workers", 0))
        pin_memory = kwargs.pop("pin_memory", None)
        if pin_memory is None:
            pin_memory = device.type == "cuda"
        pin_memory = bool(pin_memory)

        def _transform_func(
            tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]
        ) -> Dict[str, List]:
            """Tokenize a batch of input texts.

            :param PreTrainedTokenizerFast tokenizer: Tokenizer to apply.
            :param dict[str, list] x: Batch with ``input_texts``.
            :return dict[str, list]: Tokenized batch.
            """
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
        )

        non_blocking = bool(pin_memory and device.type == "cuda")
        encodings = []
        warning_context = nullcontext()
        if non_blocking:
            # Torch's current DataLoader pin-memory path still calls the
            # deprecated ``Tensor.pin_memory(device=...)`` signature internally.
            # Keep loader-side pinning for overlap, but silence that transient
            # upstream warning until PyTorch removes the deprecated call.
            warning_context = warnings.catch_warnings()

        with warning_context:
            if non_blocking:
                warnings.filterwarnings(
                    "ignore",
                    message=r"The argument 'device' of Tensor\..* is deprecated\.",
                    category=DeprecationWarning,
                )

            for batch in tqdm(
                dataloader,
                desc="encoding",
                mininterval=10,
                disable=len(sentences) < 128,
            ):
                if non_blocking:
                    batch = _pin_cpu_tensors(batch)
                input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
                int_mask = batch["attention_mask"]

                if self.config.attn_backend != "sdpa":
                    # Packed path: compute packed_seqlens on CPU to avoid CUDA sync,
                    # then pass pad_mask=None so the model uses packed attention.
                    packed_seqlens = int_mask.sum(dim=1, keepdim=True).to(
                        device="cpu", dtype=torch.int32
                    )
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
                    additive_mask = torch.where(
                        pool_mask == 1, float(0.0), float("-inf")
                    ).type(mask_dtype)
                    outputs = self.model(input_ids, additive_mask)

                if self.pooling == "avg":
                    outputs = outputs * pool_mask.unsqueeze(-1).expand(
                        -1, -1, outputs.shape[-1]
                    )
                    outputs = outputs.sum(dim=1) / pool_mask.sum(dim=1).unsqueeze(-1)
                else:
                    outputs = outputs[:, 0, :]

                encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
