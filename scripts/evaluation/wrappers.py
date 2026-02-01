"""MTEB wrapper utilities for pretrained models."""

# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

from functools import partial
from typing import Any, Dict, List, Optional

import mteb
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

DEFAULT_PROMPTS = {
    "STS": "Retrieve semantically similar text.",
    "Summarization": "Given a summary, retrieve semantically similar summaries.",
    "BitextMining": "Retrieve parallel sentences.",
    "Classification": "Given a text, classify it into main categories.",
    "Clustering": "Identify categories based on text passages",
    "Reranking": "Given a query, retrieve relevant texts.",
    "Retrieval": "Given a query, retrieve relevant texts.",
    "PairClassification": "Retrieve text that are semantically similar to the given text.",
}


class PreTrainedModelForMTEB(PreTrainedModel):
    """Wrapper that exposes encode methods for MTEB evaluations."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 8,
        pooling: str = "avg",
        tasks_to_instructions_query: Optional[Dict[str, str]] = None,
        tasks_to_instructions_corpus: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Initialize the MTEB wrapper.

        :param PreTrainedModel model: Base model to wrap.
        :param PreTrainedTokenizerFast tokenizer: Tokenizer to use.
        :param int batch_size: Encoding batch size.
        :param str pooling: Pooling strategy (avg or cls).
        :param dict[str, str] | None tasks_to_instructions_query: Query instructions.
        :param dict[str, str] | None tasks_to_instructions_corpus: Corpus instructions.
        :param Any kwargs: Unused extra arguments for compatibility.
        """
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.tasks_to_instructions_query = tasks_to_instructions_query
        self.tasks_to_instructions_corpus = tasks_to_instructions_corpus

    def encode_queries(
        self, queries: List[str], prompt_name: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        """Encode a list of queries.

        :param list[str] queries: Query strings.
        :param str | None prompt_name: Optional prompt/task name.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded query embeddings.
        """
        if prompt_name is not None and self.tasks_to_instructions_query:
            if prompt_name in self.tasks_to_instructions_query:
                instruction = self.tasks_to_instructions_query[prompt_name]
            else:
                meta = mteb.get_task(prompt_name).metadata
                instruction = DEFAULT_PROMPTS.get(meta.type, "")
            queries = [
                instruction
                + self.tokenizer.sep_token
                + " "
                + self.tokenizer.bos_token
                + sentence
                for sentence in queries
            ]

        return self.encode(
            queries,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]] | Dict[str, List[str]],
        prompt_name: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a corpus of documents.

        :param list[dict[str, str]] | dict[str, list[str]] corpus: Corpus inputs.
        :param str | None prompt_name: Optional prompt/task name.
        :param Any kwargs: Additional encoding arguments.
        :return np.ndarray: Encoded corpus embeddings.
        """
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif isinstance(corpus[0], dict):
            sentences = [
                (doc["title"] + " " + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        else:
            sentences = corpus

        if prompt_name is not None and self.tasks_to_instructions_corpus:
            if prompt_name in self.tasks_to_instructions_corpus:
                instruction = self.tasks_to_instructions_corpus[prompt_name]
            else:
                meta = mteb.get_task(prompt_name).metadata
                instruction = DEFAULT_PROMPTS.get(meta.type, "")
            sentences = [
                instruction
                + self.tokenizer.sep_token
                + " "
                + self.tokenizer.bos_token
                + sentence
                for sentence in sentences
            ]

        return self.encode(
            sentences,
            **kwargs,
        )

    @torch.no_grad()
    def encode(
        self, sentences: list[str], prompt_name: str = None, **kwargs: Any
    ) -> torch.Tensor:
        """Encode sentences using the wrapped model.

        :param list[str] sentences: Sentences to encode.
        :param str | None prompt_name: Optional prompt/task name.
        :param Any kwargs: Additional arguments passed to the encoder.
        :return torch.Tensor: Encoded sentence embeddings.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _transform_func(
            tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]
        ) -> Dict[str, List]:
            """Tokenize a batch of input texts.

            :param PreTrainedTokenizerFast tokenizer: Tokenizer instance.
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
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        encodings = []
        for batch in tqdm(
            dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = self.model(input_ids, attention_mask).last_hidden_state

            if self.pooling == "avg":
                outputs = outputs * attention_mask.unsqueeze(-1).expand(
                    -1, -1, outputs.shape[-1]
                )
                outputs = outputs.sum(dim=1) / attention_mask.to(device).sum(
                    dim=1
                ).unsqueeze(-1)
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
