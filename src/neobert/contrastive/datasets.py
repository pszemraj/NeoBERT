import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from datasets import (Dataset, concatenate_datasets, load_dataset,
                      load_from_disk)

DATASET_TO_BSZ = {
    "ALLNLI": 2,
    "AMAZONQA": 2,
    "CONCURRENTQA": 21,
    "FEVER": 2,
    "GITHUBISSUE": 2,
    "GOOAQ": 7,
    "MSMARCO": 2,
    "PAQ": 2,
    "PUBMEDQA": 22,
    "QQP": 50,
    "SENTENCECOMP": 2,
    "STACKEXCHANGE": 2,
    "STACKOVERFLOW": 50,
    "STS12": 2,
    "STSBENCHMARK": 2,
    "TRIVIAQA": 3,
    "WIKIHOW": 2,
}


def get_bsz(dataset_name, target_batch_size):
    if dataset_name not in DATASET_TO_BSZ.keys():
        raise ValueError(
            f"{dataset_name} is not a known finetuning dataset. Available splits are: {DATASET_TO_BSZ.keys()}"
        )

    return target_batch_size // DATASET_TO_BSZ[dataset_name]


class AbsDataset(ABC):
    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__()

        if dataset is None:
            self._dataset = self.from_hub()
        else:
            self._dataset = dataset

        self.size = self._dataset.num_rows

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self._dataset = dataset

    @classmethod
    def from_disk(cls, dataset_path) -> "AbsDataset":
        dataset = load_from_disk(dataset_path)
        return cls(dataset)

    @abstractmethod
    def from_hub(self) -> Dataset:
        pass


class ALLNLI(AbsDataset):
    name = "ALLNLI"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/all-nli",
            name="triplet",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("anchor", "query")
        dataset = dataset.rename_column("positive", "corpus")

        return dataset


class AMAZONQA(AbsDataset):
    name = "AMAZONQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/Amazon-QA", split="train", num_proc=num_proc
        )

        def unwrap_corpus_column(example):
            example["corpus"] = [pos[0] for pos in example["pos"]]
            return example

        dataset = dataset.map(
            unwrap_corpus_column,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class CONCURRENTQA(AbsDataset):
    name = "CONCURRENTQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "stanfordnlp/concurrentqa-retrieval", split="train", num_proc=num_proc
        )

        dataset = dataset.rename_column("question", "query")

        def process(example):
            example["corpus"] = [x[0]["text"] for x in example["pos_paras"]]
            example["negative"] = [[y["text"] for y in x] for x in example["neg_paras"]]
            return example

        dataset = dataset.map(
            process,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class FEVER(AbsDataset):
    name = "FEVER"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        dataset = load_dataset("mteb/fever", "default", split="train")
        dataset = dataset.to_pandas()

        corpus = self.load_and_prepare_corpus()
        queries = self.load_and_prepare_queries()

        dataset = dataset.merge(queries, on="query-id", how="left")
        dataset = dataset.merge(corpus, on="corpus-id", how="left")

        dataset = dataset.dropna()
        dataset = dataset[["query", "corpus"]]

        dataset = Dataset.from_pandas(dataset, preserve_index=False)

        return dataset

    def load_and_prepare_corpus(self) -> pd.DataFrame:
        corpus = load_dataset("mteb/fever", "corpus", split="corpus")
        corpus = corpus.remove_columns("title")
        corpus = corpus.rename_column("_id", "corpus-id")
        corpus = corpus.rename_column("text", "corpus")
        return corpus.to_pandas()

    def load_and_prepare_queries(self) -> pd.DataFrame:
        queries = load_dataset("mteb/fever", "queries", split="queries")
        queries = queries.rename_column("_id", "query-id")
        queries = queries.rename_column("text", "query")
        return queries.to_pandas()


class GITHUBISSUE(AbsDataset):
    name = "GITHUBISSUE"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "WhereIsAI/github-issue-similarity",
            "positive",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("anchor", "query")
        dataset = dataset.rename_column("positive", "corpus")

        return dataset


class GOOAQ(AbsDataset):
    name = "GOOAQ"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "tomaarsen/gooaq-hard-negatives",
            name="triplet-5",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("question", "query")
        dataset = dataset.rename_column("answer", "corpus")

        def aggregate_negatives(example):
            example["negative"] = [
                [example[f"negative_{i + 1}"][j] for i in range(5)]
                for j in range(len(example["query"]))
            ]
            return example

        dataset = dataset.map(
            aggregate_negatives,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class MSMARCO(AbsDataset):
    name = "MSMARCO"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        dataset = load_dataset("mteb/msmarco", "default", split="train")

        corpus = self.load_and_prepare_corpus()
        queries = self.load_and_prepare_queries()

        dataset = dataset.to_pandas()
        dataset = dataset.merge(queries, on="query-id", how="left")
        dataset = dataset.merge(corpus, on="corpus-id", how="left")

        dataset = dataset[["query", "corpus"]]

        dataset = Dataset.from_pandas(dataset, preserve_index=False)

        return dataset

    def load_and_prepare_corpus(self) -> pd.DataFrame:
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus")
        corpus = corpus.remove_columns("title")
        corpus = corpus.rename_column("_id", "corpus-id")
        corpus = corpus.rename_column("text", "corpus")
        return corpus.to_pandas()

    def load_and_prepare_queries(self) -> pd.DataFrame:
        queries = load_dataset("mteb/msmarco", "queries", split="queries")
        queries = queries.rename_column("_id", "query-id")
        queries = queries.rename_column("text", "query")
        return queries.to_pandas()


class PAQ(AbsDataset):
    name = "PAQ"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/PAQ_pairs", split="train", num_proc=num_proc
        )

        def split_set_column(example):
            example["query"] = [x[0] for x in example["set"]]
            example["corpus"] = [x[1] for x in example["set"]]
            return example

        dataset = dataset.map(
            split_set_column,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class PUBMEDQA(AbsDataset):
    name = "PUBMEDQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/pubmedqa",
            name="triplet-20",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("anchor", "query")
        dataset = dataset.rename_column("positive", "corpus")

        def aggregate_negatives(example):
            example["negative"] = [
                [example[f"negative_{i + 1}"][j] for i in range(20)]
                for j in range(len(example["query"]))
            ]
            return example

        dataset = dataset.map(
            aggregate_negatives,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class QQP(AbsDataset):
    name = "QQP"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/QQP_triplets", split="train", num_proc=num_proc
        )

        def split_set_column(example):
            example["query"] = [x["query"] for x in example["set"]]
            example["corpus"] = [x["pos"][0] for x in example["set"]]
            example["negative"] = [x["neg"] for x in example["set"]]
            return example

        dataset = dataset.map(
            split_set_column,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class SENTENCECOMP(AbsDataset):
    name = "SENTENCECOMP"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/sentence-compression", split="train", num_proc=num_proc
        )

        def split_set_column(example):
            example["query"] = [x[1] for x in example["set"]]
            example["corpus"] = [x[0] for x in example["set"]]
            return example

        dataset = dataset.map(
            split_set_column,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class STACKEXCHANGE(AbsDataset):
    name = "STACKEXCHANGE"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))

        dataset_body = load_dataset(
            "sentence-transformers/stackexchange-duplicates",
            name="body-body-pair",
            split="train",
            num_proc=num_proc,
        )
        dataset_post = load_dataset(
            "sentence-transformers/stackexchange-duplicates",
            name="post-post-pair",
            split="train",
            num_proc=num_proc,
        )

        dataset_body = dataset_body.rename_column("body1", "query")
        dataset_body = dataset_body.rename_column("body2", "corpus")

        dataset_post = dataset_post.rename_column("post1", "query")
        dataset_post = dataset_post.rename_column("post2", "corpus")

        dataset = concatenate_datasets([dataset_body, dataset_post])

        return dataset


class STACKOVERFLOW(AbsDataset):
    name = "STACKOVERFLOW"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "mteb/stackoverflowdupquestions-reranking", split="train", num_proc=num_proc
        )

        def split_set_column(example):
            example["corpus"] = [x[0] for x in example["positive"]]
            return example

        dataset = dataset.map(
            split_set_column,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ],
        )

        return dataset


class STS12(AbsDataset):
    name = "STS12"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset("mteb/sts12-sts", split="train", num_proc=num_proc)

        dataset = dataset.filter(lambda x: x["score"] > 4)

        dataset = dataset.rename_column("sentence1", "query")
        dataset = dataset.rename_column("sentence2", "corpus")

        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ]
        )

        return dataset


class STSBENCHMARK(AbsDataset):
    name = "STSBENCHMARK"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "mteb/stsbenchmark-sts", split="train", num_proc=num_proc
        )

        dataset = dataset.filter(lambda x: x["score"] > 4)

        dataset = dataset.rename_column("sentence1", "query")
        dataset = dataset.rename_column("sentence2", "corpus")

        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names
                if col not in ["query", "corpus", "negative"]
            ]
        )

        return dataset


class TRIVIAQA(AbsDataset):
    name = "TRIVIAQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/trivia-qa-triplet",
            name="triplet",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("anchor", "query")
        dataset = dataset.rename_column("positive", "corpus")

        return dataset


class WIKIHOW(AbsDataset):
    name = "WIKIHOW"

    def __init__(self, dataset: Optional[Dataset] = None):
        super().__init__(dataset=dataset)

    def from_hub(self):
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/wikihow", split="train", num_proc=num_proc
        )

        dataset = dataset.rename_column("text", "query")
        dataset = dataset.rename_column("summary", "corpus")

        return dataset
