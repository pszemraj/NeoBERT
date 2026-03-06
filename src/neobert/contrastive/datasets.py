"""Dataset wrappers and cache helpers for contrastive training tasks."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

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

_SHARED_DATASET_DEFAULT_NAME = "refinedweb"
_CONTRASTIVE_DATASET_HF_IDS: dict[str, tuple[str, ...]] = {
    "ALLNLI": ("sentence-transformers/all-nli",),
    "AMAZONQA": ("embedding-data/Amazon-QA",),
    "CONCURRENTQA": ("stanfordnlp/concurrentqa-retrieval",),
    "FEVER": ("mteb/fever",),
    "GITHUBISSUE": ("WhereIsAI/github-issue-similarity",),
    "GOOAQ": ("tomaarsen/gooaq-hard-negatives",),
    "MSMARCO": ("mteb/msmarco",),
    "PAQ": ("embedding-data/PAQ_pairs",),
    "PUBMEDQA": ("sentence-transformers/pubmedqa",),
    "QQP": ("embedding-data/QQP_triplets",),
    "SENTENCECOMP": ("embedding-data/sentence-compression",),
    "STACKEXCHANGE": ("sentence-transformers/stackexchange-duplicates",),
    "STACKOVERFLOW": ("mteb/stackoverflowdupquestions-reranking",),
    "STS12": ("mteb/sts12-sts",),
    "STSBENCHMARK": ("mteb/stsbenchmark-sts",),
    "TRIVIAQA": ("sentence-transformers/trivia-qa-triplet",),
    "WIKIHOW": ("sentence-transformers/wikihow",),
}


def get_bsz(dataset_name: str, target_batch_size: int) -> int:
    """Compute per-dataset batch size multiplier.

    :param str dataset_name: Dataset name key.
    :param int target_batch_size: Global target batch size.
    :return int: Per-dataset batch size.
    """
    if dataset_name not in DATASET_TO_BSZ.keys():
        raise ValueError(
            f"{dataset_name} is not a known finetuning dataset. Available splits are: {DATASET_TO_BSZ.keys()}"
        )

    return target_batch_size // DATASET_TO_BSZ[dataset_name]


def normalize_contrastive_dataset_name_token(value: Any) -> str:
    """Normalize a dataset selector token for registry matching.

    :param Any value: Raw dataset selector.
    :return str: Uppercase alphanumeric token.
    """
    return "".join(ch for ch in str(value).strip().upper() if ch.isalnum())


def _iter_contrastive_dataset_aliases(
    key: str,
    dataset_cls: type["AbsDataset"],
) -> Sequence[str]:
    """Yield canonical and explicit aliases for one contrastive dataset wrapper.

    :param str key: Canonical registry key.
    :param type[AbsDataset] dataset_cls: Dataset wrapper class.
    :return Sequence[str]: Ordered alias candidates.
    """
    aliases: list[str] = []
    seen: set[str] = set()
    candidates = (
        key,
        getattr(dataset_cls, "name", None),
        dataset_cls.__name__,
        *_CONTRASTIVE_DATASET_HF_IDS.get(key, ()),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        raw = str(candidate).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        aliases.append(raw)
        if "/" in raw:
            trailing = raw.rsplit("/", maxsplit=1)[-1].strip()
            if trailing and trailing not in seen:
                seen.add(trailing)
                aliases.append(trailing)
    return tuple(aliases)


def resolve_contrastive_dataset_name(requested: Any) -> str:
    """Resolve one dataset selector to a canonical contrastive registry key.

    Accepts canonical keys (for example ``ALLNLI``), class names, and the
    explicit Hugging Face dataset IDs used by the built-in wrapper registry.

    :param Any requested: Raw selector value.
    :return str: Canonical registry key.
    :raises ValueError: If the selector cannot be resolved.
    """
    aliases: dict[str, str] = {}
    for key, dataset_cls in CONTRASTIVE_DATASETS.items():
        for candidate in _iter_contrastive_dataset_aliases(key, dataset_cls):
            normalized = normalize_contrastive_dataset_name_token(candidate)
            if normalized:
                aliases.setdefault(normalized, key)

    raw_value = str(requested).strip()
    candidates = [raw_value]
    if "/" in raw_value:
        candidates.append(raw_value.rsplit("/", maxsplit=1)[-1])

    for candidate in candidates:
        normalized = normalize_contrastive_dataset_name_token(candidate)
        resolved = aliases.get(normalized)
        if resolved is not None:
            return resolved

    raise ValueError(
        "Unknown contrastive dataset name "
        f"'{requested}'. Available dataset keys: {sorted(CONTRASTIVE_DATASETS.keys())}."
    )


def resolve_contrastive_dataset_names(requested: Any) -> list[str]:
    """Resolve which contrastive datasets should participate in a run.

    Contrastive entrypoints share ``DatasetConfig`` with pretraining, whose
    default name is ``refinedweb``. When that shared default reaches a
    contrastive-only path, treat it as "unset" so omitted selectors keep the
    historical "all contrastive datasets" behavior.

    :param Any requested: Raw ``dataset.name`` value.
    :return list[str]: Dataset registry keys in request order.
    :raises TypeError: If the selector type is unsupported.
    """
    inherited_default = normalize_contrastive_dataset_name_token(
        _SHARED_DATASET_DEFAULT_NAME
    )
    if requested is None:
        return list(CONTRASTIVE_DATASETS.keys())
    if isinstance(requested, str):
        normalized = normalize_contrastive_dataset_name_token(requested)
        if normalized in {"", "ALL", inherited_default}:
            return list(CONTRASTIVE_DATASETS.keys())
        return [resolve_contrastive_dataset_name(requested)]
    if isinstance(requested, (list, tuple)):
        names: list[str] = []
        seen: set[str] = set()
        for name in requested:
            resolved = resolve_contrastive_dataset_name(name)
            if resolved in seen:
                continue
            seen.add(resolved)
            names.append(resolved)
        return names
    raise TypeError(
        "dataset.name must be a string or list of strings for contrastive workflows."
    )


def discover_cached_contrastive_dataset_names(all_dir: str | Path) -> list[str]:
    """Return cached contrastive split names with complete on-disk payloads.

    :param str | Path all_dir: Root ``all/`` directory containing cached split folders.
    :return list[str]: Cached split names in registry order.
    """
    cache_root = Path(all_dir)
    cached_names: list[str] = []
    for name in CONTRASTIVE_DATASETS:
        dataset_dir = cache_root / name
        if dataset_dir.is_dir() and (dataset_dir / "state.json").is_file():
            cached_names.append(name)
    return cached_names


def load_cached_contrastive_datasets(
    all_dir: str | Path,
    *,
    selected_names: Sequence[str],
) -> DatasetDict:
    """Load cached contrastive split directories for the requested selection.

    This path is intentionally independent from ``dataset_dict.json`` so subset
    refreshes can preserve other cached split directories without forcing them
    into subsequent training runs.

    :param str | Path all_dir: Root ``all/`` directory containing cached split folders.
    :param Sequence[str] selected_names: Canonical split names requested for this run.
    :return DatasetDict: Loaded dataset dictionary in request order.
    :raises ValueError: If any requested split is missing from the cache.
    """
    cache_root = Path(all_dir)
    if not cache_root.exists():
        raise ValueError(
            f"Cached contrastive dataset root does not exist: {cache_root}"
        )

    selected = list(selected_names)
    if not selected:
        raise ValueError("selected_names must contain at least one contrastive split.")

    available = discover_cached_contrastive_dataset_names(cache_root)
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            "Cached contrastive dataset is missing requested splits "
            f"{missing}. Available splits: {available}."
        )

    dataset_dict = {
        name: CONTRASTIVE_DATASETS[name].from_disk(str(cache_root / name)).dataset
        for name in selected
    }
    return DatasetDict(dataset_dict)


class AbsDataset(ABC):
    """Abstract dataset wrapper providing Hub/disk loading hooks."""

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize dataset wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset instance.
        """
        super().__init__()

        if dataset is None:
            self._dataset = self.from_hub()
        else:
            self._dataset = dataset

        self.size = self._dataset.num_rows

    @property
    def dataset(self) -> Dataset:
        """Return the underlying dataset.

        :return Dataset: Wrapped dataset.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        """Update the underlying dataset.

        :param Dataset dataset: Dataset to wrap.
        """
        self._dataset = dataset

    @classmethod
    def from_disk(cls, dataset_path: str) -> "AbsDataset":
        """Load dataset from disk.

        :param str dataset_path: Path to the dataset on disk.
        :return AbsDataset: Dataset wrapper instance.
        """
        dataset = load_from_disk(dataset_path)
        return cls(dataset)

    @abstractmethod
    def from_hub(self) -> Dataset:
        """Load dataset from the Hugging Face Hub.

        :return Dataset: Loaded dataset.
        """
        pass


class ALLNLI(AbsDataset):
    """ALLNLI dataset wrapper."""

    name = "ALLNLI"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the ALLNLI wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the ALLNLI triplet split.

        :return Dataset: Normalized ALLNLI dataset.
        """
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
    """Amazon QA dataset wrapper."""

    name = "AMAZONQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the AmazonQA wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the AmazonQA training split.

        :return Dataset: Normalized AmazonQA dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/Amazon-QA", split="train", num_proc=num_proc
        )

        def unwrap_corpus_column(example: dict[str, Any]) -> dict[str, Any]:
            """Flatten positive corpus entries.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized ``corpus`` values.
            """
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
    """ConcurrentQA dataset wrapper."""

    name = "CONCURRENTQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the ConcurrentQA wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the ConcurrentQA retrieval split.

        :return Dataset: Normalized ConcurrentQA dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "stanfordnlp/concurrentqa-retrieval", split="train", num_proc=num_proc
        )

        dataset = dataset.rename_column("question", "query")

        def process(example: dict[str, Any]) -> dict[str, Any]:
            """Normalize columns into query/corpus/negative fields.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized contrastive fields.
            """
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
    """FEVER dataset wrapper."""

    name = "FEVER"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the FEVER wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the FEVER training split.

        :return Dataset: Normalized FEVER dataset.
        """
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
        """Load and normalize the FEVER corpus split.

        :return pd.DataFrame: Normalized FEVER corpus table.
        """
        corpus = load_dataset("mteb/fever", "corpus", split="corpus")
        corpus = corpus.remove_columns("title")
        corpus = corpus.rename_column("_id", "corpus-id")
        corpus = corpus.rename_column("text", "corpus")
        return corpus.to_pandas()

    def load_and_prepare_queries(self) -> pd.DataFrame:
        """Load and normalize the FEVER queries split.

        :return pd.DataFrame: Normalized FEVER queries table.
        """
        queries = load_dataset("mteb/fever", "queries", split="queries")
        queries = queries.rename_column("_id", "query-id")
        queries = queries.rename_column("text", "query")
        return queries.to_pandas()


class GITHUBISSUE(AbsDataset):
    """GitHub Issue similarity dataset wrapper."""

    name = "GITHUBISSUE"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the GitHub Issue wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the GitHub issue similarity split.

        :return Dataset: Normalized GitHub issue dataset.
        """
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
    """GOOAQ hard-negative dataset wrapper."""

    name = "GOOAQ"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the GOOAQ wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the GOOAQ hard-negative split.

        :return Dataset: Normalized GOOAQ dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "tomaarsen/gooaq-hard-negatives",
            name="triplet-5",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("question", "query")
        dataset = dataset.rename_column("answer", "corpus")

        def aggregate_negatives(example: dict[str, Any]) -> dict[str, Any]:
            """Aggregate negatives into nested lists per query.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with aggregated negatives.
            """
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
    """MS MARCO dataset wrapper."""

    name = "MSMARCO"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the MS MARCO wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the MS MARCO training split.

        :return Dataset: Normalized MS MARCO dataset.
        """
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
        """Load and normalize the MS MARCO corpus split.

        :return pd.DataFrame: Normalized MS MARCO corpus table.
        """
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus")
        corpus = corpus.remove_columns("title")
        corpus = corpus.rename_column("_id", "corpus-id")
        corpus = corpus.rename_column("text", "corpus")
        return corpus.to_pandas()

    def load_and_prepare_queries(self) -> pd.DataFrame:
        """Load and normalize the MS MARCO queries split.

        :return pd.DataFrame: Normalized MS MARCO queries table.
        """
        queries = load_dataset("mteb/msmarco", "queries", split="queries")
        queries = queries.rename_column("_id", "query-id")
        queries = queries.rename_column("text", "query")
        return queries.to_pandas()


class PAQ(AbsDataset):
    """PAQ dataset wrapper."""

    name = "PAQ"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the PAQ wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the PAQ training split.

        :return Dataset: Normalized PAQ dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/PAQ_pairs", split="train", num_proc=num_proc
        )

        def split_set_column(example: dict[str, Any]) -> dict[str, Any]:
            """Split pair tuples into query/corpus columns.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized query/corpus fields.
            """
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
    """PubMedQA dataset wrapper."""

    name = "PUBMEDQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the PubMedQA wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the PubMedQA triplet split.

        :return Dataset: Normalized PubMedQA dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/pubmedqa",
            name="triplet-20",
            split="train",
            num_proc=num_proc,
        )

        dataset = dataset.rename_column("anchor", "query")
        dataset = dataset.rename_column("positive", "corpus")

        def aggregate_negatives(example: dict[str, Any]) -> dict[str, Any]:
            """Aggregate negatives into nested lists per query.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with aggregated negatives.
            """
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
    """QQP triplet dataset wrapper."""

    name = "QQP"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the QQP wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the QQP triplet split.

        :return Dataset: Normalized QQP dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/QQP_triplets", split="train", num_proc=num_proc
        )

        def split_set_column(example: dict[str, Any]) -> dict[str, Any]:
            """Split set examples into query/corpus/negative columns.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized contrastive fields.
            """
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
    """Sentence compression dataset wrapper."""

    name = "SENTENCECOMP"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the sentence compression wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the sentence compression split.

        :return Dataset: Normalized sentence compression dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "embedding-data/sentence-compression", split="train", num_proc=num_proc
        )

        def split_set_column(example: dict[str, Any]) -> dict[str, Any]:
            """Split set examples into query/corpus columns.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized query/corpus fields.
            """
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
    """StackExchange duplicate question dataset wrapper."""

    name = "STACKEXCHANGE"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the StackExchange wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize StackExchange duplicate-question pairs.

        :return Dataset: Normalized StackExchange dataset.
        """
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
    """StackOverflow duplicate question dataset wrapper."""

    name = "STACKOVERFLOW"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the StackOverflow wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize StackOverflow duplicate-question pairs.

        :return Dataset: Normalized StackOverflow dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "mteb/stackoverflowdupquestions-reranking", split="train", num_proc=num_proc
        )

        def split_set_column(example: dict[str, Any]) -> dict[str, Any]:
            """Normalize positive examples into corpus column.

            :param dict[str, Any] example: Batched dataset example.
            :return dict[str, Any]: Example with normalized ``corpus`` values.
            """
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
    """STS12 dataset wrapper filtered to high-similarity pairs."""

    name = "STS12"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the STS12 wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the high-similarity STS12 split.

        :return Dataset: Normalized STS12 dataset.
        """
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
    """STS Benchmark dataset wrapper filtered to high-similarity pairs."""

    name = "STSBENCHMARK"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the STS Benchmark wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the high-similarity STS Benchmark split.

        :return Dataset: Normalized STS Benchmark dataset.
        """
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
    """TriviaQA triplet dataset wrapper."""

    name = "TRIVIAQA"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the TriviaQA wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the TriviaQA triplet split.

        :return Dataset: Normalized TriviaQA dataset.
        """
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
    """WikiHow dataset wrapper for query/summary pairs."""

    name = "WIKIHOW"

    def __init__(self, dataset: Optional[Dataset] = None):
        """Initialize the WikiHow wrapper.

        :param Dataset | None dataset: Optional pre-loaded dataset.
        """
        super().__init__(dataset=dataset)

    def from_hub(self) -> Dataset:
        """Load and normalize the WikiHow query/summary split.

        :return Dataset: Normalized WikiHow dataset.
        """
        num_proc = len(os.sched_getaffinity(0))
        dataset = load_dataset(
            "sentence-transformers/wikihow", split="train", num_proc=num_proc
        )

        dataset = dataset.rename_column("text", "query")
        dataset = dataset.rename_column("summary", "corpus")

        return dataset


CONTRASTIVE_DATASETS: dict[str, type[AbsDataset]] = {
    "ALLNLI": ALLNLI,
    "AMAZONQA": AMAZONQA,
    "CONCURRENTQA": CONCURRENTQA,
    "FEVER": FEVER,
    "GITHUBISSUE": GITHUBISSUE,
    "GOOAQ": GOOAQ,
    "MSMARCO": MSMARCO,
    "PAQ": PAQ,
    "PUBMEDQA": PUBMEDQA,
    "QQP": QQP,
    "SENTENCECOMP": SENTENCECOMP,
    "STACKEXCHANGE": STACKEXCHANGE,
    "STACKOVERFLOW": STACKOVERFLOW,
    "STS12": STS12,
    "STSBENCHMARK": STSBENCHMARK,
    "TRIVIAQA": TRIVIAQA,
    "WIKIHOW": WIKIHOW,
}
