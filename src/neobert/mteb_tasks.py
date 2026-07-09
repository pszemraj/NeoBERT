"""Canonical task registry for the repository's MTEB evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MTEBTaskSpec:
    """Describe one aggregation task and its executable MTEB task names."""

    aggregation_name: str
    executable_names: tuple[str, ...] = ()

    @property
    def execution_names(self) -> tuple[str, ...]:
        """Return concrete MTEB task names to execute.

        :return tuple[str, ...]: Concrete task names.
        """
        return self.executable_names or (self.aggregation_name,)


@dataclass(frozen=True)
class MTEBTaskGroup:
    """Describe an MTEB reporting category and its task specifications."""

    key: str
    label: str
    tasks: tuple[MTEBTaskSpec, ...]

    @property
    def execution_names(self) -> tuple[str, ...]:
        """Return ordered concrete task names for this category.

        :return tuple[str, ...]: Concrete task names.
        """
        return tuple(name for task in self.tasks for name in task.execution_names)


def _tasks(*names: str) -> tuple[MTEBTaskSpec, ...]:
    """Build ordinary one-name task specifications.

    :param str names: MTEB task names.
    :return tuple[MTEBTaskSpec, ...]: Task specifications.
    """
    return tuple(MTEBTaskSpec(name) for name in names)


CQADUPSTACK_TASKS = (
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
)

MTEB_TASK_GROUPS = (
    MTEBTaskGroup(
        "classification",
        "Class.",
        _tasks(
            "AmazonCounterfactualClassification",
            "AmazonPolarityClassification",
            "AmazonReviewsClassification",
            "Banking77Classification",
            "EmotionClassification",
            "ImdbClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "MTOPDomainClassification",
            "MTOPIntentClassification",
            "ToxicConversationsClassification",
            "TweetSentimentExtractionClassification",
        ),
    ),
    MTEBTaskGroup(
        "clustering",
        "Clust.",
        _tasks(
            "ArxivClusteringP2P",
            "ArxivClusteringS2S",
            "BiorxivClusteringP2P",
            "BiorxivClusteringS2S",
            "MedrxivClusteringP2P",
            "MedrxivClusteringS2S",
            "RedditClustering",
            "RedditClusteringP2P",
            "StackExchangeClustering",
            "StackExchangeClusteringP2P",
            "TwentyNewsgroupsClustering",
        ),
    ),
    MTEBTaskGroup(
        "pair_classification",
        "PairClass.",
        _tasks(
            "SprintDuplicateQuestions",
            "TwitterSemEval2015",
            "TwitterURLCorpus",
        ),
    ),
    MTEBTaskGroup(
        "reranking",
        "Rerank.",
        _tasks(
            "AskUbuntuDupQuestions",
            "MindSmallReranking",
            "SciDocsRR",
            "StackOverflowDupQuestions",
        ),
    ),
    MTEBTaskGroup(
        "retrieval",
        "Retr.",
        (
            *_tasks("MSMARCO", "ArguAna", "ClimateFEVER"),
            MTEBTaskSpec("CQADupstackRetrieval", CQADUPSTACK_TASKS),
            *_tasks(
                "DBPedia",
                "FEVER",
                "FiQA2018",
                "HotpotQA",
                "NFCorpus",
                "NQ",
                "QuoraRetrieval",
                "SCIDOCS",
                "SciFact",
                "Touche2020",
                "TRECCOVID",
            ),
        ),
    ),
    MTEBTaskGroup(
        "sts",
        "STS",
        _tasks(
            "BIOSSES",
            "SICK-R",
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STS17",
            "STS22",
            "STSBenchmark",
        ),
    ),
    MTEBTaskGroup("summarization", "Summ.", _tasks("SummEval")),
)

MTEB_TASK_GROUPS_BY_KEY = {group.key: group for group in MTEB_TASK_GROUPS}
MTEB_ALL_EXECUTION_TASKS = tuple(
    name for group in MTEB_TASK_GROUPS for name in group.execution_names
)
MTEB_EXECUTION_TASKS_BY_TYPE = {
    **{group.key: group.execution_names for group in MTEB_TASK_GROUPS},
    "all": MTEB_ALL_EXECUTION_TASKS,
}


def expand_mteb_task_name(name: str) -> tuple[str, ...] | None:
    """Resolve a concrete task or aggregation alias to executable names.

    Matching is case-insensitive while returned names retain MTEB spelling.

    :param str name: Concrete task name or aggregation alias.
    :return tuple[str, ...] | None: Resolved executable names, or ``None``.
    """
    normalized = name.strip().lower()
    for group in MTEB_TASK_GROUPS:
        for task in group.tasks:
            if task.aggregation_name.lower() == normalized:
                return task.execution_names
            for execution_name in task.execution_names:
                if execution_name.lower() == normalized:
                    return (execution_name,)
    return None
