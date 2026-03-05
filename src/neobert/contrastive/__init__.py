"""Contrastive training datasets, losses, and trainer."""

__all__ = [
    "SupConLoss",
    "trainer",
    "ALLNLI",
    "AMAZONQA",
    "CONCURRENTQA",
    "FEVER",
    "GITHUBISSUE",
    "GOOAQ",
    "MSMARCO",
    "PAQ",
    "PUBMEDQA",
    "QQP",
    "SENTENCECOMP",
    "STACKEXCHANGE",
    "STACKOVERFLOW",
    "STS12",
    "STSBENCHMARK",
    "TRIVIAQA",
    "WIKIHOW",
    "CONTRASTIVE_DATASETS",
]

from .datasets import (
    ALLNLI,
    AMAZONQA,
    CONTRASTIVE_DATASETS,
    CONCURRENTQA,
    FEVER,
    GITHUBISSUE,
    GOOAQ,
    MSMARCO,
    PAQ,
    PUBMEDQA,
    QQP,
    SENTENCECOMP,
    STACKEXCHANGE,
    STACKOVERFLOW,
    STS12,
    STSBENCHMARK,
    TRIVIAQA,
    WIKIHOW,
)
from .loss import SupConLoss
from .trainer import trainer
