"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

import torch

from mteb import MTEB

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from neobert.model import NeoBERTForMTEB, NeoBERTConfig
from neobert.tokenizer import get_tokenizer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
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
]

TASK_LIST_CLUSTERING = [
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
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "MSMARCO",
    "ArguAna",
    "ClimateFEVER",
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
]

TASK_LIST_STS = [
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
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

TASK_TYPE = {
    "classification": TASK_LIST_CLASSIFICATION,
    "clustering": TASK_LIST_CLUSTERING,
    "pair_classification": TASK_LIST_PAIR_CLASSIFICATION,
    "reranking": TASK_LIST_RERANKING,
    "retrieval": TASK_LIST_RETRIEVAL,
    "sts": TASK_LIST_STS,
    "all": TASK_LIST,
}


@hydra.main(version_base=None, config_path="../../conf", config_name="mteb")
def evaluate_mteb(cfg: DictConfig):
    # Check if task type is valid
    if cfg.mteb.task_type not in TASK_TYPE.keys():
        raise ValueError(f"Task type must be one of {TASK_TYPE.keys()}.")

    # Get checkpoint number
    if cfg.model.pretrained_checkpoint != "latest":
        ckpt = cfg.model.pretrained_checkpoint
    else:
        latest_path = os.path.join(cfg.model.pretrained_checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                ckpt = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    # Define path to store results
    output_folder = os.path.join(cfg.model.pretrained_checkpoint_dir, "mteb", str(ckpt), str(cfg.tokenizer.max_length))

    # Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = get_tokenizer(**cfg.tokenizer)

    # Instantiate model
    model_pretraining_config = OmegaConf.load(cfg.model.pretrained_config_path)
    model = NeoBERTForMTEB(
        config=NeoBERTConfig(**model_pretraining_config.model, **cfg.tokenizer),
        tokenizer=tokenizer,
        batch_size=cfg.mteb.batch_size,
        pooling=cfg.mteb.pooling,
        max_length=cfg.tokenizer.max_length,
    )

    # Load pretrained weights
    if cfg.model.deepspeed:
        model = load_state_dict_from_zero_checkpoint(model, os.path.join(cfg.model.pretrained_checkpoint_dir, "model_checkpoints"), tag=str(ckpt))
    else:
        raise NotImplementedError

    model.to(device)
    model.eval()

    # Evaluate
    for task in TASK_TYPE[cfg.mteb.task_type]:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        with torch.autocast(device_type=device, dtype=torch.float16):
            evaluation.run(model, output_folder=output_folder, eval_splits=eval_splits, overwrite_results=cfg.mteb.overwrite_results)


if __name__ == "__main__":
    evaluate_mteb()
