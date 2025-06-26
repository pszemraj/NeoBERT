"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from mteb import MTEB

from neobert.config import ConfigLoader
from neobert.model import NeoBERTConfig, NeoBERTForMTEB
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


def evaluate_mteb(cfg):
    # Get MTEB-specific config (we'll add these to Config later)
    mteb_task_type = getattr(cfg, "mteb_task_type", "all")
    mteb_batch_size = getattr(cfg, "mteb_batch_size", 32)
    mteb_pooling = getattr(cfg, "mteb_pooling", "mean")
    mteb_overwrite_results = getattr(cfg, "mteb_overwrite_results", False)
    pretrained_checkpoint = getattr(cfg, "pretrained_checkpoint", "latest")
    pretrained_checkpoint_dir = cfg.trainer.output_dir
    use_deepspeed = getattr(cfg, "use_deepspeed", True)

    # Check if task type is valid
    if mteb_task_type not in TASK_TYPE.keys():
        raise ValueError(f"Task type must be one of {TASK_TYPE.keys()}.")

    # Get checkpoint number
    if pretrained_checkpoint != "latest":
        ckpt = pretrained_checkpoint
    else:
        latest_path = os.path.join(
            pretrained_checkpoint_dir, "model_checkpoints", "latest"
        )
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                ckpt = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    # Define path to store results
    output_folder = os.path.join(
        pretrained_checkpoint_dir,
        "mteb",
        str(ckpt),
        str(cfg.tokenizer.max_length),
    )

    # Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = get_tokenizer(
        name=cfg.tokenizer.name,
        path=cfg.tokenizer.path,
        max_length=cfg.tokenizer.max_length,
        padding=cfg.tokenizer.padding,
        truncation=cfg.tokenizer.truncation,
    )

    # Instantiate model
    model_config = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,
        rope=cfg.model.rope,
        rms_norm=cfg.model.rms_norm,
        hidden_act=cfg.model.hidden_act,
        dropout_prob=cfg.model.dropout_prob,
        norm_eps=cfg.model.norm_eps,
        embedding_init_range=cfg.model.embedding_init_range,
        decoder_init_range=cfg.model.decoder_init_range,
        classifier_init_range=cfg.model.classifier_init_range,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = NeoBERTForMTEB(
        config=model_config,
        tokenizer=tokenizer,
        batch_size=mteb_batch_size,
        pooling=mteb_pooling,
        max_length=cfg.tokenizer.max_length,
    )

    # Load pretrained weights
    if use_deepspeed:
        model = load_state_dict_from_zero_checkpoint(
            model,
            os.path.join(pretrained_checkpoint_dir, "model_checkpoints"),
            tag=str(ckpt),
        )
    else:
        checkpoint_path = os.path.join(
            pretrained_checkpoint_dir, "model_checkpoints", str(ckpt), "state_dict.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.to(device)
    model.eval()

    # Evaluate
    for task in TASK_TYPE[mteb_task_type]:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        with torch.autocast(device_type=device, dtype=torch.float16):
            evaluation.run(
                model,
                output_folder=output_folder,
                eval_splits=eval_splits,
                overwrite_results=mteb_overwrite_results,
            )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path")
    parser.add_argument("--task_types", type=str, default="all", help="Task types to evaluate")
    parser.add_argument("--output_folder", type=str, default="results", help="Output folder")
    
    args, remaining = parser.parse_known_args()
    
    # Load configuration
    config = ConfigLoader.load(args.config, remaining)
    config.model_name_or_path = args.model_name_or_path
    config.task_types = args.task_types.split(",") if args.task_types != "all" else None
    config.output_folder = args.output_folder
    
    # Run MTEB evaluation
    evaluate_mteb(config)


if __name__ == "__main__":
    main()
