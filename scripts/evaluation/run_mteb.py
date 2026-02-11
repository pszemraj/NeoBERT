"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from mteb import MTEB

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_deepspeed_fp32_state_dict,
    load_model_safetensors,
)
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


def _resolve_mteb_tasks(cfg: Any) -> list[str]:
    """Resolve selected MTEB task names from config and CLI overrides.

    Resolution order:
    1. ``cfg.task_types`` (CLI ``--task_types``), if provided.
    2. ``cfg.mteb_task_type`` category.

    ``cfg.task_types`` entries can be either category aliases from ``TASK_TYPE``
    (for example ``classification`` or ``sts``) or explicit task names.

    :param Any cfg: Configuration object.
    :raises ValueError: If any requested task/category is unknown.
    :return list[str]: Ordered deduplicated list of task names.
    """
    requested = getattr(cfg, "task_types", None)
    if requested is None:
        mteb_task_type = str(getattr(cfg, "mteb_task_type", "all")).strip().lower()
        if mteb_task_type not in TASK_TYPE:
            raise ValueError(f"Task type must be one of {sorted(TASK_TYPE.keys())}.")
        return list(TASK_TYPE[mteb_task_type])

    if isinstance(requested, str):
        requested_tokens = [token.strip() for token in requested.split(",")]
    else:
        requested_tokens = [str(token).strip() for token in requested]

    explicit_lookup = {task.lower(): task for task in TASK_LIST}
    selected: list[str] = []
    unknown: list[str] = []
    for token in requested_tokens:
        if not token:
            continue
        lowered = token.lower()
        if lowered == "all":
            selected.extend(TASK_LIST)
            continue
        if lowered in TASK_TYPE:
            selected.extend(TASK_TYPE[lowered])
            continue
        explicit_task = explicit_lookup.get(lowered)
        if explicit_task is not None:
            selected.append(explicit_task)
            continue
        unknown.append(token)

    if unknown:
        raise ValueError(
            "Unknown --task_types entries: "
            + ", ".join(sorted(unknown))
            + ". Valid categories: "
            + ", ".join(sorted(TASK_TYPE.keys()))
        )

    # Stable dedupe to preserve user-specified order.
    return list(dict.fromkeys(selected))


def _load_mteb_encoder_weights(
    model: NeoBERTForMTEB,
    state_dict: dict[str, torch.Tensor],
    *,
    source: str,
) -> None:
    """Load checkpoint weights for MTEB with encoder/head key tolerance.

    Pretraining checkpoints commonly include LM-head parameters (for example
    ``decoder.*``) that are not part of ``NeoBERTForMTEB``. We therefore load with
    ``strict=False`` and log any unexpected keys that are not known LM-head extras.

    :param NeoBERTForMTEB model: MTEB model instance.
    :param dict[str, torch.Tensor] state_dict: Checkpoint state dict.
    :param str source: Human-readable checkpoint source for logs.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible is None:
        # Compatibility for lightweight test doubles that don't return
        # ``_IncompatibleKeys`` from ``load_state_dict``.
        return

    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    missing_keys = list(getattr(incompatible, "missing_keys", []))

    lm_head_prefixes = ("decoder.", "model.decoder.")
    lm_head_unexpected = [
        key for key in unexpected_keys if key.startswith(lm_head_prefixes)
    ]
    remaining_unexpected = [
        key for key in unexpected_keys if key not in lm_head_unexpected
    ]

    if lm_head_unexpected:
        logger.info(
            "Ignoring %d LM-head keys while loading %s for MTEB.",
            len(lm_head_unexpected),
            source,
        )
    if remaining_unexpected:
        logger.warning(
            "Unexpected non-head keys while loading %s for MTEB: %s",
            source,
            ", ".join(sorted(remaining_unexpected)),
        )
    if missing_keys:
        logger.warning(
            "Missing model keys while loading %s for MTEB: %s",
            source,
            ", ".join(sorted(missing_keys)),
        )


def evaluate_mteb(cfg: Any) -> None:
    """Evaluate a model on the MTEB benchmark.

    :param Any cfg: Configuration object with MTEB settings.
    """
    # Get MTEB-specific config (kept at top-level Config for now)
    mteb_batch_size = getattr(cfg, "mteb_batch_size", 32)
    mteb_pooling = getattr(cfg, "mteb_pooling", "mean")
    mteb_overwrite_results = getattr(cfg, "mteb_overwrite_results", False)
    pretrained_checkpoint = getattr(cfg, "pretrained_checkpoint", "latest")
    pretrained_checkpoint_dir = Path(cfg.trainer.output_dir)
    use_deepspeed = getattr(cfg, "use_deepspeed", False)
    selected_tasks = _resolve_mteb_tasks(cfg)

    # Get checkpoint number
    if pretrained_checkpoint != "latest":
        ckpt = pretrained_checkpoint
    else:
        checkpoint_root = pretrained_checkpoint_dir / "checkpoints"
        candidates = [
            int(path.name)
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        if not candidates:
            raise ValueError(
                f"Unable to find numbered checkpoints under {checkpoint_root}"
            )
        ckpt = str(max(candidates))

    # Define path to store results
    output_folder = (
        pretrained_checkpoint_dir / "mteb" / str(ckpt) / str(cfg.tokenizer.max_length)
    )

    # Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=cfg.tokenizer.max_length,
    )

    # Instantiate model
    model_config = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        vocab_size=cfg.model.vocab_size,  # Use checkpoint's vocab_size since we always load weights
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
    checkpoint_root = pretrained_checkpoint_dir / "checkpoints"
    checkpoint_step_dir = checkpoint_root / str(ckpt)
    checkpoint_path = checkpoint_step_dir / MODEL_WEIGHTS_NAME
    if use_deepspeed:
        state_dict = load_deepspeed_fp32_state_dict(
            checkpoint_root,
            tag=str(ckpt),
        )
        _load_mteb_encoder_weights(
            model,
            state_dict,
            source=f"DeepSpeed checkpoint tag={ckpt}",
        )
    else:
        if checkpoint_path.exists():
            state_dict = load_model_safetensors(
                checkpoint_step_dir,
                map_location=device,
            )
            _load_mteb_encoder_weights(
                model,
                state_dict,
                source=f"safetensors checkpoint {checkpoint_path}",
            )
        else:
            logger.warning(
                f"{MODEL_WEIGHTS_NAME} not found at {checkpoint_path}; "
                "falling back to DeepSpeed fp32 shard conversion."
            )
            state_dict = load_deepspeed_fp32_state_dict(
                checkpoint_root,
                tag=str(ckpt),
            )
            _load_mteb_encoder_weights(
                model,
                state_dict,
                source=f"DeepSpeed fallback tag={ckpt}",
            )

    model.to(device)
    model.eval()

    # Evaluate
    for task in selected_tasks:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            evaluation.run(
                model,
                output_folder=output_folder,
                eval_splits=eval_splits,
                overwrite_results=mteb_overwrite_results,
            )


def main() -> None:
    """Run the MTEB evaluation CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Model path"
    )
    parser.add_argument(
        "--task_types", type=str, default="all", help="Task types to evaluate"
    )
    parser.add_argument(
        "--output_folder", type=str, default="results", help="Output folder"
    )

    args, remaining = parser.parse_known_args()

    # Load configuration
    config = ConfigLoader.load(args.config, remaining)
    config.trainer.output_dir = args.model_name_or_path
    config.task_types = args.task_types.split(",") if args.task_types != "all" else None
    config.output_folder = args.output_folder

    # Run MTEB evaluation
    evaluate_mteb(config)


if __name__ == "__main__":
    main()
