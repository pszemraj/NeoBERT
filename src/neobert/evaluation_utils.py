"""Shared checkpoint-local model resolution for evaluation entry points."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neobert.checkpointing import resolve_training_checkpoint_artifacts
from neobert.config import Config, ConfigLoader
from neobert.model import NeoBERTConfig
from neobert.tokenizer import get_tokenizer


@dataclass(frozen=True)
class ResolvedCheckpointModelSource:
    """Validated checkpoint-local config, tokenizer, and model configuration."""

    checkpoint_root: Path
    checkpoint_dir: Path
    checkpoint_tag: str
    config_path: Path
    tokenizer_path: Path
    training_config: Config
    tokenizer: Any
    model_config: NeoBERTConfig


def resolve_checkpoint_model_source(
    checkpoint_path: str | Path,
    checkpoint: str | int,
    *,
    max_length: int,
) -> ResolvedCheckpointModelSource:
    """Resolve and validate checkpoint-local evaluation model inputs.

    :param str | Path checkpoint_path: Run root, checkpoint root, or step directory.
    :param str | int checkpoint: Requested checkpoint selector.
    :param int max_length: Requested evaluation context length.
    :raises FileNotFoundError: If checkpoint config or tokenizer artifacts are missing.
    :raises ValueError: If tokenizer/model identity or context length is incompatible.
    :return ResolvedCheckpointModelSource: Validated local evaluation source.
    """
    checkpoint_root, checkpoint_dir, checkpoint_tag = (
        resolve_training_checkpoint_artifacts(checkpoint_path, checkpoint)
    )
    config_path = checkpoint_dir / "config.yaml"
    tokenizer_path = checkpoint_dir / "tokenizer"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint-local training config is missing: {config_path}"
        )
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint-local tokenizer is missing: {tokenizer_path}"
        )

    cfg = ConfigLoader.load(config_path)
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=str(tokenizer_path),
        max_length=max_length,
        trust_remote_code=False,
        revision=None,
        allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
    )
    if len(tokenizer) != int(cfg.model.vocab_size):
        raise ValueError(
            "Checkpoint tokenizer/model vocabulary mismatch: "
            f"tokenizer={len(tokenizer)}, model={cfg.model.vocab_size}."
        )
    if tokenizer.pad_token_id != int(cfg.model.pad_token_id):
        raise ValueError(
            "Checkpoint tokenizer/model pad-token mismatch: "
            f"tokenizer={tokenizer.pad_token_id}, model={cfg.model.pad_token_id}."
        )

    trained_max_length = int(cfg.model.max_position_embeddings)
    if not cfg.model.rope and max_length > trained_max_length:
        raise ValueError(
            f"Requested max_length={max_length} exceeds the learned position limit "
            f"({trained_max_length}) in {checkpoint_dir}."
        )
    model_max_length = max_length if cfg.model.rope else trained_max_length
    model_config = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        attn_backend="sdpa",
    )
    return ResolvedCheckpointModelSource(
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        checkpoint_tag=checkpoint_tag,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        training_config=cfg,
        tokenizer=tokenizer,
        model_config=model_config,
    )
