#!/usr/bin/env python3
"""Compact cross-task smoke coverage for core config->runtime flows."""

from pathlib import Path

import pytest
import torch

from neobert.config import ConfigLoader


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_pretraining_config_to_lm_forward_smoke() -> None:
    """Pretraining config should build LM head and run a tiny forward pass."""
    from neobert.model import NeoBERTConfig, NeoBERTLMHead

    cfg = ConfigLoader.load(str(CONFIG_DIR / "pretraining" / "test_tiny_pretrain.yaml"))

    model_cfg = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        dropout=cfg.model.dropout_prob,
        vocab_size=cfg.model.vocab_size,
        max_length=cfg.model.max_position_embeddings,
        attn_backend=cfg.model.attn_backend,
        ngpt=cfg.model.ngpt,
        hidden_act=cfg.model.hidden_act,
    )
    model = NeoBERTLMHead(model_cfg)

    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(input_ids)

    assert "logits" in out
    assert "hidden_representation" in out
    assert tuple(out["logits"].shape) == (2, 8, cfg.model.vocab_size)


def test_glue_config_to_classifier_logits_and_loss_smoke() -> None:
    """GLUE config should build classifier and compute logits/loss shapes."""
    from neobert.model import NeoBERTConfig, NeoBERTHFForSequenceClassification

    cfg = ConfigLoader.load(str(CONFIG_DIR / "evaluation" / "test_tiny_glue.yaml"))

    model_cfg = NeoBERTConfig(
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        intermediate_size=cfg.model.intermediate_size,
        dropout=cfg.model.dropout_prob,
        vocab_size=cfg.model.vocab_size,
        max_length=cfg.model.max_position_embeddings,
        attn_backend=cfg.model.attn_backend,
        ngpt=cfg.model.ngpt,
        num_labels=cfg.glue.num_labels,
        hidden_act=cfg.model.hidden_act,
    )
    model = NeoBERTHFForSequenceClassification(model_cfg)

    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 10))
    attention_mask = torch.ones(2, 10)
    labels = torch.tensor([0, 1], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    assert out.loss is not None
    assert tuple(out.logits.shape) == (2, cfg.glue.num_labels)


def test_contrastive_trainer_preflight_fails_fast_without_dataset_path() -> None:
    """Contrastive trainer should fail before dataset/network setup when path is missing."""
    from neobert.contrastive.trainer import trainer

    cfg = ConfigLoader.load(str(CONFIG_DIR / "contrastive" / "test_tiny_contrastive.yaml"))
    cfg.dataset.path = None

    with pytest.raises(ValueError, match="dataset.path"):
        trainer(cfg)
