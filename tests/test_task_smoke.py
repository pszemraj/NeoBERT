#!/usr/bin/env python3
"""Compact cross-task smoke coverage for core config->runtime flows."""

import pytest
import torch

from neobert.config import ConfigLoader


def test_pretraining_config_to_lm_forward_smoke(tiny_pretrain_config_path) -> None:
    """Pretraining config should build LM head and run a tiny forward pass."""
    from neobert.model import NeoBERTConfig, NeoBERTLMHead

    cfg = ConfigLoader.load(str(tiny_pretrain_config_path))

    model_cfg = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=cfg.model.max_position_embeddings,
        pad_token_id=cfg.model.pad_token_id,
        attn_backend=cfg.model.attn_backend,
    )
    model = NeoBERTLMHead(model_cfg)

    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(input_ids)

    assert "logits" in out
    assert "hidden_representation" in out
    assert tuple(out["logits"].shape) == (2, 8, cfg.model.vocab_size)


def test_glue_config_to_classifier_logits_and_loss_smoke(tiny_glue_config_path) -> None:
    """GLUE config should build classifier and compute logits/loss shapes."""
    from neobert.huggingface import NeoBERTHFForSequenceClassification
    from neobert.model import NeoBERTConfig

    cfg = ConfigLoader.load(str(tiny_glue_config_path))

    model_cfg = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=cfg.model.max_position_embeddings,
        pad_token_id=cfg.model.pad_token_id,
        attn_backend=cfg.model.attn_backend,
        num_labels=cfg.glue.num_labels,
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


def test_contrastive_trainer_preflight_fails_fast_without_dataset_path(
    tiny_contrastive_config_path,
) -> None:
    """Contrastive trainer should fail before dataset/network setup when path is missing."""
    from neobert.contrastive.trainer import trainer

    cfg = ConfigLoader.load(str(tiny_contrastive_config_path))
    cfg.dataset.path = None

    with pytest.raises(ValueError, match="dataset.path"):
        trainer(cfg)
