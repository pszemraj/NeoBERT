"""Model checkpoint merging utilities."""

import os

import terge
import torch

from neobert.model import NeoBERTConfig, NeoBERTForSequenceClassification

from ..config import Config


def get_merged_model(cfg: Config) -> torch.nn.Module:
    """Load and merge multiple checkpoints into a single model.

    :param Config cfg: Configuration with checkpoint list and model settings.
    :return torch.nn.Module: Merged model instance.
    """
    # Use the config model settings directly
    model_list = []
    for ckpt in cfg.model.checkpoint_list:
        state_dict = torch.load(
            os.path.join(
                cfg.model.pretrained_checkpoint_dir, str(ckpt), "state_dict.pt"
            )
        )
        num_labels = state_dict["classifier.weight"].size(0)
        # Create model config from our Config object
        model_config = NeoBERTConfig(
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            num_attention_heads=cfg.model.num_attention_heads,
            intermediate_size=cfg.model.intermediate_size,
            dropout=cfg.model.dropout_prob,
            vocab_size=cfg.model.vocab_size,
            max_length=cfg.model.max_position_embeddings,
            flash_attention=cfg.model.flash_attention,
            ngpt=cfg.model.ngpt,
            hidden_act=cfg.model.hidden_act,
            rope=cfg.model.rope,
            rms_norm=cfg.model.rms_norm,
            norm_eps=cfg.model.norm_eps,
            pad_token_id=cfg.model.pad_token_id,
        )
        model = NeoBERTForSequenceClassification(
            model_config,
            num_labels=num_labels,
            classifier_dropout=cfg.model.classifier_dropout,
            classifier_init_range=cfg.model.classifier_init_range,
        )
        model.load_state_dict(state_dict, strict=False)
        model_list.append(model)
    merged_model = terge.merge(model_list, progress=True, inplace=True)
    return merged_model
