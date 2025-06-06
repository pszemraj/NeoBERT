import os

import terge
import torch
from omegaconf import DictConfig, OmegaConf

from neobert.model import NeoBERTConfig, NeoBERTForSequenceClassification


def get_merged_model(cfg: DictConfig):
    model_pretraining_config = OmegaConf.load(cfg.model.pretrained_config_path)
    model_list = []
    for ckpt in cfg.model.checkpoint_list:
        state_dict = torch.load(
            os.path.join(
                cfg.model.pretrained_checkpoint_dir, str(ckpt), "state_dict.pt"
            )
        )
        num_labels = state_dict["classifier.weight"].size(0)
        model = NeoBERTForSequenceClassification(
            NeoBERTConfig(**model_pretraining_config.model, **cfg.tokenizer),
            num_labels=num_labels,
            classifier_dropout=cfg.model.classifier_dropout,
            classifier_init_range=cfg.model.classifier_init_range,
        )
        model.load_state_dict(state_dict, strict=False)
        model_list.append(model)
    merged_model = terge.merge(model_list, progress=True, inplace=True)
    return merged_model
