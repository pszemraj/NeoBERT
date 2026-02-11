import pytest

from neobert.config import Config
from neobert.utils import prepare_wandb_config


def test_prepare_wandb_config_scopes_pretraining_payload():
    cfg = Config()
    cfg.task = "pretraining"
    cfg.dataset.min_length = 42
    cfg.glue.task_name = "sst2"
    cfg.mteb_batch_size = 64
    cfg.use_deepspeed = True
    cfg.trainer.report_to = ["wandb"]

    payload = prepare_wandb_config(cfg)

    assert payload["task"] == "pretraining"
    assert "glue" not in payload
    assert "contrastive" not in payload
    assert "mteb_batch_size" not in payload
    assert "use_deepspeed" not in payload
    assert "dataset" in payload
    assert "min_length" not in payload["dataset"]
    assert "report_to" not in payload["trainer"]


def test_prepare_wandb_config_preserves_raw_model_dict_for_glue():
    cfg = Config()
    cfg.task = "glue"
    cfg._raw_model_dict = {"hidden_size": 384, "hidden_act": "swiglu"}

    payload = prepare_wandb_config(cfg)

    assert "_raw_model_dict" in payload
    assert payload["_raw_model_dict"] == cfg._raw_model_dict


def test_prepare_wandb_config_keeps_canonical_glue_task():
    cfg = Config()
    cfg.task = "glue"
    cfg.glue.task_name = "mnli"

    payload = prepare_wandb_config(cfg)

    assert payload["task"] == "glue"
    assert payload["glue"]["task_name"] == "mnli"


def test_prepare_wandb_config_keeps_contrastive_pretraining_prob():
    cfg = Config()
    cfg.task = "contrastive"
    cfg.contrastive.pretraining_prob = 0.42

    payload = prepare_wandb_config(cfg)

    assert payload["task"] == "contrastive"
    assert "contrastive" in payload
    assert payload["contrastive"]["pretraining_prob"] == 0.42


def test_prepare_wandb_config_requires_supported_type():
    dummy = object()

    with pytest.raises(TypeError):
        prepare_wandb_config(dummy)
