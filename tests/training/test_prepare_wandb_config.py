import pytest

from neobert.config import Config
from neobert.utils import prepare_wandb_config


def test_prepare_wandb_config_scopes_pretraining_payload():
    cfg = Config()
    cfg.task = "pretraining"
    cfg.dataset.min_length = 42
    cfg.dataset.alpha = 0.8
    cfg.glue.task_name = "sst2"
    cfg.mteb_batch_size = 64
    cfg.use_deepspeed = True
    cfg.trainer.report_to = ["wandb"]
    cfg.trainer.max_ckpt = 7
    cfg.trainer.train_batch_size = 64
    cfg.trainer.eval_batch_size = 64
    cfg.trainer.dataloader_num_workers = 4
    cfg.trainer.greater_is_better = False
    cfg.trainer.load_best_model_at_end = True
    cfg.trainer.metric_for_best_model = "loss"
    cfg.trainer.early_stopping = 2
    cfg.trainer.save_model = False
    cfg.trainer.disable_tqdm = True

    payload = prepare_wandb_config(cfg)

    assert payload["task"] == "pretraining"
    assert "glue" not in payload
    assert "contrastive" not in payload
    assert "mteb_batch_size" not in payload
    assert "use_deepspeed" not in payload
    assert "dataset" in payload
    assert "min_length" not in payload["dataset"]
    assert "alpha" not in payload["dataset"]
    assert "report_to" not in payload["trainer"]
    assert "max_ckpt" not in payload["trainer"]
    assert "train_batch_size" not in payload["trainer"]
    assert "eval_batch_size" not in payload["trainer"]
    assert "dataloader_num_workers" not in payload["trainer"]
    assert "greater_is_better" not in payload["trainer"]
    assert "load_best_model_at_end" not in payload["trainer"]
    assert "metric_for_best_model" not in payload["trainer"]
    assert "early_stopping" not in payload["trainer"]
    assert "save_model" not in payload["trainer"]
    assert "disable_tqdm" not in payload["trainer"]


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
    cfg.dataset.alpha = 0.75

    payload = prepare_wandb_config(cfg)

    assert payload["task"] == "contrastive"
    assert "contrastive" in payload
    assert payload["contrastive"]["pretraining_prob"] == 0.42
    assert payload["dataset"]["alpha"] == 0.75


def test_prepare_wandb_config_requires_supported_type():
    dummy = object()

    with pytest.raises(TypeError):
        prepare_wandb_config(dummy)
