import pytest

from neobert.config import Config
from neobert.utils import format_resolved_config, prepare_wandb_config


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


def test_format_resolved_config_compact_and_sectioned():
    payload = {
        "task": "pretraining",
        "seed": 69,
        "model": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "trainer": {
            "mixed_precision": "bf16",
            "max_steps": 100000,
            "gradient_accumulation_steps": 4,
        },
    }

    rendered = format_resolved_config(payload, width=88)
    lines = rendered.splitlines()

    assert lines[0].startswith("[meta] ")
    assert any(line.startswith("[model] ") for line in lines)
    assert any(line.startswith("[trainer] ") for line in lines)
    assert all(len(line) <= 88 for line in lines)


def test_format_resolved_config_flattens_nested_sections():
    payload = {
        "optimizer": {
            "name": "muonclip",
            "muon_config": {
                "enable_clipping": False,
                "ns_steps": 5,
            },
        }
    }

    rendered = format_resolved_config(payload, width=120)

    assert "name=muonclip" in rendered
    assert "muon_config.enable_clipping=false" in rendered
    assert "muon_config.ns_steps=5" in rendered


def test_format_resolved_config_uses_consistent_continuation_indent():
    payload = {
        "task": "pretraining",
        "seed": 69,
        "dataset": {
            "name": "EleutherAI/SmolLM2-1.7B-stage-4-100B",
            "max_seq_length": 1024,
            "shuffle_buffer_size": 10000,
            "streaming": True,
        },
        "trainer": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
            "logging_steps": 25,
        },
    }

    rendered = format_resolved_config(payload, width=88)
    lines = rendered.splitlines()

    continuation_lines = [line for line in lines if line and not line.startswith("[")]
    assert continuation_lines, "Expected at least one wrapped continuation line"

    continuation_indents = {
        len(line) - len(line.lstrip(" ")) for line in continuation_lines
    }
    assert len(continuation_indents) == 1
