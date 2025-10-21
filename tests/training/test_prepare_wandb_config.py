import pytest

from neobert.config import Config
from neobert.utils import prepare_wandb_config


def test_prepare_wandb_config_preserves_raw_model_dict():
    cfg = Config()
    cfg._raw_model_dict = {"hidden_size": 384, "hidden_act": "swiglu"}

    payload = prepare_wandb_config(cfg)

    assert "_raw_model_dict" in payload
    assert payload["_raw_model_dict"] == cfg._raw_model_dict


def test_prepare_wandb_config_requires_supported_type():
    dummy = object()

    with pytest.raises(TypeError):
        prepare_wandb_config(dummy)
