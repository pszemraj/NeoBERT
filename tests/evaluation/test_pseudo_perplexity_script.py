"""Regression tests for the pseudo-perplexity evaluation script."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset


def _load_pseudo_perplexity_module():
    """Load ``pseudo_perplexity.py`` for direct helper tests."""
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "pseudo_perplexity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "neobert_scripts_evaluation_pseudo_perplexity",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_deepspeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional legacy DeepSpeed dependencies must remain lazy."""
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deepspeed.utils.zero_to_fp32":
            raise ModuleNotFoundError("simulated missing deepspeed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    module = _load_pseudo_perplexity_module()

    assert hasattr(module, "load_step_checkpoint_state_dict")


def test_load_hub_masked_lm_preserves_learned_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub loading must not replace learned position embeddings with random weights."""
    module = _load_pseudo_perplexity_module()
    embeddings = object()
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=512),
        roberta=SimpleNamespace(embeddings=embeddings),
    )
    calls: list[tuple[str, bool]] = []

    def _from_pretrained(model_name: str, *, trust_remote_code: bool = False):
        calls.append((model_name, trust_remote_code))
        return model

    monkeypatch.setattr(
        module.AutoModelForMaskedLM, "from_pretrained", _from_pretrained
    )

    assert module._load_hub_masked_lm("roberta-base", max_length=512) is model
    assert calls == [("roberta-base", True)]
    assert model.roberta.embeddings is embeddings
    with pytest.raises(ValueError, match="exceeds.*position limit"):
        module._load_hub_masked_lm("roberta-base", max_length=1024)


def test_prepare_dataset_uses_one_consistent_length_window() -> None:
    """Filtering should retain exactly the configured inclusive character window."""
    module = _load_pseudo_perplexity_module()
    dataset = Dataset.from_dict(
        {
            "id": ["short", "minimum", "middle", "maximum", "long"],
            "text": ["a" * 4, "b" * 5, "c" * 7, "d" * 10, "e" * 11],
        }
    )

    selected = module._prepare_evaluation_dataset(
        dataset,
        text_column="text",
        min_chars=5,
        max_chars=10,
        n_sentences=10,
        num_shards=1,
        shard_index=0,
        seed=42,
    )

    assert set(selected["id"]) == {"minimum", "middle", "maximum"}


def test_local_data_path_loads_saved_dataset(
    tmp_path: Path,
) -> None:
    """The local data option should load datasets saved with ``save_to_disk``."""
    module = _load_pseudo_perplexity_module()
    data_path = tmp_path / "local-corpus"
    expected = Dataset.from_dict({"text": ["example"]})
    expected.save_to_disk(data_path)

    loaded, label = module._load_evaluation_dataset(
        data_path=data_path,
        dataset_name="ignored",
        dataset_config=None,
        split="train",
    )

    assert loaded["text"] == ["example"]
    assert label == "local-corpus_train"


def test_model_source_cli_is_exclusive_and_local_requires_checkpoint() -> None:
    """A run must select exactly one model source with complete local arguments."""
    module = _load_pseudo_perplexity_module()
    parser = module._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--hub_model", "bert-base-uncased", "--config_path", "x"])
    with pytest.raises(SystemExit):
        module.main(["--config_path", "x"])


def test_masked_batches_exclude_special_tokens() -> None:
    """Pseudo-perplexity rows should mask content tokens, never special tokens."""
    module = _load_pseudo_perplexity_module()
    dataset = Dataset.from_dict({"id": ["sample"], "text": ["hello world"]})

    class _Tokenizer:
        mask_token_id = 99

        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": torch.tensor([[101, 10, 11, 102]]),
                "special_tokens_mask": torch.tensor([[1, 0, 0, 1]]),
            }

    batches = list(
        module._iter_masked_batches(
            dataset,
            _Tokenizer(),
            text_column="text",
            id_column="id",
            batch_size=2,
            max_length=8,
        )
    )

    assert len(batches) == 1
    sample_id, input_ids, labels = batches[0]
    assert sample_id == "0:sample"
    assert input_ids.tolist() == [[101, 99, 11, 102], [101, 10, 99, 102]]
    assert labels.tolist() == [[-100, 10, -100, -100], [-100, -100, 11, -100]]


def test_build_neobert_uses_runtime_config_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local evaluation should preserve current training-model configuration fields."""
    module = _load_pseudo_perplexity_module()
    model_cfg = SimpleNamespace(
        name="tiny",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        rope=True,
        rms_norm=True,
        hidden_act="gelu",
        dropout_prob=0.1,
        norm_eps=1e-5,
        embedding_init_range=0.01,
        decoder_init_range=0.02,
        kernel_backend="torch",
        ngpt=True,
        base_scale=0.25,
    )
    cfg = SimpleNamespace(
        model=model_cfg,
        tokenizer=SimpleNamespace(
            name="tokenizer",
            trust_remote_code=False,
            revision=None,
            allow_special_token_rewrite=False,
        ),
    )
    tokenizer = SimpleNamespace(pad_token_id=7)
    captured = SimpleNamespace(config=None, loaded_state_dict=None)
    captured.load_state_dict = lambda state_dict: setattr(
        captured, "loaded_state_dict", state_dict
    )
    expected_state_dict = {"weight": torch.ones(2, 2)}
    checkpoint_calls: list[tuple[Path, str, str]] = []

    def _load_state_dict(path: Path, checkpoint: str, *, map_location: str = "cpu"):
        checkpoint_calls.append((Path(path), checkpoint, map_location))
        return expected_state_dict

    monkeypatch.setattr(module.ConfigLoader, "load", lambda _path: cfg)
    monkeypatch.setattr(module, "get_tokenizer", lambda **_kwargs: tokenizer)
    monkeypatch.setattr(
        module,
        "NeoBERTLMHead",
        lambda config: setattr(captured, "config", config) or captured,
    )
    monkeypatch.setattr(
        module,
        "load_step_checkpoint_state_dict",
        _load_state_dict,
    )

    model, loaded_tokenizer, label = module._build_neobert_masked_lm(
        tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoints",
        checkpoint="10",
        max_length=256,
    )

    assert model is captured
    assert loaded_tokenizer is tokenizer
    assert label == "tiny"
    assert captured.config.max_length == 256
    assert captured.config.dropout == 0.1
    assert captured.config.pad_token_id == 7
    assert captured.config.attn_backend == "sdpa"
    assert captured.config.kernel_backend == "torch"
    assert captured.config.ngpt is True
    assert captured.config.base_scale == 0.25
    assert captured.loaded_state_dict == expected_state_dict
    assert checkpoint_calls == [(tmp_path / "checkpoints", "10", "cpu")]
