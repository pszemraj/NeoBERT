"""Regression tests for the pseudo-perplexity evaluation script."""

from __future__ import annotations

import builtins
import csv
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from neobert import evaluation_utils
from neobert.checkpointing import save_state_dict_safetensors
from neobert.config import Config, ConfigLoader
from neobert.model import NeoBERTConfig, NeoBERTLMHead
from tests.tokenizer_utils import build_wordlevel_tokenizer


def test_shared_checkpoint_source_forwards_tokenizer_rewrite_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All evaluation entry points must reconstruct checkpoint tokenizer policy."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "10"
    (checkpoint_dir / "tokenizer").mkdir(parents=True)
    (checkpoint_dir / "config.yaml").touch()
    cfg = Config()
    cfg.tokenizer.allow_special_token_rewrite = True
    tokenizer = build_wordlevel_tokenizer()
    cfg.model.vocab_size = len(tokenizer)
    cfg.model.pad_token_id = tokenizer.pad_token_id
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        evaluation_utils,
        "resolve_training_checkpoint_artifacts",
        lambda *_args: (checkpoint_root, checkpoint_dir, "10"),
    )
    monkeypatch.setattr(evaluation_utils.ConfigLoader, "load", lambda _path: cfg)

    def _get_tokenizer(**kwargs):
        captured.update(kwargs)
        return tokenizer

    monkeypatch.setattr(evaluation_utils, "get_tokenizer", _get_tokenizer)
    evaluation_utils.resolve_checkpoint_model_source(
        checkpoint_root,
        "10",
        max_length=32,
    )

    assert captured["allow_special_token_rewrite"] is True


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


def test_build_hub_masked_lm_pins_resolved_commit_and_preserves_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub model and tokenizer loading must share one immutable source commit."""
    module = _load_pseudo_perplexity_module()
    embeddings = object()
    commit = "a" * 40
    model = SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=512,
            _commit_hash=commit,
        ),
        roberta=SimpleNamespace(embeddings=embeddings),
    )
    tokenizer = SimpleNamespace(model_max_length=None)
    model_calls: list[tuple[str, bool, str | None]] = []
    tokenizer_calls: list[tuple[str, bool, str | None]] = []

    def _load_model(
        model_name: str,
        *,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ):
        model_calls.append((model_name, trust_remote_code, revision))
        return model

    def _load_tokenizer(
        model_name: str,
        *,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ):
        tokenizer_calls.append((model_name, trust_remote_code, revision))
        return tokenizer

    monkeypatch.setattr(module.AutoModelForMaskedLM, "from_pretrained", _load_model)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", _load_tokenizer)

    source = module._build_hub_masked_lm(
        "roberta-base",
        max_length=512,
        revision="main",
    )

    assert source.model is model
    assert source.tokenizer is tokenizer
    assert source.checkpoint_label == commit
    assert source.provenance == {
        "kind": "hub",
        "model": "roberta-base",
        "requested_revision": "main",
        "commit": commit,
    }
    assert model_calls == [("roberta-base", True, "main")]
    assert tokenizer_calls == [("roberta-base", True, commit)]
    assert tokenizer.model_max_length == 512
    assert model.roberta.embeddings is embeddings
    with pytest.raises(ValueError, match="exceeds.*position limit"):
        module._build_hub_masked_lm("roberta-base", max_length=1024)


def test_hub_model_without_resolved_commit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutable Hub provenance must never enter a resumable run manifest."""
    module = _load_pseudo_perplexity_module()
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    monkeypatch.setattr(
        module.AutoModelForMaskedLM,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    with pytest.raises(RuntimeError, match="immutable Hub commit"):
        module._build_hub_masked_lm("roberta-base", max_length=512)


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
    original_fingerprint = module._dataset_fingerprint(loaded)

    replacement = Dataset.from_dict({"text": ["different"]})
    shutil.rmtree(data_path)
    replacement.save_to_disk(data_path)
    replaced, _ = module._load_evaluation_dataset(
        data_path=data_path,
        dataset_name="ignored",
        dataset_config=None,
        split="train",
    )

    assert module._dataset_fingerprint(replaced) != original_fingerprint


def test_model_source_cli_is_exclusive_and_local_requires_checkpoint() -> None:
    """A run must select exactly one model source with complete local arguments."""
    module = _load_pseudo_perplexity_module()
    parser = module._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--hub_model", "bert-base-uncased", "--checkpoint_path", "x"]
        )
    assert parser.parse_args(["--hub_model", "bert-base-uncased"]).bf16 is True
    assert (
        parser.parse_args(["--hub_model", "bert-base-uncased", "--no-bf16"]).bf16
        is False
    )
    with pytest.raises(SystemExit):
        module.main(["--checkpoint", "10"])


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
        hidden_act="swiglu",
        dropout_prob=0.1,
        norm_eps=1e-5,
        embedding_init_range=0.01,
        decoder_init_range=0.02,
        classifier_init_range=0.03,
        kernel_backend="torch",
        max_position_embeddings=128,
        pad_token_id=7,
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

    class _Tokenizer:
        pad_token_id = 7

        def __len__(self):
            return 128

    tokenizer = _Tokenizer()
    captured = SimpleNamespace(config=None, loaded_state_dict=None)
    captured.load_state_dict = lambda state_dict: setattr(
        captured, "loaded_state_dict", state_dict
    )
    expected_state_dict = {"weight": torch.ones(2, 2)}
    checkpoint_calls: list[tuple[Path, str, str]] = []

    def _load_state_dict(path: Path, checkpoint: str, *, map_location: str = "cpu"):
        checkpoint_calls.append((Path(path), checkpoint, map_location))
        return expected_state_dict

    checkpoint_dir = tmp_path / "checkpoints" / "10"
    (checkpoint_dir / "tokenizer").mkdir(parents=True)
    (checkpoint_dir / "config.yaml").touch()
    resolved_model_config = NeoBERTConfig.from_model_config(
        model_cfg,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        attn_backend="sdpa",
    )
    monkeypatch.setattr(
        module,
        "resolve_checkpoint_model_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            checkpoint_root=tmp_path / "checkpoints",
            checkpoint_dir=checkpoint_dir,
            checkpoint_tag="10",
            config_path=checkpoint_dir / "config.yaml",
            tokenizer_path=checkpoint_dir / "tokenizer",
            training_config=cfg,
            tokenizer=tokenizer,
            model_config=resolved_model_config,
        ),
    )
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

    source = module._build_neobert_masked_lm(
        checkpoint_path=tmp_path / "checkpoints",
        checkpoint="10",
        max_length=256,
    )

    assert source.model is captured
    assert source.tokenizer is tokenizer
    assert source.model_label == "tiny"
    assert source.checkpoint_label == "10"
    assert captured.config.max_length == 256
    assert captured.config.dropout == 0.1
    assert captured.config.pad_token_id == 7
    assert captured.config.attn_backend == "sdpa"
    assert captured.config.kernel_backend == "torch"
    assert captured.loaded_state_dict == expected_state_dict
    assert checkpoint_calls == [(tmp_path / "checkpoints", "10", "cpu")]


def test_non_rope_local_model_keeps_trained_position_table(tmp_path: Path) -> None:
    """Shorter evaluation does not rebuild learned position embeddings."""
    module = _load_pseudo_perplexity_module()
    checkpoint_dir = tmp_path / "checkpoints" / "10"
    tokenizer = build_wordlevel_tokenizer()
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    cfg = Config()
    cfg.model.hidden_size = 16
    cfg.model.num_hidden_layers = 1
    cfg.model.num_attention_heads = 2
    cfg.model.intermediate_size = 32
    cfg.model.vocab_size = len(tokenizer)
    cfg.model.rope = False
    cfg.model.max_position_embeddings = 8
    cfg.model.pad_token_id = tokenizer.pad_token_id
    cfg.model.attn_backend = "sdpa"
    cfg.model.hidden_act = "gelu"
    cfg.dataset.max_seq_length = 8
    cfg.dataset.min_length = 1
    cfg.tokenizer.max_length = 8
    ConfigLoader.save(cfg, checkpoint_dir / "config.yaml")
    model_config = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=8,
        pad_token_id=tokenizer.pad_token_id,
        attn_backend="sdpa",
    )
    trained_model = NeoBERTLMHead(model_config)
    save_state_dict_safetensors(trained_model.state_dict(), checkpoint_dir)

    source = module._build_neobert_masked_lm(
        checkpoint_path=tmp_path,
        checkpoint="latest",
        max_length=4,
    )

    assert source.checkpoint_label == "10"
    assert source.model.config.max_length == 8
    with pytest.raises(ValueError, match="learned position limit"):
        module._build_neobert_masked_lm(
            checkpoint_path=tmp_path,
            checkpoint="latest",
            max_length=16,
        )


def test_existing_results_require_matching_run_manifest(tmp_path: Path) -> None:
    """CSV resume cannot mix checkpoints or scoring contracts."""
    module = _load_pseudo_perplexity_module()
    manifest_path = tmp_path / "scores.manifest.json"
    expected = {"schema_version": 1, "model": {"checkpoint_tag": "10"}}

    module._ensure_run_manifest(manifest_path, expected, results_exist=False)
    module._ensure_run_manifest(manifest_path, expected, results_exist=True)

    with pytest.raises(RuntimeError, match="different run contract"):
        module._ensure_run_manifest(
            manifest_path,
            {"schema_version": 1, "model": {"checkpoint_tag": "20"}},
            results_exist=True,
        )


def test_completed_ids_repair_incomplete_csv_tail(tmp_path: Path) -> None:
    """A truncated append must be discarded so its sample is recomputed."""
    module = _load_pseudo_perplexity_module()
    output_file = tmp_path / "scores.csv"
    with output_file.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(module.RESULT_FIELDS)
    module._write_score(output_file, "0:complete", [0.25, 0.75])
    with output_file.open("a", encoding="utf-8") as file:
        file.write('1:interrupted,2.0,0.5,"[0.25')

    with pytest.warns(RuntimeWarning, match="Discarding incomplete"):
        completed = module._read_completed_ids(output_file)

    assert completed == {"0:complete"}
    with output_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["sample_id"] for row in rows] == ["0:complete"]

    module._write_score(output_file, "1:interrupted", [0.5])
    assert module._read_completed_ids(output_file) == {
        "0:complete",
        "1:interrupted",
    }


@pytest.mark.parametrize(
    "row",
    [
        ["", "1.0", "0.0", "[0.0]"],
        ["0:sample"],
        ["0:sample", "invalid", "0.0", "[0.0]"],
        ["0:sample", "1.0", "invalid", "[0.0]"],
        ["0:sample", "1.0", "0.0", "not-json"],
        ["0:sample", "1.0", "0.5", "[0.0]"],
        ["0:sample", "1.0", "0.0", json.dumps([float("nan")])],
    ],
)
def test_completed_ids_reject_malformed_score_fields(
    tmp_path: Path,
    row: list[str],
) -> None:
    """Every persisted field must be valid before a sample counts as complete."""
    module = _load_pseudo_perplexity_module()
    output_file = tmp_path / "scores.csv"
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(module.RESULT_FIELDS)
        writer.writerow(row)

    with pytest.warns(RuntimeWarning, match="Discarding incomplete"):
        assert module._read_completed_ids(output_file) == set()

    assert output_file.read_text(encoding="utf-8").splitlines() == [
        ",".join(module.RESULT_FIELDS)
    ]
