"""Preprocess and tokenize contrastive datasets."""

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from neobert.config import load_config_from_args
from neobert.contrastive.datasets import (
    CONTRASTIVE_DATASETS,
    discover_cached_contrastive_dataset_names,
    load_cached_contrastive_datasets,
    normalize_contrastive_dataset_name_token,
    resolve_contrastive_dataset_name,
    resolve_contrastive_dataset_names,
)
from neobert.tokenization_cache import (
    build_tokenization_manifest,
    validate_tokenized_cache_manifest,
    write_tokenized_cache_manifest,
)
from neobert.tokenizer import get_tokenizer, tokenize


def _normalize_dataset_name_token(value: Any) -> str:
    """Normalize a dataset selector token for registry matching.

    :param Any value: Raw dataset selector.
    :return str: Uppercase alphanumeric token.
    """
    return normalize_contrastive_dataset_name_token(value)


def _resolve_single_dataset_name(requested: Any) -> str:
    """Resolve one dataset selector to a canonical contrastive registry key.

    Accepts canonical keys (for example ``ALLNLI``), class names, and common
    built-in Hugging Face dataset IDs such as ``sentence-transformers/all-nli``.

    :param Any requested: Raw selector value.
    :return str: Canonical registry key.
    :raises ValueError: If the selector cannot be resolved.
    """
    return resolve_contrastive_dataset_name(requested)


def _resolve_dataset_names(cfg: Any) -> list[str]:
    """Resolve which contrastive datasets should be prepared.

    :param Any cfg: Runtime config.
    :return list[str]: Dataset registry keys to prepare.
    """
    dataset_cfg = getattr(cfg, "dataset", None)
    requested = getattr(dataset_cfg, "name", None)
    return resolve_contrastive_dataset_names(requested)


def _discover_cached_dataset_names(all_dir: Path) -> list[str]:
    """Return cached contrastive split names with complete on-disk payloads.

    :param Path all_dir: Root ``all/`` directory containing cached split folders.
    :return list[str]: Cached split names in registry order.
    """
    return discover_cached_contrastive_dataset_names(all_dir)


def _write_cached_dataset_manifest(all_dir: Path) -> list[str]:
    """Write the cached contrastive split manifest from directories on disk.

    :param Path all_dir: Root ``all/`` directory containing cached split folders.
    :return list[str]: Manifest split names that were written.
    """
    cached_names = _discover_cached_dataset_names(all_dir)
    with (all_dir / "dataset_dict.json").open("w", encoding="utf-8") as handle:
        json.dump({"splits": cached_names}, handle, indent=2)
    return cached_names


def pipeline(cfg: Any) -> DatasetDict:
    """Run dataset preprocessing and tokenization.

    :param Any cfg: Configuration object with dataset/tokenizer settings.
    :return DatasetDict: Prepared dataset dictionary.
    """
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        raise ValueError("Config must define a dataset section.")

    dataset_path = getattr(dataset_cfg, "path", None)
    if not dataset_path:
        raise ValueError(
            "Contrastive preprocessing requires dataset.path pointing to a dataset root."
        )

    dataset_root = Path(dataset_path)
    all_dir = dataset_root / "all"
    load_all_from_disk = bool(getattr(dataset_cfg, "load_all_from_disk", False))
    force_redownload = bool(getattr(dataset_cfg, "force_redownload", False))
    selected_names = _resolve_dataset_names(cfg)

    if load_all_from_disk:
        dataset = load_cached_contrastive_datasets(
            all_dir,
            selected_names=selected_names,
        )

    else:
        all_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = get_tokenizer(
            pretrained_model_name_or_path=getattr(cfg.tokenizer, "path", None)
            or cfg.tokenizer.name,
            max_length=getattr(cfg.tokenizer, "max_length", 512),
            trust_remote_code=getattr(cfg.tokenizer, "trust_remote_code", False),
            revision=getattr(cfg.tokenizer, "revision", None),
            allow_special_token_rewrite=getattr(
                cfg.tokenizer, "allow_special_token_rewrite", False
            ),
        )

        # Load and tokenize subdatasets if necessary
        dataset_dict = {}
        for name in selected_names:
            dataset_cls = CONTRASTIVE_DATASETS[name]
            dataset_dir = all_dir / name
            if dataset_dir.is_dir() and not force_redownload:
                print(f"Loading tokenized {name} from disk...")
                subdataset = dataset_cls.from_disk(dataset_dir).dataset
                token_columns = tuple(
                    col.removeprefix("input_ids_")
                    for col in subdataset.column_names
                    if col.startswith("input_ids_")
                )
                manifest = build_tokenization_manifest(
                    tokenizer,
                    dataset_name=name,
                    dataset_config=None,
                    dataset_path=dataset_path,
                    column_name=token_columns,
                    max_length=getattr(cfg.tokenizer, "max_length", 512),
                    truncation=getattr(cfg.tokenizer, "truncation", True),
                    add_special_tokens=True,
                    return_special_tokens_mask=False,
                )
                validate_tokenized_cache_manifest(dataset_dir, manifest)
            else:
                if dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir)
                print(f"Loading {name} from huggingface and preprocessing...")
                subdataset = dataset_cls().dataset
                print(f"Tokenizing {name}...")
                token_columns = tuple(
                    col
                    for col in subdataset.column_names
                    if col in {"query", "corpus", "negative"}
                )
                if not token_columns:
                    raise ValueError(
                        f"{name} has no tokenizable columns in "
                        f"{subdataset.column_names!r}."
                    )
                manifest = build_tokenization_manifest(
                    tokenizer,
                    dataset_name=name,
                    dataset_config=None,
                    dataset_path=dataset_path,
                    column_name=token_columns,
                    max_length=getattr(cfg.tokenizer, "max_length", 512),
                    truncation=getattr(cfg.tokenizer, "truncation", True),
                    add_special_tokens=True,
                    return_special_tokens_mask=False,
                )
                subdataset = tokenize(
                    subdataset,
                    tokenizer,
                    column_name=token_columns[0]
                    if len(token_columns) == 1
                    else token_columns,
                    max_length=getattr(cfg.tokenizer, "max_length", 512),
                    truncation=getattr(cfg.tokenizer, "truncation", True),
                    remove_columns=True,
                )
                subdataset.save_to_disk(dataset_dir)
                write_tokenized_cache_manifest(dataset_dir, manifest)

            dataset_dict[name] = subdataset

        # Aggregate datasets
        dataset = DatasetDict(dataset_dict)
        cached_names = _write_cached_dataset_manifest(all_dir)
        print(
            "Prepared contrastive dataset selection "
            f"{list(dataset.keys())}. Cached manifest now lists {cached_names}."
        )

    return dataset


def main() -> None:
    """Run preprocessing from a CLI config."""
    # Load configuration from command line arguments
    config = load_config_from_args(require_config=True)

    # Run contrastive preprocessing
    pipeline(config)


if __name__ == "__main__":
    main()
