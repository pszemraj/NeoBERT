"""Preprocess and tokenize contrastive datasets."""

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from neobert.config import load_config_from_args
from neobert.contrastive.datasets import CONTRASTIVE_DATASETS
from neobert.tokenizer import get_tokenizer, tokenize


def _resolve_dataset_names(cfg: Any) -> list[str]:
    """Resolve which contrastive datasets should be prepared.

    :param Any cfg: Runtime config.
    :return list[str]: Dataset registry keys to prepare.
    """
    dataset_cfg = getattr(cfg, "dataset", None)
    requested = getattr(dataset_cfg, "name", None)
    if requested is None:
        return list(CONTRASTIVE_DATASETS.keys())
    if isinstance(requested, str):
        normalized = requested.strip().upper()
        if normalized in {"", "ALL"}:
            return list(CONTRASTIVE_DATASETS.keys())
        if normalized in CONTRASTIVE_DATASETS:
            return [normalized]
        print(
            "dataset.name is not a known contrastive key "
            f"('{requested}'); preparing all contrastive datasets."
        )
        return list(CONTRASTIVE_DATASETS.keys())
    if isinstance(requested, (list, tuple)):
        names = [str(name).strip().upper() for name in requested]
        invalid = [name for name in names if name not in CONTRASTIVE_DATASETS]
        if invalid:
            raise ValueError(
                "Unknown contrastive dataset names: "
                f"{invalid}. Available: {sorted(CONTRASTIVE_DATASETS.keys())}."
            )
        return names
    raise TypeError(
        "dataset.name must be a string or list of strings for contrastive preprocess."
    )


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
        dataset = load_from_disk(all_dir)

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

            dataset_dict[name] = subdataset

        # Aggregate datasets
        dataset = DatasetDict(dataset_dict)
        with (all_dir / "dataset_dict.json").open("w", encoding="utf-8") as handle:
            json.dump({"splits": list(dataset.keys())}, handle, indent=2)
        print(
            "Global contrastive dataset is ready. It contains subdatasets "
            f"{list(dataset.keys())}."
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
