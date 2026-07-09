"""Compute pseudo-perplexity scores for masked language models."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from neobert.checkpointing import load_step_checkpoint_state_dict
from neobert.config import ConfigLoader
from neobert.model import NeoBERTConfig, NeoBERTLMHead
from neobert.tokenizer import get_tokenizer


def _load_hub_masked_lm(model_name: str, *, max_length: int) -> Any:
    """Load a Hub masked-language model without modifying its learned embeddings.

    :param str model_name: Hub model identifier/path.
    :param int max_length: Requested evaluation context length.
    :raises ValueError: If the requested length exceeds the model's position limit.
    :return Any: Loaded masked-language-model instance.
    """
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    position_limit = getattr(model.config, "max_position_embeddings", None)
    if position_limit is not None and max_length > int(position_limit):
        raise ValueError(
            f"Requested max_length={max_length} exceeds {model_name}'s learned "
            f"position limit ({position_limit})."
        )
    return model


def _build_neobert_masked_lm(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    checkpoint: str,
    max_length: int,
) -> tuple[NeoBERTLMHead, Any, str]:
    """Build a NeoBERT MLM from a training config and checkpoint.

    :param str | Path config_path: Training configuration path.
    :param str | Path checkpoint_path: Checkpoint root or step directory.
    :param str checkpoint: Checkpoint selector.
    :param int max_length: Evaluation context length.
    :return tuple[NeoBERTLMHead, Any, str]: Model, tokenizer, and output label.
    """
    cfg = ConfigLoader.load(config_path)
    tokenizer = get_tokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.name,
        max_length=max_length,
        trust_remote_code=cfg.tokenizer.trust_remote_code,
        revision=cfg.tokenizer.revision,
        allow_special_token_rewrite=cfg.tokenizer.allow_special_token_rewrite,
    )
    model_config = NeoBERTConfig.from_model_config(
        cfg.model,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        attn_backend="sdpa",
    )
    model = NeoBERTLMHead(model_config)
    state_dict = load_step_checkpoint_state_dict(
        checkpoint_path,
        checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model_label = str(cfg.model.name or Path(checkpoint_path).resolve().name).replace(
        "/", "_"
    )
    return model, tokenizer, model_label


def _load_evaluation_dataset(
    *,
    data_path: Path | None,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
) -> tuple[Any, str]:
    """Load a local or Hub evaluation dataset.

    :param Path | None data_path: Optional path saved with ``save_to_disk``.
    :param str dataset_name: Hub dataset name.
    :param str | None dataset_config: Optional Hub dataset subset/config.
    :param str split: Dataset split.
    :raises ValueError: If a local dataset dictionary lacks the requested split.
    :return tuple[Any, str]: Dataset split and filesystem-safe source label.
    """
    if data_path is None:
        if dataset_name == "wikipedia" and dataset_config is None:
            dataset_config = "20220301.en"
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        label_parts = [dataset_name.replace("/", "_")]
        if dataset_config is not None:
            label_parts.append(dataset_config.replace("/", "_"))
        label_parts.append(split)
        return dataset, "_".join(label_parts)

    loaded = load_from_disk(str(data_path))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise ValueError(
                f"Local dataset {data_path} has no {split!r} split; "
                f"available splits: {sorted(loaded)}."
            )
        loaded = loaded[split]
    return loaded, f"{data_path.resolve().name}_{split}"


def _prepare_evaluation_dataset(
    dataset: Any,
    *,
    text_column: str,
    min_chars: int,
    max_chars: int,
    n_sentences: int,
    num_shards: int,
    shard_index: int,
    seed: int,
) -> Any:
    """Filter, shuffle, shard, and bound an evaluation dataset.

    :param Any dataset: Input dataset.
    :param str text_column: Text column name.
    :param int min_chars: Inclusive minimum text length.
    :param int max_chars: Inclusive maximum text length.
    :param int n_sentences: Maximum examples to retain per shard.
    :param int num_shards: Number of deterministic dataset shards.
    :param int shard_index: Zero-based shard index.
    :param int seed: Shuffle seed.
    :raises ValueError: If the selection parameters or result are invalid.
    :return Any: Prepared dataset.
    """
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Dataset has no {text_column!r} column; available columns: "
            f"{dataset.column_names}."
        )
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("Expected 0 <= min_chars <= max_chars.")
    if n_sentences <= 0:
        raise ValueError("n_sentences must be positive.")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Expected num_shards > 0 and 0 <= shard_index < num_shards.")

    selected = dataset.filter(
        lambda example: min_chars <= len(example[text_column]) <= max_chars
    ).shuffle(seed=seed)
    if num_shards > 1:
        selected = selected.shard(num_shards=num_shards, index=shard_index)
    selected = selected.select(range(min(n_sentences, len(selected))))
    if not selected:
        raise ValueError(
            f"No examples remain after filtering {text_column!r} to "
            f"{min_chars}..{max_chars} characters."
        )
    return selected


def _iter_masked_batches(
    dataset: Any,
    tokenizer: Any,
    *,
    text_column: str,
    id_column: str,
    batch_size: int,
    max_length: int,
    skip_ids: set[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield batches with exactly one non-special token masked per row.

    :param Any dataset: Prepared text dataset.
    :param Any tokenizer: Masked-language-model tokenizer.
    :param str text_column: Text column name.
    :param str id_column: Preferred sample identifier column.
    :param int batch_size: Number of masked positions per model batch.
    :param int max_length: Tokenization limit.
    :param set[str] | None skip_ids: Completed sample identifiers to skip.
    :return Iterator[tuple[str, torch.Tensor, torch.Tensor]]: IDs, inputs, and labels.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Pseudo-perplexity requires a tokenizer mask token.")
    completed = skip_ids or set()
    for row_index, example in enumerate(dataset):
        source_id = example.get(id_column, row_index)
        sample_id = f"{row_index}:{source_id}"
        if sample_id in completed:
            continue
        tokenized = tokenizer(
            example[text_column],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        special_mask = tokenized["special_tokens_mask"].to(torch.bool)
        positions = (~special_mask[0]).nonzero(as_tuple=False).flatten()
        for position_batch in positions.split(batch_size):
            masked_inputs = input_ids.repeat(len(position_batch), 1)
            labels = torch.full_like(masked_inputs, -100)
            rows = torch.arange(len(position_batch))
            labels[rows, position_batch] = masked_inputs[rows, position_batch]
            masked_inputs[rows, position_batch] = tokenizer.mask_token_id
            yield sample_id, masked_inputs, labels


def _read_completed_ids(output_file: Path) -> set[str]:
    """Read completed sample IDs from an existing result file.

    :param Path output_file: CSV result path.
    :return set[str]: Completed sample IDs.
    """
    if not output_file.exists():
        return set()
    with output_file.open(newline="", encoding="utf-8") as file:
        return {row["sample_id"] for row in csv.DictReader(file)}


def _write_score(output_file: Path, sample_id: str, losses: Sequence[float]) -> None:
    """Append one pseudo-perplexity result row.

    :param Path output_file: CSV result path.
    :param str sample_id: Dataset sample identifier.
    :param Sequence[float] losses: Per-token cross-entropy values.
    """
    mean_loss = float(np.mean(losses))
    with output_file.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [sample_id, float(np.exp(mean_loss)), mean_loss, json.dumps(losses)]
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the pseudo-perplexity command-line parser.

    :return argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--hub_model", help="Hub masked-LM identifier")
    model_source.add_argument(
        "--config_path", type=Path, help="NeoBERT training config"
    )
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)

    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--data_path", type=Path, help="Dataset saved to disk")
    data_source.add_argument("--dataset_name", default="wikipedia")
    parser.add_argument("--dataset_config")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--min_chars", type=int, default=500)
    parser.add_argument("--max_chars", type=int, default=20000)
    parser.add_argument("--n_sentences", type=int, default=10000)
    parser.add_argument("--num_dataset_shards", type=int, default=1)
    parser.add_argument("--dataset_shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/pseudo_perplexity"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run pseudo-perplexity evaluation.

    :param Sequence[str] | None argv: Optional command-line arguments.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config_path is not None and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required with --config_path")
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("--batch_size and --max_length must be positive")

    if args.hub_model is not None:
        model = _load_hub_masked_lm(args.hub_model, max_length=args.max_length)
        tokenizer = AutoTokenizer.from_pretrained(
            args.hub_model,
            trust_remote_code=True,
        )
        tokenizer.model_max_length = args.max_length
        model_label = args.hub_model.replace("/", "_")
        checkpoint_label = "hub"
    else:
        model, tokenizer, model_label = _build_neobert_masked_lm(
            args.config_path,
            checkpoint_path=args.checkpoint_path,
            checkpoint=args.checkpoint,
            max_length=args.max_length,
        )
        checkpoint_label = args.checkpoint

    dataset, dataset_label = _load_evaluation_dataset(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
    )
    dataset = _prepare_evaluation_dataset(
        dataset,
        text_column=args.text_column,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        n_sentences=args.n_sentences,
        num_shards=args.num_dataset_shards,
        shard_index=args.dataset_shard_index,
        seed=args.seed,
    )

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    if args.compile:
        model = torch.compile(model)

    output_dir = args.output_path / model_label / str(checkpoint_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"{dataset_label}_chars-{args.min_chars}-{args.max_chars}_"
        f"tokens-{args.max_length}_seed-{args.seed}_ppl_"
        f"{args.dataset_shard_index}-of-{args.num_dataset_shards}.csv"
    )
    completed_ids = _read_completed_ids(output_file)
    if not output_file.exists():
        with output_file.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ["sample_id", "pseudo_perplexity", "mean_cross_entropy", "token_losses"]
            )

    batches = _iter_masked_batches(
        dataset,
        tokenizer,
        text_column=args.text_column,
        id_column=args.id_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        skip_ids=completed_ids,
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    current_id: str | None = None
    current_losses: list[float] = []
    progress = tqdm(total=max(0, len(dataset) - len(completed_ids)))
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.bf16,
        ),
    ):
        for sample_id, input_ids, labels in batches:
            if current_id is not None and sample_id != current_id:
                _write_score(output_file, current_id, current_losses)
                progress.update(1)
                current_losses = []
            current_id = sample_id
            output = model(input_ids.to(device))
            logits = output["logits"] if isinstance(output, dict) else output.logits
            losses = loss_fn(logits.transpose(1, 2), labels.to(device)).sum(-1)
            current_losses.extend(losses.float().cpu().tolist())

    if current_id is not None:
        _write_score(output_file, current_id, current_losses)
        progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()
