"""Compute pseudo-perplexity scores for masked language models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import stat
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from neobert.checkpointing import load_step_checkpoint_state_dict
from neobert.evaluation_utils import resolve_checkpoint_model_source
from neobert.model import NeoBERTLMHead

RESULT_FIELDS = (
    "sample_id",
    "pseudo_perplexity",
    "mean_cross_entropy",
    "token_losses",
)


def _build_hub_masked_lm(
    model_name: str,
    *,
    max_length: int,
    revision: str | None = None,
) -> _ResolvedModelSource:
    """Build a Hub masked LM and tokenizer from one immutable commit.

    :param str model_name: Hub model identifier/path.
    :param int max_length: Requested evaluation context length.
    :param str | None revision: Optional Hub revision.
    :raises RuntimeError: If the loaded source has no immutable Hub commit.
    :raises ValueError: If the requested length exceeds the model's position limit.
    :return _ResolvedModelSource: Resolved Hub model source.
    """
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=revision,
    )
    position_limit = getattr(model.config, "max_position_embeddings", None)
    if position_limit is not None and max_length > int(position_limit):
        raise ValueError(
            f"Requested max_length={max_length} exceeds {model_name}'s learned "
            f"position limit ({position_limit})."
        )
    commit = getattr(model.config, "_commit_hash", None)
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
    ):
        raise RuntimeError(
            f"Could not resolve {model_name!r} to an immutable Hub commit."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=commit,
    )
    tokenizer.model_max_length = max_length
    return _ResolvedModelSource(
        model=model,
        tokenizer=tokenizer,
        model_label=model_name.replace("/", "_"),
        checkpoint_label=commit,
        provenance={
            "kind": "hub",
            "model": model_name,
            "requested_revision": revision,
            "commit": commit,
        },
    )


class _ResolvedModelSource:
    """Concrete model, tokenizer, output identity, and provenance."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_label: str,
        checkpoint_label: str,
        provenance: dict[str, Any],
    ) -> None:
        """Store a fully resolved evaluation source.

        :param Any model: Loaded masked-language model.
        :param Any tokenizer: Checkpoint-matched tokenizer.
        :param str model_label: Filesystem-safe model label.
        :param str checkpoint_label: Concrete checkpoint tag.
        :param dict[str, Any] provenance: Serializable source identity.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_label = model_label
        self.checkpoint_label = checkpoint_label
        self.provenance = provenance


def _dataset_fingerprint(dataset: Any) -> str:
    """Return the concrete Hugging Face dataset fingerprint.

    :param Any dataset: Loaded or prepared Hugging Face dataset.
    :raises RuntimeError: If the dataset exposes no usable fingerprint.
    :return str: Dataset content/transformation fingerprint.
    """
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError(
            "Loaded evaluation dataset has no fingerprint; resumable scoring "
            "cannot safely identify its contents."
        )
    return fingerprint


def _build_neobert_masked_lm(
    *,
    checkpoint_path: str | Path,
    checkpoint: str,
    max_length: int,
) -> _ResolvedModelSource:
    """Build a NeoBERT MLM from checkpoint-local config and tokenizer artifacts.

    :param str | Path checkpoint_path: Checkpoint root or step directory.
    :param str checkpoint: Checkpoint selector.
    :param int max_length: Evaluation context length.
    :return _ResolvedModelSource: Resolved local model source.
    :raises FileNotFoundError: If checkpoint config or tokenizer artifacts are missing.
    :raises ValueError: If context length or tokenizer/model identity is incompatible.
    """
    resolved = resolve_checkpoint_model_source(
        checkpoint_path,
        checkpoint,
        max_length=max_length,
    )
    model = NeoBERTLMHead(resolved.model_config)
    state_dict = load_step_checkpoint_state_dict(
        resolved.checkpoint_root,
        resolved.checkpoint_tag,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    run_name = (
        resolved.checkpoint_dir.parent.parent.name
        if resolved.checkpoint_dir.parent.name == "checkpoints"
        else resolved.checkpoint_dir.parent.name
    )
    model_label = str(resolved.training_config.model.name or run_name).replace("/", "_")
    return _ResolvedModelSource(
        model=model,
        tokenizer=resolved.tokenizer,
        model_label=model_label,
        checkpoint_label=resolved.checkpoint_tag,
        provenance={
            "kind": "checkpoint",
            "checkpoint": str(resolved.checkpoint_dir.resolve()),
            "checkpoint_tag": resolved.checkpoint_tag,
            "config": str(resolved.config_path.resolve()),
            "tokenizer": str(resolved.tokenizer_path.resolve()),
        },
    )


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
        if positions.numel() == 0:
            warnings.warn(
                f"Skipping sample {sample_id!r}: tokenization produced no "
                "non-special tokens to score.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for position_batch in positions.split(batch_size):
            masked_inputs = input_ids.repeat(len(position_batch), 1)
            labels = torch.full_like(masked_inputs, -100)
            rows = torch.arange(len(position_batch))
            labels[rows, position_batch] = masked_inputs[rows, position_batch]
            masked_inputs[rows, position_batch] = tokenizer.mask_token_id
            yield sample_id, masked_inputs, labels


def _read_completed_ids(output_file: Path) -> set[str]:
    """Read valid completed IDs and atomically discard an incomplete tail.

    :param Path output_file: CSV result path.
    :raises RuntimeError: If an existing result file has an unexpected header.
    :return set[str]: Completed sample IDs.
    """
    if not output_file.exists():
        return set()

    completed: set[str] = set()
    valid_rows: list[dict[str, str]] = []
    invalid_record: int | None = None
    with output_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, strict=True)
        if reader.fieldnames is None:
            invalid_record = 1
        elif tuple(reader.fieldnames) != RESULT_FIELDS:
            raise RuntimeError(
                f"Existing pseudo-perplexity results have an invalid CSV header: "
                f"{output_file}"
            )
        else:
            try:
                for record_number, row in enumerate(reader, start=2):
                    try:
                        sample_id = _validate_result_row(row)
                    except (TypeError, ValueError):
                        invalid_record = record_number
                        break
                    if sample_id in completed:
                        invalid_record = record_number
                        break
                    completed.add(sample_id)
                    valid_rows.append({field: row[field] for field in RESULT_FIELDS})
            except csv.Error:
                invalid_record = max(2, reader.line_num)

    if invalid_record is not None:
        warnings.warn(
            f"Discarding incomplete pseudo-perplexity CSV records from record "
            f"{invalid_record} onward: {output_file}",
            RuntimeWarning,
            stacklevel=2,
        )
        _replace_result_rows(output_file, valid_rows)
    return completed


def _validate_result_row(row: dict[str | None, str | list[str] | None]) -> str:
    """Validate one persisted pseudo-perplexity result row.

    :param dict row: CSV row keyed by result field.
    :raises ValueError: If any field is missing, malformed, or inconsistent.
    :return str: Validated sample identifier.
    """
    if set(row) != set(RESULT_FIELDS) or any(
        not isinstance(row[field], str) for field in RESULT_FIELDS
    ):
        raise ValueError("Result row does not contain exactly the required fields.")

    sample_id = row["sample_id"]
    if not sample_id:
        raise ValueError("Result row has no sample ID.")

    pseudo_perplexity = float(row["pseudo_perplexity"])
    mean_cross_entropy = float(row["mean_cross_entropy"])
    token_losses = json.loads(row["token_losses"])
    if (
        not math.isfinite(pseudo_perplexity)
        or pseudo_perplexity <= 0
        or not math.isfinite(mean_cross_entropy)
        or mean_cross_entropy < 0
        or not isinstance(token_losses, list)
        or not token_losses
        or any(
            isinstance(loss, bool) or not isinstance(loss, int | float)
            for loss in token_losses
        )
    ):
        raise ValueError("Result row contains invalid score values.")

    losses = np.asarray(token_losses, dtype=np.float64)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("Result row contains non-finite token losses.")
    expected_mean = float(np.mean(losses))
    try:
        expected_perplexity = math.exp(expected_mean)
    except OverflowError:
        expected_perplexity = math.inf
    if not math.isfinite(expected_perplexity) or not math.isclose(
        mean_cross_entropy,
        expected_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Result row mean cross-entropy is inconsistent.")
    if not math.isclose(
        pseudo_perplexity,
        expected_perplexity,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Result row pseudo-perplexity is inconsistent.")
    return sample_id


def _replace_result_rows(
    output_file: Path,
    rows: Sequence[dict[str, str]],
) -> None:
    """Atomically replace a result CSV with a validated row prefix.

    :param Path output_file: CSV result path.
    :param Sequence[dict[str, str]] rows: Valid result rows to preserve.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        if output_file.exists():
            temporary_path.chmod(stat.S_IMODE(output_file.stat().st_mode))
        temporary_path.replace(output_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_score(output_file: Path, sample_id: str, losses: Sequence[float]) -> None:
    """Append one pseudo-perplexity result row.

    :param Path output_file: CSV result path.
    :param str sample_id: Dataset sample identifier.
    :param Sequence[float] losses: Per-token cross-entropy values.
    """
    numeric_losses = [float(loss) for loss in losses]
    if (
        not sample_id
        or not numeric_losses
        or any(not math.isfinite(loss) or loss < 0 for loss in numeric_losses)
    ):
        raise ValueError("Cannot persist an incomplete or non-finite score row.")
    mean_loss = float(np.mean(numeric_losses))
    try:
        pseudo_perplexity = math.exp(mean_loss)
    except OverflowError:
        pseudo_perplexity = math.inf
    if not math.isfinite(mean_loss) or not math.isfinite(pseudo_perplexity):
        raise ValueError("Cannot persist an incomplete or non-finite score row.")
    with output_file.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [sample_id, pseudo_perplexity, mean_loss, json.dumps(numeric_losses)]
        )


def _ensure_run_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    results_exist: bool,
) -> None:
    """Write or validate the identity of a resumable pseudo-perplexity run.

    :param Path manifest_path: Sidecar manifest path.
    :param dict[str, Any] manifest: Expected run identity and scoring contract.
    :param bool results_exist: Whether a result CSV already exists.
    :raises RuntimeError: If existing results have missing or different provenance.
    """
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise RuntimeError(
                f"Existing pseudo-perplexity results use a different run contract: "
                f"{manifest_path}"
            )
        return
    if results_exist:
        raise RuntimeError(
            f"Existing pseudo-perplexity results have no provenance manifest: "
            f"{manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the pseudo-perplexity command-line parser.

    :return argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--hub_model", help="Hub masked-LM identifier")
    model_source.add_argument(
        "--checkpoint_path",
        type=Path,
        help="NeoBERT run root, checkpoints root, or concrete step directory",
    )
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--revision", help="Optional Hub model/tokenizer revision")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bf16 autocast; use --no-bf16 for fp32 evaluation",
    )
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
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("--batch_size and --max_length must be positive")

    if args.hub_model is not None:
        source = _build_hub_masked_lm(
            args.hub_model,
            max_length=args.max_length,
            revision=args.revision,
        )
    else:
        source = _build_neobert_masked_lm(
            checkpoint_path=args.checkpoint_path,
            checkpoint=args.checkpoint,
            max_length=args.max_length,
        )
    model = source.model
    tokenizer = source.tokenizer
    model_label = source.model_label
    checkpoint_label = source.checkpoint_label
    model_provenance = source.provenance

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
    dataset_fingerprint = _dataset_fingerprint(dataset)

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
    manifest = {
        "schema_version": 3,
        "model": model_provenance,
        "dataset": {
            "data_path": str(args.data_path.resolve()) if args.data_path else None,
            "dataset_name": args.dataset_name if args.data_path is None else None,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "dataset_label": dataset_label,
            "fingerprint": dataset_fingerprint,
            "text_column": args.text_column,
            "id_column": args.id_column,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "n_sentences": args.n_sentences,
            "num_shards": args.num_dataset_shards,
            "shard_index": args.dataset_shard_index,
            "seed": args.seed,
        },
        "scoring": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "bf16": args.bf16,
        },
    }
    _ensure_run_manifest(
        output_file.with_suffix(".manifest.json"),
        manifest,
        results_exist=output_file.exists(),
    )
    completed_ids = _read_completed_ids(output_file)
    if not output_file.exists():
        _replace_result_rows(output_file, [])

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
