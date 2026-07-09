"""Preprocessing utilities for supported GLUE and GLUE-like tasks."""

from typing import Any

from neobert.glue.tasks import get_glue_task_spec


def get_sentences(
    examples: dict[str, Any], task: str
) -> tuple[list[str], list[str] | None]:
    """Extract the sentence columns for a supported task.

    :param dict[str, Any] examples: Batch of dataset examples.
    :param str task: Supported task name.
    :return tuple[list[str], list[str] | None]: Sentence pairs or single sentences.
    """
    key1, key2 = get_glue_task_spec(task).sentence_keys
    return examples[key1], None if key2 is None else examples[key2]


def process_function(
    examples: dict[str, Any], cfg: Any, tokenizer: Any
) -> dict[str, Any]:
    """Tokenize a batch for a supported GLUE or GLUE-like task.

    :param dict[str, Any] examples: Batch of dataset examples.
    :param Any cfg: Task configuration with tokenizer settings.
    :param Any tokenizer: Tokenizer with sequence-classification support.
    :return dict[str, Any]: Tokenized batch with labels when applicable.
    """
    task = str(getattr(cfg.glue, "task_name", getattr(cfg, "task", "glue"))).strip()
    result = tokenizer(
        *get_sentences(examples, task),
        padding=False,
        max_length=int(cfg.tokenizer.max_length),
        truncation=bool(getattr(cfg.tokenizer, "truncation", True)),
    )
    if cfg.mode in {"train", "eval"}:
        result["labels"] = examples["label"]
    return result
