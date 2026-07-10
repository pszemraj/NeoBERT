"""Canonical task metadata and scoring rules for GLUE fine-tuning."""

from dataclasses import dataclass
from numbers import Real
from typing import Mapping


@dataclass(frozen=True)
class GlueTaskSpec:
    """Describe one supported GLUE or GLUE-like task.

    :param str name: Canonical task name.
    :param tuple[str, str | None] sentence_keys: Dataset text column names.
    :param int num_labels: Number of classifier outputs.
    :param str checkpoint_metric: Metric used for checkpoint selection and early stopping.
    :param tuple[str, ...] score_metrics: Metrics averaged for the official GLUE score.
    :param str score_label: Human-readable official-score label.
    :param str | None transfer_from: Optional source task for classifier transfer.
    :param int config_eval_steps: Default evaluation interval for generated configs.
    :param int | None config_logging_steps: Optional logging interval override.
    """

    name: str
    sentence_keys: tuple[str, str | None]
    num_labels: int
    checkpoint_metric: str
    score_metrics: tuple[str, ...] = ()
    score_label: str = "Accuracy"
    transfer_from: str | None = None
    config_eval_steps: int = 500
    config_logging_steps: int | None = None


GLUE_TASK_SPECS: Mapping[str, GlueTaskSpec] = {
    "cola": GlueTaskSpec(
        "cola",
        ("sentence", None),
        2,
        "matthews_correlation",
        ("matthews_correlation",),
        "Matthews Corr",
        config_eval_steps=200,
        config_logging_steps=50,
    ),
    "sst2": GlueTaskSpec(
        "sst2", ("sentence", None), 2, "accuracy", ("accuracy",), config_eval_steps=500
    ),
    "mrpc": GlueTaskSpec(
        "mrpc",
        ("sentence1", "sentence2"),
        2,
        "f1",
        ("accuracy", "f1"),
        "Acc/F1 (avg)",
        transfer_from="mnli",
        config_eval_steps=100,
    ),
    "stsb": GlueTaskSpec(
        "stsb",
        ("sentence1", "sentence2"),
        1,
        "pearson",
        ("pearson", "spearmanr"),
        "Pearson/Spearman (avg)",
        transfer_from="mnli",
        config_eval_steps=150,
    ),
    "qqp": GlueTaskSpec(
        "qqp",
        ("question1", "question2"),
        2,
        "f1",
        ("accuracy", "f1"),
        "Acc/F1 (avg)",
        config_eval_steps=1000,
    ),
    "mnli": GlueTaskSpec(
        "mnli",
        ("premise", "hypothesis"),
        3,
        "accuracy",
        ("accuracy", "accuracy_mm"),
        "MNLI-m/mm (avg)",
        transfer_from="snli",
        config_eval_steps=1000,
    ),
    "qnli": GlueTaskSpec(
        "qnli",
        ("question", "sentence"),
        2,
        "accuracy",
        ("accuracy",),
        transfer_from="mnli",
        config_eval_steps=500,
    ),
    "rte": GlueTaskSpec(
        "rte",
        ("sentence1", "sentence2"),
        2,
        "accuracy",
        ("accuracy",),
        transfer_from="mnli",
        config_eval_steps=50,
    ),
    "wnli": GlueTaskSpec(
        "wnli",
        ("sentence1", "sentence2"),
        2,
        "accuracy",
        ("accuracy",),
        transfer_from="allnli",
        config_eval_steps=20,
    ),
}

GLUE_LIKE_TASK_SPECS: Mapping[str, GlueTaskSpec] = {
    "snli": GlueTaskSpec("snli", ("premise", "hypothesis"), 3, "accuracy"),
    "allnli": GlueTaskSpec("allnli", ("premise", "hypothesis"), 2, "accuracy"),
}

SUPPORTED_GLUE_TASK_SPECS: Mapping[str, GlueTaskSpec] = {
    **GLUE_TASK_SPECS,
    **GLUE_LIKE_TASK_SPECS,
}


def get_glue_task_spec(task: str) -> GlueTaskSpec:
    """Return metadata for a supported task.

    :param str task: Task name.
    :raises ValueError: If the task is unsupported.
    :return GlueTaskSpec: Canonical task metadata.
    """
    normalized = str(task).strip().lower()
    try:
        return SUPPORTED_GLUE_TASK_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_GLUE_TASK_SPECS))
        raise ValueError(
            f"Unsupported GLUE task {task!r}; choose one of: {supported}"
        ) from exc


def normalize_glue_metrics(metrics: Mapping[str, object] | None) -> dict[str, float]:
    """Normalize numeric GLUE metric keys by removing an ``eval_`` prefix.

    :param Mapping[str, object] | None metrics: Raw evaluation metrics.
    :return dict[str, float]: Normalized numeric metrics.
    """
    normalized: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            continue
        metric_key = key.removeprefix("eval_")
        normalized[metric_key] = float(value)
    return normalized


def compute_official_glue_score(
    task: str, metrics: Mapping[str, object] | None
) -> float | None:
    """Compute a task's official GLUE score when all components are present.

    :param str task: Official GLUE task name.
    :param Mapping[str, object] | None metrics: Raw evaluation metrics.
    :return float | None: Official score, or ``None`` for incomplete/non-GLUE metrics.
    """
    spec = get_glue_task_spec(task)
    if not spec.score_metrics:
        return None

    normalized = normalize_glue_metrics(metrics)
    values = [normalized.get(metric) for metric in spec.score_metrics]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def get_checkpoint_selection_score(
    task: str, metrics: Mapping[str, object] | None
) -> float | None:
    """Return the configured checkpoint-selection metric for a task.

    :param str task: Supported task name.
    :param Mapping[str, object] | None metrics: Raw evaluation metrics.
    :return float | None: Selection score when the preferred metric is present.
    """
    spec = get_glue_task_spec(task)
    return normalize_glue_metrics(metrics).get(spec.checkpoint_metric)
