"""Metric aggregation helpers for pretraining."""

import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict

import torch
from accelerate import Accelerator


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Format metrics for logging with sensible precision.

    :param dict[str, Any] metrics: Raw metrics dictionary.
    :return dict[str, Any]: Metrics with rounded floats for logging.
    """
    formatted: Dict[str, Any] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value):
                formatted[key] = value
                continue
            if "learning_rate" in key:
                formatted[key] = float(f"{value:.6g}")
                continue
            if "tokens_per_sec" in key:
                formatted[key] = round(value, 2)
                continue
            if any(
                token in key
                for token in ("loss", "perplexity", "accuracy", "grad_norm")
            ):
                formatted[key] = round(value, 4)
                continue
            abs_val = abs(value)
            if abs_val >= 1000:
                formatted[key] = round(value, 1)
            elif abs_val >= 100:
                formatted[key] = round(value, 2)
            elif abs_val >= 10:
                formatted[key] = round(value, 3)
            else:
                formatted[key] = round(value, 4)
        else:
            formatted[key] = value
    return formatted


class Metrics(defaultdict):
    """Dictionary-like metrics container with distributed aggregation helpers."""

    # Internal counters that should never be emitted to experiment trackers.
    TRACKER_EXCLUDE_KEYS = {
        "train/steps",
        "train/compute_accuracy",
        "train/batches",
        "train/batches_in_epoch",
        "train/samples",
        "train/masked_tokens",
        "train/epochs",
    }

    LOCAL_COUNT_KEYS = (
        "train/local_samples",
        "train/local_tokens",
        "train/local_num_pred",
        "train/local_num_correct",
    )
    LOCAL_FLOAT_KEYS = ("train/local_sum_loss",)

    def __init__(self):
        """Initialize metrics storage with integer defaults."""
        super().__init__(int)
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
        self["train/compute_accuracy"] = 1
        self._last_log_time: float | None = None
        self._last_log_step: int | None = None

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable copy of the metrics.

        :return dict[str, Any]: Metrics state.
        """
        return dict(self)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load metrics from a serialized state.

        :param dict[str, Any] state_dict: Metrics state to load.
        """
        for k, v in state_dict.items():
            self[k] = v
        self._last_log_time = None
        self._last_log_step = None

    def log(
        self,
        accelerator: Accelerator,
        *,
        emit_console: bool = False,
        console_fn: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """Aggregate and log metrics across devices.

        :param Accelerator accelerator: Accelerator used for reduction/logging.
        :param bool emit_console: Whether to print formatted metrics to console.
        :param Callable[[str], None] | None console_fn: Optional console emit function.
        :return dict[str, Any]: Formatted metrics logged for this step.
        """
        # Aggregate only the local counters using a fixed key order.
        count_tensor = torch.tensor(
            [self.get(k, 0) for k in self.LOCAL_COUNT_KEYS],
            device=accelerator.device,
            dtype=torch.long,
        )
        count_tensor = accelerator.reduce(count_tensor, reduction="sum")
        float_tensor = torch.tensor(
            [self.get(k, 0.0) for k in self.LOCAL_FLOAT_KEYS],
            device=accelerator.device,
            dtype=torch.float64,
        )
        float_tensor = accelerator.reduce(float_tensor, reduction="sum")

        count_vals = count_tensor.detach().cpu().tolist()
        float_vals = float_tensor.detach().cpu().tolist()
        metrics_agg = {
            **{k: int(v) for k, v in zip(self.LOCAL_COUNT_KEYS, count_vals)},
            **{k: float(v) for k, v in zip(self.LOCAL_FLOAT_KEYS, float_vals)},
        }

        # Update global values
        self["train/samples"] = (
            self["train/samples"] + metrics_agg["train/local_samples"]
        )
        self["train/tokens"] = self["train/tokens"] + metrics_agg["train/local_tokens"]
        self["train/masked_tokens"] = (
            self["train/masked_tokens"] + metrics_agg["train/local_num_pred"]
        )

        # Build the metrics to log (use aggregated local counters).
        metrics_log = dict(self)
        for key, value in metrics_agg.items():
            metrics_log[key] = value

        compute_accuracy = bool(self.get("train/compute_accuracy", 1))
        if not compute_accuracy:
            metrics_log.pop("train/local_num_correct", None)

        if metrics_agg["train/local_num_pred"] > 0:
            metrics_log["train/loss"] = (
                metrics_agg["train/local_sum_loss"]
                / metrics_agg["train/local_num_pred"]
            )
            metrics_log["train/perplexity"] = math.exp(metrics_log["train/loss"])
            if compute_accuracy:
                metrics_log["train/accuracy"] = (
                    metrics_agg["train/local_num_correct"]
                    / metrics_agg["train/local_num_pred"]
                )

        # Extract the step value to pass separately to accelerator.log
        current_step = self.get("train/steps", 0)
        now = time.perf_counter()
        if self._last_log_time is not None:
            elapsed = now - self._last_log_time
            if elapsed > 0:
                metrics_log["train/tokens_per_sec"] = (
                    metrics_agg["train/local_tokens"] / elapsed
                )

        # Log metrics with the current step while keeping some runtime/internal
        # fields out of external trackers.
        formatted = format_metrics(metrics_log)
        tracker_payload = dict(formatted)
        for key in self.TRACKER_EXCLUDE_KEYS:
            tracker_payload.pop(key, None)
        for key in list(tracker_payload):
            if key.startswith("train/local_"):
                tracker_payload.pop(key, None)
            if key.startswith("train/loss_path_"):
                tracker_payload.pop(key, None)
        if not compute_accuracy:
            tracker_payload.pop("train/accuracy", None)
        accelerator.log(tracker_payload, step=current_step)
        if emit_console and accelerator.is_main_process:
            if console_fn is None:
                console_fn = print
            keys = (
                "train/steps",
                "train/loss",
                "train/perplexity",
                "train/accuracy",
                "train/tokens_per_sec",
                "train/learning_rate",
                "train/grad_norm",
            )
            fields = [f"{key}={formatted[key]}" for key in keys if key in formatted]
            if fields:
                console_fn(" | ".join(fields))
        self._last_log_time = now
        self._last_log_step = current_step

        # Reset the local counters
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
        return formatted
