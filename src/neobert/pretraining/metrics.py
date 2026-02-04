"""Metric aggregation helpers for pretraining."""

import math
import time
from collections import defaultdict
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch import Tensor


class Metrics(defaultdict):
    """Dictionary-like metrics container with distributed aggregation helpers."""

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

    def log(self, accelerator: Accelerator) -> None:
        """Aggregate and log metrics across devices.

        :param Accelerator accelerator: Accelerator used for reduction/logging.
        """
        # Aggregate only the local counters using a fixed key order.
        count_tensor = Tensor([self.get(k, 0) for k in self.LOCAL_COUNT_KEYS]).to(
            accelerator.device, dtype=torch.long, non_blocking=True
        )
        count_tensor = accelerator.reduce(count_tensor, reduction="sum")
        float_tensor = Tensor([self.get(k, 0.0) for k in self.LOCAL_FLOAT_KEYS]).to(
            accelerator.device, dtype=torch.float64, non_blocking=True
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

        if metrics_agg["train/local_num_pred"] > 0:
            metrics_log["train/loss"] = (
                metrics_agg["train/local_sum_loss"]
                / metrics_agg["train/local_num_pred"]
            )
            metrics_log["train/perplexity"] = math.exp(metrics_log["train/loss"])
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
                steps_in_window = (
                    current_step - self._last_log_step
                    if self._last_log_step is not None
                    else 0
                )
                if steps_in_window > 0:
                    metrics_log["train/steps_per_sec"] = steps_in_window / elapsed
                    metrics_log["train/samples_per_step"] = (
                        metrics_agg["train/local_samples"] / steps_in_window
                    )
                    metrics_log["train/tokens_per_step"] = (
                        metrics_agg["train/local_tokens"] / steps_in_window
                    )
                    metrics_log["train/masked_tokens_per_step"] = (
                        metrics_agg["train/local_num_pred"] / steps_in_window
                    )
                metrics_log["train/samples_per_sec"] = (
                    metrics_agg["train/local_samples"] / elapsed
                )
                metrics_log["train/tokens_per_sec"] = (
                    metrics_agg["train/local_tokens"] / elapsed
                )
                metrics_log["train/masked_tokens_per_sec"] = (
                    metrics_agg["train/local_num_pred"] / elapsed
                )

        # Log the metrics with the current step
        accelerator.log(metrics_log, step=current_step)
        self._last_log_time = now
        self._last_log_step = current_step

        # Reset the local counters
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
