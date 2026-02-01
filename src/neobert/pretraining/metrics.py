"""Metric aggregation helpers for pretraining."""

import math
from collections import defaultdict
from typing import Any, Dict

from accelerate import Accelerator
from torch import Tensor


class Metrics(defaultdict):
    """Dictionary-like metrics container with distributed aggregation helpers."""

    def __init__(self):
        """Initialize metrics storage with integer defaults."""
        super().__init__(int)

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

    def log(self, accelerator: Accelerator) -> None:
        """Aggregate and log metrics across devices.

        :param Accelerator accelerator: Accelerator used for reduction/logging.
        """
        # Aggregate ALL metrics across devices (only required for local counters!)
        metrics_agg = Tensor(list(self.values())).to(
            accelerator.device, non_blocking=True
        )
        metrics_agg = (
            accelerator.reduce(metrics_agg, reduction="sum").detach().cpu().numpy()
        )
        metrics_agg = {k: v for k, v in zip(self.keys(), metrics_agg)}

        # Update global values
        self["train/samples"] = (
            self["train/samples"] + metrics_agg["train/local_samples"]
        )
        self["train/tokens"] = self["train/tokens"] + metrics_agg["train/local_tokens"]
        self["train/masked_tokens"] = (
            self["train/masked_tokens"] + metrics_agg["train/local_num_pred"]
        )

        # Build the metrics to log
        metrics_log = dict()
        for key in self.keys():
            metrics_log[key] = self[key]

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

        # Log the metrics with the current step
        # Extract the step value to pass separately to accelerator.log
        current_step = self.get("train/steps", 0)
        accelerator.log(metrics_log, step=current_step)

        # Reset the local counters
        for k in metrics_agg.keys():
            if "local" in k:
                self.pop(k)
