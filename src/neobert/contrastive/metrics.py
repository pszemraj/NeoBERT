"""Metric aggregation helpers for contrastive training."""

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

        # Build the metrics to log
        metrics_log = dict()
        metrics_log["train/epochs"] = self["train/epochs"]
        metrics_log["train/steps"] = self["train/steps"]
        metrics_log["train/grad_norm"] = self["train/grad_norm"]
        metrics_log["train/weight_norm"] = self["train/weight_norm"]
        metrics_log["train/learning_rate"] = self["train/learning_rate"]
        metrics_log["train/samples"] = self["train/samples"]
        if metrics_agg["train/local_samples"] > 0:
            metrics_log["train/loss"] = (
                metrics_agg["train/local_sum_loss"] / metrics_agg["train/local_samples"]
            )
            # metrics_log["train/perplexity"] = math.exp(metrics_log["train/loss"])

        metrics_log |= {key: value for key, value in self.items() if "batches" in key}

        # Log the metrics
        accelerator.log(metrics_log)

        # Reset the local counters
        for k in metrics_agg.keys():
            if "local" in k:
                self.pop(k)
