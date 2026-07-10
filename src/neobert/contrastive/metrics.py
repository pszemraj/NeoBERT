"""Metric aggregation helpers for contrastive training."""

from typing import Any, Dict

from accelerate import Accelerator

from neobert.metrics import BaseTrainingMetrics


class Metrics(BaseTrainingMetrics):
    """Dictionary-like metrics container with distributed aggregation helpers."""

    LOCAL_COUNT_KEYS = ("train/local_samples",)
    LOCAL_FLOAT_KEYS = ("train/local_sum_loss",)
    STATE_CONTEXT = "contrastive"

    def log(self, accelerator: Accelerator) -> None:
        """Aggregate local counters and log already-global diagnostics as-is.

        Only per-rank counters participate in distributed reduction here.
        Scalars that are already global, such as FSDP-aware grad/weight norms,
        are forwarded unchanged so they are not double-counted.

        :param Accelerator accelerator: Accelerator used for reduction/logging.
        """
        metrics_agg = self.reduce_local_counters(accelerator)

        self["train/samples"] = (
            self["train/samples"] + metrics_agg["train/local_samples"]
        )

        metrics_log: Dict[str, Any] = {
            "train/epochs": self["train/epochs"],
            "train/steps": self["train/steps"],
            "train/learning_rate": self["train/learning_rate"],
            "train/samples": self["train/samples"],
        }
        if "train/grad_norm" in self:
            metrics_log["train/grad_norm"] = self["train/grad_norm"]
        if "train/weight_norm" in self:
            metrics_log["train/weight_norm"] = self["train/weight_norm"]
        if metrics_agg["train/local_samples"] > 0:
            metrics_log["train/loss"] = (
                metrics_agg["train/local_sum_loss"] / metrics_agg["train/local_samples"]
            )

        metrics_log |= {key: value for key, value in self.items() if "batches" in key}
        accelerator.log(metrics_log, step=self["train/steps"])

        self.reset_local_counters()
