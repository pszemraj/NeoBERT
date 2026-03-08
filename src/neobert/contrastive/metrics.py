"""Metric aggregation helpers for contrastive training."""

from collections import defaultdict
from typing import Any, Dict

import torch
from accelerate import Accelerator


class Metrics(defaultdict):
    """Dictionary-like metrics container with distributed aggregation helpers."""

    LOCAL_COUNT_KEYS = ("train/local_samples",)
    LOCAL_FLOAT_KEYS = ("train/local_sum_loss",)

    def __init__(self):
        """Initialize metrics storage with stable numeric defaults."""
        super().__init__(int)
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0

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
        """Aggregate local counters and log already-global diagnostics as-is.

        Only per-rank counters participate in distributed reduction here.
        Scalars that are already global, such as FSDP-aware grad/weight norms,
        are forwarded unchanged so they are not double-counted.

        :param Accelerator accelerator: Accelerator used for reduction/logging.
        """
        count_tensor = torch.tensor(
            [self.get(key, 0) for key in self.LOCAL_COUNT_KEYS],
            device=accelerator.device,
            dtype=torch.long,
        )
        count_tensor = accelerator.reduce(count_tensor, reduction="sum")
        float_tensor = torch.tensor(
            [self.get(key, 0.0) for key in self.LOCAL_FLOAT_KEYS],
            device=accelerator.device,
            dtype=torch.float64,
        )
        float_tensor = accelerator.reduce(float_tensor, reduction="sum")

        metrics_agg = {
            **{
                key: int(value)
                for key, value in zip(
                    self.LOCAL_COUNT_KEYS, count_tensor.detach().cpu().tolist()
                )
            },
            **{
                key: float(value)
                for key, value in zip(
                    self.LOCAL_FLOAT_KEYS, float_tensor.detach().cpu().tolist()
                )
            },
        }

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

        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
