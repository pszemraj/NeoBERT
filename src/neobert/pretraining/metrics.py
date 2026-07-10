"""Metric aggregation helpers for pretraining."""

import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
from accelerate import Accelerator

_METRICS_STATE_VERSION = 1
_METRICS_STATE_VERSION_KEY = "metrics_state_version"
_METRICS_STATE_WORLD_SIZE_KEY = "world_size"
_METRICS_STATE_PAYLOAD_KEY = "metrics"
_RESUME_POSITION_KEYS = (
    "train/steps",
    "train/epochs",
    "train/batches_in_epoch",
    "train/dataloader_batches_in_epoch",
)


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
        "train/dataloader_batches_in_epoch",
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

    def state_dict(self) -> Dict[str, Any]:
        """Return versioned metrics after validating distributed resume position.

        :raises RuntimeError: If distributed ranks have different resume cursors.
        :return dict[str, Any]: Metrics state.
        """
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            if world_size > 1:
                local_position = {
                    key: int(self.get(key, 0)) for key in _RESUME_POSITION_KEYS
                }
                rank_positions: list[dict[str, int] | None] = [None] * world_size
                dist.all_gather_object(rank_positions, local_position)
                if any(
                    position != rank_positions[0] for position in rank_positions[1:]
                ):
                    raise RuntimeError(
                        "Distributed pretraining ranks disagree on checkpoint resume "
                        f"position: {rank_positions}. Refusing to persist a shared cursor."
                    )

        return {
            _METRICS_STATE_VERSION_KEY: _METRICS_STATE_VERSION,
            _METRICS_STATE_WORLD_SIZE_KEY: world_size,
            _METRICS_STATE_PAYLOAD_KEY: dict(self),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load versioned metrics state for the current distributed topology.

        :param dict[str, Any] state_dict: Metrics state to load.
        :raises ValueError: If the schema or distributed world size differs.
        """
        version = state_dict.get(_METRICS_STATE_VERSION_KEY)
        if version != _METRICS_STATE_VERSION:
            raise ValueError(
                "Unsupported pretraining metrics checkpoint version "
                f"{version!r}; expected {_METRICS_STATE_VERSION}. Older loop state "
                "cannot prove that distributed packed-data cursors were aligned."
            )
        checkpoint_world_size = int(state_dict.get(_METRICS_STATE_WORLD_SIZE_KEY, 0))
        runtime_world_size = (
            int(dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if checkpoint_world_size != runtime_world_size:
            raise ValueError(
                "Cannot restore pretraining metrics with a different world size: "
                f"checkpoint={checkpoint_world_size}, runtime={runtime_world_size}."
            )
        payload = state_dict.get(_METRICS_STATE_PAYLOAD_KEY)
        if not isinstance(payload, dict):
            raise ValueError(
                "Pretraining metrics checkpoint payload must be a mapping."
            )

        self.clear()
        for k, v in payload.items():
            self[k] = v
        self._last_log_time = None

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

        # Reset the local counters
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
        return formatted
