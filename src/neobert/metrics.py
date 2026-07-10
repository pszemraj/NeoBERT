"""Shared distributed metric state and reduction primitives."""

from collections import defaultdict
from typing import Any

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


class BaseTrainingMetrics(defaultdict):
    """Checkpoint-safe metrics mapping with fixed-order distributed reductions."""

    LOCAL_COUNT_KEYS: tuple[str, ...] = ()
    LOCAL_FLOAT_KEYS: tuple[str, ...] = ()
    STATE_CONTEXT = "training"

    def __init__(self) -> None:
        """Initialize local metric counters with stable numeric defaults."""
        super().__init__(int)
        self.reset_local_counters()

    def state_dict(self) -> dict[str, Any]:
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
                        f"Distributed {self.STATE_CONTEXT} ranks disagree on checkpoint "
                        f"resume position: {rank_positions}. Refusing to persist a "
                        "shared cursor."
                    )
        return {
            _METRICS_STATE_VERSION_KEY: _METRICS_STATE_VERSION,
            _METRICS_STATE_WORLD_SIZE_KEY: world_size,
            _METRICS_STATE_PAYLOAD_KEY: dict(self),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load versioned metrics state for the current distributed topology.

        :param dict[str, Any] state_dict: Metrics state to load.
        :raises ValueError: If the schema or distributed world size differs.
        """
        version = state_dict.get(_METRICS_STATE_VERSION_KEY)
        if version != _METRICS_STATE_VERSION:
            raise ValueError(
                f"Unsupported {self.STATE_CONTEXT} metrics checkpoint version "
                f"{version!r}; expected {_METRICS_STATE_VERSION}. Older loop state "
                "cannot prove that distributed data cursors were aligned."
            )
        checkpoint_world_size = int(state_dict.get(_METRICS_STATE_WORLD_SIZE_KEY, 0))
        runtime_world_size = (
            int(dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if checkpoint_world_size != runtime_world_size:
            raise ValueError(
                f"Cannot restore {self.STATE_CONTEXT} metrics with a different world "
                f"size: checkpoint={checkpoint_world_size}, runtime={runtime_world_size}."
            )
        payload = state_dict.get(_METRICS_STATE_PAYLOAD_KEY)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{self.STATE_CONTEXT.capitalize()} metrics checkpoint payload must "
                "be a mapping."
            )
        self.clear()
        self.update(payload)
        self._after_load()

    def _after_load(self) -> None:
        """Reset non-checkpointed runtime state after loading persisted metrics."""

    def reduce_local_counters(self, accelerator: Accelerator) -> dict[str, int | float]:
        """Sum fixed-order rank-local counters across processes.

        :param Accelerator accelerator: Accelerator used for reductions.
        :return dict[str, int | float]: Aggregated local counter values.
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
        return {
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

    def reset_local_counters(self) -> None:
        """Reset rank-local counters after logging or initial construction."""
        for key in self.LOCAL_COUNT_KEYS:
            self[key] = 0
        for key in self.LOCAL_FLOAT_KEYS:
            self[key] = 0.0
