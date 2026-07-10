"""Checkpointable state for the GLUE fine-tuning loop."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

_GLUE_LOOP_STATE_VERSION = 1


@dataclass
class GlueLoopState:
    """Track optimizer-boundary progress and metric-selection state."""

    world_size: int
    completed_steps: int = 0
    epoch: int = 0
    microbatches_in_epoch: int = 0
    total_loss: float = 0.0
    best_selection_score: float | None = None
    early_stopping_counter: int = 0
    last_train_metrics: dict[str, Any] = field(default_factory=dict)
    last_val_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate construction-time process topology."""
        self.world_size = int(self.world_size)
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}.")

    def record_update(
        self,
        *,
        completed_steps: int,
        epoch: int,
        microbatches_in_epoch: int,
        total_loss: float,
    ) -> None:
        """Record the next unconsumed dataloader position after an update.

        :param int completed_steps: Completed optimizer updates.
        :param int epoch: Current zero-based data epoch.
        :param int microbatches_in_epoch: Consumed microbatches in the epoch.
        :param float total_loss: Cumulative mean-per-update training loss.
        """
        completed_steps = int(completed_steps)
        epoch = int(epoch)
        microbatches_in_epoch = int(microbatches_in_epoch)
        total_loss = float(total_loss)
        if min(completed_steps, epoch, microbatches_in_epoch) < 0:
            raise ValueError("GLUE loop update counters must be non-negative.")
        if not math.isfinite(total_loss):
            raise ValueError(f"GLUE cumulative loss must be finite: {total_loss}.")
        self.completed_steps = completed_steps
        self.epoch = epoch
        self.microbatches_in_epoch = microbatches_in_epoch
        self.total_loss = total_loss

    def update_selection_score(self, score: float) -> bool:
        """Update best-score and early-stopping state.

        :param float score: Current task-specific checkpoint-selection score.
        :return bool: True when the score is the first or best score seen.
        """
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(
                f"GLUE checkpoint-selection score must be finite: {score}."
            )
        improved = (
            self.best_selection_score is None or score > self.best_selection_score
        )
        if improved:
            self.best_selection_score = score
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        return improved

    def state_dict(self) -> dict[str, Any]:
        """Return a versioned checkpoint payload.

        :return dict[str, Any]: Serializable loop state.
        """
        return {
            "version": _GLUE_LOOP_STATE_VERSION,
            "world_size": self.world_size,
            "completed_steps": self.completed_steps,
            "epoch": self.epoch,
            "microbatches_in_epoch": self.microbatches_in_epoch,
            "total_loss": self.total_loss,
            "best_selection_score": self.best_selection_score,
            "early_stopping_counter": self.early_stopping_counter,
            "last_train_metrics": deepcopy(self.last_train_metrics),
            "last_val_metrics": deepcopy(self.last_val_metrics),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore loop state while enforcing version and process topology.

        :param Mapping[str, Any] state_dict: Saved state payload.
        :raises ValueError: If the payload or world size is incompatible.
        """
        if not isinstance(state_dict, Mapping):
            raise ValueError("GLUE loop state must be a mapping.")
        version = state_dict.get("version")
        if version != _GLUE_LOOP_STATE_VERSION:
            raise ValueError(
                f"Unsupported GLUE loop state version {version!r}; "
                f"expected {_GLUE_LOOP_STATE_VERSION}."
            )
        checkpoint_world_size = int(state_dict.get("world_size", 0))
        if checkpoint_world_size != self.world_size:
            raise ValueError(
                "Cannot resume GLUE with a different world size: "
                f"checkpoint={checkpoint_world_size}, runtime={self.world_size}."
            )

        completed_steps = int(state_dict.get("completed_steps", -1))
        epoch = int(state_dict.get("epoch", -1))
        microbatches_in_epoch = int(state_dict.get("microbatches_in_epoch", -1))
        early_stopping_counter = int(state_dict.get("early_stopping_counter", -1))
        if (
            min(
                completed_steps,
                epoch,
                microbatches_in_epoch,
                early_stopping_counter,
            )
            < 0
        ):
            raise ValueError("GLUE loop counters must be non-negative.")

        last_train_metrics = state_dict.get("last_train_metrics", {})
        last_val_metrics = state_dict.get("last_val_metrics", {})
        if not isinstance(last_train_metrics, Mapping) or not isinstance(
            last_val_metrics, Mapping
        ):
            raise ValueError("GLUE loop metric snapshots must be mappings.")

        self.completed_steps = completed_steps
        self.epoch = epoch
        self.microbatches_in_epoch = microbatches_in_epoch
        total_loss = float(state_dict.get("total_loss", 0.0))
        if not math.isfinite(total_loss):
            raise ValueError(f"GLUE cumulative loss must be finite: {total_loss}.")
        self.total_loss = total_loss
        best_score = state_dict.get("best_selection_score")
        self.best_selection_score = None if best_score is None else float(best_score)
        if self.best_selection_score is not None and not math.isfinite(
            self.best_selection_score
        ):
            raise ValueError(
                "GLUE checkpoint-selection score must be finite: "
                f"{self.best_selection_score}."
            )
        self.early_stopping_counter = early_stopping_counter
        self.last_train_metrics = deepcopy(dict(last_train_metrics))
        self.last_val_metrics = deepcopy(dict(last_val_metrics))
