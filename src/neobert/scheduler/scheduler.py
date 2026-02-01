"""Learning-rate scheduler factory for training runs."""

import torch
from typing import Any
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    decay: str,
    warmup_steps: int,
    decay_steps: int,
    final_ratio: float = 0.1,
    constant_steps: int = 0,
    **kwargs: Any,
) -> SequentialLR:
    """Create a chained warmup/decay scheduler.

    :param torch.optim.Optimizer optimizer: Optimizer to schedule.
    :param float lr: Base learning rate.
    :param str decay: Decay type (``cosine`` or ``linear``).
    :param int warmup_steps: Steps for linear warmup.
    :param int decay_steps: Steps for decay schedule.
    :param float final_ratio: Final LR multiplier after decay.
    :param int constant_steps: Optional constant plateau steps.
    :param Any kwargs: Unused extra scheduler arguments.
    :return SequentialLR: Configured scheduler.
    """

    if decay.lower() not in ["cosine", "linear"]:
        raise ValueError(
            f"Decay {decay} is not a valid type. Options are cosine and linear."
        )

    assert (constant_steps == 0 and warmup_steps < decay_steps) or (
        warmup_steps < constant_steps and constant_steps < decay_steps
    ), (
        "warmup_steps, constant_steps and decay_steps are milestones parameters, not total number of steps for each scheduler."
    )

    schedulers = []
    milestones = []

    # Warmup scheduler
    schedulers.append(
        LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    )
    milestones.append(warmup_steps)

    # Optional constant scheduler at peak learning rate
    if constant_steps:
        schedulers.append(LambdaLR(optimizer, lr_lambda=lambda _: 1))
        milestones.append(constant_steps)

    # Decay scheduler
    schedulers.append(
        CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=lr * final_ratio)
        if decay == "cosine"
        else LinearLR(
            optimizer, start_factor=1.0, end_factor=final_ratio, total_iters=decay_steps
        )
    )

    milestones.append(decay_steps)

    # Final constant scheduler at lowest learning rate
    def _constant_min_lr(_: int) -> float:
        """Return a constant LR multiplier at the minimum ratio.

        :param int _: Current epoch or step index.
        :return float: Final LR multiplier.
        """
        return final_ratio

    schedulers.append(LambdaLR(optimizer, lr_lambda=_constant_min_lr))

    return SequentialLR(optimizer, schedulers, milestones)
