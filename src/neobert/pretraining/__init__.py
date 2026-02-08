"""Pretraining entry points and objective utilities."""

__all__ = ["MaskedPositionsOnlyMLMObjective", "trainer"]

from .masked_objective import MaskedPositionsOnlyMLMObjective
from .trainer import trainer
