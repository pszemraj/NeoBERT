"""Shared distributed-tensor capability and placement predicates."""

from typing import Any

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except Exception:  # pragma: no cover - older torch builds without DTensor APIs
    DTensor = None  # type: ignore[assignment]
    Shard = None  # type: ignore[assignment]


def is_dtensor_like(value: Any) -> bool:
    """Return whether a value exposes DTensor semantics.

    :param Any value: Tensor or parameter candidate.
    :return bool: True when the value is a DTensor or compatible test double.
    """
    if DTensor is not None and isinstance(value, DTensor):
        return True
    return (
        hasattr(value, "device_mesh")
        and hasattr(value, "placements")
        and callable(getattr(value, "to_local", None))
    )


def is_row_shard_placement(placement: Any) -> bool:
    """Return whether a placement represents ``Shard(0)``.

    :param Any placement: DTensor placement descriptor.
    :return bool: True when the placement is a row shard.
    """
    if Shard is not None and isinstance(placement, Shard):
        return int(getattr(placement, "dim", -1)) == 0
    placement_name = type(placement).__name__.lower()
    shard_dim = getattr(placement, "dim", None)
    try:
        return placement_name.endswith("shard") and int(shard_dim) == 0
    except (TypeError, ValueError):
        return False
