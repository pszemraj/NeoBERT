"""Focused kernel backend dispatch regression tests."""

import pytest
import torch

from neobert.kernels.backend import (
    canonicalize_kernel_backend,
    get_cross_entropy_loss,
    get_rmsnorm,
    resolve_kernel_backend,
    swiglu_forward,
)


def test_resolve_kernel_backend_rejects_invalid_values() -> None:
    """Kernel backend resolvers should fail fast on unknown values."""
    assert resolve_kernel_backend("torch") == "torch"
    assert canonicalize_kernel_backend("auto") == "auto"

    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        resolve_kernel_backend("bad_backend")
    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        canonicalize_kernel_backend("bad_backend")


def test_cpu_dispatch_paths_produce_expected_shapes() -> None:
    """Core CPU dispatch paths should remain numerically valid."""
    norm = get_rmsnorm(64, 1e-5, "torch")
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape

    gate = torch.randn(2, 10, 64)
    up = torch.randn(2, 10, 64)
    swiglu_out = swiglu_forward(gate, up, "torch")
    assert swiglu_out.shape == gate.shape

    loss_fn = get_cross_entropy_loss(reduction="mean", backend="torch")
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_dispatch_helpers_reject_invalid_backend_consistently() -> None:
    """All kernel dispatch entrypoints should reject unknown backend strings."""
    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        get_rmsnorm(64, 1e-5, "bad_backend")

    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        swiglu_forward(torch.randn(2, 4, 8), torch.randn(2, 4, 8), "bad_backend")

    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        get_cross_entropy_loss(reduction="mean", backend="bad_backend")
