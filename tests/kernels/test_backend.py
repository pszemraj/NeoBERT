"""Tests for kernel backend dispatch (torch fallback + auto dispatch)."""

import pytest
import torch
from torch import nn

from neobert.kernels.backend import (
    LIGER_AVAILABLE,
    _AdaptiveRMSNorm,
    canonicalize_kernel_backend,
    get_cross_entropy_loss,
    get_rmsnorm,
    resolve_kernel_backend,
    swiglu_forward,
)


class TestResolveKernelBackend:
    """Tests for resolve_kernel_backend()."""

    def test_torch_always_resolves(self):
        assert resolve_kernel_backend("torch") == "torch"

    def test_auto_resolves(self):
        result = resolve_kernel_backend("auto")
        assert result in ("torch", "liger")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel_backend"):
            resolve_kernel_backend("invalid_backend")

    def test_liger_without_cuda_raises(self):
        if not torch.cuda.is_available():
            if LIGER_AVAILABLE:
                with pytest.raises(RuntimeError, match="CUDA"):
                    resolve_kernel_backend("liger")


class TestCanonicalizeKernelBackend:
    """Tests for canonicalize_kernel_backend()."""

    def test_valid_values(self):
        assert canonicalize_kernel_backend("torch") == "torch"
        assert canonicalize_kernel_backend("auto") == "auto"
        assert canonicalize_kernel_backend("liger") == "liger"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel_backend"):
            canonicalize_kernel_backend("bad_backend")


class TestGetRMSNorm:
    """Tests for get_rmsnorm() dispatch."""

    def test_torch_returns_rmsnorm(self):
        norm = get_rmsnorm(64, 1e-5, "torch")
        assert isinstance(norm, nn.Module)
        assert hasattr(norm, "weight")

    def test_torch_rmsnorm_forward(self):
        norm = get_rmsnorm(64, 1e-5, "torch")
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_torch_rmsnorm_numerics(self):
        """Verify RMSNorm produces unit-scale outputs."""
        norm = get_rmsnorm(64, 1e-5, "torch")
        x = torch.randn(4, 8, 64)
        out = norm(x)
        rms = (out**2).mean(dim=-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.5)

    def test_auto_returns_adaptive_on_liger_machine(self):
        """When Liger is available, auto should return _AdaptiveRMSNorm."""
        norm = get_rmsnorm(64, 1e-5, "auto")
        if LIGER_AVAILABLE:
            assert isinstance(norm, _AdaptiveRMSNorm)
        else:
            assert isinstance(norm, nn.Module)

    def test_auto_rmsnorm_works_on_cpu(self):
        """auto backend must work on CPU even when Liger is available."""
        norm = get_rmsnorm(64, 1e-5, "auto")
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_auto_rmsnorm_numerics_on_cpu(self):
        norm = get_rmsnorm(64, 1e-5, "auto")
        x = torch.randn(4, 8, 64)
        out = norm(x)
        rms = (out**2).mean(dim=-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.5)

    def test_rmsnorm_rejects_fp16_on_cpu(self):
        """NeoBERT policy rejects fp16 runtime tensors."""
        norm = _AdaptiveRMSNorm(64, 1e-5)
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="fp16/float16"):
            norm(x)


class TestSwiGLUForward:
    """Tests for swiglu_forward() dispatch."""

    def test_torch_matches_manual(self):
        gate = torch.randn(2, 10, 64)
        up = torch.randn(2, 10, 64)
        result = swiglu_forward(gate, up, "torch")
        expected = nn.functional.silu(gate) * up
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        gate = torch.randn(3, 5, 32)
        up = torch.randn(3, 5, 32)
        result = swiglu_forward(gate, up, "torch")
        assert result.shape == (3, 5, 32)

    def test_auto_works_on_cpu(self):
        """auto backend must work on CPU."""
        gate = torch.randn(2, 10, 64)
        up = torch.randn(2, 10, 64)
        result = swiglu_forward(gate, up, "auto")
        expected = nn.functional.silu(gate) * up
        torch.testing.assert_close(result, expected)


class TestGetCrossEntropyLoss:
    """Tests for get_cross_entropy_loss() dispatch."""

    def test_torch_returns_ce_loss(self):
        loss_fn = get_cross_entropy_loss(reduction="mean", backend="torch")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_torch_ce_numerics(self):
        loss_fn = get_cross_entropy_loss(reduction="mean", backend="torch")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_auto_ce_works_on_cpu(self):
        """auto backend must work on CPU."""
        loss_fn = get_cross_entropy_loss(reduction="mean", backend="auto")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


@pytest.mark.parametrize(
    ("call", "args", "kwargs"),
    [
        (get_rmsnorm, (64, 1e-5, "bad_backend"), {}),
        (
            swiglu_forward,
            (torch.randn(2, 4, 8), torch.randn(2, 4, 8), "bad_backend"),
            {},
        ),
        (get_cross_entropy_loss, (), {"reduction": "mean", "backend": "bad_backend"}),
    ],
)
def test_invalid_backend_raises_across_dispatchers(
    call,
    args,
    kwargs,
):
    """Ensure all kernel dispatch entrypoints reject invalid backend strings."""
    with pytest.raises(ValueError, match="Unknown kernel_backend"):
        call(*args, **kwargs)
