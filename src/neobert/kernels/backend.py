"""Liger Kernel / PyTorch dispatch for RMSNorm, SwiGLU, and CrossEntropy.

All dispatch functions accept the raw ``kernel_backend`` string (``"auto"``,
``"liger"``, or ``"torch"``) and resolve at **call time** based on tensor
device.  ``"auto"`` uses Liger Triton kernels when tensors are on CUDA and
liger-kernel is installed, otherwise falls back to native PyTorch.

RMSNorm is special: ``get_rmsnorm`` returns a single ``nn.Module`` whose
``forward`` dispatches per-call so the module can be constructed on CPU and
later moved to GPU without issues.
"""

import logging
from typing import Any, Callable, Literal, Optional, TypeVar

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import-time Liger availability check
# ---------------------------------------------------------------------------
LIGER_AVAILABLE: bool = False
LIGER_ERROR: Optional[str] = None

_LigerRMSNormFunction = None
_LigerSiLUMulFunction = None
_LigerCrossEntropyLoss = None


_F = TypeVar("_F", bound=Callable[..., Any])


def _identity_decorator(fn: _F) -> _F:
    """Return *fn* unchanged.

    :param callable fn: Function to decorate.
    :return callable: Original function.
    """
    return fn


try:
    _compile_disable = torch.compiler.disable  # type: ignore[attr-defined]
except AttributeError:
    try:
        import torch._dynamo as _dynamo

        _compile_disable = _dynamo.disable
    except ImportError:
        _compile_disable = _identity_decorator

try:
    from liger_kernel.ops.rms_norm import (
        LigerRMSNormFunction as _LigerRMSNormFunction,  # type: ignore[no-redef]
    )

    LIGER_AVAILABLE = True
except (ImportError, RuntimeError) as exc:
    LIGER_ERROR = str(exc)

if LIGER_AVAILABLE:
    try:
        from liger_kernel.ops.swiglu import (
            LigerSiLUMulFunction as _LigerSiLUMulFunction,
        )  # type: ignore[no-redef]
    except ImportError:
        try:
            from liger_kernel.transformers.functional import (
                liger_swiglu as _liger_swiglu_fn,
            )  # type: ignore[import]

            class _LigerSiLUMulFunction:  # type: ignore[no-redef]
                """Thin wrapper to present the same ``apply(gate, up)`` API."""

                @staticmethod
                def apply(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
                    """Apply fused SwiGLU pointwise op.

                    :param torch.Tensor gate: Gate tensor.
                    :param torch.Tensor up: Up tensor.
                    :return torch.Tensor: Activated tensor.
                    """
                    return _liger_swiglu_fn(gate, up)

        except ImportError:
            _LigerSiLUMulFunction = None
            logger.debug("Liger SiLUMul not found; SwiGLU will use torch fallback.")

    try:
        from liger_kernel.transformers import (
            LigerCrossEntropyLoss as _LigerCrossEntropyLoss,
        )  # type: ignore[no-redef]
    except ImportError:
        _LigerCrossEntropyLoss = None
        logger.debug("LigerCrossEntropyLoss not found; CE will use torch fallback.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_use_liger(backend: str, tensor: torch.Tensor) -> bool:
    """Decide whether to dispatch to Liger for a given backend string + tensor.

    :param str backend: ``"auto"``, ``"liger"``, or ``"torch"``.
    :param torch.Tensor tensor: A representative input tensor (for device check).
    :return bool: True if Liger should be used.
    """
    backend = canonicalize_kernel_backend(backend)
    if backend == "torch":
        return False
    if not LIGER_AVAILABLE:
        if backend == "liger":
            raise ImportError(
                f"kernel_backend='liger' requested but liger-kernel is not available: "
                f"{LIGER_ERROR}"
            )
        return False
    if not tensor.is_cuda:
        if backend == "liger":
            raise RuntimeError(
                "kernel_backend='liger' requires CUDA tensors but input is on "
                f"{tensor.device}."
            )
        return False
    # backend is "auto" or "liger", tensor is on CUDA, Liger available
    return True


def canonicalize_kernel_backend(
    requested: str,
) -> Literal["auto", "liger", "torch"]:
    """Canonicalize the kernel backend string without environment checks.

    :param str requested: One of ``"auto"``, ``"liger"``, or ``"torch"``.
    :return str: Canonical backend name.
    :raises ValueError: If *requested* is unknown.
    """
    normalized = str(requested).lower().strip()
    if normalized in {"auto", "liger", "torch"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(
        f"Unknown kernel_backend '{requested}'. Expected: 'auto', 'liger', or 'torch'."
    )


def resolve_kernel_backend(
    requested: str,
) -> Literal["torch", "liger"]:
    """Eagerly resolve the kernel backend (for contexts without a tensor).

    Prefer using the dispatch functions directly (``get_rmsnorm``,
    ``swiglu_forward``, ``get_cross_entropy_loss``) which resolve per-call.
    This helper is still useful for logging / config validation.

    :param str requested: One of ``"auto"``, ``"liger"``, or ``"torch"``.
    :return str: Resolved backend name.
    """
    requested = canonicalize_kernel_backend(requested)
    if requested == "torch":
        return "torch"
    if requested == "liger":
        if not LIGER_AVAILABLE:
            raise ImportError(
                f"kernel_backend='liger' requested but liger-kernel is not available: "
                f"{LIGER_ERROR}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "kernel_backend='liger' requires CUDA but no CUDA device is available."
            )
        return "liger"
    if requested == "auto":
        if LIGER_AVAILABLE and torch.cuda.is_available():
            return "liger"
        return "torch"
    raise AssertionError(f"Unhandled kernel_backend state: {requested}")


@_compile_disable
def _liger_rmsnorm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Run Liger RMSNorm outside torch.compile graphs.

    :param torch.Tensor x: Input tensor.
    :param torch.Tensor weight: RMSNorm weight.
    :param float eps: Numerical epsilon.
    :return torch.Tensor: Normalized tensor.
    """
    assert _LigerRMSNormFunction is not None
    return _LigerRMSNormFunction.apply(x, weight, eps)


@_compile_disable
def _liger_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Run Liger SwiGLU pointwise op outside torch.compile graphs.

    :param torch.Tensor gate: Gate tensor.
    :param torch.Tensor up: Up tensor.
    :return torch.Tensor: Activated tensor.
    """
    assert _LigerSiLUMulFunction is not None
    return _LigerSiLUMulFunction.apply(gate, up)


@_compile_disable
def _liger_cross_entropy_forward(
    loss_module: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Run Liger cross-entropy outside torch.compile graphs.

    :param nn.Module loss_module: Liger loss module.
    :param torch.Tensor input: Input logits.
    :param torch.Tensor target: Target labels.
    :return torch.Tensor: Scalar/tensor loss.
    """
    return loss_module(input, target)


# ---------------------------------------------------------------------------
# RMSNorm dispatch
# ---------------------------------------------------------------------------


class _AdaptiveRMSNorm(nn.Module):
    """RMSNorm that uses Liger's Triton kernel on CUDA, native torch on CPU.

    A single ``weight`` parameter is shared across both code-paths so the
    module can be constructed on CPU, moved to GPU, and seamlessly use the
    faster Triton kernel — with no checkpoint-compatibility issues.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the adaptive RMSNorm module.

        :param int dim: Feature dimension.
        :param float eps: Numerical epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization with dynamic kernel dispatch.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Normalized tensor.
        """
        if x.is_cuda and _LigerRMSNormFunction is not None:
            return _liger_rmsnorm_forward(x, self.weight, self.eps)
        # Native torch path (CPU or Liger unavailable)
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight

    def extra_repr(self) -> str:
        """Return a concise module representation for debugging.

        :return str: Extra module representation.
        """
        return (
            f"{self.weight.shape[0]}, eps={self.eps}, "
            f"liger={'available' if _LigerRMSNormFunction is not None else 'unavailable'}"
        )


def get_rmsnorm(
    dim: int,
    eps: float,
    backend: str,
) -> nn.Module:
    """Return a RMSNorm module.

    When *backend* is ``"auto"`` or ``"liger"``, returns an
    ``_AdaptiveRMSNorm`` that uses Liger's Triton kernel on CUDA inputs and
    native torch on CPU inputs.  When ``"torch"``, returns the plain native
    implementation.

    :param int dim: Feature dimension.
    :param float eps: Normalization epsilon.
    :param str backend: ``"auto"``, ``"liger"``, or ``"torch"``.
    :return nn.Module: RMSNorm instance.
    """
    backend = canonicalize_kernel_backend(backend)
    if backend in ("liger", "auto") and _LigerRMSNormFunction is not None:
        return _AdaptiveRMSNorm(dim, eps=eps)

    from ..model.rmsnorm import RMSNorm

    return RMSNorm(dim, eps=eps)


# ---------------------------------------------------------------------------
# SwiGLU pointwise dispatch
# ---------------------------------------------------------------------------


def swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` using Liger or native ops.

    Resolves ``"auto"`` at call time based on tensor device.

    :param torch.Tensor gate: Gate projection output (w1(x)).
    :param torch.Tensor up: Up projection output (w2(x)).
    :param str backend: ``"auto"``, ``"liger"``, or ``"torch"``.
    :return torch.Tensor: Activated tensor.
    """
    backend = canonicalize_kernel_backend(backend)
    if _LigerSiLUMulFunction is not None and _should_use_liger(backend, gate):
        return _liger_swiglu_forward(gate, up)
    return nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# CrossEntropy dispatch
# ---------------------------------------------------------------------------


class _AdaptiveCrossEntropyLoss(nn.Module):
    """CrossEntropy loss that dispatches to Liger on CUDA, PyTorch on CPU."""

    def __init__(self, reduction: str = "mean", **kwargs: Any) -> None:
        """Initialize the adaptive CE loss module.

        :param str reduction: Loss reduction mode.
        :param Any kwargs: Forwarded kwargs for CE constructors.
        """
        super().__init__()
        self._torch_ce = nn.CrossEntropyLoss(reduction=reduction, **kwargs)
        self._liger_ce = (
            _LigerCrossEntropyLoss(reduction=reduction, **kwargs)
            if _LigerCrossEntropyLoss is not None
            else None
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CE loss with dynamic kernel dispatch.

        :param torch.Tensor input: Input logits.
        :param torch.Tensor target: Target labels.
        :return torch.Tensor: Computed loss.
        """
        if input.is_cuda and self._liger_ce is not None:
            return _liger_cross_entropy_forward(self._liger_ce, input, target)
        return self._torch_ce(input, target)


def get_cross_entropy_loss(
    reduction: str = "mean",
    backend: str = "torch",
    **kwargs: Any,
) -> nn.Module:
    """Return a CrossEntropyLoss module.

    When *backend* is ``"auto"`` or ``"liger"``, returns an adaptive wrapper
    that uses Liger on CUDA inputs and PyTorch on CPU inputs.

    :param str reduction: Loss reduction mode.
    :param str backend: ``"auto"``, ``"liger"``, or ``"torch"``.
    :param Any kwargs: Forwarded kwargs for CE constructors.
    :return nn.Module: CrossEntropyLoss instance.
    """
    backend = canonicalize_kernel_backend(backend)
    if backend in ("liger", "auto") and _LigerCrossEntropyLoss is not None:
        return _AdaptiveCrossEntropyLoss(reduction=reduction, **kwargs)
    return nn.CrossEntropyLoss(reduction=reduction, **kwargs)
