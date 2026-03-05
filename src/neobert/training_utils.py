"""Shared helpers for training loops (pretraining, GLUE, contrastive)."""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

logger = logging.getLogger(__name__)

_LOW_PRECISION_LINEAR_PROBE_CACHE: dict[tuple[int, str, str], bool] = {}


def _probe_cuda_linear_dtype(dtype: torch.dtype) -> bool:
    """Probe whether CUDA linear GEMM works for a low-precision dtype.

    :param torch.dtype dtype: Probe dtype (typically ``torch.bfloat16``).
    :return bool: ``True`` when a tiny linear op succeeds end-to-end.
    """
    if not torch.cuda.is_available():
        return False
    device_index = torch.cuda.current_device()
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    blas_tag = "unknown"
    if callable(preferred_blas):
        try:
            blas_tag = str(preferred_blas())
        except Exception:
            blas_tag = "unknown"
    cache_key = (int(device_index), str(dtype), blas_tag)
    cached = _LOW_PRECISION_LINEAR_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with torch.no_grad():
            x = torch.randn((16, 64), device="cuda", dtype=dtype)
            w = torch.randn((128, 64), device="cuda", dtype=dtype)
            _ = torch.nn.functional.linear(x, w)
            torch.cuda.synchronize()
        _LOW_PRECISION_LINEAR_PROBE_CACHE[cache_key] = True
        return True
    except RuntimeError:
        _LOW_PRECISION_LINEAR_PROBE_CACHE[cache_key] = False
        return False


def stabilize_cuda_mixed_precision(
    *,
    mixed_precision: str,
    log: logging.Logger,
) -> str:
    """Stabilize mixed precision policy for CUDA runtimes with broken bf16 GEMM.

    Some driver/toolchain/GPU combinations report bf16 support but fail at
    runtime in default cuBLAS GEMM. We first probe bf16 linear; on failure we
    try switching to cuBLASLt, and only then fall back to fp32.

    :param str mixed_precision: Requested mixed precision mode.
    :param logging.Logger log: Logger for runtime policy warnings.
    :return str: Effective mixed precision mode (``"bf16"`` or ``"no"``).
    """
    if mixed_precision != "bf16" or not torch.cuda.is_available():
        return mixed_precision

    if _probe_cuda_linear_dtype(torch.bfloat16):
        return mixed_precision

    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if callable(preferred_blas):
        try:
            preferred_blas("cublaslt")
            if _probe_cuda_linear_dtype(torch.bfloat16):
                log.warning(
                    "bf16 CUDA linear GEMM failed with default cuBLAS on this runtime; "
                    "switched torch.backends.cuda.preferred_blas_library('cublaslt')."
                )
                return mixed_precision
        except Exception as exc:
            log.warning(
                "Failed to switch CUDA BLAS backend to cuBLASLt for bf16 stability: "
                f"{exc}"
            )

    log.warning(
        "bf16 CUDA linear GEMM is unavailable on this runtime; "
        "falling back to mixed_precision='no' to avoid immediate crashes."
    )
    return "no"


def resolve_wandb_watch_mode(
    *,
    wandb_mode: str,
    config_value: Optional[str],
    env_value: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve effective ``wandb.watch`` mode with sane defaults.

    Behavior:
    - Precedence: ``WANDB_WATCH`` env var > ``wandb.watch`` config > default.
    - Default is ``"gradients"`` when mode is online.
    - False-like values disable watching.
    - If set to ``weights``, map to ``parameters`` (W&B API naming).
    - If set to an unsupported value, disable watching and return a warning.

    :param str wandb_mode: Effective W&B run mode (online/offline/disabled).
    :param str | None config_value: Config value from ``wandb.watch``.
    :param str | None env_value: Raw ``WANDB_WATCH`` environment value.
    :return tuple[str | None, str | None]: (watch mode, optional warning message).
    """
    resolved_mode = str(wandb_mode).strip().lower()
    if resolved_mode != "online":
        return None, None

    raw_mode = env_value if env_value is not None else config_value
    if raw_mode is None:
        raw_mode = "gradients"
    watch_mode = str(raw_mode).strip().lower()
    if watch_mode in {"", "false", "0", "none", "off"}:
        return None, None
    if watch_mode == "disabled":
        return None, None
    if watch_mode == "weights":
        return "parameters", None
    if watch_mode in {"gradients", "parameters", "all"}:
        return watch_mode, None
    return (
        None,
        f"Unrecognized wandb watch mode '{raw_mode}'; skipping wandb.watch().",
    )


def _unwrap_optimizer(opt: Any) -> Any:
    """Return the underlying optimizer if wrapped by Accelerate.

    :param Any opt: Optimizer instance to unwrap.
    :return Any: Unwrapped optimizer.
    """
    return getattr(opt, "optimizer", opt)


def create_accelerator(
    *,
    use_cpu: bool,
    log: logging.Logger,
    accelerator_factory: Callable[..., Accelerator] = Accelerator,
    **kwargs: Any,
) -> Accelerator:
    """Create an Accelerator and gracefully handle mixed cpu/cuda test processes.

    When ``use_cpu=True`` is requested after Accelerate has already initialized a
    CUDA-backed shared state (common in unit-test suites), Accelerate raises a
    ValueError about changing ``cpu=True``. In that case we warn once and retry
    without forcing CPU so the existing process state remains usable.

    :param bool use_cpu: Whether to request CPU execution.
    :param logging.Logger log: Logger for fallback warnings.
    :param Callable[..., Accelerator] accelerator_factory: Accelerator constructor.
    :param kwargs: Additional ``Accelerator(...)`` keyword arguments.
    :return Accelerator: Initialized accelerator.
    """
    accelerator_kwargs = dict(kwargs)
    if use_cpu:
        accelerator_kwargs["cpu"] = True
    try:
        return accelerator_factory(**accelerator_kwargs)
    except ValueError as exc:
        if use_cpu and "cpu=True" in str(exc):
            log.warning(
                "trainer.use_cpu=true requested but AcceleratorState is already "
                "initialized on a non-CPU device in this process; continuing with "
                "the existing Accelerate device state."
            )
            accelerator_kwargs.pop("cpu", None)
            return accelerator_factory(**accelerator_kwargs)
        raise


def validate_muon_distributed_compatibility(
    *,
    accelerator: Accelerator,
    optimizer_name: str,
    log: logging.Logger,
    context: str,
) -> None:
    """Validate MuonClip compatibility for the active distributed runtime.

    MuonClip supports FSDP2 owner-compute orthogonalization for single-node
    sharded training. FSDP v1 and DeepSpeed ZeRO-2/3 remain unsupported.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param str optimizer_name: Configured optimizer name.
    :param logging.Logger log: Logger for compatibility warnings.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError: If MuonClip is enabled with incompatible sharding.
    """
    optimizer_key = str(optimizer_name).strip().lower()
    if optimizer_key not in {"muonclip", "muon-clip", "muon_clip"}:
        return

    distributed_type = getattr(accelerator, "distributed_type", None)
    if distributed_type is DistributedType.FSDP:
        state = getattr(accelerator, "state", None)
        fsdp_plugin = getattr(state, "fsdp_plugin", None) if state is not None else None
        raw_version = getattr(fsdp_plugin, "fsdp_version", None)
        try:
            fsdp_version = int(raw_version) if raw_version is not None else 1
        except (TypeError, ValueError):
            fsdp_version = 1

        if fsdp_version < 2:
            raise RuntimeError(
                "MuonClip requires FSDP v2 in "
                f"{context}. Detected FSDP v{fsdp_version}; set fsdp_version=2."
            )
        return

    if distributed_type is not DistributedType.DEEPSPEED:
        return

    deepspeed_plugin = getattr(
        getattr(accelerator, "state", None), "deepspeed_plugin", None
    )
    zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
    if zero_stage is None:
        log.warning(
            "MuonClip enabled with DeepSpeed in "
            f"{context}, but ZeRO stage is unknown. Ensure ZeRO stage < 2."
        )
        return
    if int(zero_stage) >= 2:
        raise RuntimeError(
            "MuonClip is not compatible with DeepSpeed ZeRO stage >= 2 in "
            f"{context} (sharded grads/params). Use ZeRO stage 0/1 or disable MuonClip."
        )


def _maybe_prepare_for_forward(
    optimizer: Any,
    *,
    update_step: int,
    is_last_microbatch: bool,
) -> None:
    """Invoke MuonClip hook gating if supported by the optimizer.

    :param Any optimizer: Optimizer instance (possibly wrapped).
    :param int update_step: Current optimizer update step.
    :param bool is_last_microbatch: Whether this microbatch will sync gradients.
    """
    inner = _unwrap_optimizer(optimizer)
    fn = getattr(inner, "prepare_for_forward", None)
    if fn is None:
        return
    fn(update_step=int(update_step), is_last_microbatch=bool(is_last_microbatch))


def _maybe_compile_model(
    model: torch.nn.Module,
    cfg: Any,
    accelerator: Accelerator,
    log: logging.Logger,
) -> torch.nn.Module:
    """Apply torch.compile if configured and compatible.

    :param torch.nn.Module model: Model to compile.
    :param Any cfg: Training config with ``trainer.torch_compile``.
    :param Accelerator accelerator: Accelerator instance.
    :param logging.Logger log: Logger for warnings/info.
    :return torch.nn.Module: Possibly compiled model.
    """
    if not getattr(cfg.trainer, "torch_compile", False):
        return model
    if not hasattr(torch, "compile"):
        log.warning(
            "trainer.torch_compile is enabled but torch.compile is unavailable; skipping."
        )
        return model
    if accelerator.distributed_type is DistributedType.DEEPSPEED:
        log.warning(
            "trainer.torch_compile is enabled but DeepSpeed is active; skipping torch.compile."
        )
        return model
    compile_backend = str(
        getattr(cfg.trainer, "torch_compile_backend", "inductor")
    ).lower()
    if compile_backend not in {"inductor", "aot_eager", "eager"}:
        log.warning(
            f"Unknown trainer.torch_compile_backend='{compile_backend}'; using 'inductor'."
        )
        compile_backend = "inductor"
    dynamic_override = getattr(cfg.trainer, "torch_compile_dynamic", None)
    if dynamic_override is None:
        # Prefer static-shape compilation by default. In packed mode this avoids
        # aggressive shape-specialization/recompile churn when occasional short
        # batches slip through; users can still opt into dynamic mode explicitly.
        use_dynamic = False
    else:
        use_dynamic = bool(dynamic_override)
    log.info(
        f"Compiling model with torch.compile (backend={compile_backend}, "
        f"dynamic={use_dynamic})."
    )
    return torch.compile(model, backend=compile_backend, dynamic=use_dynamic)


def _resolve_resume_checkpoint(
    resume_from_checkpoint: Optional[str],
    checkpoint_dir: str,
    output_dir: str,
) -> Tuple[Optional[str], int]:
    """Resolve an explicit or latest checkpoint path for resuming.

    :param str | None resume_from_checkpoint: Configured resume value.
    :param str checkpoint_dir: Default checkpoint directory to scan for latest.
    :param str output_dir: Output directory for relative path resolution.
    :return tuple[str | None, int]: Resolved checkpoint path and iteration.
    """
    if not resume_from_checkpoint:
        return None, 0

    checkpoint_dir_path = Path(checkpoint_dir)
    output_dir_path = Path(output_dir)

    if isinstance(resume_from_checkpoint, str):
        resume_value = resume_from_checkpoint.strip()
        if resume_value.lower() not in {"true", "latest", "auto"}:
            resume_path = Path(resume_value)
            if not resume_path.is_absolute():
                candidate = output_dir_path / resume_path
                if candidate.exists():
                    resume_path = candidate
            base = resume_path.name
            iteration = int(base) + 1 if base.isdigit() else 0
            return str(resume_path), iteration

    if not checkpoint_dir_path.exists() or not any(checkpoint_dir_path.iterdir()):
        return None, 0

    folders = [
        folder
        for folder in checkpoint_dir_path.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ]
    if not folders:
        return None, 0

    latest_step = max(
        int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder.name)[0])
        for folder in folders
    )
    return str(checkpoint_dir_path / str(latest_step)), latest_step + 1
