"""Shared helpers for training loops (pretraining, GLUE, contrastive)."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DistributedType

logger = logging.getLogger(__name__)

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except Exception:  # pragma: no cover - older torch builds without DTensor APIs
    DTensor = None  # type: ignore[assignment]
    Shard = None  # type: ignore[assignment]

_LOW_PRECISION_LINEAR_PROBE_CACHE: dict[tuple[int, str, str], bool] = {}
_LOCAL_RANK_ENV_VARS = ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID")
_ACCELERATOR_STATE_REINIT_PREFIX = (
    "AcceleratorState has already been initialized and cannot be changed"
)


def _resolve_cuda_probe_device_index() -> int:
    """Resolve a stable CUDA device index for startup dtype probes.

    Before Accelerate initializes per-rank devices, ``torch.cuda.current_device()``
    often defaults to ``0`` on every rank. Prefer rank-local environment metadata
    so probes run on the GPU owned by this process.

    :return int: CUDA device index to probe.
    """
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return 0

    for env_var in _LOCAL_RANK_ENV_VARS:
        raw_value = os.environ.get(env_var)
        if raw_value is None:
            continue
        try:
            local_rank = int(raw_value)
        except ValueError:
            continue
        if 0 <= local_rank < device_count:
            return local_rank

    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = 0
    if 0 <= current_device < device_count:
        return current_device
    return 0


def _probe_cuda_linear_dtype(dtype: torch.dtype) -> bool:
    """Probe whether CUDA linear GEMM works for a low-precision dtype.

    :param torch.dtype dtype: Probe dtype (typically ``torch.bfloat16``).
    :return bool: ``True`` when a tiny linear op succeeds end-to-end.
    """
    if not torch.cuda.is_available():
        return False
    device_index = _resolve_cuda_probe_device_index()
    probe_device = torch.device("cuda", device_index)
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
            x = torch.randn((16, 64), device=probe_device, dtype=dtype)
            w = torch.randn((128, 64), device=probe_device, dtype=dtype)
            _ = torch.nn.functional.linear(x, w)
            torch.cuda.synchronize(device=probe_device)
        _LOW_PRECISION_LINEAR_PROBE_CACHE[cache_key] = True
        return True
    except RuntimeError:
        _LOW_PRECISION_LINEAR_PROBE_CACHE[cache_key] = False
        return False


def stabilize_cuda_mixed_precision(
    *,
    mixed_precision: str,
    log: logging.Logger,
    use_cpu: bool = False,
) -> str:
    """Stabilize mixed precision policy for CUDA runtimes with broken bf16 GEMM.

    Some driver/toolchain/GPU combinations report bf16 support but fail at
    runtime in default cuBLAS GEMM. We first probe bf16 linear; on failure we
    try switching to cuBLASLt, and only then fall back to fp32.

    :param str mixed_precision: Requested mixed precision mode.
    :param logging.Logger log: Logger for runtime policy warnings.
    :param bool use_cpu: Whether the run is explicitly targeting CPU execution.
    :return str: Effective mixed precision mode (``"bf16"`` or ``"no"``).
    """
    if use_cpu or mixed_precision != "bf16" or not torch.cuda.is_available():
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


def resolve_runtime_mixed_precision_and_attn_backend(
    *,
    mixed_precision: str,
    attn_backend: str,
    log: logging.Logger,
    use_cpu: bool = False,
) -> tuple[str, str]:
    """Resolve stable mixed precision and attention backend policy for CUDA.

    :param str mixed_precision: Requested mixed precision mode.
    :param str attn_backend: Requested attention backend.
    :param logging.Logger log: Logger for runtime warnings.
    :param bool use_cpu: Whether the run is explicitly targeting CPU execution.
    :return tuple[str, str]: Effective ``(mixed_precision, attn_backend)``.
    """
    effective_precision = stabilize_cuda_mixed_precision(
        mixed_precision=mixed_precision,
        log=log,
        use_cpu=use_cpu,
    )
    effective_backend = str(attn_backend)
    normalized_backend = effective_backend.strip().lower()
    if use_cpu and normalized_backend == "flash_attn_varlen":
        log.warning(
            "attn_backend='flash_attn_varlen' requires CUDA tensors, but "
            "trainer.use_cpu=true; falling back to attn_backend='sdpa'."
        )
        effective_backend = "sdpa"
        normalized_backend = "sdpa"
    if effective_precision == "no" and normalized_backend == "flash_attn_varlen":
        log.warning(
            "attn_backend='flash_attn_varlen' with mixed_precision='no' is unsupported; "
            "falling back to attn_backend='sdpa'."
        )
        effective_backend = "sdpa"
    return effective_precision, effective_backend


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


def _is_muonclip_optimizer(optimizer_name: str) -> bool:
    """Return whether ``optimizer_name`` selects MuonClip.

    :param str optimizer_name: Configured optimizer name.
    :return bool: ``True`` when MuonClip is selected.
    """
    optimizer_key = str(optimizer_name).strip().lower()
    return optimizer_key in {"muonclip", "muon-clip", "muon_clip"}


def _looks_like_dtensor(param: Any) -> bool:
    """Return whether ``param`` exposes DTensor-like metadata.

    :param Any param: Parameter candidate.
    :return bool: ``True`` when the object behaves like a DTensor.
    """
    if DTensor is not None and isinstance(param, DTensor):
        return True
    return (
        hasattr(param, "device_mesh")
        and hasattr(param, "placements")
        and callable(getattr(param, "to_local", None))
    )


def _is_row_shard_placement(placement: Any) -> bool:
    """Return whether ``placement`` represents ``Shard(0)``.

    :param Any placement: DTensor placement descriptor.
    :return bool: ``True`` when the placement is a row shard.
    """
    if Shard is not None and isinstance(placement, Shard):
        return int(getattr(placement, "dim", -1)) == 0

    placement_name = type(placement).__name__.lower()
    shard_dim = getattr(placement, "dim", None)
    try:
        return placement_name.endswith("shard") and int(shard_dim) == 0
    except (TypeError, ValueError):
        return False


def _is_accelerator_state_reinit_error(exc: Exception) -> bool:
    """Return whether ``exc`` indicates stale Accelerate singleton state.

    :param Exception exc: Exception raised while constructing ``Accelerator``.
    :return bool: ``True`` when Accelerate requests a runtime restart.
    """
    return isinstance(exc, ValueError) and (
        _ACCELERATOR_STATE_REINIT_PREFIX in str(exc)
    )


def _reset_accelerate_runtime_state() -> None:
    """Reset Accelerate singleton state for sequential in-process trainer reuse."""
    GradientState._reset_state()
    AcceleratorState._reset_state(reset_partial_state=True)


def create_accelerator(
    *,
    use_cpu: bool,
    log: logging.Logger,
    accelerator_factory: Callable[..., Accelerator] = Accelerator,
    **kwargs: Any,
) -> Accelerator:
    """Create an Accelerator and handle stale Accelerate singleton state.

    In long-lived processes (tests, notebooks, agent loops), a previous trainer
    invocation may leave Accelerate singleton state initialized with different
    runtime settings such as ``cpu=True`` or ``mixed_precision='bf16'``. When
    that happens, we reset Accelerate's shared state and recreate the
    accelerator so the new run honors its requested runtime policy.

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
        if _is_accelerator_state_reinit_error(exc):
            log.warning(
                "AcceleratorState is already initialized with incompatible runtime "
                "settings for this process (requested cpu=%s, mixed_precision=%r). "
                "Resetting Accelerate singleton state and recreating the accelerator.",
                bool(accelerator_kwargs.get("cpu", False)),
                accelerator_kwargs.get("mixed_precision"),
            )
            _reset_accelerate_runtime_state()
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

    MuonClip supports distributed execution only through the FSDP2
    owner-compute path used in this repo. FSDP v1 and all DeepSpeed runtimes
    remain unsupported.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param str optimizer_name: Configured optimizer name.
    :param logging.Logger log: Logger for compatibility warnings.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError: If MuonClip is enabled with incompatible sharding.
    """
    if not _is_muonclip_optimizer(optimizer_name):
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

        parallelism_config = getattr(accelerator, "parallelism_config", None)
        if parallelism_config is None and state is not None:
            parallelism_config = getattr(state, "parallelism_config", None)
        enabled_axes = [
            axis_name
            for axis_name, attr_name in (
                ("tensor parallelism", "tp_enabled"),
                ("context parallelism", "cp_enabled"),
            )
            if bool(getattr(parallelism_config, attr_name, False))
        ]
        if enabled_axes:
            axes = ", ".join(enabled_axes)
            raise RuntimeError(
                "MuonClip FSDP v2 currently supports only a 1D row-sharded device "
                f"mesh in {context}. Disable {axes} for MuonClip runs."
            )
        return

    if distributed_type is not DistributedType.DEEPSPEED:
        return

    deepspeed_plugin = getattr(
        getattr(accelerator, "state", None), "deepspeed_plugin", None
    )
    zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
    zero_suffix = ""
    if zero_stage is not None:
        zero_suffix = f" (ZeRO stage {int(zero_stage)})"
    raise RuntimeError(
        "MuonClip distributed mode is FSDP2-only in "
        f"{context}; DeepSpeed{zero_suffix} is not supported. "
        "Use Accelerate FSDP v2 or switch optimizers."
    )


def validate_distributed_runtime_policy(
    *,
    accelerator: Accelerator,
    log: logging.Logger,
    context: str,
) -> None:
    """Reject distributed runtimes that this repo no longer supports.

    DeepSpeed execution support has been removed in favor of Accelerate-managed
    FSDP2 paths. Legacy DeepSpeed checkpoint conversion remains supported
    separately via checkpoint-loading helpers.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param logging.Logger log: Logger for policy warnings/errors.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError: If DeepSpeed is selected as the active runtime backend.
    """
    distributed_type = getattr(accelerator, "distributed_type", None)
    if distributed_type is not DistributedType.DEEPSPEED:
        return

    deepspeed_plugin = getattr(
        getattr(accelerator, "state", None), "deepspeed_plugin", None
    )
    zero_stage = getattr(deepspeed_plugin, "zero_stage", None)
    zero_suffix = ""
    if zero_stage is not None:
        zero_suffix = f" (ZeRO stage {int(zero_stage)})"
    raise RuntimeError(
        "DeepSpeed runtime is unsupported in "
        f"{context}{zero_suffix}. Use Accelerate FSDP v2 for distributed runs; "
        "legacy DeepSpeed checkpoint conversion remains available separately."
    )


def validate_muon_runtime_topology(
    *,
    accelerator: Accelerator,
    optimizer: Any,
    optimizer_name: str,
    log: logging.Logger,
    context: str,
) -> None:
    """Validate prepared MuonClip DTensor topology after ``accelerator.prepare()``.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param Any optimizer: Prepared optimizer (possibly wrapped by Accelerate).
    :param str optimizer_name: Configured optimizer name.
    :param logging.Logger log: Logger for topology warnings.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError:
        If prepared MuonClip params use unsupported DTensor layout, or if a
        multi-process FSDP2 run failed to expose DTensor Muon parameters at all.
    """
    if not _is_muonclip_optimizer(optimizer_name):
        return
    if getattr(accelerator, "distributed_type", None) is not DistributedType.FSDP:
        return

    inner = _unwrap_optimizer(optimizer)
    saw_dtensor = False
    for group in getattr(inner, "param_groups", ()):
        if not group.get("use_muon", False):
            continue

        for param in group.get("params", ()):
            if not _looks_like_dtensor(param):
                continue
            saw_dtensor = True

            mesh = getattr(param, "device_mesh", None)
            if mesh is None:
                raise RuntimeError(
                    "MuonClip encountered a DTensor-like FSDP2 parameter without a "
                    f"device mesh in {context}."
                )

            mesh_ndim = getattr(mesh, "ndim", None)
            if mesh_ndim is None:
                log.warning(
                    "MuonClip could not determine FSDP2 device_mesh.ndim in %s; "
                    "continuing because runtime topology metadata is incomplete.",
                    context,
                )
            elif int(mesh_ndim) != 1:
                raise RuntimeError(
                    "MuonClip FSDP v2 currently supports only 1D row-sharded device "
                    f"meshes in {context}; got device_mesh.ndim={int(mesh_ndim)}."
                )

            placements = tuple(getattr(param, "placements", ()))
            if len(placements) != 1 or not _is_row_shard_placement(placements[0]):
                raise RuntimeError(
                    "MuonClip FSDP v2 currently supports only Shard(0) DTensor "
                    f"placements in {context}; got placements={placements!r}."
                )

    if (
        getattr(accelerator, "num_processes", 1) > 1
        and any(
            group.get("use_muon", False) for group in getattr(inner, "param_groups", ())
        )
        and not saw_dtensor
    ):
        raise RuntimeError(
            "MuonClip expected DTensor Muon parameters after accelerator.prepare() "
            f"in {context}, but none were observed. Refusing to continue because "
            "the distributed owner-compute path would be inactive."
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
