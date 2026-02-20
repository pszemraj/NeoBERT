"""Shared helpers for training loops (pretraining, GLUE, contrastive)."""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

try:
    from torch.distributed.tensor import DTensor, DeviceMesh
except Exception:  # pragma: no cover - import safety for stripped torch builds
    DTensor = None  # type: ignore[assignment]
    DeviceMesh = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
    """Validate optimizer compatibility for the active distributed runtime.

    MuonClip orthogonalizes full 2D tensors. Sharded-parameter modes (notably
    FSDP and DeepSpeed ZeRO-2/3) violate this assumption because each rank holds
    only slices of the matrix.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param str optimizer_name: Configured optimizer name.
    :param logging.Logger log: Logger for compatibility warnings.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError: If optimizer is enabled with incompatible sharding.
    """
    optimizer_key = str(optimizer_name).strip().lower()
    muon_enabled = optimizer_key in {"muonclip", "muon-clip", "muon_clip"}
    dion2_enabled = optimizer_key in {"dion2", "dion-2", "dion_2"}
    if not (muon_enabled or dion2_enabled):
        return

    distributed_type = getattr(accelerator, "distributed_type", None)
    if muon_enabled and distributed_type is DistributedType.FSDP:
        raise RuntimeError(
            "MuonClip is not compatible with FSDP sharded parameters in "
            f"{context}. Use AdamW or disable FSDP for MuonClip runs."
        )
    if dion2_enabled and distributed_type is DistributedType.DEEPSPEED:
        raise RuntimeError(
            "Dion2 is not compatible with DeepSpeed in "
            f"{context}. Use FSDP2 or DDP/no-sharding."
        )

    if not muon_enabled or distributed_type is not DistributedType.DEEPSPEED:
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


def finalize_dion2_distributed_mesh(
    *,
    optimizer: Any,
    log: logging.Logger,
) -> None:
    """Bind Dion2 to the prepared FSDP2 DTensor mesh after ``accelerator.prepare``.

    Dion2 can be instantiated before FSDP wrapping, but when parameters are
    converted to DTensor during prepare-time, the optimizer must use the same
    1D sharded DeviceMesh for communication.

    :param Any optimizer: Optimizer instance (possibly wrapped by Accelerate).
    :param logging.Logger log: Logger for informational messages.
    :raises RuntimeError: If DTensor parameters use an unsupported mesh layout.
    """
    inner = _unwrap_optimizer(optimizer)
    cls = inner.__class__
    is_dion2 = bool(getattr(inner, "_is_neobert_dion2", False)) or (
        cls.__name__ == "Dion2" and cls.__module__.startswith("dion")
    )
    if not is_dion2:
        return

    dtensor_param = None
    for group in getattr(inner, "param_groups", []):
        for param in group.get("params", []):
            if DTensor is not None and isinstance(param, DTensor):
                dtensor_param = param
                break
        if dtensor_param is not None:
            break

    if dtensor_param is None:
        # DDP or single-process execution path; WORLD pg is set in get_optimizer.
        return
    if DeviceMesh is None:
        raise RuntimeError(
            "Detected DTensor parameters for Dion2, but torch.distributed.tensor "
            "DeviceMesh is unavailable in this runtime."
        )

    mesh = dtensor_param.device_mesh
    if not isinstance(mesh, DeviceMesh):
        raise RuntimeError(
            "Detected DTensor parameters for Dion2 but could not resolve a "
            "DeviceMesh from parameter metadata."
        )
    if int(mesh.ndim) != 1:
        raise RuntimeError(
            "Dion2 requires a 1D sharded DeviceMesh under FSDP2. "
            f"Got {mesh.ndim}D mesh {mesh}. "
            "Use the 1D shard sub-mesh for fully_shard()/FSDP2."
        )

    process_group = mesh.get_group()
    if process_group is None:
        raise RuntimeError(
            "Dion2 could not resolve process group from FSDP2 DeviceMesh."
        )
    try:
        device_rank = int(mesh.get_local_rank())
    except TypeError:
        device_rank = int(mesh.get_local_rank(0))
    world_size = int(mesh.size())

    inner._distributed_mesh = mesh
    inner._process_group = process_group
    inner._device_rank = device_rank
    inner._world_size = world_size
    log.info(
        "Dion2 distributed mesh finalized from prepared DTensor parameters "
        f"(world_size={world_size}, local_rank={device_rank})."
    )


def finalize_dion2_qk_clipping_runtime(
    *,
    optimizer: Any,
    model: Any,
    log: logging.Logger,
) -> None:
    """Rebind Dion2 MuonClip QK runtime hooks after ``accelerator.prepare``.

    Dion2's optional MuonClip QK clipping path registers forward hooks and
    parameter references. After model preparation/wrapping, those references
    should be refreshed to point at the active model/parameter objects.

    :param Any optimizer: Optimizer instance (possibly wrapped by Accelerate).
    :param Any model: Prepared model object (preferably unwrapped base module).
    :param logging.Logger log: Logger for informational messages.
    """
    inner = _unwrap_optimizer(optimizer)
    runtime = getattr(inner, "_neobert_dion2_qk_runtime", None)
    if runtime is None:
        return

    rebind = getattr(runtime, "rebind_model", None)
    if not callable(rebind):
        return

    rebind(model)
    log.info("Dion2 MuonClip QK clipping runtime rebound to prepared model.")


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
