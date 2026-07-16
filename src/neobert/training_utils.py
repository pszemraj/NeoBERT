"""Shared helpers for training loops (pretraining, GLUE, contrastive)."""

import json
import logging
import os
import signal
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import wandb
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import DataLoaderConfiguration, DistributedType

from neobert.checkpointing import (
    OPTIMIZER_PARAM_NAMES_MANIFEST,
    checkpoint_resume_errors,
    invalidate_checkpoint_completion,
    is_resumable_checkpoint,
    is_step_checkpoint_name,
    mark_checkpoint_complete,
    optimizer_param_name_manifest_schema_errors,
    save_accelerate_state,
    save_portable_checkpoint_weights,
    strip_runtime_prefixes,
)
from neobert.distributed import is_dtensor_like, is_row_shard_placement

try:
    from transformers import BatchEncoding
except Exception:  # pragma: no cover - transformers import should succeed in repo env
    BatchEncoding = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_ACCELERATOR_STATE_REINIT_PREFIX = (
    "AcceleratorState has already been initialized and cannot be changed"
)


@dataclass
class PreemptionState:
    """Signal-safe termination intent synchronized at optimizer boundaries."""

    requested_signum: int = 0

    def request(self, signum: int, frame: FrameType | None) -> None:
        """Record preemption intent without I/O, CUDA work, or collectives.

        :param int signum: Received signal number.
        :param FrameType | None frame: Interrupted Python frame.
        """
        del frame
        self.requested_signum = int(signum)

    def synchronize(self, accelerator: Any) -> bool:
        """Return whether any rank requested preemption.

        :param Any accelerator: Active Accelerator instance.
        :return bool: True when at least one rank requested termination.
        """
        if accelerator.num_processes == 1:
            return self.requested_signum != 0

        # Every rank must participate on every completed update: gating this
        # collective on local state would deadlock when only one rank receives SIGTERM.
        local_request = torch.tensor(
            int(self.requested_signum != 0),
            device=accelerator.device,
            dtype=torch.int32,
        )
        requests = accelerator.reduce(local_request, reduction="sum")
        return bool(int(requests.item()))


@contextmanager
def preserve_sigterm_handler() -> Iterator[None]:
    """Restore the process SIGTERM handler after a training entry point exits.

    :return collections.abc.Iterator[None]: Context-managed trainer execution.
    """
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def build_dataloader_config(
    *,
    seed: int,
    dispatch_batches: bool | None = None,
) -> DataLoaderConfiguration:
    """Build deterministic Accelerate dataloader settings shared by trainers.

    :param int seed: Global seed used by Accelerate's seedable sampler.
    :param bool | None dispatch_batches: Optional dispatch policy override.
    :return DataLoaderConfiguration: Deterministic dataloader configuration.
    """
    return DataLoaderConfiguration(
        dispatch_batches=dispatch_batches,
        use_seedable_sampler=True,
        data_seed=int(seed),
    )


def initialize_wandb_trackers(
    *,
    cfg: Any,
    accelerator: Any,
    tracker_config: dict[str, Any],
    log: logging.Logger,
) -> None:
    """Initialize W&B tracking and upload the source configuration when present.

    :param Any cfg: Runtime configuration with a ``wandb`` section.
    :param Any accelerator: Active Accelerator instance.
    :param dict[str, Any] tracker_config: Resolved configuration payload.
    :param logging.Logger log: Logger used for missing source-config warnings.
    """
    Path(cfg.wandb.dir).mkdir(parents=True, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "name": cfg.wandb.name,
                "entity": cfg.wandb.entity,
                "config": tracker_config,
                "tags": cfg.wandb.tags,
                "dir": cfg.wandb.dir,
                "mode": cfg.wandb.mode,
                "resume": cfg.wandb.resume,
            }
        },
    )
    if not accelerator.is_main_process or wandb.run is None:
        return
    wandb.run.config.update(tracker_config, allow_val_change=True)
    config_path = getattr(cfg, "config_path", None)
    if not config_path:
        return
    abs_config_path = Path(config_path).expanduser().resolve()
    if not abs_config_path.is_file():
        log.warning(
            "Configured config_path '%s' not found; skipping wandb artifact upload",
            config_path,
        )
        return
    artifact = wandb.Artifact(
        name=f"{wandb.run.id}-config",
        type="config",
        metadata={"source": str(abs_config_path)},
    )
    artifact.add_file(str(abs_config_path))
    wandb.run.log_artifact(artifact)


def resolve_runtime_mixed_precision_and_attn_backend(
    *,
    mixed_precision: str,
    attn_backend: str,
    log: logging.Logger,
    use_cpu: bool = False,
) -> tuple[str, str]:
    """Resolve attention backend policy that depends on runtime precision/CPU.

    :param str mixed_precision: Requested mixed precision mode.
    :param str attn_backend: Requested attention backend.
    :param logging.Logger log: Logger for runtime warnings.
    :param bool use_cpu: Whether the run is explicitly targeting CPU execution.
    :return tuple[str, str]: Effective ``(mixed_precision, attn_backend)``.
    """
    effective_precision = str(mixed_precision)
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


def _resolve_cuda_pin_memory(
    requested_pin_memory: bool,
    *,
    device: torch.device,
) -> tuple[bool, list[str]]:
    """Resolve effective pinned CPU staging for training/eval dataloaders.

    NeoBERT keeps pinned host buffers enabled on CUDA so both automatic
    device-placement paths and manual non-blocking H2D copies can overlap
    transfers with compute. Call sites can choose whether that staging happens
    inside the ``DataLoader`` or via a final batch repin just before transfer.

    :param bool requested_pin_memory: User-configured pinned staging toggle.
    :param torch.device device: Active accelerator device.
    :return tuple[bool, list[str]]: Effective setting plus informational notes.
    """
    pin_memory = bool(requested_pin_memory)
    notes: list[str] = []
    if device.type == "cuda" and not pin_memory:
        pin_memory = True
        notes.append(
            "dataset.pin_memory was false; enabling pinned CPU staging on CUDA "
            "to improve host->device transfer overlap."
        )
    return pin_memory, notes


def _pin_cpu_tensors(value: Any) -> Any:
    """Recursively pin CPU tensors for non-blocking host-to-device copies.

    :param Any value: Tensor, nested container, or scalar to pin.
    :return Any: Value with CPU tensors pinned when supported.
    """

    def _pin(inner: Any) -> tuple[Any, bool]:
        """Pin a nested value and report whether anything changed.

        :param Any inner: Candidate tensor/container/scalar.
        :return tuple[Any, bool]: Pinned value and whether a change was made.
        """
        if torch.is_tensor(inner):
            if inner.device.type != "cpu" or inner.is_pinned():
                return inner, False
            return inner.pin_memory(), True

        if BatchEncoding is not None and isinstance(inner, BatchEncoding):
            updated_data: dict[Any, Any] = {}
            changed = False
            for key, nested in inner.items():
                pinned_nested, nested_changed = _pin(nested)
                updated_data[key] = pinned_nested
                changed = changed or nested_changed
            if not changed:
                return inner, False
            return (
                BatchEncoding(
                    data=updated_data,
                    encoding=inner.encodings,
                    n_sequences=inner.n_sequences,
                ),
                True,
            )

        if isinstance(inner, Mapping):
            updated: dict[Any, Any] = {}
            changed = False
            for key, nested in inner.items():
                pinned_nested, nested_changed = _pin(nested)
                updated[key] = pinned_nested
                changed = changed or nested_changed
            if not changed:
                return inner, False
            if isinstance(inner, dict):
                return updated, True
            try:
                return type(inner)(updated), True
            except TypeError:
                return updated, True

        if isinstance(inner, list):
            updated_list: list[Any] = []
            changed = False
            for nested in inner:
                pinned_nested, nested_changed = _pin(nested)
                updated_list.append(pinned_nested)
                changed = changed or nested_changed
            if not changed:
                return inner, False
            return updated_list, True

        if isinstance(inner, tuple):
            updated_items: list[Any] = []
            changed = False
            for nested in inner:
                pinned_nested, nested_changed = _pin(nested)
                updated_items.append(pinned_nested)
                changed = changed or nested_changed
            if not changed:
                return inner, False
            return tuple(updated_items), True

        return inner, False

    pinned_value, _ = _pin(value)
    return pinned_value


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


def _placement_requires_norm_reduction(placement: Any) -> bool:
    """Return whether a DTensor placement contributes only a local partial norm.

    :param Any placement: DTensor placement descriptor.
    :return bool: ``True`` when values must be reduced across ranks.
    """
    if is_row_shard_placement(placement):
        return True

    placement_name = type(placement).__name__.lower()
    return placement_name.endswith("shard") or placement_name.endswith("partial")


def _dtensor_requires_norm_reduction(value: Any) -> bool:
    """Return whether a DTensor-like value needs cross-rank norm reduction.

    :param Any value: DTensor-like tensor or parameter.
    :return bool: ``True`` when local values represent only a shard/partial.
    """
    placements = tuple(getattr(value, "placements", ()))
    return any(
        _placement_requires_norm_reduction(placement) for placement in placements
    )


def _tensor_l2_sumsq(value: torch.Tensor) -> torch.Tensor:
    """Compute a numerically stable squared L2 contribution for one local tensor.

    :param torch.Tensor value: Local tensor or shard to accumulate.
    :return torch.Tensor: Scalar squared-norm contribution on ``value.device``.
    """
    tensor = value.detach()
    if tensor.is_sparse:
        tensor = tensor.coalesce().values()
    return tensor.double().pow(2).sum()


def _accumulate_scalar_norm(
    accumulator: Optional[torch.Tensor],
    contribution: torch.Tensor,
) -> torch.Tensor:
    """Accumulate scalar norm contributions while preserving device placement.

    :param torch.Tensor | None accumulator: Existing scalar accumulator.
    :param torch.Tensor contribution: New scalar contribution to add.
    :return torch.Tensor: Updated accumulator tensor.
    """
    if accumulator is None:
        return contribution
    return accumulator + contribution.to(device=accumulator.device)


def _compute_l2_norm_for_logging(
    parameters: Iterable[Any],
    accelerator: Accelerator,
    *,
    grad: bool = False,
) -> Optional[float]:
    """Compute a global L2 norm for parameters or gradients in logging paths.

    FSDP2 exposes sharded parameters as DTensors. Their local tensor values are
    only partial shards, so logging must sum squared local contributions and
    reduce only the sharded subset across ranks. Replicated tensors are kept
    local so their contributions are not over-counted.

    :param Iterable[Any] parameters: Parameter-like objects to inspect.
    :param Accelerator accelerator: Active accelerator runtime.
    :param bool grad: Whether to read ``param.grad`` instead of the parameter.
    :return float | None: Global L2 norm or ``None`` when no tensors are present.
    """
    fsdp_multi_process = (
        getattr(accelerator, "distributed_type", None) is DistributedType.FSDP
        and int(getattr(accelerator, "num_processes", 1)) > 1
    )
    local_sumsq: Optional[torch.Tensor] = None
    sharded_sumsq: Optional[torch.Tensor] = None
    saw_tensor = False

    for param in parameters:
        value = getattr(param, "grad", None) if grad else param
        if value is None:
            continue

        local_value = value.to_local() if is_dtensor_like(value) else value.detach()
        if not torch.is_tensor(local_value):
            continue

        saw_tensor = True
        contribution = _tensor_l2_sumsq(local_value)
        requires_reduction = False
        if is_dtensor_like(value):
            requires_reduction = _dtensor_requires_norm_reduction(value)
        elif is_dtensor_like(param):
            # FSDP2 gradients are typically local tensors attached to DTensor params.
            requires_reduction = _dtensor_requires_norm_reduction(param)
        elif fsdp_multi_process:
            requires_reduction = True

        if requires_reduction:
            sharded_sumsq = _accumulate_scalar_norm(sharded_sumsq, contribution)
        else:
            local_sumsq = _accumulate_scalar_norm(local_sumsq, contribution)

    if not saw_tensor:
        return None

    if sharded_sumsq is not None and fsdp_multi_process:
        reduce_fn = getattr(accelerator, "reduce", None)
        if callable(reduce_fn):
            sharded_sumsq = reduce_fn(sharded_sumsq, reduction="sum")
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sharded_sumsq)

    total_sumsq = local_sumsq
    if sharded_sumsq is not None:
        total_sumsq = _accumulate_scalar_norm(total_sumsq, sharded_sumsq)

    if total_sumsq is None:
        return None
    return float(total_sumsq.sqrt().item())


def _update_global_norm_metric_for_logging(
    metrics: dict[str, Any],
    *,
    key: str,
    parameters: Iterable[Any],
    accelerator: Accelerator,
    enabled: bool,
    grad: bool = False,
) -> None:
    """Collect a norm metric on all ranks but only emit it on the main process.

    FSDP-aware norm helpers may execute collectives, so every rank must
    participate even when only rank 0 should publish the resulting metric.

    :param dict[str, Any] metrics: Mutable metrics mapping to update in place.
    :param str key: Metric key to populate or clear.
    :param Iterable[Any] parameters: Parameters/gradients to inspect.
    :param Accelerator accelerator: Active accelerator runtime.
    :param bool enabled: Whether this metric is enabled for the current window.
    :param bool grad: Whether to read gradients instead of parameter values.
    """
    if not enabled:
        metrics.pop(key, None)
        return

    norm_value = _compute_l2_norm_for_logging(
        parameters,
        accelerator,
        grad=grad,
    )
    if accelerator.is_main_process and norm_value is not None:
        metrics[key] = norm_value
    else:
        metrics.pop(key, None)


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


def _maybe_set_local_cuda_device(*, use_cpu: bool, log: logging.Logger) -> None:
    """Bind the current process to its LOCAL_RANK CUDA device before init.

    :param bool use_cpu: Whether the run is explicitly targeting CPU execution.
    :param logging.Logger log: Logger for malformed-rank warnings.
    """
    if use_cpu or not torch.cuda.is_available():
        return

    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None:
        return

    try:
        torch.cuda.set_device(int(local_rank_raw))
    except (TypeError, ValueError):
        log.warning(
            "Ignoring invalid LOCAL_RANK=%r while binding the CUDA device.",
            local_rank_raw,
        )


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
    _maybe_set_local_cuda_device(use_cpu=use_cpu, log=log)
    try:
        accelerator = accelerator_factory(**accelerator_kwargs)
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
            accelerator = accelerator_factory(**accelerator_kwargs)
        else:
            raise

    device = getattr(accelerator, "device", None)
    if use_cpu and getattr(device, "type", None) == "cuda":
        log.warning(
            "Accelerator returned CUDA device despite trainer.use_cpu=true, likely "
            "because stale Accelerate singleton state was reused. Resetting state "
            "and recreating the accelerator."
        )
        _reset_accelerate_runtime_state()
        accelerator = accelerator_factory(**accelerator_kwargs)
        device = getattr(accelerator, "device", None)
        if getattr(device, "type", None) == "cuda":
            raise RuntimeError(
                "Accelerator still resolved to CUDA after resetting state while "
                "trainer.use_cpu=true."
            )

    return accelerator


def resolve_fsdp_version(accelerator: Accelerator) -> int:
    """Resolve FSDP version from Accelerate state.

    Missing or malformed plugin metadata is treated as FSDP v1 so callers fail
    closed when they require FSDP2 behavior.

    :param Accelerator accelerator: Active Accelerator runtime.
    :return int: FSDP plugin version.
    """
    state = getattr(accelerator, "state", None)
    fsdp_plugin = getattr(state, "fsdp_plugin", None) if state is not None else None
    raw_version = getattr(fsdp_plugin, "fsdp_version", None)
    try:
        return int(raw_version) if raw_version is not None else 1
    except (TypeError, ValueError):
        return 1


def validate_muon_distributed_compatibility(
    *,
    accelerator: Accelerator,
    optimizer_name: str,
    context: str,
) -> None:
    """Validate MuonClip compatibility for the active distributed runtime.

    MuonClip supports FSDP execution only through the FSDP2 owner-compute path
    used in this repo. The repository-wide runtime policy rejects unsupported
    distributed backends separately.

    :param Accelerator accelerator: Active Accelerator runtime.
    :param str optimizer_name: Configured optimizer name.
    :param str context: Human-readable task context for error messages.
    :raises RuntimeError: If MuonClip is enabled with incompatible sharding.
    """
    if not _is_muonclip_optimizer(optimizer_name):
        return

    distributed_type = getattr(accelerator, "distributed_type", None)
    if distributed_type is DistributedType.FSDP:
        state = getattr(accelerator, "state", None)
        fsdp_version = resolve_fsdp_version(accelerator)

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


def validate_distributed_runtime_policy(
    *,
    accelerator: Accelerator,
    context: str,
) -> None:
    """Reject distributed runtimes that this repo no longer supports.

    DeepSpeed execution support has been removed in favor of Accelerate-managed
    FSDP2 paths. Legacy DeepSpeed checkpoint conversion remains supported
    separately via checkpoint-loading helpers.

    :param Accelerator accelerator: Active Accelerator runtime.
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
            if not is_dtensor_like(param):
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
            if len(placements) != 1 or not is_row_shard_placement(placements[0]):
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


def _copy_checkpoint_config_fields(
    dst_obj: Any,
    src_obj: Any,
    fields: tuple[str, ...],
    *,
    section: str,
) -> list[str]:
    """Copy selected config fields and report changed fully qualified names.

    :param Any dst_obj: Mutable runtime config section.
    :param Any src_obj: Checkpoint config section.
    :param tuple[str, ...] fields: Field names to copy.
    :param str section: Human-readable section prefix.
    :return list[str]: Changed ``section.field`` names.
    """
    changed: list[str] = []
    for field_name in fields:
        if not hasattr(dst_obj, field_name) or not hasattr(src_obj, field_name):
            continue
        checkpoint_value = deepcopy(getattr(src_obj, field_name))
        if getattr(dst_obj, field_name) != checkpoint_value:
            changed.append(f"{section}.{field_name}")
        setattr(dst_obj, field_name, checkpoint_value)
    return changed


def _report_launch_controlled_config_drift(
    dst_obj: Any,
    src_obj: Any,
    fields: tuple[str, ...],
    *,
    section: str,
) -> list[str]:
    """Report launch-config fields that differ from the checkpoint without overriding.

    Used for resume fields the operator legitimately changes on a continuation
    pass - a different or annealing corpus, or a longer context window. The
    launch config's value is kept as-is; drift is only surfaced for logging so
    the change is visible but not silently reverted.

    :param Any dst_obj: Mutable runtime config section (left unchanged).
    :param Any src_obj: Checkpoint config section.
    :param tuple[str, ...] fields: Field names to compare.
    :param str section: Human-readable section prefix.
    :return list[str]: Drifted ``section.field`` names.
    """
    drifted: list[str] = []
    for field_name in fields:
        if not hasattr(dst_obj, field_name) or not hasattr(src_obj, field_name):
            continue
        if getattr(dst_obj, field_name) != getattr(src_obj, field_name):
            drifted.append(f"{section}.{field_name}")
    return drifted


def sync_resume_source_of_truth(
    cfg: Any,
    resume_checkpoint_path: str | Path | None,
    *,
    task: str,
    log: logging.Logger,
) -> set[str]:
    """Make checkpoint metadata authoritative for resume-sensitive config fields.

    A resumable checkpoint contains optimizer and scheduler state, so silently
    combining it with a different tokenizer, model shape, masking contract, or
    contrastive objective is not a faithful continuation. Trainer controls that
    do not define saved state (precision, compile flags, loss path, ``max_steps``,
    logging cadence, ...) remain launch-controlled. Cursor-sensitive geometry is
    checkpoint-controlled: pretraining forces its per-device batch size, while
    GLUE forces its seed, per-device batch size, gradient accumulation,
    optimizer/scheduler construction, and task/head semantics so loaded state
    and the saved epoch/microbatch cursor retain their meaning.

    Continuation data sources are likewise launch-controlled (the launch config
    wins, drift is warned but not reverted): the training corpus identity
    (``dataset.name``/``config``/``path``/``text_column``) and the optional
    contrastive SimCSE source (``contrastive.pretraining_dataset_path``), which
    never touches checkpointed model or optimizer state, and - for RoPE models
    only - the context window (``model.max_position_embeddings``,
    ``tokenizer.max_length``, ``dataset.max_seq_length``), which RoPE makes
    weight-compatible. This supports continued pretraining (annealing to a new
    corpus, or extending context) without silently undoing the operator's
    intent. When pretraining switches corpus identity, its loader/split/eval
    selection also stays launch-controlled because the data cursor is reset to
    zero; restoring those fields from the old corpus could select nonexistent
    splits on the new source. Non-RoPE sequence length stays
    checkpoint-authoritative because it sizes a learned positional table and
    would break the strict weight load.

    :param Any cfg: Mutable runtime config.
    :param str | Path | None resume_checkpoint_path: Resolved checkpoint step path.
    :param str task: Training task name.
    :param logging.Logger log: Logger for drift warnings.
    :return set[str]: Launch-controlled fields that differ from the checkpoint.
    :raises RuntimeError: If checkpoint ``config.yaml`` is missing.
    """
    if resume_checkpoint_path is None:
        return set()

    from neobert.config import ConfigLoader

    checkpoint_path = Path(resume_checkpoint_path)
    checkpoint_config_path = checkpoint_path / "config.yaml"
    if not checkpoint_config_path.is_file():
        raise RuntimeError(
            f"{checkpoint_config_path} is missing; refusing to resume with current "
            "tokenizer/model/objective settings."
        )

    checkpoint_cfg = ConfigLoader.load(str(checkpoint_config_path))
    if checkpoint_cfg.task != task:
        raise RuntimeError(
            f"Checkpoint task {checkpoint_cfg.task!r} does not match requested "
            f"resume task {task!r}."
        )
    changed: list[str] = []

    # Sequence-length geometry (context window) is operator-controlled on resume
    # only for RoPE models: RoPE has no learned positional-embedding table, so
    # lengthening context on a continuation pass is weight-compatible and changes
    # no checkpointed parameter. Non-RoPE models size a positional table by
    # length, so a length change would break the strict weight load - keep it
    # forced there.
    rope_enabled = bool(
        getattr(checkpoint_cfg.model, "rope", False)
        and getattr(cfg.model, "rope", False)
    )
    corpus_identity_fields = ("name", "config", "path", "text_column")
    corpus_identity_drift = _report_launch_controlled_config_drift(
        cfg.dataset,
        checkpoint_cfg.dataset,
        corpus_identity_fields,
        section="dataset",
    )
    pretraining_corpus_changed = bool(task == "pretraining" and corpus_identity_drift)

    model_forced = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "rope",
        "rms_norm",
        "hidden_act",
        "dropout_prob",
        "norm_eps",
        "embedding_init_range",
        "decoder_init_range",
        "classifier_init_range",
        "attn_backend",
        "kernel_backend",
        "pad_token_id",
    ]
    tokenizer_forced = [
        "name",
        "path",
        "padding",
        "truncation",
        "trust_remote_code",
        "revision",
        "allow_special_token_rewrite",
    ]
    dataset_forced = [
        "trust_remote_code",
        "streaming",
        "validation_split",
        "train_split",
        "eval_split",
        "eval_samples",
        "shuffle_buffer_size",
        "load_all_from_disk",
        "min_length",
        "alpha",
    ]
    corpus_selection_fields = (
        "trust_remote_code",
        "streaming",
        "validation_split",
        "train_split",
        "eval_split",
        "eval_samples",
        "shuffle_buffer_size",
    )
    if pretraining_corpus_changed:
        dataset_forced = [
            field for field in dataset_forced if field not in corpus_selection_fields
        ]
    if not rope_enabled:
        model_forced.append("max_position_embeddings")
        tokenizer_forced.append("max_length")
        dataset_forced.append("max_seq_length")

    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.model,
            checkpoint_cfg.model,
            tuple(model_forced),
            section="model",
        )
    )
    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.tokenizer,
            checkpoint_cfg.tokenizer,
            tuple(tokenizer_forced),
            section="tokenizer",
        )
    )
    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.dataset,
            checkpoint_cfg.dataset,
            tuple(dataset_forced),
            section="dataset",
        )
    )
    datacollator_forced = [
        "mlm_probability",
        "pad_to_multiple_of",
        "mask_all",
        "pack_sequences",
    ]
    if not rope_enabled:
        datacollator_forced.append("max_length")

    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.datacollator,
            checkpoint_cfg.datacollator,
            tuple(datacollator_forced),
            section="datacollator",
        )
    )
    if task == "pretraining":
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.trainer,
                checkpoint_cfg.trainer,
                ("per_device_train_batch_size",),
                section="trainer",
            )
        )
    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.optimizer,
            checkpoint_cfg.optimizer,
            ("name", "lr", "weight_decay", "betas", "eps", "muon_config"),
            section="optimizer",
        )
    )
    changed.extend(
        _copy_checkpoint_config_fields(
            cfg.scheduler,
            checkpoint_cfg.scheduler,
            (
                "name",
                "warmup_steps",
                "total_steps",
                "decay_steps",
                "final_lr_ratio",
                "warmup_percent",
                "decay_percent",
            ),
            section="scheduler",
        )
    )
    if task == "contrastive":
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.contrastive,
                checkpoint_cfg.contrastive,
                (
                    "temperature",
                    "pooling",
                    "pretraining_prob",
                ),
                section="contrastive",
            )
        )
    if task == "glue":
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.model,
                checkpoint_cfg.model,
                ("name", "from_hub", "max_position_embeddings"),
                section="model",
            )
        )
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.tokenizer,
                checkpoint_cfg.tokenizer,
                ("max_length",),
                section="tokenizer",
            )
        )
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.glue,
                checkpoint_cfg.glue,
                (
                    "task_name",
                    "num_labels",
                    "max_seq_length",
                    "classifier_dropout",
                    "classifier_init_range",
                    "allow_random_weights",
                ),
                section="glue",
            )
        )
        changed.extend(
            _copy_checkpoint_config_fields(
                cfg.trainer,
                checkpoint_cfg.trainer,
                (
                    "per_device_train_batch_size",
                    "gradient_accumulation_steps",
                ),
                section="trainer",
            )
        )
        if int(cfg.seed) != int(checkpoint_cfg.seed):
            changed.append("seed")
        cfg.seed = int(checkpoint_cfg.seed)

    # Operator-controlled resume fields: the launch config wins so a deliberate
    # continuation change is honored instead of silently reverted. Corpus
    # identity never touches checkpointed model/optimizer state; sequence length
    # is included only when RoPE makes it weight-compatible (see above). Drift is
    # surfaced as a warning, not overridden.
    operator_controlled: list[str] = list(
        _report_launch_controlled_config_drift(
            cfg.dataset,
            checkpoint_cfg.dataset,
            ("name", "config", "path", "cache_dir", "text_column"),
            section="dataset",
        )
    )
    if task == "contrastive":
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.contrastive,
                checkpoint_cfg.contrastive,
                ("pretraining_dataset_path",),
                section="contrastive",
            )
        )
    if pretraining_corpus_changed:
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.dataset,
                checkpoint_cfg.dataset,
                corpus_selection_fields,
                section="dataset",
            )
        )
    if rope_enabled:
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.model,
                checkpoint_cfg.model,
                ("max_position_embeddings",),
                section="model",
            )
        )
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.tokenizer,
                checkpoint_cfg.tokenizer,
                ("max_length",),
                section="tokenizer",
            )
        )
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.dataset,
                checkpoint_cfg.dataset,
                ("max_seq_length",),
                section="dataset",
            )
        )
        operator_controlled.extend(
            _report_launch_controlled_config_drift(
                cfg.datacollator,
                checkpoint_cfg.datacollator,
                ("max_length",),
                section="datacollator",
            )
        )

    checkpoint_tokenizer_dir = checkpoint_path / "tokenizer"
    if checkpoint_tokenizer_dir.is_dir():
        tokenizer_path = str(checkpoint_tokenizer_dir)
        if getattr(cfg.tokenizer, "path", None) != tokenizer_path:
            changed.append("tokenizer.path")
        cfg.tokenizer.path = tokenizer_path

    if changed:
        log.warning(
            "Resume config drift detected in %s; checkpoint values are the source "
            "of truth.",
            ", ".join(sorted(set(changed))),
        )
    if operator_controlled:
        log.warning(
            "Resume: keeping launch-config values for operator-controlled fields "
            "(%s); the checkpoint's values are not restored. This is intended for "
            "continued pretraining (a different/annealing corpus, or - for RoPE - "
            "context extension).",
            ", ".join(sorted(set(operator_controlled))),
        )
    return set(operator_controlled)


def _optimizer_param_groups(optimizer: Any) -> list[dict[str, Any]]:
    """Return optimizer parameter groups from plain or Accelerate-wrapped objects.

    :param Any optimizer: Optimizer-like object.
    :return list[dict[str, Any]]: Optimizer parameter groups.
    :raises TypeError: If no parameter groups can be found.
    """
    if hasattr(optimizer, "param_groups"):
        return list(optimizer.param_groups)
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "param_groups"):
        return list(inner.param_groups)
    raise TypeError(f"Optimizer object {type(optimizer).__name__} has no param_groups.")


def attach_optimizer_param_names(
    model: torch.nn.Module,
    optimizer: Any,
) -> None:
    """Attach ordered parameter names to optimizer groups for resume validation.

    :param torch.nn.Module model: Model whose named parameters define the order.
    :param Any optimizer: Optimizer or Accelerate optimizer wrapper.
    :raises RuntimeError: If an optimizer parameter cannot be mapped to a model name.
    """
    name_by_id = {
        id(param): strip_runtime_prefixes(str(name))
        for name, param in model.named_parameters()
    }
    for group in _optimizer_param_groups(optimizer):
        names: list[str] = []
        for param in group["params"]:
            try:
                names.append(name_by_id[id(param)])
            except KeyError as exc:
                raise RuntimeError(
                    "Optimizer contains a parameter that is not present in "
                    "model.named_parameters(); cannot build resume manifest."
                ) from exc
        group["param_names"] = names


def optimizer_param_name_groups(optimizer: Any) -> list[list[str]]:
    """Return the ordered parameter-name manifest for optimizer groups.

    :param Any optimizer: Optimizer or Accelerate optimizer wrapper.
    :return list[list[str]]: Parameter names per group.
    :raises RuntimeError: If names were not attached first.
    """
    payload: list[list[str]] = []
    for group_idx, group in enumerate(_optimizer_param_groups(optimizer)):
        names = group.get("param_names")
        if names is None:
            raise RuntimeError(
                "Optimizer parameter names are missing for group "
                f"{group_idx}; call attach_optimizer_param_names() before checkpointing."
            )
        payload.append([strip_runtime_prefixes(str(name)) for name in names])
    return payload


def optimizer_state_semantics(optimizer: Any) -> str:
    """Return the tag naming how the optimizer's per-parameter state is defined.

    Optimizers that change their update rule in a way that reinterprets saved
    state declare a ``STATE_SEMANTICS`` class attribute and bump it on such
    changes. Optimizers whose update rule additionally depends on configuration
    (for example MuonClip's norm-factor selection) shadow the class tag with a
    qualified instance attribute so config drift is rejected on resume too.
    Optimizers without an explicit tag get a stable default derived from the
    class name (their state semantics are pinned by the framework).

    :param Any optimizer: Optimizer or Accelerate optimizer wrapper.
    :return str: State-semantics tag recorded in resume manifests.
    """
    unwrapped = _unwrap_optimizer(optimizer)
    semantics = getattr(unwrapped, "STATE_SEMANTICS", None)
    if semantics is not None:
        return str(semantics)
    return f"{type(unwrapped).__name__.lower()}-v1"


def should_save_step_checkpoint(
    *,
    step: int,
    max_steps: int,
    save_steps: int,
    save_model: bool,
    save_strategy: str,
) -> bool:
    """Whether to write a step checkpoint after ``step``.

    Saves on every ``save_steps`` tick and always on the terminal step, so a run
    whose ``max_steps`` is not a multiple of ``save_steps`` still persists its
    final trained weights instead of leaving ``latest`` at a stale earlier step.

    :param int step: Post-update global step count.
    :param int max_steps: Configured terminal step count.
    :param int save_steps: Step-based save interval.
    :param bool save_model: Whether model saving is enabled.
    :param str save_strategy: Configured save strategy; only ``steps`` saves here.
    :return bool: ``True`` when a checkpoint should be written for this step.
    """
    if not save_model or save_strategy != "steps":
        return False
    if step >= max_steps:
        return True
    return save_steps > 0 and step % save_steps == 0


def save_optimizer_param_name_manifest(
    optimizer: Any,
    checkpoint_path: str | Path,
) -> Path:
    """Persist optimizer parameter ordering and state semantics beside a checkpoint.

    :param Any optimizer: Optimizer or Accelerate optimizer wrapper.
    :param str | Path checkpoint_path: Checkpoint step directory.
    :return Path: Written manifest path.
    """
    path = Path(checkpoint_path) / OPTIMIZER_PARAM_NAMES_MANIFEST
    payload = {
        "schema_version": 1,
        "state_semantics": optimizer_state_semantics(optimizer),
        "param_name_groups": optimizer_param_name_groups(optimizer),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_training_checkpoint(
    *,
    task: str,
    checkpoint_path: str | Path,
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: Any,
    save_metadata: Callable[[Path], None],
) -> Path:
    """Save one complete self-contained training checkpoint across all ranks.

    ``save_metadata`` owns task-specific config/tokenizer artifacts and runs only
    on the main process. All synchronization and completion-marker ordering stays
    centralized here so trainers cannot drift on partial-checkpoint behavior.

    :param str task: Checkpoint task recorded in the completion marker.
    :param str | Path checkpoint_path: Destination step directory.
    :param Any accelerator: Active Accelerator instance.
    :param torch.nn.Module model: Prepared model to export.
    :param Any optimizer: Prepared optimizer whose parameter manifest is saved.
    :param Callable[[Path], None] save_metadata: Main-process metadata callback.
    :return Path: Completed checkpoint step directory.
    """
    checkpoint_path = Path(checkpoint_path)
    if accelerator.is_main_process:
        invalidate_checkpoint_completion(checkpoint_path)
    accelerator.wait_for_everyone()
    save_accelerate_state(accelerator, checkpoint_path)
    accelerator.wait_for_everyone()
    metadata_exception: Exception | None = None
    if accelerator.is_main_process:
        try:
            save_optimizer_param_name_manifest(optimizer, checkpoint_path)
            save_metadata(checkpoint_path)
        except Exception as exc:
            metadata_exception = exc

    reduce_fn = getattr(accelerator, "reduce", None)
    if callable(reduce_fn):
        failed = torch.tensor(
            int(metadata_exception is not None),
            device=accelerator.device,
            dtype=torch.int32,
        )
        metadata_failure_count = int(reduce_fn(failed, reduction="sum").item())
    else:
        metadata_failure_count = int(metadata_exception is not None)
    if metadata_failure_count:
        message = (
            f"failed to save checkpoint metadata at {checkpoint_path}: "
            f"{metadata_exception}"
            if metadata_exception is not None
            else "checkpoint metadata save failed on another rank; inspect the "
            "main-rank error for details"
        )
        raise RuntimeError(message) from metadata_exception

    save_portable_checkpoint_weights(model, accelerator, checkpoint_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        mark_checkpoint_complete(checkpoint_path, task=task)
    accelerator.wait_for_everyone()
    return checkpoint_path


def validate_optimizer_param_name_manifest(
    optimizer: Any,
    checkpoint_path: str | Path,
) -> None:
    """Fail fast if optimizer state cannot be faithfully restored from a checkpoint.

    PyTorch optimizer state is positional inside parameter groups, so a model
    refactor or construction-order change silently hands same-shaped parameters
    the wrong buffers. The manifest also records state-semantics tags so state
    written under a different update rule (for example a momentum-scale change)
    is rejected instead of silently mis-scaling post-resume updates. There is
    deliberately no fallback for checkpoints without a current-schema manifest:
    their optimizer state is unverifiable and this repo does not carry
    checkpoint back-compat before a stable release.

    :param Any optimizer: Optimizer or Accelerate optimizer wrapper.
    :param str | Path checkpoint_path: Checkpoint step directory.
    :raises RuntimeError: If the manifest is missing, outdated, or does not match.
    """
    path = Path(checkpoint_path) / OPTIMIZER_PARAM_NAMES_MANIFEST
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing; refusing a silent positional optimizer resume. "
            "This checkpoint predates the optimizer resume manifest, so its "
            "optimizer state cannot be verified against the current optimizer "
            "(parameter order and state semantics are unrecorded). Start a new "
            "run, or continue from the model weights without optimizer state."
        )

    saved = json.loads(path.read_text(encoding="utf-8"))
    schema_errors = optimizer_param_name_manifest_schema_errors(saved)
    if schema_errors:
        raise RuntimeError(
            f"{path} uses an outdated manifest schema ({'; '.join(schema_errors)}); "
            "refusing to resume optimizer state written by an older trainer. Start "
            "a new run, or re-save the checkpoint with a current trainer."
        )

    saved_semantics = saved.get("state_semantics")
    current_semantics = optimizer_state_semantics(optimizer)
    if saved_semantics != current_semantics:
        raise RuntimeError(
            "Optimizer state semantics changed since the checkpoint was written "
            f"(checkpoint: {saved_semantics!r}, current: {current_semantics!r}). "
            "Refusing to reinterpret saved optimizer state under a different "
            "update rule."
        )

    saved_param_name_groups = [
        [strip_runtime_prefixes(str(name)) for name in group]
        for group in saved["param_name_groups"]
    ]
    if saved_param_name_groups != optimizer_param_name_groups(optimizer):
        raise RuntimeError(
            "Optimizer parameter order changed since the checkpoint was written. "
            "Refusing to load optimizer state positionally."
        )


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
            is_step_selector = is_step_checkpoint_name(resume_value)
            if is_step_selector:
                resume_path = checkpoint_dir_path / resume_value
            else:
                resume_path = Path(resume_value)
            if not resume_path.is_absolute() and not is_step_selector:
                candidate = output_dir_path / resume_path
                if candidate.exists():
                    resume_path = candidate
            errors = checkpoint_resume_errors(resume_path)
            if errors:
                raise RuntimeError(
                    f"Checkpoint {resume_path} is not resumable: " + "; ".join(errors)
                )
            base = resume_path.name
            iteration = int(base) + 1 if is_step_checkpoint_name(base) else 0
            return str(resume_path), iteration

    if not checkpoint_dir_path.exists() or not any(checkpoint_dir_path.iterdir()):
        raise FileNotFoundError(
            f"No checkpoints found under requested resume root {checkpoint_dir_path}."
        )

    numeric_folders = [
        folder
        for folder in checkpoint_dir_path.iterdir()
        if folder.is_dir() and is_step_checkpoint_name(folder.name)
    ]
    folders = [folder for folder in numeric_folders if is_resumable_checkpoint(folder)]
    if not folders:
        if numeric_folders:
            details = "; ".join(
                f"{folder.name}: {', '.join(checkpoint_resume_errors(folder))}"
                for folder in sorted(
                    numeric_folders,
                    key=lambda item: (int(item.name), item.name),
                )
            )
            raise RuntimeError(
                f"No complete resumable checkpoints found under {checkpoint_dir_path} "
                f"({details})."
            )
        raise FileNotFoundError(
            "No numbered checkpoints found under requested resume root "
            f"{checkpoint_dir_path}."
        )

    latest_folder = max(folders, key=lambda folder: (int(folder.name), folder.name))
    latest_step = int(latest_folder.name)
    return str(latest_folder), latest_step + 1
