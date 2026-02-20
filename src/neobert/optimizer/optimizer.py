"""Factory helpers for constructing optimizers used in NeoBERT training."""

import logging
from dataclasses import asdict, is_dataclass
from types import MethodType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import DistributedType
from torch.optim import Adam, AdamW

from neobert.optimizer.muon_clip import MuonClipConfig, MuonClipOptimizer

# from .soap.soap import SOAP  # TODO: Add SOAP optimizer implementation

logger = logging.getLogger(__name__)


_POLAR_EXPRESS_BASE_COEFFS: tuple[Tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.948690853482295, -2.908902115962949, 0.5518191394370137),
    (3.318419657370602, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.668903984574749, 0.4188073119525673),
    (1.891301407787398, -1.267995827194587, 0.3768040894852483),
    (1.875001480853448, -1.250001645399949, 0.3750001645474248),
    (1.875000000000000, -1.250000000000000, 0.375000000000000),
)


def _build_adamw_param_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups for AdamW.

    :param torch.nn.Module model: Model whose parameters will be grouped.
    :param float weight_decay: Weight decay value for decayed parameters.
    :return list[dict[str, Any]]: Parameter groups for AdamW.
    """
    decay_params = []
    no_decay_params = []
    embedding_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for param in module.parameters(recurse=False)
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if (
            param.ndim < 2
            or name_lower.endswith(".bias")
            or "norm" in name_lower
            or id(param) in embedding_param_ids
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _build_dion2_param_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    scalar_algorithm: str,
    fraction: float,
    ef_decay: float,
    adjust_lr: Optional[str],
    flatten: bool,
) -> list[dict[str, Any]]:
    """Build Dion2 parameter groups with matrix/scalar splits.

    Matrix parameters (``ndim >= 2`` except embeddings) use Dion2 updates.
    Scalar parameters and embeddings use the configured scalar optimizer
    (AdamW/Lion), split into decay and no-decay groups.

    :param torch.nn.Module model: Model whose parameters will be grouped.
    :param float lr: Base learning rate.
    :param float weight_decay: Weight decay for decayed scalar/matrix groups.
    :param tuple[float, float] betas: Scalar optimizer betas.
    :param float eps: Scalar optimizer epsilon.
    :param str scalar_algorithm: Scalar optimizer algorithm (adamw/lion).
    :param float fraction: Dion2 submatrix fraction.
    :param float ef_decay: Dion2 error-feedback decay.
    :param str | None adjust_lr: Dion2 learning-rate adjustment mode.
    :param bool flatten: Dion2 flatten toggle.
    :return list[dict[str, Any]]: Dion2-compatible parameter groups.
    """
    matrix_params: list[torch.nn.Parameter] = []
    scalar_decay_params: list[torch.nn.Parameter] = []
    scalar_no_decay_params: list[torch.nn.Parameter] = []
    embedding_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for param in module.parameters(recurse=False)
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()

        if id(param) in embedding_param_ids:
            scalar_no_decay_params.append(param)
            continue

        if param.ndim >= 2:
            matrix_params.append(param)
            continue

        if name_lower.endswith(".bias") or "norm" in name_lower:
            scalar_no_decay_params.append(param)
        else:
            scalar_decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "algorithm": "dion2",
                "lr": lr,
                "fraction": fraction,
                "ef_decay": ef_decay,
                "weight_decay": weight_decay,
                "epsilon": eps,
                "adjust_lr": adjust_lr,
                "flatten": flatten,
            }
        )

    scalar_common = {
        "algorithm": scalar_algorithm,
        "lr": lr,
        "beta1": betas[0],
        "beta2": betas[1],
        "epsilon": eps,
    }
    if scalar_decay_params:
        groups.append(
            {
                "params": scalar_decay_params,
                "weight_decay": weight_decay,
                **scalar_common,
            }
        )
    if scalar_no_decay_params:
        groups.append(
            {
                "params": scalar_no_decay_params,
                "weight_decay": 0.0,
                **scalar_common,
            }
        )

    if not groups:
        raise ValueError("No trainable parameters found for Dion2 optimizer.")

    return groups


def _get_polar_express_coefficients(steps: int) -> list[Tuple[float, float, float]]:
    """Return dampened coefficient schedule for Polar Express iterations.

    :param int steps: Number of coefficient triples requested.
    :return list[tuple[float, float, float]]: Coefficient schedule.
    """
    steps = max(1, int(steps))
    dampening_factor = 1.01
    coeffs = [
        (
            a / dampening_factor,
            b / (dampening_factor**3),
            c / (dampening_factor**5),
        )
        for (a, b, c) in _POLAR_EXPRESS_BASE_COEFFS[:-1]
    ]
    coeffs.append(_POLAR_EXPRESS_BASE_COEFFS[-1])
    if steps <= len(coeffs):
        return coeffs[:steps]
    return coeffs + [coeffs[-1]] * (steps - len(coeffs))


def _polar_express_orthogonalize(
    grad: torch.Tensor,
    *,
    steps: int,
    epsilon: float,
) -> torch.Tensor:
    """Approximate orthogonalization via Polar Express polynomial iterations.

    The callable shape matches upstream Dion2's ``newton_schulz_func`` contract.

    :param torch.Tensor grad: Input matrix (or batch if upstream forwards it).
    :param int steps: Iteration count.
    :param float epsilon: Numerical stability epsilon.
    :return torch.Tensor: Orthogonalized matrix update.
    """
    if grad.ndim != 2:
        return grad

    is_transpose = grad.size(0) > grad.size(1)
    working = grad.T if is_transpose else grad
    original_dtype = working.dtype
    if working.dtype == torch.float16:
        working = working.float()

    norm = torch.linalg.norm(working)
    norm_value = float(norm.item())
    if norm_value == 0.0 or not torch.isfinite(norm).item():
        return torch.zeros_like(grad)

    working = working / (norm * 1.01 + float(epsilon))
    for a, b, c in _get_polar_express_coefficients(steps):
        matrix = working @ working.T
        working = a * working + (b * matrix + c * (matrix @ matrix)) @ working

    scale_factor = 0.4 * max(working.size(0), working.size(1)) ** 0.5
    working = scale_factor * working

    if working.dtype != original_dtype:
        working = working.to(original_dtype)
    return working.T if is_transpose else working


class _Dion2QKClippingRuntime:
    """Bridge MuonClip QK clipping hooks/logic onto a Dion2 optimizer."""

    def __init__(
        self,
        *,
        qk_clipper: MuonClipOptimizer,
    ) -> None:
        """Initialize runtime bridge.

        :param MuonClipOptimizer qk_clipper: MuonClip helper used for QK clipping.
        """
        self._qk_clipper = qk_clipper
        self._last_update_step = 0

    def rebind_model(self, model: torch.nn.Module) -> None:
        """Rebuild MuonClip QK hook/parameter references for prepared models.

        :param torch.nn.Module model: Prepared (possibly wrapped) model instance.
        """
        old_hook_system = getattr(self._qk_clipper, "hook_system", None)
        if old_hook_system is not None:
            old_hook_system.clear()
            old_hook_system.remove_hooks()

        self._qk_clipper = MuonClipOptimizer(
            model,
            self._qk_clipper.model_config,
            self._qk_clipper.config,
        )

    def prepare_for_forward(self, *, update_step: int, is_last_microbatch: bool) -> bool:
        """Forward pre-forward capture gating into MuonClip hook system.

        :param int update_step: Optimizer update index (0-based).
        :param bool is_last_microbatch: Whether this microbatch is sync/last.
        :return bool: Whether capture is enabled for this forward.
        """
        self._last_update_step = int(update_step)
        # Keep MuonClip's internal step aligned so schedule warnings are meaningful.
        self._qk_clipper._step = int(update_step)
        return self._qk_clipper.prepare_for_forward(
            update_step=int(update_step),
            is_last_microbatch=bool(is_last_microbatch),
        )

    def post_step(self) -> None:
        """Run MuonClip QK clipping immediately after Dion2 parameter updates."""
        clipper = self._qk_clipper
        should_clip = clipper.should_clip_update(int(self._last_update_step))
        if should_clip:
            if clipper.hook_system is None or not clipper.hook_system.has_captured_inputs():
                logger.warning(
                    "Dion2 MuonClip QK clipping scheduled at update_step="
                    f"{self._last_update_step} but no activations were captured. "
                    "Clipping will be skipped. This usually means "
                    "prepare_for_forward() was not called on the correct microbatch."
                )
                clipper._last_metrics.clear()
            else:
                clipper._apply_qk_clipping()
        else:
            clipper._last_metrics.clear()

        if clipper.hook_system is not None:
            clipper.hook_system.clear()
            clipper.hook_system.set_enabled(False, clear_cache_when_disabling=False)

        clipper._step = int(self._last_update_step) + 1

    def get_metrics(self) -> Dict[str, float]:
        """Return QK clipping metrics for trainer logging."""
        return self._qk_clipper.get_metrics()


def _attach_dion2_qk_clipping_runtime(
    optimizer: torch.optim.Optimizer,
    qk_clipper: MuonClipOptimizer,
) -> None:
    """Attach MuonClip QK clipping behavior to a Dion2 optimizer instance.

    :param torch.optim.Optimizer optimizer: Dion2 optimizer instance.
    :param MuonClipOptimizer qk_clipper: MuonClip helper with registered hooks.
    """
    runtime = _Dion2QKClippingRuntime(
        qk_clipper=qk_clipper,
    )
    original_step = optimizer.step

    def _step_with_qk_clipping(
        _self: torch.optim.Optimizer, *args: Any, **kwargs: Any
    ) -> Any:
        del _self
        loss = original_step(*args, **kwargs)
        runtime.post_step()
        return loss

    # Keep ``optimizer.step`` method-like (bound to optimizer) so PyTorch LR
    # schedulers can inspect ``__func__`` and wrap it correctly.
    setattr(optimizer, "step", MethodType(_step_with_qk_clipping, optimizer))
    setattr(optimizer, "prepare_for_forward", runtime.prepare_for_forward)
    setattr(optimizer, "get_metrics", runtime.get_metrics)
    setattr(optimizer, "_neobert_dion2_qk_runtime", runtime)


def get_optimizer(
    model: torch.nn.Module,
    distributed_type: DistributedType,
    model_config: Optional[Any] = None,
    muon_config: Optional[Any] = None,
    dion2_config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Construct an optimizer configured for the current training run.

    :param torch.nn.Module model: Model whose parameters will be optimized.
    :param DistributedType distributed_type: Distributed execution mode.
    :param Any | None model_config: Optional model config (required for MuonClip).
    :param Any | None muon_config: MuonClip overrides or dataclass.
    :param Any | None dion2_config: Dion2 overrides or dataclass.
    :param Any kwargs: Additional optimizer-specific keyword arguments.
    :return torch.optim.Optimizer: Initialized optimizer instance.
    :raises ValueError: If MuonClip is requested without ``model_config``.
    :raises ValueError: If an unsupported optimizer name is provided.
    """
    optimizer_name = kwargs.pop("name").lower()

    logger.info(f"Initializing optimizer: {optimizer_name}")

    match optimizer_name:
        case "adamw":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is adamw; ignoring"
                )
            if dion2_config is not None:
                logger.warning(
                    "optimizer.dion2_config provided but optimizer is adamw; ignoring"
                )
            weight_decay = kwargs.pop("weight_decay", 0.0)
            param_groups = _build_adamw_param_groups(model, weight_decay)
            optimizer = AdamW(param_groups, **kwargs)
            logger.info(f"AdamW initialized with lr={kwargs.get('lr', 'default')}")
            return optimizer

        case "adam":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is adam; ignoring"
                )
            if dion2_config is not None:
                logger.warning(
                    "optimizer.dion2_config provided but optimizer is adam; ignoring"
                )
            optimizer = Adam(model.parameters(), **kwargs)
            logger.info(f"Adam initialized with lr={kwargs.get('lr', 'default')}")
            return optimizer

        case "muonclip" | "muon-clip" | "muon_clip":
            if model_config is None:
                raise ValueError(
                    "MuonClip requires model_config to be passed. "
                    "Update get_optimizer() call in trainer to include model_config argument."
                )
            if dion2_config is not None:
                logger.warning(
                    "optimizer.dion2_config provided but optimizer is muonclip; ignoring"
                )

            # Build MuonClipConfig from kwargs
            lr = kwargs.pop("lr", 1e-4)
            weight_decay = kwargs.pop("weight_decay", 0.0)
            betas = kwargs.pop("betas", (0.9, 0.98))
            eps = kwargs.pop("eps", 1e-10)

            extra_args = {k: v for k, v in kwargs.items()}
            if extra_args:
                logger.warning(
                    "Ignoring unused optimizer kwargs for MuonClip: "
                    f"{', '.join(sorted(extra_args))}"
                )

            muon_kwargs: Dict[str, Any] = {}
            if muon_config is not None:
                if is_dataclass(muon_config):
                    muon_kwargs = asdict(muon_config)
                elif isinstance(muon_config, dict):
                    muon_kwargs = dict(muon_config)
                else:
                    raise TypeError(
                        "optimizer.muon_config must be a mapping or dataclass"
                    )

            muon_clip_cfg = MuonClipConfig(
                lr=lr,
                adam_betas=betas,
                adam_decay=weight_decay,
                adam_eps=eps,
                **muon_kwargs,
            )

            logger.info(
                f"MuonClip configuration:\n"
                f"  - Learning rate: {muon_clip_cfg.lr}\n"
                f"  - Clipping enabled: {muon_clip_cfg.enable_clipping}\n"
                f"  - Clipping threshold: {muon_clip_cfg.clipping_threshold}\n"
                f"  - Clipping interval: {muon_clip_cfg.clipping_interval}\n"
                f"  - QK chunk size: {muon_clip_cfg.clipping_qk_chunk_size}\n"
                f"  - Capture last microbatch only: {muon_clip_cfg.capture_last_microbatch_only}\n"
                f"  - Newton-Schulz steps: {muon_clip_cfg.ns_steps}\n"
                f"  - Alpha (Q/K balance): {muon_clip_cfg.clipping_alpha}\n"
                f"  - Orthogonalization: {muon_clip_cfg.orthogonalization}"
            )

            return MuonClipOptimizer(model, model_config, muon_clip_cfg)

        case "dion2" | "dion-2" | "dion_2":
            if muon_config is not None:
                logger.warning(
                    "optimizer.muon_config provided but optimizer is dion2; ignoring"
                )
            try:
                from dion import Dion2
            except ImportError as exc:
                raise ImportError(
                    "optimizer.name='dion2' requires the optional 'dion' package. "
                    "Install with: pip install -e \".[dion]\" "
                    "or pip install \"dion @ git+https://github.com/microsoft/dion.git\". "
                    "Note: upstream dion requires a recent PyTorch + Triton runtime."
                ) from exc

            lr = kwargs.pop("lr", 1e-4)
            weight_decay = kwargs.pop("weight_decay", 0.01)
            betas = tuple(kwargs.pop("betas", (0.9, 0.95)))
            eps = kwargs.pop("eps", 1e-8)
            if len(betas) != 2:
                raise ValueError(
                    f"Dion2 expects exactly 2 beta values, got {betas!r}."
                )
            extra_args = {k: v for k, v in kwargs.items()}
            if extra_args:
                logger.warning(
                    "Ignoring unused optimizer kwargs for Dion2: "
                    f"{', '.join(sorted(extra_args))}"
                )

            dion2_kwargs: Dict[str, Any] = {}
            if dion2_config is not None:
                if is_dataclass(dion2_config):
                    dion2_kwargs = asdict(dion2_config)
                elif isinstance(dion2_config, dict):
                    dion2_kwargs = dict(dion2_config)
                else:
                    raise TypeError(
                        "optimizer.dion2_config must be a mapping or dataclass"
                    )
            known_dion2_keys = {
                "fraction",
                "ef_decay",
                "adjust_lr",
                "orthogonalization",
                "ns_steps",
                "flatten",
                "use_triton",
                "enable_clipping",
                "clipping_threshold",
                "clipping_alpha",
                "clipping_warmup_steps",
                "clipping_interval",
                "clipping_qk_chunk_size",
                "capture_last_microbatch_only",
                "clipping_layers_mapping",
                "verbose",
                "scalar_algorithm",
            }
            unknown_dion2_keys = sorted(set(dion2_kwargs) - known_dion2_keys)
            if unknown_dion2_keys:
                logger.warning(
                    "Ignoring unknown optimizer.dion2_config keys: "
                    f"{', '.join(unknown_dion2_keys)}"
                )
            fraction = float(dion2_kwargs.get("fraction", 0.25))
            ef_decay = float(dion2_kwargs.get("ef_decay", 0.95))
            adjust_lr = dion2_kwargs.get("adjust_lr", "spectral_norm")
            if isinstance(adjust_lr, str):
                adjust_lr = adjust_lr.strip().lower()
            if adjust_lr not in {"spectral_norm", "rms_norm", None}:
                raise ValueError(
                    "optimizer.dion2_config.adjust_lr must be "
                    "{'spectral_norm', 'rms_norm', None}, "
                    f"got {adjust_lr!r}."
                )
            orthogonalization = str(
                dion2_kwargs.get("orthogonalization", "newton_schulz")
            ).strip().lower()
            orthogonalization_aliases = {
                "polar": "polar_express",
                "polar_express": "polar_express",
                "newton_schulz": "newton_schulz",
                "newton-schulz": "newton_schulz",
                "ns": "newton_schulz",
            }
            if orthogonalization not in orthogonalization_aliases:
                raise ValueError(
                    "optimizer.dion2_config.orthogonalization must be one of "
                    "{'newton_schulz', 'polar_express'}, got "
                    f"{orthogonalization!r}."
                )
            orthogonalization = orthogonalization_aliases[orthogonalization]
            ns_steps = int(dion2_kwargs.get("ns_steps", 5))
            if ns_steps < 1:
                raise ValueError(
                    "optimizer.dion2_config.ns_steps must be >= 1, got "
                    f"{ns_steps}."
                )
            flatten = bool(dion2_kwargs.get("flatten", False))
            use_triton = bool(dion2_kwargs.get("use_triton", False))
            verbose = bool(dion2_kwargs.get("verbose", False))
            enable_qk_clipping = bool(dion2_kwargs.get("enable_clipping", False))
            clipping_threshold = float(dion2_kwargs.get("clipping_threshold", 50.0))
            if not (0.0 < clipping_threshold <= 1000.0):
                raise ValueError(
                    "optimizer.dion2_config.clipping_threshold must be in (0, 1000], "
                    f"got {clipping_threshold}."
                )
            clipping_alpha = float(dion2_kwargs.get("clipping_alpha", 0.5))
            if not (0.0 <= clipping_alpha <= 1.0):
                raise ValueError(
                    "optimizer.dion2_config.clipping_alpha must be in [0, 1], "
                    f"got {clipping_alpha}."
                )
            clipping_warmup_steps = int(dion2_kwargs.get("clipping_warmup_steps", 0))
            if clipping_warmup_steps < 0:
                raise ValueError(
                    "optimizer.dion2_config.clipping_warmup_steps must be >= 0, "
                    f"got {clipping_warmup_steps}."
                )
            clipping_interval = int(dion2_kwargs.get("clipping_interval", 10))
            if clipping_interval < 1:
                raise ValueError(
                    "optimizer.dion2_config.clipping_interval must be >= 1, got "
                    f"{clipping_interval}."
                )
            clipping_qk_chunk_size = int(
                dion2_kwargs.get("clipping_qk_chunk_size", 1024)
            )
            if clipping_qk_chunk_size < 1:
                raise ValueError(
                    "optimizer.dion2_config.clipping_qk_chunk_size must be >= 1, got "
                    f"{clipping_qk_chunk_size}."
                )
            capture_last_microbatch_only = bool(
                dion2_kwargs.get("capture_last_microbatch_only", True)
            )
            clipping_layers_mapping = dion2_kwargs.get("clipping_layers_mapping", {})
            if clipping_layers_mapping is None:
                clipping_layers_mapping = {}
            if not isinstance(clipping_layers_mapping, dict):
                raise TypeError(
                    "optimizer.dion2_config.clipping_layers_mapping must be a mapping"
                )
            clipping_layers_mapping = {
                str(key).lower(): str(value)
                for key, value in clipping_layers_mapping.items()
            }
            scalar_algorithm = str(
                dion2_kwargs.get("scalar_algorithm", "adamw")
            ).strip().lower()
            if scalar_algorithm not in {"adamw", "lion"}:
                raise ValueError(
                    "optimizer.dion2_config.scalar_algorithm must be 'adamw' or "
                    f"'lion', got {scalar_algorithm!r}."
                )
            newton_schulz_func = None
            if orthogonalization == "polar_express":
                if use_triton:
                    logger.warning(
                        "optimizer.dion2_config.use_triton=true is ignored when "
                        "orthogonalization='polar_express'."
                    )
                use_triton = False

                def _polar_func(
                    input_tensor: torch.Tensor, epsilon: float = 1e-7
                ) -> torch.Tensor:
                    return _polar_express_orthogonalize(
                        input_tensor,
                        steps=ns_steps,
                        epsilon=float(epsilon),
                    )

                newton_schulz_func = _polar_func

            distributed_mesh = None
            if (
                distributed_type is not DistributedType.FSDP
                and dist.is_available()
                and dist.is_initialized()
            ):
                distributed_mesh = dist.group.WORLD

            param_groups = _build_dion2_param_groups(
                model,
                lr=float(lr),
                weight_decay=float(weight_decay),
                betas=(float(betas[0]), float(betas[1])),
                eps=float(eps),
                scalar_algorithm=scalar_algorithm,
                fraction=fraction,
                ef_decay=ef_decay,
                adjust_lr=adjust_lr,
                flatten=flatten,
            )
            optimizer = Dion2(
                param_groups,
                distributed_mesh=distributed_mesh,
                lr=float(lr),
                fraction=fraction,
                ef_decay=ef_decay,
                betas=(float(betas[0]), float(betas[1])),
                weight_decay=float(weight_decay),
                epsilon=float(eps),
                adjust_lr=adjust_lr,
                flatten=flatten,
                use_triton=use_triton,
                newton_schulz_func=newton_schulz_func,
                verbose=verbose,
            )
            setattr(optimizer, "_is_neobert_dion2", True)
            if enable_qk_clipping:
                if model_config is None:
                    raise ValueError(
                        "optimizer.dion2_config.enable_clipping=true requires "
                        "model_config to enable MuonClip QK clipping hooks."
                    )
                qk_clip_cfg = MuonClipConfig(
                    lr=float(lr),
                    ns_steps=ns_steps,
                    enable_clipping=True,
                    clipping_threshold=clipping_threshold,
                    clipping_alpha=clipping_alpha,
                    clipping_warmup_steps=clipping_warmup_steps,
                    clipping_interval=clipping_interval,
                    clipping_qk_chunk_size=clipping_qk_chunk_size,
                    capture_last_microbatch_only=capture_last_microbatch_only,
                    clipping_layers_mapping=clipping_layers_mapping,
                )
                qk_clipper = MuonClipOptimizer(model, model_config, qk_clip_cfg)
                _attach_dion2_qk_clipping_runtime(optimizer, qk_clipper)
                logger.info(
                    "Dion2 MuonClip QK clipping enabled "
                    f"(threshold={clipping_threshold}, alpha={clipping_alpha}, "
                    f"interval={clipping_interval}, warmup={clipping_warmup_steps}, "
                    f"chunk_size={clipping_qk_chunk_size}, "
                    f"capture_last_microbatch_only={capture_last_microbatch_only})."
                )
            logger.info(
                f"Dion2 initialized with lr={lr}, fraction={fraction}, "
                f"ef_decay={ef_decay}, adjust_lr={adjust_lr}, "
                f"orthogonalization={orthogonalization}, ns_steps={ns_steps}, "
                f"scalar_algorithm={scalar_algorithm}, use_triton={use_triton}, "
                f"qk_clipping={enable_qk_clipping}"
            )
            return optimizer

        # case "SOAP":
        #     assert distributed_type is not DistributedType.DEEPSPEED, (
        #         "SOAP does not support DeepSpeed"
        #     )
        #     return SOAP(model.parameters(), **kwargs)

        case _:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw, muonclip, dion2"
            )
