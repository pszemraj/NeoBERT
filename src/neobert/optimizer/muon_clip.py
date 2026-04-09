"""MuonClip optimizer for NeoBERT encoders.

Adapted for fused QKV projections, attention hooks, and distributed training.
References: Kimi K2 report, Muon, MuonClip, DISCO, Polar Express.
"""

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
except Exception:  # pragma: no cover - older torch builds without DTensor APIs
    DTensor = None  # type: ignore[assignment]
    Shard = None  # type: ignore[assignment]

try:
    _dynamo_disable = torch._dynamo.disable  # pyright: ignore[reportAttributeAccessIssue]
except Exception:

    def _dynamo_disable(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``fn`` unchanged when torch Dynamo is unavailable.

        :param Callable[..., Any] fn: Function to wrap.
        :return Callable[..., Any]: The original function ``fn``.
        """
        return fn


# Configuration


@dataclass
class MuonClipConfig:
    """Configuration container for the MuonClip optimizer."""

    # Learning rates
    lr: float = 1e-4

    # Muon parameters (for hidden-layer 2D weight matrices)
    muon_beta: float = 0.95  # Momentum coefficient
    nesterov: bool = True
    muon_decay: float = 0.0  # Weight decay for Muon params
    ns_steps: int = 5  # Newton-Schulz iterations (3-9 recommended)

    # Adam parameters (for 1D params: biases, LayerNorm)
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    adam_decay: float = 0.0  # Weight decay for Adam params
    adam_eps: float = 1e-10

    # QK-Clipping parameters
    enable_clipping: bool = True
    clipping_threshold: float = 50.0  # Conservative for encoders
    clipping_alpha: float = 0.5  # Q/K scaling balance (0.5 = equal)
    clipping_warmup_steps: int = 0  # Disable clipping for N steps
    clipping_interval: int = 10  # Apply clipping every N steps to cap overhead
    clipping_qk_chunk_size: int = (
        1024  # Chunk size for QK max tiling to avoid S^2 peaks
    )
    capture_last_microbatch_only: bool = (
        True  # Capture activations only on last microbatch
    )

    # Architecture adaptation
    clipping_layers_mapping: Dict[str, str] = field(default_factory=dict)

    # Monitoring and debugging
    detect_anomalies: bool = False  # Enable gradient anomaly detection

    # Orthogonalization / routing control
    orthogonalization: str = "polar_express"
    norm_factor: str = "legacy_compat"
    param_policy: str = "hidden_2d"
    algorithm: Optional[str] = None  # Alias for orthogonalization
    polar_express: Optional[bool] = None  # Legacy toggle

    def __post_init__(self) -> None:
        """Validate configuration.

        IMPORTANT: do not use ``assert`` for runtime validation. Python can be
        executed with ``-O``, which strips asserts entirely.
        """
        if not (0 < self.lr < 1):
            raise ValueError(f"lr must be in (0, 1), got {self.lr}")
        if not (0 <= self.muon_beta < 1):
            raise ValueError(f"muon_beta must be in [0, 1), got {self.muon_beta}")
        if not (0 <= self.muon_decay < 1):
            raise ValueError(f"muon_decay must be in [0, 1), got {self.muon_decay}")
        if not (1 <= self.ns_steps <= 20):
            raise ValueError(f"ns_steps must be in [1, 20], got {self.ns_steps}")
        if not (0 < self.clipping_threshold <= 1000):
            raise ValueError(
                "clipping_threshold must be in (0, 1000], got "
                f"{self.clipping_threshold}"
            )
        if not (0 <= self.clipping_alpha <= 1):
            raise ValueError(
                f"clipping_alpha must be in [0, 1], got {self.clipping_alpha}"
            )
        if self.clipping_interval < 1:
            raise ValueError(
                f"clipping_interval must be >= 1, got {self.clipping_interval}"
            )
        if self.clipping_qk_chunk_size < 1:
            raise ValueError(
                "clipping_qk_chunk_size must be >= 1, got "
                f"{self.clipping_qk_chunk_size}"
            )

        if self.algorithm is not None:
            warnings.warn(
                "MuonClipConfig.algorithm is deprecated; use orthogonalization instead.",
                UserWarning,
                stacklevel=2,
            )
        if self.polar_express is not None:
            warnings.warn(
                "MuonClipConfig.polar_express is deprecated; use orthogonalization instead.",
                UserWarning,
                stacklevel=2,
            )

        # Warnings for suboptimal settings
        if self.ns_steps < 3:
            warnings.warn(
                f"ns_steps={self.ns_steps} may not provide sufficient orthogonalization. "
                "Recommended: 5-9",
                UserWarning,
                stacklevel=2,
            )
        if self.clipping_threshold > 200:
            warnings.warn(
                f"clipping_threshold={self.clipping_threshold} is very high. "
                "You may not see clipping effects.",
                UserWarning,
                stacklevel=2,
            )
        if self.clipping_threshold < 30:
            warnings.warn(
                f"clipping_threshold={self.clipping_threshold} is very low. "
                "Risk of over-constraining attention.",
                UserWarning,
                stacklevel=2,
            )

        if self.clipping_layers_mapping is None:
            self.clipping_layers_mapping = {}
        elif not isinstance(self.clipping_layers_mapping, dict):
            raise TypeError("clipping_layers_mapping must be a dict[str, str]")
        else:
            normalised_mapping: Dict[str, str] = {}
            for key, value in self.clipping_layers_mapping.items():
                normalised_mapping[str(key).lower()] = str(value)
            unsupported = set(normalised_mapping) - {"q_proj", "k_proj", "v_proj"}
            if unsupported:
                warnings.warn(
                    "Unsupported keys in clipping_layers_mapping: "
                    + ", ".join(sorted(unsupported)),
                    UserWarning,
                    stacklevel=2,
                )
            self.clipping_layers_mapping = normalised_mapping

        if not isinstance(self.adam_betas, (tuple, list)) or len(self.adam_betas) != 2:
            raise TypeError(
                "adam_betas must be a 2-tuple of floats, got "
                f"{type(self.adam_betas).__name__}={self.adam_betas!r}"
            )
        _ = self.adam_betas[1]

        # Resolve orthogonalization algorithm
        algo_source = self.algorithm or self.orthogonalization
        if self.polar_express is not None:
            algo_source = "polar_express" if self.polar_express else "newton_schulz"

        if not algo_source:
            algo_source = "polar_express"

        algo_normalized = str(algo_source).replace("-", "_").lower()
        alias_map = {
            "ns5": "newton_schulz",
            "newton_schulz5": "newton_schulz",
            "newton_schulz_5": "newton_schulz",
            "polar": "polar_express",
        }
        algo = alias_map.get(algo_normalized, algo_normalized)

        valid_algos = {"polar_express", "newton_schulz"}
        if algo not in valid_algos:
            raise ValueError(
                f"Unsupported orthogonalization algorithm '{algo_source}'. "
                f"Valid options: {', '.join(sorted(valid_algos))}"
            )

        norm_factor = str(self.norm_factor).strip().replace("-", "_").lower()
        valid_norm_factors = {
            "legacy_compat",
            "original",
            "spectral",
            "match_rms_adamw",
            "none",
        }
        if norm_factor not in valid_norm_factors:
            raise ValueError(
                f"Unsupported norm_factor '{self.norm_factor}'. "
                f"Valid options: {', '.join(sorted(valid_norm_factors))}"
            )

        param_policy = str(self.param_policy).strip().replace("-", "_").lower()
        param_policy = {"transformer_only": "hidden_2d"}.get(param_policy, param_policy)
        valid_param_policies = {"all_2d", "hidden_2d"}
        if param_policy not in valid_param_policies:
            raise ValueError(
                f"Unsupported param_policy '{self.param_policy}'. "
                f"Valid options: {', '.join(sorted(valid_param_policies))}"
            )

        self.orthogonalization = algo
        self.norm_factor = norm_factor
        self.param_policy = param_policy
        self.algorithm = algo
        # Reset explicit toggle to prevent downstream confusion
        self.polar_express = None


# Attention hooks


class NeoBERTAttentionHooks:
    """Capture attention inputs needed for QK clipping.

    Stores normalized QKV inputs plus pad mask/rotary/packed metadata. Expensive
    stats are computed lazily during the optimizer step.
    """

    def __init__(
        self, model_config: Any, layer_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize the hook system.

        :param Any model_config: Model configuration with attention settings.
        :param dict[str, str] | None layer_mapping: Optional Q/K/V name mapping.
        """
        self.config = model_config
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.layer_mapping = layer_mapping or {}

        self.layer_inputs: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_pad_masks: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_freqs: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_packed_seqlens: Dict[int, Optional[list[list[int]]]] = {}
        self.layers: Dict[int, torch.nn.Module] = {}
        self._module_to_layer_idx: Dict[int, int] = {}

        self.enabled = True
        self.hook_handles: List[RemovableHandle] = []

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that the attention configuration is consistent."""
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.config.num_attention_heads})"
            )

    def set_enabled(
        self, enabled: bool, *, clear_cache_when_disabling: bool = True
    ) -> None:
        """Enable/disable activation capture for hooks.

        :param bool enabled: Whether capture should be enabled.
        :param bool clear_cache_when_disabling: Clear cached tensors when disabling.
        """
        enabled = bool(enabled)
        if self.enabled == enabled:
            return

        self.enabled = enabled
        if not enabled and clear_cache_when_disabling:
            self.clear()

    def register_hooks(self, model: torch.nn.Module) -> int:
        """Register hooks across all transformer encoder layers.

        Returns
        -------
        int
            Number of hook handles that were successfully registered.

        Raises
        ------
        RuntimeError
            If no transformer layers are found or hook registration fails.
            On failure, any hooks registered before the error are cleaned up.
        """
        layers = self._resolve_transformer_layers(model)
        if not layers:
            raise RuntimeError("No transformer layers found for MuonClip hooks")

        num_hooks = 0
        registered_handles: List[RemovableHandle] = []
        try:
            for idx, layer in enumerate(layers):
                self.layers[idx] = layer
                self.layer_inputs[idx] = None
                self.layer_pad_masks[idx] = None
                self.layer_freqs[idx] = None
                self.layer_packed_seqlens[idx] = None
                self._module_to_layer_idx[id(layer)] = idx

                if hasattr(layer, "qkv"):
                    self._module_to_layer_idx[id(layer.qkv)] = idx
                    handle = layer.qkv.register_forward_hook(self._qkv_input_hook)
                    registered_handles.append(handle)
                    num_hooks += 1
                else:
                    q_proj_name = self.layer_mapping.get("q_proj")
                    if not q_proj_name:
                        raise RuntimeError(
                            "Encoder block lacks fused qkv projection; provide "
                            "'clipping_layers_mapping' with q_proj entry."
                        )
                    q_proj = getattr(layer, q_proj_name, None)
                    if q_proj is None:
                        raise RuntimeError(
                            f"Encoder block missing projection '{q_proj_name}'"
                        )
                    self._module_to_layer_idx[id(q_proj)] = idx
                    handle = q_proj.register_forward_hook(self._qkv_input_hook)
                    registered_handles.append(handle)
                    num_hooks += 1

                block_handle = layer.register_forward_hook(self._block_context_hook)
                registered_handles.append(block_handle)
                num_hooks += 1

                logger.debug(f"Registered MuonClip hooks on layer {idx}")

            # All hooks registered successfully; transfer to instance list.
            self.hook_handles.extend(registered_handles)
            logger.info(f"Registered {num_hooks} MuonClip hooks")
            return num_hooks

        except Exception as e:
            # Clean up any hooks registered before the failure to prevent dangling hooks.
            for handle in registered_handles:
                handle.remove()
            self._module_to_layer_idx.clear()
            raise RuntimeError(f"Hook registration failed on layer {idx}: {e}") from e

    def _resolve_transformer_layers(
        self, model: torch.nn.Module
    ) -> Optional[Sequence[torch.nn.Module]]:
        """Return the sequence of encoder layers exposed by ``model``.

        :param torch.nn.Module model: Model to inspect.
        :return Sequence[torch.nn.Module] | None: Encoder layers if found.
        """
        if hasattr(model, "transformer_encoder"):
            return model.transformer_encoder

        for attr_name in ("model", "base", "backbone"):
            submodule = getattr(model, attr_name, None)
            if submodule is None:
                continue
            layers = self._resolve_transformer_layers(submodule)
            if layers is not None:
                return layers

        return None

    @_dynamo_disable
    def _module_layer_idx(self, module: torch.nn.Module) -> Optional[int]:
        """Resolve the cached layer index for a hook module.

        :param torch.nn.Module module: Hooked module.
        :return int | None: Layer index when present.
        """
        layer_idx = self._module_to_layer_idx.get(id(module), None)
        if layer_idx is None:
            return None
        try:
            return int(layer_idx)
        except (TypeError, ValueError):
            return None

    @_dynamo_disable
    def _detach_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Detach ``tensor`` and cache it on CPU with async-friendly semantics.

        CUDA tensors are copied into pinned host buffers with ``non_blocking=True``
        so D2H transfer can overlap with device work. CPU tensors are detached in
        place without extra copies.

        :param torch.Tensor tensor: Tensor to detach/cache.
        :return torch.Tensor: Detached CPU tensor.
        """
        detached = tensor.detach()
        if detached.device.type != "cuda":
            return detached.to("cpu")

        pinned = torch.empty_like(detached, device="cpu", pin_memory=True)
        pinned.copy_(detached, non_blocking=True)
        return pinned

    @_dynamo_disable
    def _qkv_input_hook(
        self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        """Store QKV input activations for the layer owning ``module``.

        :param torch.nn.Module module: Hooked qkv (or q_proj) module.
        :param tuple[Any, ...] inputs: Forward inputs.
        :param Any output: Forward output (unused).
        """
        del output
        if not self.enabled or not inputs:
            return
        layer_idx = self._module_layer_idx(module)
        if layer_idx is None:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        # During gradient accumulation, we intentionally keep only the latest
        # microbatch activation to bound memory/compute overhead. This biases
        # QK clipping stats toward the most recent microbatch by design.
        # We cache pre-projection inputs (not Q/K) to avoid duplicating large
        # QKV tensors in CPU memory; projections are recomputed only on clip
        # steps because prepare_for_forward() interval-gates hook enablement.
        # Offload to pinned host memory so CUDA->CPU transfer can be async.
        self.layer_inputs[layer_idx] = self._detach_to_cpu(x)

    @_dynamo_disable
    def _block_context_hook(
        self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> None:
        """Store layer-level context tensors for clipping diagnostics.

        :param torch.nn.Module module: Hooked encoder block module.
        :param tuple[Any, ...] inputs: Forward inputs.
        :param Any output: Forward output (unused).
        """
        del output
        if not self.enabled or not inputs:
            return
        layer_idx = self._module_layer_idx(module)
        if layer_idx is None:
            return

        pad_mask = inputs[1] if len(inputs) > 1 else None
        freqs_cis = inputs[2] if len(inputs) > 2 else None
        packed_seqlens = inputs[3] if len(inputs) > 3 else None

        self.layer_pad_masks[layer_idx] = (
            self._detach_to_cpu(pad_mask) if torch.is_tensor(pad_mask) else pad_mask
        )
        self.layer_freqs[layer_idx] = (
            self._detach_to_cpu(freqs_cis) if torch.is_tensor(freqs_cis) else freqs_cis
        )
        if packed_seqlens is not None:
            if torch.is_tensor(packed_seqlens):
                if packed_seqlens.numel() == 0:
                    packed_seqlens = [[] for _ in range(packed_seqlens.shape[0])]
                else:
                    cpu = packed_seqlens.detach().cpu()
                    packed_seqlens = [
                        [int(x) for x in row[row > 0].tolist()] for row in cpu
                    ]
            else:
                packed_seqlens = [
                    list(map(int, seg_lens)) for seg_lens in packed_seqlens
                ]
        self.layer_packed_seqlens[layer_idx] = packed_seqlens

    def get_layer_data(
        self, layer_idx: int
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[list[list[int]]],
    ]:
        """Return cached tensors for a given layer.

        :param int layer_idx: Encoder layer index.
        :return tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
            list[list[int]] | None]: Input, pad mask, rotary frequencies, and packed
            sequence lengths.
        """
        return (
            self.layer_inputs.get(layer_idx),
            self.layer_pad_masks.get(layer_idx),
            self.layer_freqs.get(layer_idx),
            self.layer_packed_seqlens.get(layer_idx),
        )

    def clear(self) -> None:
        """Clear cached tensors from all layers."""
        for layer_idx in self.layer_inputs:
            self.layer_inputs[layer_idx] = None
        for layer_idx in self.layer_pad_masks:
            self.layer_pad_masks[layer_idx] = None
        for layer_idx in self.layer_freqs:
            self.layer_freqs[layer_idx] = None
        for layer_idx in self.layer_packed_seqlens:
            self.layer_packed_seqlens[layer_idx] = None

    def has_captured_inputs(self) -> bool:
        """Return whether any layer currently has captured activations.

        :return bool: True when at least one layer input tensor is cached.
        """
        return any(inputs is not None for inputs in self.layer_inputs.values())

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._module_to_layer_idx.clear()


# Optimizer


class MuonClipOptimizer(Optimizer):
    """MuonClip optimizer for NeoBERT encoders.

    Uses Muon for 2D params, Adam for 1D params, with optional QK clipping.
    """

    def __init__(
        self, model: torch.nn.Module, model_config: Any, config: MuonClipConfig
    ) -> None:
        """Initialize the optimizer and attach attention hooks as needed.

        :param torch.nn.Module model: Model to optimize.
        :param Any model_config: Model configuration.
        :param MuonClipConfig config: Optimizer configuration.
        """
        self.config = config
        self.model_config = model_config
        self._step = 0
        self._last_metrics: Dict[str, float] = {}
        self._layer_mapping = dict(self.config.clipping_layers_mapping)
        self._polar_coeff_cache: Dict[int, List[Tuple[float, float, float]]] = {}
        self._polar_work_dtype: Optional[torch.dtype] = None
        self._muon_owner_rank_cache: Dict[
            Tuple[Tuple[int, ...], int], Dict[int, int]
        ] = {}
        self._dtensor_row_count_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...], int], List[int]
        ] = {}
        self._runtime_clipping_enabled = bool(self.config.enable_clipping)
        self._clipping_disabled_warning_emitted = False
        if torch.cuda.is_available():
            try:
                # bfloat16 offers good perf/stability balance for polar iteration
                self._polar_work_dtype = torch.bfloat16
            except TypeError:
                self._polar_work_dtype = None

        # Validate model architecture
        self._validate_model(model)

        # Setup hooks for QK-clipping
        if config.enable_clipping:
            logger.info("Initializing attention hook system...")
            self.hook_system = NeoBERTAttentionHooks(
                model_config, layer_mapping=self._layer_mapping
            )
            num_hooks = self.hook_system.register_hooks(self.model_base)
            logger.info(f"Hook system ready: {num_hooks} hooks registered")
        else:
            self.hook_system = None
            logger.info("QK-clipping disabled, no hooks registered")

        # Build parameter groups
        param_groups = self._build_param_groups(model)
        logger.info(
            f"Parameter groups: {len(param_groups)} groups, "
            f"{sum(len(g['params']) for g in param_groups)} parameters"
        )

        # Initialize base optimizer
        super().__init__(param_groups, {})

        # Gradient anomaly detection
        if config.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)
            logger.warning(
                "Gradient anomaly detection enabled - training will be slower"
            )

    def should_clip_update(self, update_step: int) -> bool:
        """Return True if QK clipping should run on this optimizer update step.

        :param int update_step: Optimizer update index (0-based).
        :return bool: Whether clipping should run.
        """
        if not self._runtime_clipping_enabled:
            return False

        warmup = int(getattr(self.config, "clipping_warmup_steps", 0))
        if update_step < warmup:
            return False

        interval = int(getattr(self.config, "clipping_interval", 1))
        if interval <= 0:
            raise ValueError(f"clipping_interval must be >= 1, got {interval}")

        return ((update_step - warmup) % interval) == 0

    def prepare_for_forward(
        self, *, update_step: int, is_last_microbatch: bool
    ) -> bool:
        """Enable/disable hook capture before a forward pass.

        :param int update_step: Optimizer update index (0-based).
        :param bool is_last_microbatch: Whether this microbatch completes an update.
        :return bool: Whether capture is enabled for this forward.
        """
        if self.hook_system is None:
            return False

        if int(update_step) != int(self._step):
            logger.warning(
                "MuonClip step desync: trainer update_step="
                f"{int(update_step)} != optimizer._step={int(self._step)}. "
                "Clipping/capture schedules may misalign."
            )

        should_clip = self.should_clip_update(int(update_step))
        capture_last_only = bool(
            getattr(self.config, "capture_last_microbatch_only", True)
        )
        capture_enabled = (
            should_clip and bool(is_last_microbatch)
            if capture_last_only
            else should_clip
        )
        self.hook_system.set_enabled(capture_enabled, clear_cache_when_disabling=True)
        return capture_enabled

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Validate model architecture compatibility.

        :param torch.nn.Module model: Model to validate.
        """
        base_model, transformer_layers = self._resolve_transformer_stack(model)
        if base_model is None or transformer_layers is None:
            raise ValueError(
                "Model must expose 'transformer_encoder' either directly or via a "
                "'base' attribute. Is this a NeoBERT model?"
            )

        self.model_base = base_model
        self.transformer_layers = transformer_layers

        # Check first layer projections
        if len(self.transformer_layers) == 0:
            raise ValueError("Model has no transformer layers")

        first_layer = self.transformer_layers[0]
        if hasattr(first_layer, "qkv"):
            logger.debug("Detected fused QKV attention block")
        else:
            q_proj = self._layer_mapping.get("q_proj")
            k_proj = self._layer_mapping.get("k_proj")
            if not q_proj or not k_proj:
                raise ValueError(
                    "EncoderBlock lacks fused 'qkv' projection and no "
                    "'clipping_layers_mapping' overrides were provided."
                )
            missing = [
                name
                for name, attr in (("q_proj", q_proj), ("k_proj", k_proj))
                if not hasattr(first_layer, attr)
            ]
            if missing:
                raise ValueError(
                    "EncoderBlock missing required projection(s): " + ", ".join(missing)
                )
            logger.debug(
                f"Detected separate attention projections using mapping {self._layer_mapping}"
            )

        logger.debug("Model architecture validation passed")

    def _use_adam_weight_decay(
        self,
        *,
        name: str,
        param: torch.nn.Parameter,
        embedding_param_ids: set[int],
    ) -> bool:
        """Return whether a non-Muon parameter should receive Adam weight decay.

        This mirrors the repo's standard AdamW grouping policy so MuonClip's
        Adam fallback does not silently decay embeddings, biases, or norm gains.

        :param str name: Fully qualified parameter name.
        :param torch.nn.Parameter param: Parameter to classify.
        :param set[int] embedding_param_ids: IDs belonging to embedding weights.
        :return bool: ``True`` when decoupled Adam weight decay should apply.
        """
        name_lower = name.lower()
        return not (
            param.ndim < 2
            or name_lower.endswith(".bias")
            or "norm" in name_lower
            or id(param) in embedding_param_ids
        )

    def _build_param_groups(self, model: torch.nn.Module) -> List[Dict]:
        """Build parameter groups for hybrid Muon+Adam optimization.

        Policies:
        - ``all_2d``: route every trainable rank-2 parameter to Muon. This
          restores the v0.1.3 scope and remains available as an explicit
          compatibility mode.
        - ``hidden_2d``: route only hidden transformer-layer rank-2 weights to
          Muon. Embeddings / output matrices fall back to AdamW-style grouping.
          This is the runtime default and matches the original Muon guidance.

        :param torch.nn.Module model: Model to inspect.
        :return list[dict]: Parameter groups for the optimizer.
        """
        muon_params: List[torch.nn.Parameter] = []
        muon_param_info: List[Dict[str, Any]] = []
        adam_decay_params: List[torch.nn.Parameter] = []
        adam_decay_param_info: List[Dict[str, Any]] = []
        adam_no_decay_params: List[torch.nn.Parameter] = []
        adam_no_decay_param_info: List[Dict[str, Any]] = []

        param_policy = str(getattr(self.config, "param_policy", "hidden_2d")).strip()
        param_policy = param_policy.replace("-", "_").lower()
        param_policy = {"transformer_only": "hidden_2d"}.get(param_policy, param_policy)
        valid_param_policies = {"all_2d", "hidden_2d"}
        if param_policy not in valid_param_policies:
            raise ValueError(
                f"Unsupported param_policy '{param_policy}'. "
                f"Valid options: {', '.join(sorted(valid_param_policies))}"
            )

        transformer_param_ids: set[int] = set()
        layer_idx_by_param_id: Dict[int, int] = {}
        proj_type_by_param_id: Dict[int, str] = {}

        for idx, layer in enumerate(self.transformer_layers):
            for param in layer.parameters():
                param_id = id(param)
                transformer_param_ids.add(param_id)
                layer_idx_by_param_id[param_id] = idx

            if hasattr(layer, "qkv"):
                qkv_module = getattr(layer, "qkv", None)
                if qkv_module is not None and hasattr(qkv_module, "weight"):
                    proj_type_by_param_id[id(qkv_module.weight)] = "qkv"
            else:
                for canonical, attr_name in self._layer_mapping.items():
                    if not attr_name:
                        continue
                    module = getattr(layer, attr_name, None)
                    if module is None or not hasattr(module, "weight"):
                        continue

                    if canonical.lower().startswith("q"):
                        proj_type = "q"
                    elif canonical.lower().startswith("k"):
                        proj_type = "k"
                    elif canonical.lower().startswith("v"):
                        proj_type = "v"
                    else:
                        continue
                    proj_type_by_param_id[id(module.weight)] = proj_type

        embedding_param_ids = {
            id(param)
            for module in model.modules()
            if isinstance(module, torch.nn.Embedding)
            for param in module.parameters(recurse=False)
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_id = id(param)
            layer_idx = layer_idx_by_param_id.get(param_id)

            proj_type = proj_type_by_param_id.get(param_id)
            if proj_type is None and "qkv" in name:
                proj_type = "qkv"
            if proj_type is None:
                for canonical, pattern in self._layer_mapping.items():
                    if not pattern or pattern not in name:
                        continue
                    if canonical.lower().startswith("q"):
                        proj_type = "q"
                    elif canonical.lower().startswith("k"):
                        proj_type = "k"
                    elif canonical.lower().startswith("v"):
                        proj_type = "v"
                    if proj_type is not None:
                        break

            # Track parameters that participate in Q/K scaling
            is_qkv = proj_type in {"qkv", "q", "k"}

            # Keep only serializable metadata here; ``group["params"]`` remains
            # the source of truth for live tensors and may be rewritten in-place
            # by wrappers such as Accelerate FSDP2 during ``prepare()``.
            param_info = {
                "name": name,
                "layer_idx": layer_idx,
                "is_qkv": is_qkv,
                "proj_type": proj_type,
            }

            if param_policy == "all_2d":
                use_muon = param.ndim == 2
            else:
                use_muon = (
                    param.ndim == 2
                    and param_id in transformer_param_ids
                    and param_id not in embedding_param_ids
                )
            if use_muon:
                muon_params.append(param)
                muon_param_info.append(param_info)
            else:
                if self._use_adam_weight_decay(
                    name=name,
                    param=param,
                    embedding_param_ids=embedding_param_ids,
                ):
                    adam_decay_params.append(param)
                    adam_decay_param_info.append(param_info)
                else:
                    adam_no_decay_params.append(param)
                    adam_no_decay_param_info.append(param_info)

        groups = []

        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "param_info": muon_param_info,
                    "lr": self.config.lr,
                    "beta": self.config.muon_beta,
                    "nesterov": self.config.nesterov,
                    "weight_decay": self.config.muon_decay,
                    "use_muon": True,
                    "param_policy": param_policy,
                }
            )
            logger.info(
                "Muon group: %s parameters (policy=%s)",
                len(muon_params),
                param_policy,
            )

        if adam_decay_params:
            groups.append(
                {
                    "params": adam_decay_params,
                    "param_info": adam_decay_param_info,
                    "lr": self.config.lr,
                    "betas": self.config.adam_betas,
                    "eps": self.config.adam_eps,
                    "weight_decay": self.config.adam_decay,
                    "use_muon": False,
                }
            )
            logger.info(
                "Adam decay group: %s parameters",
                len(adam_decay_params),
            )
        if adam_no_decay_params:
            groups.append(
                {
                    "params": adam_no_decay_params,
                    "param_info": adam_no_decay_param_info,
                    "lr": self.config.lr,
                    "betas": self.config.adam_betas,
                    "eps": self.config.adam_eps,
                    "weight_decay": 0.0,
                    "use_muon": False,
                }
            )
            logger.info(
                "Adam no-decay group: %s parameters",
                len(adam_no_decay_params),
            )

        if not groups:
            raise ValueError("No trainable parameters found")

        return groups

    def _resolve_transformer_stack(
        self, model: torch.nn.Module
    ) -> Tuple[Optional[torch.nn.Module], Optional[Sequence[torch.nn.Module]]]:
        """Discover the base encoder module and its layers inside ``model``.

        :param torch.nn.Module model: Model to inspect.
        :return tuple[torch.nn.Module | None, Sequence[torch.nn.Module] | None]:
            Base model and encoder layers.
        """
        if hasattr(model, "transformer_encoder"):
            return model, model.transformer_encoder

        for attr_name in ("model", "base", "backbone"):
            submodule = getattr(model, attr_name, None)
            if submodule is None:
                continue
            base_model, layers = self._resolve_transformer_stack(submodule)
            if base_model is not None:
                return base_model, layers

        return None, None

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        """Perform an optimization step.

        :param Callable | None closure: Optional closure to recompute loss.
        :return torch.Tensor | None: Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply parameter updates
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)

        # Apply QK-clipping
        if self._runtime_clipping_enabled and self.hook_system:
            should_clip = self.should_clip_update(self._step)
            if should_clip:
                if not self.hook_system.has_captured_inputs():
                    logger.warning(
                        "MuonClip scheduled at update_step="
                        f"{self._step} but no activations were captured. Clipping will be "
                        "skipped. This usually means prepare_for_forward() was not called on "
                        "the correct microbatch (or hooks were disabled/wrapped)."
                    )
                    self._last_metrics.clear()
                else:
                    self._apply_qk_clipping()
            else:
                self._last_metrics.clear()

            # Always clear cached activations and disable capture between steps.
            self.hook_system.clear()
            self.hook_system.set_enabled(False, clear_cache_when_disabling=False)

        self._step += 1
        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state including the MuonClip update counter.

        :return dict[str, Any]: Optimizer state dictionary.
        """
        base = super().state_dict()
        base["muonclip_step"] = int(self._step)
        return base

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state including the MuonClip update counter.

        Sharded FSDP2 Muon resumes must materialize optimizer state via
        Accelerate/DCP helpers before calling into this optimizer. A raw
        ``optimizer.state_dict()`` checkpoint for DTensor params does not
        preserve DTensor metadata and is rejected below.

        :param dict[str, Any] state_dict: Optimizer state dictionary.
        """
        # Copy the outer mapping so MuonClip can strip its own metadata without
        # mutating the caller's state_dict object.
        payload = dict(state_dict)
        muonclip_step = payload.pop("muonclip_step", None)
        saved_groups = payload.get("param_groups", None)
        if saved_groups is not None:
            payload["param_groups"] = copy.deepcopy(saved_groups)
            self._restore_missing_group_metadata(payload["param_groups"])
        super().load_state_dict(payload)
        self._validate_loaded_muon_state_topology()

        if muonclip_step is not None:
            self._step = int(muonclip_step)
            return

        inferred = 0
        for state in self.state.values():
            if isinstance(state, dict) and "step" in state:
                try:
                    inferred = max(inferred, int(state["step"]))
                except Exception:
                    continue
        self._step = inferred

    def _restore_missing_group_metadata(
        self, saved_groups: Sequence[Dict[str, Any]]
    ) -> None:
        """Fill in custom group metadata that generic checkpoint loaders may drop.

        Some optimizer checkpoint backends normalize ``param_groups`` down to the
        standard optimizer fields. MuonClip depends on additional group metadata
        such as ``use_muon``, ``beta``, and ``param_info`` to restore the correct
        update path after resume, so merge any missing keys back from the current
        optimizer groups before calling ``Optimizer.load_state_dict()``.

        :param Sequence[dict[str, Any]] saved_groups: Loaded optimizer param groups.
        """
        if len(saved_groups) != len(self.param_groups):
            return

        for current_group, saved_group in zip(
            self.param_groups, saved_groups, strict=True
        ):
            for key, value in current_group.items():
                if key == "params":
                    continue
                if key not in saved_group:
                    saved_group[key] = copy.deepcopy(value)

    def _muon_step(self, group: Dict[str, Any]) -> None:
        """Apply Muon update with orthogonalization.

        :param dict[str, Any] group: Parameter group metadata.
        """
        if self._runtime_clipping_enabled and any(
            self._is_dtensor(param) for param in group["params"]
        ):
            self._disable_clipping_for_sharded_runtime()

        group_param_info = group.get("param_info")
        for param_idx, param in enumerate(group["params"]):
            if param.grad is None:
                continue
            param_info = None
            if group_param_info is not None and param_idx < len(group_param_info):
                param_info = group_param_info[param_idx]

            # Check for NaN/Inf in gradients
            if self.config.detect_anomalies:
                if not torch.isfinite(param.grad).all():
                    logger.error(f"Non-finite gradient detected in {param.shape}")
                    continue

            # Get optimizer state
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["step"] = 0

            state["step"] += 1
            grad = param.grad
            if grad is None:
                continue

            # Apply momentum
            state["momentum_buffer"].mul_(group["beta"]).add_(grad)

            momentum_buffer = state["momentum_buffer"]
            muon_input = self._muon_input(
                grad=grad,
                momentum_buffer=momentum_buffer,
                beta=float(group["beta"]),
                nesterov=bool(group.get("nesterov", True)),
            )
            # Parameter topology is the source of truth; loaded state must match it.
            param_is_dtensor = self._is_dtensor(param)
            buffer_is_dtensor = self._is_dtensor(momentum_buffer)
            if param_is_dtensor:
                if not buffer_is_dtensor:
                    raise RuntimeError(
                        "MuonClip found a local Tensor momentum buffer for DTensor "
                        "parameter state during a sharded Muon update. Restore this "
                        "optimizer via Accelerate/DCP state-dict helpers instead of "
                        "a raw local-tensor load_state_dict path."
                    )
                update = self._orthogonalize_dtensor_update(
                    muon_input=muon_input,
                    param=param,
                    group_params=group["params"],
                    group_param_info=group.get("param_info"),
                    param_info=param_info,
                )
            else:
                if buffer_is_dtensor:
                    raise RuntimeError(
                        "MuonClip found a DTensor momentum buffer attached to a "
                        "non-DTensor parameter. Optimizer state topology does not "
                        "match the current model parameters."
                    )
                update = self._orthogonalize_muon_input(
                    muon_input=muon_input,
                    param_shape=param.shape,
                    param_info=param_info,
                )

            # Weight decay
            if group["weight_decay"] > 0:
                param.mul_(1 - group["lr"] * group["weight_decay"])

            # Parameter update
            param.add_(update, alpha=-group["lr"])

    def _adam_step(self, group: Dict[str, Any]) -> None:
        """Apply standard Adam update to 1D parameters.

        :param dict[str, Any] group: Parameter group metadata.
        """
        for param in group["params"]:
            if param.grad is None:
                continue

            # Check for anomalies
            if self.config.detect_anomalies:
                if not torch.isfinite(param.grad).all():
                    logger.error(f"Non-finite gradient detected in {param.shape}")
                    continue

            # Get optimizer state
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0

            state["step"] += 1
            grad = param.grad
            if grad is None:
                continue

            beta1, beta2 = group["betas"]

            # Exponential moving averages
            state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # Step size with bias correction
            step_size = group["lr"] / bias_correction1

            # Weight decay
            if group["weight_decay"] > 0:
                param.mul_(1 - group["lr"] * group["weight_decay"])

            # Denominator with eps for numerical stability
            denom = (state["exp_avg_sq"].sqrt() / (bias_correction2**0.5)).add_(
                group["eps"]
            )

            # Update parameters
            param.addcdiv_(state["exp_avg"], denom, value=-step_size)

    def _is_dtensor(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is a DTensor instance.

        :param torch.Tensor tensor: Candidate tensor.
        :return bool: ``True`` when ``tensor`` is a DTensor.
        """
        return DTensor is not None and isinstance(tensor, DTensor)

    def _dtensor_mesh_signature(self, tensor: Any) -> Tuple[Any, ...]:
        """Build a comparable device-mesh signature for a DTensor-like object.

        :param Any tensor: DTensor-like value with ``device_mesh`` metadata.
        :return tuple[Any, ...]: Mesh ndim plus flattened mesh contents when known.
        """
        mesh = getattr(tensor, "device_mesh", None)
        mesh_ndim = getattr(mesh, "ndim", None)
        mesh_layout: Tuple[int, ...] = ()
        mesh_shape: Tuple[int, ...] = ()
        mesh_tensor = getattr(mesh, "mesh", None)
        if isinstance(mesh_tensor, torch.Tensor):
            mesh_shape = tuple(int(dim) for dim in mesh_tensor.shape)
            mesh_layout = tuple(int(rank) for rank in mesh_tensor.reshape(-1).tolist())
        return (
            int(mesh_ndim) if mesh_ndim is not None else None,
            mesh_shape,
            mesh_layout,
        )

    def _dtensor_placement_signature(self, tensor: Any) -> Tuple[Tuple[str, Any], ...]:
        """Build a comparable placement signature for a DTensor-like object.

        :param Any tensor: DTensor-like value with ``placements`` metadata.
        :return tuple[tuple[str, Any], ...]: Placement type names and dimensions.
        """
        placements = tuple(getattr(tensor, "placements", ()))
        return tuple(
            (type(placement).__name__, getattr(placement, "dim", None))
            for placement in placements
        )

    def _validate_loaded_muon_state_topology(self) -> None:
        """Reject Muon state loads whose momentum-buffer topology mismatches params.

        :raises RuntimeError: If sharded Muon params do not have matching DTensor
            momentum buffers, or if loaded DTensor state targets local params.
        """
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            for param, info in self._iter_group_params_with_info(group):
                state = self.state.get(param, {})
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    continue

                param_name = str(info.get("name", "<unknown>"))
                param_is_dtensor = self._is_dtensor(param)
                buffer_is_dtensor = self._is_dtensor(momentum_buffer)
                if param_is_dtensor and not buffer_is_dtensor:
                    raise RuntimeError(
                        "MuonClip loaded a local Tensor momentum buffer for DTensor "
                        f"parameter '{param_name}'. Restore sharded Muon state via "
                        "Accelerate/DCP helpers so optimizer topology matches the "
                        "prepared FSDP2 model."
                    )
                if buffer_is_dtensor and not param_is_dtensor:
                    raise RuntimeError(
                        "MuonClip loaded a DTensor momentum buffer for non-sharded "
                        f"parameter '{param_name}'. Optimizer state topology does not "
                        "match the current model."
                    )

                param_shape = getattr(param, "shape", None)
                buffer_shape = getattr(momentum_buffer, "shape", None)
                if param_shape is not None and buffer_shape is not None:
                    normalized_param_shape = tuple(int(dim) for dim in param_shape)
                    normalized_buffer_shape = tuple(int(dim) for dim in buffer_shape)
                    if normalized_param_shape != normalized_buffer_shape:
                        raise RuntimeError(
                            "MuonClip loaded momentum state with shape "
                            f"{normalized_buffer_shape} for parameter "
                            f"'{param_name}' with shape {normalized_param_shape}."
                        )

                if not param_is_dtensor:
                    continue

                param_mesh = self._dtensor_mesh_signature(param)
                buffer_mesh = self._dtensor_mesh_signature(momentum_buffer)
                param_placements = self._dtensor_placement_signature(param)
                buffer_placements = self._dtensor_placement_signature(momentum_buffer)
                if param_mesh != buffer_mesh or param_placements != buffer_placements:
                    raise RuntimeError(
                        "MuonClip loaded DTensor momentum state with mesh/placement "
                        f"metadata that does not match parameter '{param_name}'."
                    )

    def _disable_clipping_for_sharded_runtime(self) -> None:
        """Disable QK clipping once when sharded Muon updates are detected."""
        if not self._runtime_clipping_enabled:
            return
        if not self._clipping_disabled_warning_emitted:
            logger.warning(
                "MuonClip QK clipping is disabled under FSDP2 sharded Muon updates. "
                "Proceeding with Muon-only optimization."
            )
            self._clipping_disabled_warning_emitted = True

        self._runtime_clipping_enabled = False
        if self.hook_system is not None:
            self.hook_system.clear()
            self.hook_system.set_enabled(False, clear_cache_when_disabling=False)

    def _process_group_cache_key(
        self, process_group: dist.ProcessGroup
    ) -> Tuple[int, ...]:
        """Build a stable cache key for a process group.

        :param dist.ProcessGroup process_group: Process group for collective ops.
        :return tuple[int, ...]: Deterministic key based on member global ranks.
        """
        get_ranks = getattr(dist, "get_process_group_ranks", None)
        if callable(get_ranks):
            ranks = tuple(int(rank) for rank in get_ranks(process_group))
            if ranks:
                return ranks
        # Fallback for older torch builds lacking get_process_group_ranks.
        return (int(id(process_group)),)

    def _resolve_owner_rank(
        self,
        *,
        param: torch.nn.Parameter,
        group_params: Sequence[torch.nn.Parameter],
        group_param_info: Optional[Sequence[Dict[str, Any]]] = None,
        world_size: int,
        process_group: dist.ProcessGroup,
    ) -> int:
        """Resolve a stable owner rank (group-local) for a parameter.

        :param torch.nn.Parameter param: Parameter to assign.
        :param Sequence[torch.nn.Parameter] group_params: Muon parameter group members.
        :param Sequence[dict[str, Any]] | None group_param_info:
            Static metadata aligned with ``group_params``.
        :param int world_size: Group-local world size.
        :param dist.ProcessGroup process_group: Process group backing the shard mesh.
        :return int: Owner rank in the process group.
        """
        cache_key = (self._process_group_cache_key(process_group), int(world_size))
        owners = self._muon_owner_rank_cache.get(cache_key)
        if owners is None or id(param) not in owners:
            owners = {}
            loads = [0] * world_size
            if group_param_info is not None:
                if len(group_params) != len(group_param_info):
                    raise RuntimeError(
                        "MuonClip owner assignment expected param_info to stay "
                        "aligned with params, but their lengths differ."
                    )
                ordered = [
                    group_param
                    for _, _, group_param in sorted(
                        (
                            (
                                int(group_param.numel()),
                                str(info.get("name", "")),
                                group_param,
                            )
                            for group_param, info in zip(
                                group_params,
                                group_param_info,
                                strict=True,
                            )
                        ),
                        key=lambda item: (-item[0], item[1]),
                    )
                ]
            else:
                ordered = sorted(
                    group_params, key=lambda p: int(p.numel()), reverse=True
                )
            for group_param in ordered:
                owner = min(
                    range(world_size),
                    key=lambda rank_idx: (loads[rank_idx], rank_idx),
                )
                owners[id(group_param)] = int(owner)
                loads[owner] += int(group_param.numel())
            self._muon_owner_rank_cache[cache_key] = owners

        owner = owners.get(id(param))
        if owner is None:
            raise RuntimeError(
                "Failed to resolve owner rank for Muon DTensor parameter; "
                "owner assignment cache does not contain this parameter."
            )
        return int(owner)

    def _get_dtensor_row_counts(
        self,
        *,
        local_shard: torch.Tensor,
        global_shape: torch.Size,
        process_group: dist.ProcessGroup,
        world_size: int,
        rank: int,
    ) -> List[int]:
        """Return cached per-rank row counts for a row-sharded DTensor.

        :param torch.Tensor local_shard: Local shard tensor on this rank.
        :param torch.Size global_shape: Global DTensor shape.
        :param dist.ProcessGroup process_group: Process group backing the DTensor mesh.
        :param int world_size: Group-local world size.
        :param int rank: Group-local rank.
        :return list[int]: Row count owned by each rank in process-group order.
        """
        key = (
            self._process_group_cache_key(process_group),
            tuple(int(dim) for dim in global_shape),
            int(world_size),
        )
        row_counts = self._dtensor_row_count_cache.get(key)
        local_rows = int(local_shard.shape[0])
        if row_counts is None or len(row_counts) != world_size:
            total_rows = int(global_shape[0]) if len(global_shape) > 0 else 0
            base_rows, remainder = divmod(total_rows, world_size)
            row_counts = [
                base_rows + (1 if rank_idx < remainder else 0)
                for rank_idx in range(world_size)
            ]
            self._dtensor_row_count_cache[key] = row_counts

        if int(row_counts[rank]) != local_rows:
            raise RuntimeError(
                "MuonClip DTensor path expected canonical Shard(0) row counts "
                f"{row_counts}, but local rank {rank} owns {local_rows} rows."
            )

        if not self.config.detect_anomalies:
            return row_counts

        # Debug-only validation path: compare the analytical Shard(0) split
        # against the runtime shards when anomaly detection is enabled.
        local_row_count = torch.tensor(
            [local_rows],
            device=local_shard.device,
            dtype=torch.int64,
        )
        gathered_counts = [torch.zeros_like(local_row_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_row_count, group=process_group)
        gathered_row_counts = [int(count.item()) for count in gathered_counts]
        if gathered_row_counts != row_counts:
            raise RuntimeError(
                "MuonClip DTensor path expected analytical row counts "
                f"{row_counts}, but observed runtime row counts "
                f"{gathered_row_counts}."
            )
        return row_counts

    def _muon_input(
        self,
        *,
        grad: Any,
        momentum_buffer: Any,
        beta: float,
        nesterov: bool,
    ) -> Any:
        """Build the Muon direction from the current grad and momentum state.

        :param Any grad: Current gradient tensor or DTensor.
        :param Any momentum_buffer: Momentum buffer after the current update.
        :param float beta: Muon momentum coefficient.
        :param bool nesterov: Whether to use standard Nesterov Muon momentum.
        :return Any: Tensor-like value to orthogonalize.
        """
        if nesterov:
            return grad.add(momentum_buffer, alpha=beta)
        return momentum_buffer

    def _orthogonalize_dtensor_update(
        self,
        *,
        muon_input: Any,
        param: torch.nn.Parameter,
        group_params: Sequence[torch.nn.Parameter],
        group_param_info: Optional[Sequence[Dict[str, Any]]] = None,
        param_info: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Orthogonalize a row-sharded DTensor via owner-compute gather/scatter.

        :param Any muon_input: DTensor Muon input (raw momentum or Nesterov direction).
        :param torch.nn.Parameter param: Parameter being updated.
        :param Sequence[torch.nn.Parameter] group_params: Muon parameter group members.
        :param Sequence[dict[str, Any]] | None group_param_info:
            Static metadata aligned with ``group_params``.
        :param dict[str, Any] | None param_info: Metadata for the current parameter.
        :return Any: DTensor update with the original placement metadata.
        """
        if DTensor is None or Shard is None:
            raise RuntimeError(
                "DTensor Muon path requested but DTensor APIs are unavailable in this torch build."
            )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "DTensor Muon update requires torch.distributed to be initialized."
            )

        placements = muon_input.placements
        if len(placements) != 1 or not isinstance(placements[0], Shard):
            raise RuntimeError(
                "MuonClip DTensor path currently supports only 1D row-sharded DTensors."
            )
        shard_dim = int(placements[0].dim)
        if shard_dim != 0:
            raise RuntimeError(
                "MuonClip DTensor path currently supports only Shard(0) placement."
            )

        mesh = muon_input.device_mesh
        mesh_ndim = getattr(mesh, "ndim", None)
        if mesh_ndim is not None and int(mesh_ndim) != 1:
            raise RuntimeError(
                "MuonClip DTensor path currently supports only 1D FSDP2 device meshes "
                f"with row-sharded parameters, got device_mesh.ndim={int(mesh_ndim)}."
            )
        process_group = mesh.get_group()
        if process_group is None:
            raise RuntimeError(
                "Failed to resolve process group for DTensor device mesh."
            )

        world_size = int(dist.get_world_size(process_group))
        rank = int(dist.get_rank(process_group))
        owner_rank = self._resolve_owner_rank(
            param=param,
            group_params=group_params,
            group_param_info=group_param_info,
            world_size=world_size,
            process_group=process_group,
        )

        local_shard = muon_input.to_local()
        if local_shard.ndim != 2:
            raise RuntimeError(
                "MuonClip DTensor path expects rank-2 local shards, got "
                f"shape={tuple(local_shard.shape)}."
            )
        local_rows = int(local_shard.shape[0])
        row_counts = self._get_dtensor_row_counts(
            local_shard=local_shard,
            global_shape=muon_input.shape,
            process_group=process_group,
            world_size=world_size,
            rank=rank,
        )
        global_param_shape = torch.Size(muon_input.shape)
        try:
            global_param_stride = tuple(
                int(dim_stride) for dim_stride in muon_input.stride()
            )
        except Exception as exc:
            raise RuntimeError(
                "MuonClip DTensor path failed to resolve global stride metadata "
                "for DTensor reconstruction."
            ) from exc
        max_rows = max(row_counts)
        pad_rows = max_rows - local_rows
        if pad_rows > 0:
            pad = local_shard.new_zeros((pad_rows, local_shard.shape[1]))
            local_padded = torch.cat((local_shard, pad), dim=0).contiguous()
        else:
            local_padded = local_shard.contiguous()

        gather_list: Optional[List[torch.Tensor]]
        if rank == owner_rank:
            gather_list = [torch.empty_like(local_padded) for _ in range(world_size)]
        else:
            gather_list = None
        # TODO(phase3): pipeline or batch these owner-compute collectives across
        # Muon parameters so multi-node runs do not serialize one gather/scatter
        # pair per matrix update.
        dist.gather(
            tensor=local_padded,
            gather_list=gather_list,
            group=process_group,
            group_dst=owner_rank,
        )

        scatter_list: Optional[List[torch.Tensor]]
        if rank == owner_rank:
            if gather_list is None:
                raise RuntimeError(
                    "gather_list must not be None on owner rank during DTensor Muon gather."
                )
            pieces = [
                gather_list[idx][: row_counts[idx]]
                for idx in range(world_size)
                if row_counts[idx] > 0
            ]
            if pieces:
                full_matrix = torch.cat(pieces, dim=0)
            else:
                full_matrix = local_padded.new_zeros((0, local_padded.shape[1]))

            update_full = self._orthogonalize_muon_input(
                muon_input=full_matrix,
                param_shape=global_param_shape,
                param_info=param_info,
            )

            scatter_list = []
            row_start = 0
            for rows in row_counts:
                rows_i = int(rows)
                row_end = row_start + rows_i
                chunk = update_full[row_start:row_end]
                row_start = row_end
                if rows_i < max_rows:
                    padding = chunk.new_zeros((max_rows - rows_i, chunk.shape[1]))
                    chunk = torch.cat((chunk, padding), dim=0)
                scatter_list.append(chunk.contiguous())
        else:
            scatter_list = None

        local_update_padded = torch.empty_like(local_padded)
        dist.scatter(
            tensor=local_update_padded,
            scatter_list=scatter_list,
            group=process_group,
            group_src=owner_rank,
        )
        local_update = local_update_padded[:local_rows].contiguous()

        return DTensor.from_local(
            local_update,
            device_mesh=mesh,
            placements=placements,
            run_check=False,
            shape=global_param_shape,
            stride=global_param_stride,
        )

    def _orthogonalize_muon_input(
        self,
        *,
        muon_input: torch.Tensor,
        param_shape: torch.Size,
        param_info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Orthogonalize and normalize a local Muon input tensor.

        :param torch.Tensor muon_input: Tensor to orthogonalize.
        :param torch.Size param_shape: Shape of the owning parameter.
        :param dict[str, Any] | None param_info: Static metadata for the parameter.
        :return torch.Tensor: Orthogonalized and normalized update tensor.
        """
        if self._uses_fused_qkv_muon_split(
            param_shape=param_shape, param_info=param_info
        ):
            return self._orthogonalize_fused_qkv_update(muon_input)
        update = self._orthogonalize_update(muon_input)
        return self._normalize_muon_update(update, param_shape)

    def _uses_fused_qkv_muon_split(
        self,
        *,
        param_shape: torch.Size,
        param_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return whether a parameter should split fused QKV for Muon.

        :param torch.Size param_shape: Shape of the owning parameter.
        :param dict[str, Any] | None param_info: Static metadata for the parameter.
        :return bool: ``True`` when the update should be split into Q/K/V matrices.
        """
        if len(param_shape) != 2:
            return False
        if (param_info or {}).get("proj_type") != "qkv":
            return False
        return True

    def _orthogonalize_fused_qkv_update(self, muon_input: torch.Tensor) -> torch.Tensor:
        """Orthogonalize fused QKV matrices as separate Q, K, and V projections.

        :param torch.Tensor muon_input: Interleaved fused-QKV update tensor.
        :return torch.Tensor: Fused update tensor rebuilt from per-projection Muon.
        """
        q_update, k_update, v_update = self._split_interleaved_qkv_matrix(muon_input)
        q_update = self._normalize_muon_update(
            self._orthogonalize_update(q_update),
            q_update.shape,
        )
        k_update = self._normalize_muon_update(
            self._orthogonalize_update(k_update),
            k_update.shape,
        )
        v_update = self._normalize_muon_update(
            self._orthogonalize_update(v_update),
            v_update.shape,
        )
        return self._merge_interleaved_qkv_matrix(q_update, k_update, v_update)

    def _split_interleaved_qkv_matrix(
        self, matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split an interleaved fused QKV matrix into per-projection matrices.

        :param torch.Tensor matrix: Fused QKV tensor shaped like ``qkv.weight``.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Separate Q, K, and V matrices.
        """
        matrix_view, head_dim, hidden_size = self._view_interleaved_qkv_matrix(matrix)
        q_slice = slice(0, head_dim)
        k_slice = slice(head_dim, 2 * head_dim)
        v_slice = slice(2 * head_dim, 3 * head_dim)
        q_matrix = (
            matrix_view[:, q_slice].reshape(hidden_size, hidden_size).contiguous()
        )
        k_matrix = (
            matrix_view[:, k_slice].reshape(hidden_size, hidden_size).contiguous()
        )
        v_matrix = (
            matrix_view[:, v_slice].reshape(hidden_size, hidden_size).contiguous()
        )
        return q_matrix, k_matrix, v_matrix

    def _merge_interleaved_qkv_matrix(
        self,
        q_matrix: torch.Tensor,
        k_matrix: torch.Tensor,
        v_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Merge per-projection Q, K, and V matrices into interleaved fused rows.

        :param torch.Tensor q_matrix: Query projection update.
        :param torch.Tensor k_matrix: Key projection update.
        :param torch.Tensor v_matrix: Value projection update.
        :return torch.Tensor: Interleaved fused-QKV update tensor.
        """
        hidden_size = int(self.model_config.hidden_size)
        num_heads = int(self.model_config.num_attention_heads)
        head_dim = int(self.model_config.dim_head)
        cols = int(q_matrix.shape[1])

        expected_projection_shape = (hidden_size, cols)
        for name, matrix in (
            ("q", q_matrix),
            ("k", k_matrix),
            ("v", v_matrix),
        ):
            if tuple(matrix.shape) != expected_projection_shape:
                raise RuntimeError(
                    f"Unexpected {name}-projection shape {tuple(matrix.shape)}; "
                    f"expected {expected_projection_shape}."
                )

        fused = q_matrix.new_empty((3 * hidden_size, cols))
        fused_view = fused.view(num_heads, 3 * head_dim, cols)
        fused_view[:, :head_dim].copy_(q_matrix.view(num_heads, head_dim, cols))
        fused_view[:, head_dim : 2 * head_dim].copy_(
            k_matrix.view(num_heads, head_dim, cols)
        )
        fused_view[:, 2 * head_dim : 3 * head_dim].copy_(
            v_matrix.view(num_heads, head_dim, cols)
        )
        return fused

    def _view_interleaved_qkv_matrix(
        self, matrix: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Return the per-head fused-QKV view used by NeoBERT attention blocks.

        :param torch.Tensor matrix: Fused QKV tensor shaped like ``qkv.weight``.
        :return tuple[torch.Tensor, int, int]:
            View shaped ``[heads, 3 * head_dim, hidden_size]``, ``head_dim``,
            and ``hidden_size``.
        """
        hidden_size = int(self.model_config.hidden_size)
        num_heads = int(self.model_config.num_attention_heads)
        head_dim = int(self.model_config.dim_head)
        expected_shape = (3 * hidden_size, hidden_size)
        if tuple(matrix.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected fused QKV parameter layout "
                f"{tuple(matrix.shape)}; expected {expected_shape} for per-head interleaved "
                "rows [Q_h, K_h, V_h]."
            )
        return matrix.view(num_heads, 3 * head_dim, hidden_size), head_dim, hidden_size

    def _newton_schulz_update(self, grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Apply Newton-Schulz orthogonalization to a gradient.

        :param torch.Tensor grad: Gradient tensor.
        :param int steps: Number of iteration steps.
        :return torch.Tensor: Orthogonalized update.
        """
        if grad.ndim != 2:
            return grad

        # Handle matrix orientation
        is_transpose = grad.size(0) > grad.size(1)
        working = grad.T if is_transpose else grad

        original_dtype = working.dtype
        if working.dtype == torch.float16:
            raise RuntimeError(
                "fp16/float16 gradients are not supported by MuonClip "
                "orthogonalization. Use bf16 or fp32."
            )
        if working.dtype == torch.bfloat16:
            # NS path computes in fp32 for stability, then casts back.
            working = working.float()
        norm = torch.linalg.norm(working)
        if norm == 0:
            return torch.zeros_like(grad)

        # Newton-Schulz iteration coefficients from Polar Express appendix.
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = working / (norm + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if X.dtype != original_dtype:
            X = X.to(original_dtype)

        return X.T if is_transpose else X

    def _normalize_muon_update(
        self, update: torch.Tensor, param_shape: torch.Size
    ) -> torch.Tensor:
        """Normalize orthogonalized update magnitude before applying it.

        Named modes:
        - ``legacy_compat``: NeoBERT compatibility scaling
        - ``original``: reference Muon scaling
        - ``spectral``: scale by sqrt(d_out / d_in)
        - ``match_rms_adamw``: reduced legacy-style scale
        - ``none``: no extra scaling

        :param torch.Tensor update: Orthogonalized update tensor.
        :param torch.Size param_shape: Shape of the owning parameter.
        :return torch.Tensor: Normalized update tensor.
        """
        if update.ndim != 2:
            return update

        d_out, d_in = int(param_shape[0]), int(param_shape[1])
        norm_factor = (
            str(getattr(self.config, "norm_factor", "legacy_compat"))
            .strip()
            .replace("-", "_")
            .lower()
        )
        if norm_factor == "none":
            scale = 1.0
        elif norm_factor == "legacy_compat":
            scale = 0.4 * max(d_out, d_in) ** 0.5
        elif norm_factor == "original":
            scale = max(1.0, d_out / max(d_in, 1)) ** 0.5
        elif norm_factor == "spectral":
            scale = (d_out / max(d_in, 1)) ** 0.5
        elif norm_factor == "match_rms_adamw":
            scale = 0.2 * max(d_out, d_in) ** 0.5
        else:
            raise ValueError(
                f"Unsupported norm_factor '{norm_factor}'. "
                "Expected one of: legacy_compat, original, spectral, "
                "match_rms_adamw, none."
            )

        return update * scale

    def _orthogonalize_update(self, grad: torch.Tensor) -> torch.Tensor:
        """Dispatch orthogonalization based on configuration.

        ``orthogonalization`` affects both the iteration scheme and the compute
        dtype on CUDA. ``polar_express`` prefers a fast CUDA work dtype
        (typically bf16), while ``newton_schulz`` upcasts bf16 inputs to fp32
        for the iteration before casting back to the original dtype.

        :param torch.Tensor grad: Gradient tensor.
        :return torch.Tensor: Orthogonalized update.
        """
        algo = getattr(self.config, "orthogonalization", "polar_express")
        if algo == "polar_express":
            return self._polar_express_update(grad, self.config.ns_steps)
        return self._newton_schulz_update(grad, self.config.ns_steps)

    def _polar_express_update(
        self, grad: torch.Tensor, steps: int = 5, eps: float = 1e-7
    ) -> torch.Tensor:
        """Apply Polar Express orthogonalization with adaptive coefficients.

        :param torch.Tensor grad: Gradient tensor.
        :param int steps: Number of iteration steps.
        :param float eps: Numerical stability epsilon.
        :return torch.Tensor: Orthogonalized update.
        """
        if grad.ndim != 2:
            return grad

        steps = max(1, steps)

        is_transpose = grad.size(0) > grad.size(1)
        working = grad.T if is_transpose else grad

        original_dtype = working.dtype
        if (
            self._polar_work_dtype is not None
            and working.is_cuda
            and working.dtype != self._polar_work_dtype
        ):
            # Polar path intentionally stays in a fast CUDA work dtype (bf16).
            working = working.to(self._polar_work_dtype)

        # Frobenius norm provides the needed scale control without expensive SVD
        spectral_norm = torch.linalg.norm(working)
        if spectral_norm == 0 or not torch.isfinite(spectral_norm):
            return torch.zeros_like(grad)

        working = working / (spectral_norm * 1.01 + eps)

        coeffs = self._get_polar_coefficients(steps)
        for a, b, c in coeffs:
            A = working @ working.T
            B = b * A + c * (A @ A)
            working = a * working + B @ working

        if working.dtype != original_dtype:
            working = working.to(original_dtype)

        return working.T if is_transpose else working

    def _get_polar_coefficients(self, steps: int) -> List[Tuple[float, float, float]]:
        """Return dampened coefficient schedule for Polar Express.

        Coefficients follow the Polar Express appendix schedule (see module references).

        :param int steps: Number of coefficients to return.
        :return list[tuple[float, float, float]]: Coefficient schedule.
        """
        cache = self._polar_coeff_cache
        if steps in cache:
            return cache[steps]

        coeffs_base: List[Tuple[float, float, float]] = [
            (8.28721201814563, -23.595886519098837, 17.300387312530933),
            (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
            (3.948690853482295, -2.908902115962949, 0.5518191394370137),
            (3.318419657370602, -2.488488024314874, 0.51004894012372),
            (2.300652019954817, -1.668903984574749, 0.4188073119525673),
            (1.891301407787398, -1.267995827194587, 0.3768040894852483),
            (1.875001480853448, -1.250001645399949, 0.3750001645474248),
            (1.875000000000000, -1.250000000000000, 0.375000000000000),
        ]

        # Dampening factor from Polar Express implementation; keeps updates stable.
        dampening_factor = 1.01
        coeffs = [
            (
                a / dampening_factor,
                b / (dampening_factor**3),
                c / (dampening_factor**5),
            )
            for (a, b, c) in coeffs_base[:-1]
        ]
        coeffs.append(coeffs_base[-1])

        if steps <= len(coeffs):
            result = coeffs[:steps]
            cache[steps] = result
            return result

        if not getattr(self, "_warned_polar_coeff_extrapolation", False):
            logger.warning(
                "Polar Express coefficients defined for "
                f"{len(coeffs)} steps; repeating final coefficient for steps={steps}. "
                "Consider lowering ns_steps."
            )
            self._warned_polar_coeff_extrapolation = True

        coeffs.extend([coeffs[-1]] * (steps - len(coeffs)))
        cache[steps] = coeffs
        return coeffs

    def _apply_qk_clipping(self) -> None:
        """Apply per-head QK-clipping using cached activations."""
        if not self.hook_system:
            return

        # Note: we scale weights *after* the optimizer step by design so clipping is
        # decoupled from the optimizer's momentum buffers; moving this earlier would
        # change the update dynamics.
        self._last_metrics.clear()
        max_attention_logit: Optional[float] = None

        layer_entries: Dict[int, Dict[str, torch.nn.Parameter]] = {}
        for group in self.param_groups:
            if not group["use_muon"]:
                continue

            for param, info in self._iter_group_params_with_info(group):
                if not info["is_qkv"] or info["layer_idx"] is None:
                    continue

                params = layer_entries.setdefault(info["layer_idx"], {})
                proj_type = info.get("proj_type", "qkv") or "qkv"
                params[proj_type] = param

        if not layer_entries:
            self.hook_system.clear()
            return

        for layer_idx, param_dict in layer_entries.items():
            (
                inputs,
                pad_mask,
                freqs_cis,
                packed_seqlens,
            ) = self.hook_system.get_layer_data(layer_idx)
            if inputs is None:
                continue
            layer = self.hook_system.layers.get(layer_idx)

            if "qkv" in param_dict:
                eta_per_head, layer_max = self._compute_eta_for_fused(
                    inputs=inputs,
                    param=param_dict["qkv"],
                    pad_mask=pad_mask,
                    freqs_cis=freqs_cis,
                    packed_seqlens=packed_seqlens,
                    layer=layer,
                )
                if eta_per_head is not None:
                    self._scale_qkv_weights(
                        param=param_dict["qkv"],
                        eta_per_head=eta_per_head,
                        alpha=self.config.clipping_alpha,
                    )
                    if layer_max is not None:
                        max_attention_logit = (
                            layer_max
                            if max_attention_logit is None
                            else max(max_attention_logit, layer_max)
                        )
                continue

            q_param = param_dict.get("q")
            k_param = param_dict.get("k")
            if q_param is None or k_param is None:
                continue

            eta_per_head, layer_max = self._compute_eta_for_separate(
                inputs=inputs,
                q_param=q_param,
                k_param=k_param,
                pad_mask=pad_mask,
                freqs_cis=freqs_cis,
                packed_seqlens=packed_seqlens,
                layer=layer,
            )
            if eta_per_head is None:
                continue

            self._scale_separate_projection(
                param=q_param,
                eta_per_head=eta_per_head,
                alpha=self.config.clipping_alpha,
                proj_type="q",
            )
            self._scale_separate_projection(
                param=k_param,
                eta_per_head=eta_per_head,
                alpha=self.config.clipping_alpha,
                proj_type="k",
            )
            if layer_max is not None:
                max_attention_logit = (
                    layer_max
                    if max_attention_logit is None
                    else max(max_attention_logit, layer_max)
                )

        self.hook_system.clear()
        if max_attention_logit is not None:
            self._last_metrics["train/max_attention_logit"] = max_attention_logit

    def _iter_group_params_with_info(
        self, group: Dict[str, Any]
    ) -> Iterator[Tuple[torch.nn.Parameter, Dict[str, Any]]]:
        """Yield current group parameters alongside their static metadata.

        :param dict[str, Any] group: Optimizer parameter group.
        :return Iterator[tuple[torch.nn.Parameter, dict[str, Any]]]:
            Current parameter and its associated metadata.
        :raises RuntimeError: If group params and metadata become desynchronized.
        """
        params = list(group.get("params", ()))
        param_info = list(group.get("param_info", ()))
        if len(params) != len(param_info):
            raise RuntimeError(
                "MuonClip parameter metadata is desynchronized from optimizer "
                "params; expected matching lengths for param_info and params."
            )
        return iter(zip(params, param_info, strict=True))

    def _compute_eta_for_fused(
        self,
        inputs: torch.Tensor,
        param: torch.nn.Parameter,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        packed_seqlens: Optional[list[list[int]]],
        layer: Optional[torch.nn.Module],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Compute per-head scaling factors for fused QKV weights.

        :param torch.Tensor inputs: Cached layer inputs.
        :param torch.nn.Parameter param: Fused QKV weight parameter.
        :param torch.Tensor | None pad_mask: Optional pad mask.
        :param torch.Tensor | None freqs_cis: Optional rotary frequencies.
        :param list[list[int]] | None packed_seqlens: Optional packed segment lengths.
        :param torch.nn.Module | None layer: Optional encoder layer (ngpt metadata).
        :return tuple[torch.Tensor | None, float | None]: Eta per head and max logit.
        """
        try:
            inputs = inputs.to(device=param.device, dtype=param.dtype)
            projections = torch.matmul(inputs, param.transpose(0, 1))
        except RuntimeError as exc:
            logger.error(f"Failed to compute fused QKV projections: {exc}")
            return None, None

        batch, seq_len, _ = projections.shape
        expected = 3 * self.model_config.hidden_size
        if projections.shape[-1] != expected:
            logger.error(
                "Unexpected fused QKV projection shape "
                f"{projections.shape} (expected last dim {expected})"
            )
            return None, None

        proj = projections.view(
            batch,
            seq_len,
            self.model_config.num_attention_heads,
            self.model_config.dim_head * 3,
        )
        xq, xk, _ = proj.chunk(3, dim=-1)

        return self._compute_eta_from_qk(
            xq=xq,
            xk=xk,
            pad_mask=pad_mask,
            freqs_cis=freqs_cis,
            packed_seqlens=packed_seqlens,
            layer=layer,
        )

    def _compute_eta_for_separate(
        self,
        inputs: torch.Tensor,
        q_param: torch.nn.Parameter,
        k_param: torch.nn.Parameter,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        packed_seqlens: Optional[list[list[int]]],
        layer: Optional[torch.nn.Module],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Compute per-head scaling factors for separate Q/K projections.

        :param torch.Tensor inputs: Cached layer inputs.
        :param torch.nn.Parameter q_param: Query projection weights.
        :param torch.nn.Parameter k_param: Key projection weights.
        :param torch.Tensor | None pad_mask: Optional pad mask.
        :param torch.Tensor | None freqs_cis: Optional rotary frequencies.
        :param list[list[int]] | None packed_seqlens: Optional packed segment lengths.
        :param torch.nn.Module | None layer: Optional encoder layer (ngpt metadata).
        :return tuple[torch.Tensor | None, float | None]: Eta per head and max logit.
        """
        try:
            inputs_q = inputs.to(device=q_param.device, dtype=q_param.dtype)
            inputs_k = inputs.to(device=k_param.device, dtype=k_param.dtype)
            q_proj = torch.matmul(inputs_q, q_param.transpose(0, 1))
            k_proj = torch.matmul(inputs_k, k_param.transpose(0, 1))
        except RuntimeError as exc:
            logger.error(f"Failed to compute separate Q/K projections: {exc}")
            return None, None

        batch, seq_len, _ = q_proj.shape
        head_dim = self.model_config.dim_head
        q_proj = q_proj.view(
            batch, seq_len, self.model_config.num_attention_heads, head_dim
        )
        k_proj = k_proj.view(
            batch, seq_len, self.model_config.num_attention_heads, head_dim
        )

        return self._compute_eta_from_qk(
            xq=q_proj,
            xk=k_proj,
            pad_mask=pad_mask,
            freqs_cis=freqs_cis,
            packed_seqlens=packed_seqlens,
            layer=layer,
        )

    def _compute_eta_from_qk(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        packed_seqlens: Optional[list[list[int]]],
        layer: Optional[torch.nn.Module],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Derive per-head eta values from Q and K projections.

        :param torch.Tensor xq: Query projections.
        :param torch.Tensor xk: Key projections.
        :param torch.Tensor | None pad_mask: Optional pad mask.
        :param torch.Tensor | None freqs_cis: Optional rotary frequencies.
        :param list[list[int]] | None packed_seqlens: Optional packed segment lengths.
        :param torch.nn.Module | None layer: Optional encoder layer (ngpt metadata).
        :return tuple[torch.Tensor | None, float | None]: Eta per head and max logit.
        """
        if self.model_config.rope and freqs_cis is not None:
            from neobert.model.rotary import apply_rotary_emb

            freqs_cis = freqs_cis.to(device=xq.device)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if self.model_config.ngpt:
            if layer is None or not hasattr(layer, "sqk"):
                logger.warning(
                    "MuonClip QK clipping skipped: ngpt enabled but sqk metadata missing."
                )
                return None, None
            xq, xk = self._apply_ngpt_qk_transform(xq, xk, layer)

        xq_heads = xq.transpose(1, 2)
        xk_heads = xk.transpose(1, 2)

        if self.model_config.ngpt:
            scale = (
                self.model_config.hidden_size / self.model_config.num_attention_heads
            ) ** 0.5
        else:
            scale = 1.0 / (self.model_config.dim_head**0.5)

        if packed_seqlens is not None:
            if pad_mask is not None:
                raise ValueError(
                    "packed_seqlens was provided but pad_mask is not None. "
                    "Packed attention uses block-diagonal bias; pad_mask should be None."
                )
            per_step_max = self._packed_attention_logit_max(
                xq_heads=xq_heads,
                xk_heads=xk_heads,
                packed_seqlens=packed_seqlens,
                scale=scale,
            )
        else:
            per_step_max = self._attention_logit_max(
                xq_heads=xq_heads,
                xk_heads=xk_heads,
                scale=scale,
                pad_mask=pad_mask,
            )
        mean_per_head = per_step_max.mean(dim=0)
        denom = torch.clamp(mean_per_head, min=1e-6)
        eta_per_head = (self.config.clipping_threshold / denom).clamp(max=1.0)

        global_max: Optional[float] = None
        if per_step_max.numel() > 0:
            candidate = per_step_max.max()
            if torch.isfinite(candidate):
                global_max = float(candidate.item())
        return eta_per_head, global_max

    def _attention_logit_max(
        self,
        xq_heads: torch.Tensor,
        xk_heads: torch.Tensor,
        scale: float,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample/per-head max attention logits with 2D tiling.

        :param torch.Tensor xq_heads: Query tensor of shape [B, H, S, D].
        :param torch.Tensor xk_heads: Key tensor of shape [B, H, S, D].
        :param float scale: Multiplicative scale applied to QK logits.
        :param torch.Tensor | None pad_mask: Optional additive pad mask.
        :return torch.Tensor: Per-sample max logits of shape [B, H].
        """
        if xq_heads.ndim != 4 or xk_heads.ndim != 4:
            raise ValueError(
                "Expected xq_heads/xk_heads to be rank-4 [B,H,S,D], got "
                f"xq={tuple(xq_heads.shape)} xk={tuple(xk_heads.shape)}"
            )
        if xq_heads.shape != xk_heads.shape:
            raise ValueError(
                "xq_heads and xk_heads must have the same shape, got "
                f"xq={tuple(xq_heads.shape)} xk={tuple(xk_heads.shape)}"
            )

        batch, heads, seq_len, _ = xq_heads.shape
        chunk_size = min(int(self.config.clipping_qk_chunk_size), seq_len)
        if chunk_size <= 0:
            raise ValueError(
                "clipping_qk_chunk_size must be >= 1, got "
                f"{self.config.clipping_qk_chunk_size}"
            )

        per_step_max = torch.full(
            (batch, heads),
            float("-inf"),
            device=xq_heads.device,
            dtype=xq_heads.dtype,
        )
        bias = None
        bias_is_full = False
        if pad_mask is not None:
            if not torch.is_tensor(pad_mask):
                raise TypeError("pad_mask must be a tensor when provided")
            bias = pad_mask.to(device=xq_heads.device, dtype=xq_heads.dtype)
            if bias.dim() == 2:
                bias = bias.view(batch, 1, 1, seq_len)
            elif bias.dim() == 4:
                if bias.shape[-1] != seq_len:
                    raise ValueError(
                        "pad_mask last dim must equal seq_len "
                        f"({bias.shape[-1]} != {seq_len})"
                    )
                if bias.shape[-2] not in (1, seq_len):
                    raise ValueError(
                        "pad_mask must have shape (B,1,1,S) or (B,1,S,S), got "
                        f"{tuple(bias.shape)}"
                    )
                bias_is_full = bias.shape[-2] == seq_len
            else:
                raise ValueError(
                    "pad_mask must have shape (B,S), (B,1,1,S), or (B,1,S,S); got "
                    f"{tuple(bias.shape)}"
                )

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(seq_len, q_start + chunk_size)
            q = xq_heads[:, :, q_start:q_end, :]
            q_max = torch.full(
                (batch, heads),
                float("-inf"),
                device=xq_heads.device,
                dtype=xq_heads.dtype,
            )
            for k_start in range(0, seq_len, chunk_size):
                k_end = min(seq_len, k_start + chunk_size)
                k = xk_heads[:, :, k_start:k_end, :]
                logits = torch.matmul(q, k.transpose(-2, -1)) * scale

                if bias is not None:
                    if bias_is_full:
                        logits = logits + bias[:, :, q_start:q_end, k_start:k_end]
                    else:
                        logits = logits + bias[..., k_start:k_end]

                chunk_max = logits.amax(dim=(-2, -1))
                q_max = torch.maximum(q_max, chunk_max)

            per_step_max = torch.maximum(per_step_max, q_max)

        return per_step_max

    def _packed_attention_logit_max(
        self,
        xq_heads: torch.Tensor,
        xk_heads: torch.Tensor,
        packed_seqlens: list[list[int]],
        scale: float,
    ) -> torch.Tensor:
        """Compute per-sample/per-head max attention logits for packed sequences.

        Packed mode is block-diagonal attention by segment; cross-segment logits
        must be ignored.

        :param torch.Tensor xq_heads: Query tensor of shape [B, H, S, D].
        :param torch.Tensor xk_heads: Key tensor of shape [B, H, S, D].
        :param list[list[int]] packed_seqlens: Segment lengths per batch item.
        :param float scale: Multiplicative scale applied to QK logits.
        :return torch.Tensor: Per-sample max logits of shape [B, H].
        """
        if xq_heads.ndim != 4 or xk_heads.ndim != 4:
            raise ValueError(
                "Expected xq_heads/xk_heads to be rank-4 [B,H,S,D], got "
                f"xq={tuple(xq_heads.shape)} xk={tuple(xk_heads.shape)}"
            )
        if xq_heads.shape != xk_heads.shape:
            raise ValueError(
                "xq_heads and xk_heads must have the same shape, got "
                f"xq={tuple(xq_heads.shape)} xk={tuple(xk_heads.shape)}"
            )

        batch, heads, seq_len, _ = xq_heads.shape
        if len(packed_seqlens) != batch:
            raise ValueError(
                "packed_seqlens must have length equal to batch size: "
                f"len(packed_seqlens)={len(packed_seqlens)} vs batch={batch}"
            )

        per_step_max = torch.full(
            (batch, heads),
            float("-inf"),
            device=xq_heads.device,
            dtype=xq_heads.dtype,
        )

        chunk_size = min(int(self.config.clipping_qk_chunk_size), seq_len)
        if chunk_size <= 0:
            raise ValueError(
                "clipping_qk_chunk_size must be >= 1, got "
                f"{self.config.clipping_qk_chunk_size}"
            )

        for batch_idx in range(batch):
            start = 0
            for seg_len in packed_seqlens[batch_idx]:
                seg_len_i = int(seg_len)
                if seg_len_i <= 0:
                    continue
                end = start + seg_len_i
                if end > seq_len:
                    raise ValueError(
                        "packed_seqlens contains segment lengths that exceed seq_len: "
                        f"sample={batch_idx} start={start} seg_len={seg_len_i} "
                        f"seq_len={seq_len}"
                    )

                q = xq_heads[batch_idx, :, start:end, :]
                k = xk_heads[batch_idx, :, start:end, :]
                for q_start in range(0, seg_len_i, chunk_size):
                    q_end = min(seg_len_i, q_start + chunk_size)
                    q_chunk = q[:, q_start:q_end, :]
                    q_max = torch.full(
                        (heads,),
                        float("-inf"),
                        device=xq_heads.device,
                        dtype=xq_heads.dtype,
                    )
                    for k_start in range(0, seg_len_i, chunk_size):
                        k_end = min(seg_len_i, k_start + chunk_size)
                        k_chunk = k[:, k_start:k_end, :]
                        logits = (
                            torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
                        )
                        seg_max = logits.amax(dim=(-2, -1))
                        q_max = torch.maximum(q_max, seg_max)
                    per_step_max[batch_idx] = torch.maximum(
                        per_step_max[batch_idx], q_max
                    )
                start = end

        return per_step_max

    def _apply_ngpt_qk_transform(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        layer: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply NormNeoBERT (nGPT) Q/K normalization and scaling.

        :param torch.Tensor xq: Query projections.
        :param torch.Tensor xk: Key projections.
        :param torch.nn.Module layer: Encoder layer with sqk parameters.
        :return tuple[torch.Tensor, torch.Tensor]: Transformed Q and K tensors.
        """
        sqk = layer.sqk
        sqk_init_value = getattr(layer, "sqk_init_value", 1.0)
        sqk_init_scaling = getattr(layer, "sqk_init_scaling", 1.0)
        sqk = (sqk * (sqk_init_value / sqk_init_scaling)).view(
            1,
            1,
            self.model_config.num_attention_heads,
            self.model_config.dim_head,
        )
        sqk = sqk.to(device=xq.device, dtype=xq.dtype)

        def _justnorm(x: torch.Tensor) -> torch.Tensor:
            """Match NormEncoderBlock justnorm behavior.

            :param torch.Tensor x: Input tensor.
            :return torch.Tensor: L2-normalized tensor.
            """
            return x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        return sqk * _justnorm(xq), sqk * _justnorm(xk)

    def _scale_qkv_weights(
        self, param: torch.nn.Parameter, eta_per_head: torch.Tensor, alpha: float
    ) -> None:
        """Scale Q and K portions of fused QKV weight matrix.

        :param torch.nn.Parameter param: QKV weight parameter.
        :param torch.Tensor eta_per_head: Scaling factors per head.
        :param float alpha: Q/K scaling balance (0.5 = equal).
        """
        param_view, head_dim, _ = self._view_interleaved_qkv_matrix(param)
        num_heads = int(self.model_config.num_attention_heads)

        # Ensure scaling factors are finite and on the right device/dtype
        eta = eta_per_head.to(device=param.device, dtype=param.dtype)
        eta = torch.clamp(eta, min=1e-6, max=1.0)

        # Compute per-head scaling powers
        eta_q = eta.pow(alpha).view(num_heads, 1, 1)
        eta_k = eta.pow(1 - alpha).view(num_heads, 1, 1)

        q_slice = slice(0, head_dim)
        k_slice = slice(head_dim, 2 * head_dim)
        param_view[:, q_slice].mul_(eta_q)  # Query rows per head
        param_view[:, k_slice].mul_(eta_k)  # Key rows per head
        # Value rows remain unchanged (slice 2*head_dim:3*head_dim)

    def _scale_separate_projection(
        self,
        param: torch.nn.Parameter,
        eta_per_head: torch.Tensor,
        alpha: float,
        proj_type: str,
    ) -> None:
        """Scale separate Q or K projection weights.

        :param torch.nn.Parameter param: Projection weight parameter.
        :param torch.Tensor eta_per_head: Scaling factors per head.
        :param float alpha: Q/K scaling balance (0.5 = equal).
        :param str proj_type: Projection type ('q' or 'k').
        """
        hidden_size = self.model_config.hidden_size
        num_heads = self.model_config.num_attention_heads
        head_dim = hidden_size // num_heads

        if param.shape[0] != hidden_size:
            raise RuntimeError(
                f"Unexpected {proj_type}-projection shape {tuple(param.shape)}; "
                f"expected first dimension {hidden_size}"
            )

        eta = eta_per_head.to(device=param.device, dtype=param.dtype)
        eta = torch.clamp(eta, min=1e-6, max=1.0)

        power = alpha if proj_type == "q" else (1 - alpha)
        eta_power = eta.pow(power).view(num_heads, 1, 1)

        param_view = param.view(num_heads, head_dim, -1)
        param_view.mul_(eta_power)

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics for logging and clear internal storage.

        :return dict[str, float]: Metrics collected during optimization.
        """
        metrics = dict(self._last_metrics)
        self._last_metrics.clear()
        return metrics

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients and hook statistics.

        :param bool set_to_none: Whether to set grads to None.
        """
        super().zero_grad(set_to_none=set_to_none)
        if self.hook_system:
            self.hook_system.clear()
