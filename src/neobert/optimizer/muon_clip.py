"""MuonClip optimizer for NeoBERT encoders.

Adapted for fused QKV projections, attention hooks, and distributed training.
References: Kimi K2 report, Muon, MuonClip, DISCO, Polar Express.
"""

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


# Configuration


@dataclass
class MuonClipConfig:
    """Configuration container for the MuonClip optimizer."""

    # Learning rates
    lr: float = 1e-4

    # Muon parameters (for 2D weight matrices)
    muon_beta: float = 0.95  # Momentum coefficient
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

    # Orthogonalization control
    orthogonalization: str = "polar_express"
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

        self.orthogonalization = algo
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

        self.layer_inputs: Dict[int, torch.Tensor] = {}
        self.layer_pad_masks: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_freqs: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_packed_seqlens: Dict[int, Optional[list[list[int]]]] = {}
        self.layers: Dict[int, torch.nn.Module] = {}

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

                if hasattr(layer, "qkv"):
                    handle = layer.qkv.register_forward_hook(
                        self._create_qkv_input_hook(idx)
                    )
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
                    handle = q_proj.register_forward_hook(
                        self._create_qkv_input_hook(idx)
                    )
                    registered_handles.append(handle)
                    num_hooks += 1

                block_handle = layer.register_forward_hook(
                    self._create_block_context_hook(idx)
                )
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

    def _create_qkv_input_hook(
        self, layer_idx: int
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        """Create a hook capturing QKV input activations.

        :param int layer_idx: Encoder layer index.
        :return Callable: Hook function for forward pass.
        """

        def hook_fn(
            module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            """Store the QKV input tensor for a layer.

            :param torch.nn.Module module: Hooked module.
            :param tuple[Any, ...] inputs: Forward inputs.
            :param Any output: Forward output (unused).
            """
            if not self.enabled or not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            # During gradient accumulation, we intentionally keep only the latest
            # microbatch activation to bound memory/compute overhead. This biases
            # QK clipping stats toward the most recent microbatch by design.
            # We cache pre-projection inputs (not Q/K) to avoid duplicating large
            # QKV tensors in CPU memory; projections are recomputed only on clip
            # steps (interval-gated) to keep the steady-state overhead low.
            # Move to CPU to avoid retaining GPU activations when clipping is enabled.
            self.layer_inputs[layer_idx] = x.detach().to("cpu")

        return hook_fn

    def _create_block_context_hook(
        self, layer_idx: int
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        """Create a hook capturing pad mask and rotary embeddings.

        :param int layer_idx: Encoder layer index.
        :return Callable: Hook function for forward pass.
        """

        def hook_fn(
            module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            """Store pad mask and rotary embeddings for a layer.

            :param torch.nn.Module module: Hooked module.
            :param tuple[Any, ...] inputs: Forward inputs.
            :param Any output: Forward output (unused).
            """
            if not self.enabled or not inputs:
                return

            pad_mask = inputs[1] if len(inputs) > 1 else None
            freqs_cis = inputs[2] if len(inputs) > 2 else None
            packed_seqlens = inputs[3] if len(inputs) > 3 else None

            self.layer_pad_masks[layer_idx] = (
                pad_mask.detach().to("cpu") if torch.is_tensor(pad_mask) else pad_mask
            )
            self.layer_freqs[layer_idx] = (
                freqs_cis.detach().to("cpu")
                if torch.is_tensor(freqs_cis)
                else freqs_cis
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

        return hook_fn

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
        self.layer_inputs.clear()
        self.layer_pad_masks.clear()
        self.layer_freqs.clear()
        self.layer_packed_seqlens.clear()

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


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
        if not getattr(self.config, "enable_clipping", False):
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
                "MuonClip step desync: trainer update_step=%s != optimizer._step=%s. "
                "Clipping/capture schedules may misalign.",
                int(update_step),
                int(self._step),
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
                "Detected separate attention projections using mapping %s",
                self._layer_mapping,
            )

        logger.debug("Model architecture validation passed")

    def _build_param_groups(self, model: torch.nn.Module) -> List[Dict]:
        """Build parameter groups for hybrid Muon+Adam optimization.

        :param torch.nn.Module model: Model to inspect.
        :return list[dict]: Parameter groups for the optimizer.
        """
        muon_params = []
        adam_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Extract layer index for QKV tracking
            layer_idx = None
            if "transformer_encoder" in name:
                match = re.search(r"transformer_encoder\.(\d+)", name)
                if match:
                    layer_idx = int(match.group(1))

            proj_type = None
            if "qkv" in name:
                proj_type = "qkv"
            else:
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

            param_info = {
                "param": param,
                "name": name,
                "layer_idx": layer_idx,
                "is_qkv": is_qkv,
                "proj_type": proj_type,
            }

            # 2D parameters -> Muon, 1D parameters -> Adam
            if param.ndim == 2:
                muon_params.append(param_info)
            else:
                adam_params.append(param_info)

        groups = []

        if muon_params:
            groups.append(
                {
                    "params": [p["param"] for p in muon_params],
                    "param_info": muon_params,
                    "lr": self.config.lr,
                    "beta": self.config.muon_beta,
                    "weight_decay": self.config.muon_decay,
                    "use_muon": True,
                }
            )
            logger.info(f"Muon group: {len(muon_params)} parameters")

        if adam_params:
            groups.append(
                {
                    "params": [p["param"] for p in adam_params],
                    "param_info": adam_params,
                    "lr": self.config.lr,
                    "betas": self.config.adam_betas,
                    "eps": self.config.adam_eps,
                    "weight_decay": self.config.adam_decay,
                    "use_muon": False,
                }
            )
            logger.info(f"Adam group: {len(adam_params)} parameters")

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
        if self.config.enable_clipping and self.hook_system:
            should_clip = self.should_clip_update(self._step)
            if should_clip:
                if not self.hook_system.layer_inputs:
                    logger.warning(
                        "MuonClip scheduled at update_step=%d but no activations were captured. "
                        "Clipping will be skipped. This usually means prepare_for_forward() "
                        "was not called on the correct microbatch (or hooks were disabled/wrapped).",
                        self._step,
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

        :param dict[str, Any] state_dict: Optimizer state dictionary.
        """
        payload = dict(state_dict)
        muonclip_step = payload.pop("muonclip_step", None)
        super().load_state_dict(payload)

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

    def _muon_step(self, group: Dict[str, Any]) -> None:
        """Apply Muon update with orthogonalization.

        :param dict[str, Any] group: Parameter group metadata.
        """
        for param in group["params"]:
            if param.grad is None:
                continue

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
            state["momentum_buffer"].mul_(group["beta"]).add_(
                grad, alpha=1 - group["beta"]
            )

            # Orthogonalize 2D gradients using Newton-Schulz
            update = self._orthogonalize_update(state["momentum_buffer"])

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
        if working.dtype in (torch.float16, torch.bfloat16):
            working = working.float()
        norm = torch.linalg.norm(working)
        if norm == 0:
            return torch.zeros_like(grad)

        # Newton-Schulz iteration coefficients from Polar Express appendix.
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = working / (norm + 1e-5)

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        # RMS scaling for Adam lr compatibility
        # Factor: 0.4 * sqrt(max_dim) (Polar Express appendix).
        scale_factor = 0.4 * max(working.size(0), working.size(1)) ** 0.5
        X = scale_factor * X
        if X.dtype != original_dtype:
            X = X.to(original_dtype)

        return X.T if is_transpose else X

    def _orthogonalize_update(self, grad: torch.Tensor) -> torch.Tensor:
        """Dispatch orthogonalization based on configuration.

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

        scale_factor = 0.4 * max(working.size(0), working.size(1)) ** 0.5
        working = scale_factor * working

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
                "Polar Express coefficients defined for %s steps; repeating final "
                "coefficient for steps=%s. Consider lowering ns_steps.",
                len(coeffs),
                steps,
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

            for info in group["param_info"]:
                if not info["is_qkv"] or info["layer_idx"] is None:
                    continue

                params = layer_entries.setdefault(info["layer_idx"], {})
                proj_type = info.get("proj_type", "qkv") or "qkv"
                params[proj_type] = info["param"]

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
                "Unexpected fused QKV projection shape %s (expected last dim %d)",
                projections.shape,
                expected,
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
            from ..model.rotary import apply_rotary_emb

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
        # Dimensions derived from model config
        hidden_size = self.model_config.hidden_size
        num_heads = self.model_config.num_attention_heads
        head_dim = hidden_size // num_heads

        if param.numel() != 3 * hidden_size * hidden_size:
            raise RuntimeError(
                f"Unexpected QKV parameter shape {tuple(param.shape)} for hidden_size={hidden_size}"
            )

        # Ensure scaling factors are finite and on the right device/dtype
        eta = eta_per_head.to(device=param.device, dtype=param.dtype)
        eta = torch.clamp(eta, min=1e-6, max=1.0)

        # Compute per-head scaling powers
        eta_q = eta.pow(alpha).view(num_heads, 1, 1)
        eta_k = eta.pow(1 - alpha).view(num_heads, 1, 1)

        # Reshape parameter to [num_heads, 3 * head_dim, hidden_size] to match
        # the model's per-head interleaved fused-QKV layout.
        param_view = param.view(num_heads, 3 * head_dim, hidden_size)
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
