"""
MuonClip Optimizer for NeoBERT Encoder Models

Adapted for bidirectional encoders with:
- Fused QKV projection support
- Memory-efficient attention hook system
- Full distributed training support (DDP, DeepSpeed)
- Comprehensive error handling and validation

Author: Peter Szemraj
Date: 2025-01-10

References:
- Kimi K2 Technical Report: https://moonshotai.github.io/Kimi-K2/
- Original Muon: https://github.com/KellerJordan/Muon
- MuonClip: https://github.com/GAD-cell/muon-clip
- DISCO optimizer: https://github.com/SDLAML/disco
- Polar Express: https://arxiv.org/abs/2505.16932
"""

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


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

    # Architecture adaptation
    clipping_layers_mapping: Dict[str, str] = field(default_factory=dict)

    # Monitoring and debugging
    detect_anomalies: bool = False  # Enable gradient anomaly detection

    # Orthogonalization control
    orthogonalization: str = "polar_express"
    algorithm: Optional[str] = None  # Alias for orthogonalization
    polar_express: Optional[bool] = None  # Legacy toggle

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.lr < 1, f"lr must be in (0, 1), got {self.lr}"
        assert 0 <= self.muon_beta < 1, (
            f"muon_beta must be in [0, 1), got {self.muon_beta}"
        )
        assert 0 <= self.muon_decay < 1, (
            f"muon_decay must be in [0, 1), got {self.muon_decay}"
        )
        assert 1 <= self.ns_steps <= 20, (
            f"ns_steps must be in [1, 20], got {self.ns_steps}"
        )
        assert 0 < self.clipping_threshold <= 1000, (
            f"clipping_threshold must be in (0, 1000], got {self.clipping_threshold}"
        )
        assert 0 <= self.clipping_alpha <= 1, (
            f"clipping_alpha must be in [0, 1], got {self.clipping_alpha}"
        )

        # Warnings for suboptimal settings
        if self.ns_steps < 3:
            warnings.warn(
                f"ns_steps={self.ns_steps} may not provide sufficient orthogonalization. Recommended: 5-9"
            )
        if self.clipping_threshold > 200:
            warnings.warn(
                f"clipping_threshold={self.clipping_threshold} is very high. You may not see clipping effects."
            )
        if self.clipping_threshold < 30:
            warnings.warn(
                f"clipping_threshold={self.clipping_threshold} is very low. Risk of over-constraining attention."
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
                )
            self.clipping_layers_mapping = normalised_mapping

        _, beta2 = self.adam_betas
        if beta2 < 0.98:
            warnings.warn(
                f"adam_betas second moment {beta2} is unusually low; "
                "encoder pretraining typically uses >= 0.98.",
                RuntimeWarning,
            )

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


# ============================================================================
# Attention Hook System for NeoBERT
# ============================================================================


class NeoBERTAttentionHooks:
    """
    Lightweight hook system to capture attention inputs for QK clipping.

    We record:
      - The normalized attention input fed into each layer's QKV (or Q/K) projection.
      - The pad mask and rotary embeddings passed to the encoder block.
    The expensive QK statistics are computed lazily during the optimizer step.
    """

    def __init__(self, model_config, layer_mapping: Optional[Dict[str, str]] = None):
        self.config = model_config
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.layer_mapping = layer_mapping or {}

        self.layer_inputs: Dict[int, torch.Tensor] = {}
        self.layer_pad_masks: Dict[int, Optional[torch.Tensor]] = {}
        self.layer_freqs: Dict[int, Optional[torch.Tensor]] = {}
        self.layers: Dict[int, torch.nn.Module] = {}

        self.enabled = True
        self.hook_handles: List[RemovableHandle] = []

        self._validate_config()

    def _validate_config(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.config.num_attention_heads})"
            )

    def register_hooks(self, model: torch.nn.Module) -> int:
        """Register hooks across all transformer encoder layers.

        Returns
        -------
        int
            Number of hook handles that were successfully registered.
        """
        layers = self._resolve_transformer_layers(model)
        if not layers:
            raise RuntimeError("No transformer layers found for MuonClip hooks")

        num_hooks = 0
        for idx, layer in enumerate(layers):
            self.layers[idx] = layer

            if hasattr(layer, "qkv"):
                handle = layer.qkv.register_forward_hook(
                    self._create_qkv_input_hook(idx)
                )
                self.hook_handles.append(handle)
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
                handle = q_proj.register_forward_hook(self._create_qkv_input_hook(idx))
                self.hook_handles.append(handle)
                num_hooks += 1

            block_handle = layer.register_forward_hook(
                self._create_block_context_hook(idx)
            )
            self.hook_handles.append(block_handle)
            num_hooks += 1

            logger.debug(f"Registered MuonClip hooks on layer {idx}")

        logger.info(f"Registered {num_hooks} MuonClip hooks")
        return num_hooks

    def _resolve_transformer_layers(
        self, model: torch.nn.Module
    ) -> Optional[Sequence[torch.nn.Module]]:
        """Return the sequence of encoder layers exposed by ``model``."""
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

    def _create_qkv_input_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not self.enabled or not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            self.layer_inputs[layer_idx] = x.detach()

        return hook_fn

    def _create_block_context_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if not self.enabled or not inputs:
                return

            pad_mask = inputs[1] if len(inputs) > 1 else None
            freqs_cis = inputs[2] if len(inputs) > 2 else None

            self.layer_pad_masks[layer_idx] = (
                pad_mask.detach() if torch.is_tensor(pad_mask) else pad_mask
            )
            self.layer_freqs[layer_idx] = (
                freqs_cis.detach() if torch.is_tensor(freqs_cis) else freqs_cis
            )

        return hook_fn

    def get_layer_data(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return (
            self.layer_inputs.get(layer_idx),
            self.layer_pad_masks.get(layer_idx),
            self.layer_freqs.get(layer_idx),
        )

    def clear(self):
        self.layer_inputs.clear()
        self.layer_pad_masks.clear()
        self.layer_freqs.clear()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


# ============================================================================
# Optimizer Implementation
# ============================================================================


class MuonClipOptimizer(Optimizer):
    """
    MuonClip optimizer for NeoBERT encoder models.

    Combines:
    - Muon (orthogonalized gradients) for 2D parameters
    - Adam for 1D parameters
    - Optional QK-clipping for attention stability

    Adapted for NeoBERT's architecture:
    - Fused QKV projections
    - Bidirectional attention
    - transformer_encoder layer structure
    """

    def __init__(
        self, model: torch.nn.Module, model_config: Any, config: MuonClipConfig
    ):
        """Initialize the optimizer and attach attention hooks as needed."""
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

    def _validate_model(self, model):
        """Validate model architecture compatibility."""
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

    def _build_param_groups(self, model) -> List[Dict]:
        """
        Build parameter groups for hybrid Muon+Adam optimization.

        Muon: 2D weight matrices (QKV, W_o, FFN weights)
        Adam: 1D parameters (biases, LayerNorm weights)
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
        """Discover the base encoder module and its layers inside ``model``."""
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
    def step(self, closure=None):
        """
        Perform optimization step.

        Order of operations:
        1. Apply Muon updates to 2D parameters
        2. Apply Adam updates to 1D parameters
        3. Apply QK-clipping (if enabled and past warmup)
        4. Collect metrics
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
            if self._step >= self.config.clipping_warmup_steps:
                self._apply_qk_clipping()
            else:
                self.hook_system.clear()
                self._last_metrics.clear()

        self._step += 1
        return loss

    def _muon_step(self, group):
        """Apply Muon update with Newton-Schulz orthogonalization."""
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

    def _adam_step(self, group):
        """Apply standard Adam update to 1D parameters."""
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
        """
        Apply Newton-Schulz orthogonalization to gradient.

        This orthogonalizes 2D gradients to make updates scale-free.
        """
        if grad.ndim != 2:
            return grad

        # Handle matrix orientation
        is_transpose = grad.size(0) > grad.size(1)
        working = grad.T if is_transpose else grad

        norm = torch.linalg.norm(working)
        if norm == 0:
            return torch.zeros_like(grad)

        # Newton-Schulz iteration
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = working / (norm + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        # RMS scaling for Adam lr compatibility
        # Factor: 0.4 * sqrt(max_dim)
        scale_factor = 0.4 * max(working.size(0), working.size(1)) ** 0.5
        X = scale_factor * X

        return X.T if is_transpose else X

    def _orthogonalize_update(self, grad: torch.Tensor) -> torch.Tensor:
        """Dispatch orthogonalization based on configuration."""
        algo = getattr(self.config, "orthogonalization", "polar_express")
        if algo == "polar_express":
            return self._polar_express_update(grad, self.config.ns_steps)
        return self._newton_schulz_update(grad, self.config.ns_steps)

    def _polar_express_update(
        self, grad: torch.Tensor, steps: int = 5, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Apply Polar Express orthogonalization with adaptive coefficient schedule.
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
        """Return dampened coefficient schedule for Polar Express algorithm."""
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

        coeffs.extend([coeffs[-1]] * (steps - len(coeffs)))
        cache[steps] = coeffs
        return coeffs

    def _apply_qk_clipping(self):
        """
        Apply per-head QK-clipping using cached activations.
        """
        if not self.hook_system:
            return

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
            inputs, pad_mask, freqs_cis = self.hook_system.get_layer_data(layer_idx)
            if inputs is None:
                continue

            if "qkv" in param_dict:
                eta_per_head, layer_max = self._compute_eta_for_fused(
                    inputs,
                    param_dict["qkv"],
                    pad_mask,
                    freqs_cis,
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
                inputs,
                q_param,
                k_param,
                pad_mask,
                freqs_cis,
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
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Compute per-head scaling factors for fused QKV weights."""
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
            3,
            self.model_config.num_attention_heads,
            self.model_config.dim_head,
        )
        xq = proj[:, :, 0]
        xk = proj[:, :, 1]

        return self._compute_eta_from_qk(xq, xk, pad_mask, freqs_cis)

    def _compute_eta_for_separate(
        self,
        inputs: torch.Tensor,
        q_param: torch.nn.Parameter,
        k_param: torch.nn.Parameter,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Compute per-head scaling factors for separate Q/K projections."""
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

        return self._compute_eta_from_qk(q_proj, k_proj, pad_mask, freqs_cis)

    def _compute_eta_from_qk(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Derive per-head eta values from Q and K projections."""
        if self.model_config.rope and freqs_cis is not None:
            from ..model.rotary import apply_rotary_emb

            freqs_cis = freqs_cis.to(device=xq.device)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq_heads = xq.transpose(1, 2)
        xk_heads = xk.transpose(1, 2)

        attn_logits = torch.matmul(xq_heads, xk_heads.transpose(-2, -1))
        attn_logits = attn_logits / (self.model_config.dim_head**0.5)

        if pad_mask is not None:
            attn_logits = attn_logits + pad_mask.to(
                device=attn_logits.device, dtype=attn_logits.dtype
            )

        per_step_max = attn_logits.amax(dim=(-2, -1))  # [batch, heads]
        mean_per_head = per_step_max.mean(dim=0)
        denom = torch.clamp(mean_per_head, min=1e-6)
        eta_per_head = (self.config.clipping_threshold / denom).clamp(max=1.0)

        global_max = per_step_max.max().item() if per_step_max.numel() > 0 else None
        return eta_per_head, global_max

    def _scale_qkv_weights(
        self, param: torch.nn.Parameter, eta_per_head: torch.Tensor, alpha: float
    ):
        """
        Scale Q and K portions of fused QKV weight matrix.

        QKV weight shape: [hidden_size, 3*hidden_size]
        Need to scale Q and K per-head, leave V unchanged.

        Args:
            param: QKV weight parameter
            eta_per_head: Scaling factors per head [num_heads]
            alpha: Q/K scaling balance (0.5 = equal)
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

        # Reshape parameter to [3, num_heads, head_dim, hidden_size]
        # In-place scaling keeps the original tensor layout intact.
        param_view = param.view(3, num_heads, head_dim, hidden_size)

        param_view[0].mul_(eta_q)  # Query rows
        param_view[1].mul_(eta_k)  # Key rows
        # Value rows remain unchanged (param_view[2])

    def _scale_separate_projection(
        self,
        param: torch.nn.Parameter,
        eta_per_head: torch.Tensor,
        alpha: float,
        proj_type: str,
    ):
        """
        Scale separate Q or K projection weights when architectures do not use fused QKV.

        Args:
            param: Projection weight parameter
            eta_per_head: Scaling factors per head [num_heads]
            alpha: Q/K scaling balance (0.5 = equal)
            proj_type: 'q' or 'k'
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

    def get_metrics(self) -> Dict:
        """
        Get metrics for logging.

        Returns metrics dict and clears internal storage.
        """
        metrics = dict(self._last_metrics)
        self._last_metrics.clear()
        return metrics

    def zero_grad(self, set_to_none: bool = True):
        """
        Override zero_grad to clear hook statistics.

        Important: Hook statistics should be cleared after each step
        to prevent memory leaks.
        """
        super().zero_grad(set_to_none=set_to_none)
        if self.hook_system:
            self.hook_system.clear()
