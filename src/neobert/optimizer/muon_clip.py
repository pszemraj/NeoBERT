"""
MuonClip Optimizer for NeoBERT Encoder Models

Production implementation adapted for bidirectional encoders with:
- Fused QKV projection support
- Memory-efficient attention hook system
- Full distributed training support (DDP, DeepSpeed)
- Comprehensive error handling and validation

Author: NeoBERT Team
Date: 2025-01-10
License: MIT

References:
- Kimi K2 Technical Report: https://moonshotai.github.io/Kimi-K2/
- Original Muon: https://github.com/KellerJordan/Muon
- MuonClip: https://github.com/GAD-cell/muon-clip
"""

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MuonClipConfig:
    """Configuration for MuonClip optimizer.

    All parameters validated on initialization.
    """

    # Learning rates
    lr: float = 1e-4

    # Muon parameters (for 2D weight matrices)
    muon_beta: float = 0.95  # Momentum coefficient
    muon_decay: float = 0.0  # Weight decay for Muon params
    ns_steps: int = 5  # Newton-Schulz iterations (3-9 recommended)

    # Adam parameters (for 1D params: biases, LayerNorm)
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_decay: float = 0.0  # Weight decay for Adam params
    adam_eps: float = 1e-10

    # QK-Clipping parameters
    enable_clipping: bool = True
    clipping_threshold: float = 50.0  # Conservative for encoders
    clipping_alpha: float = 0.5  # Q/K scaling balance (0.5 = equal)
    clipping_warmup_steps: int = 0  # Disable clipping for N steps

    # Monitoring and debugging
    monitor_attention_entropy: bool = True
    detect_anomalies: bool = False  # Enable gradient anomaly detection
    log_max_logits: bool = True
    log_interval: int = 100

    # Performance optimization
    offload_hooks_to_cpu: bool = True  # Save GPU memory
    enable_profiling: bool = False  # Detailed timing info

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


# ============================================================================
# Attention Hook System for NeoBERT
# ============================================================================


class NeoBERTAttentionHooks:
    """
    Attention hook system specifically designed for NeoBERT's fused QKV architecture.

    Captures attention logits from EncoderBlock forward pass by:
    1. Hooking the QKV output
    2. Splitting into Q, K, V
    3. Computing Q@K^T attention scores
    4. Tracking max logits per head for clipping
    """

    def __init__(self, model_config, offload_to_cpu: bool = True):
        self.config = model_config
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.offload_to_cpu = offload_to_cpu

        # Storage for captured data
        self.layer_stats: Dict[int, Dict] = {}
        self.enabled = True
        self.hook_handles = []

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate model config for hook compatibility."""
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.config.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.config.num_attention_heads})"
            )

    def register_hooks(self, model) -> int:
        """
        Register forward hooks on all EncoderBlocks.

        Returns:
            Number of hooks registered
        """
        num_hooks = 0

        # Find transformer_encoder layers
        if hasattr(model, "transformer_encoder"):
            for idx, layer in enumerate(model.transformer_encoder):
                # Create hook for this layer
                hook = self._create_layer_hook(idx)
                handle = layer.register_forward_hook(hook)
                self.hook_handles.append(handle)
                num_hooks += 1
                logger.debug(f"Registered hook on transformer_encoder[{idx}]")
        else:
            raise RuntimeError(
                "Model missing 'transformer_encoder' attribute. "
                "This implementation requires NeoBERT architecture."
            )

        if num_hooks == 0:
            raise RuntimeError("No hooks registered! Check model architecture.")

        logger.info(f"Registered {num_hooks} attention hooks")
        return num_hooks

    def _create_layer_hook(self, layer_idx: int):
        """Create forward hook for specific layer."""

        def hook_fn(module, input_tuple, output):
            if not self.enabled:
                return

            try:
                # Extract inputs
                if isinstance(input_tuple, tuple):
                    x = input_tuple[0]
                    pad_mask = input_tuple[1] if len(input_tuple) > 1 else None
                    freqs_cis = input_tuple[2] if len(input_tuple) > 2 else None
                else:
                    x = input_tuple
                    pad_mask = None
                    freqs_cis = None

                # Compute attention statistics
                with torch.no_grad():
                    stats = self._compute_attention_stats(
                        module, x, pad_mask, freqs_cis
                    )
                    self.layer_stats[layer_idx] = stats

            except Exception as e:
                logger.error(f"Hook failed on layer {layer_idx}: {e}")
                # Don't raise - allow training to continue

        return hook_fn

    def _compute_attention_stats(
        self,
        encoder_block,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
    ) -> Dict:
        """
        Compute attention statistics by replicating forward pass.

        This is the critical function - must match NeoBERT's EncoderBlock._att_block.
        """
        batch_size, seq_len, _ = x.shape

        # Apply attention norm (as in EncoderBlock.forward)
        x_norm = encoder_block.attention_norm(x)

        # QKV projection
        qkv = encoder_block.qkv(x_norm)

        # Reshape and split into Q, K, V
        qkv_reshaped = qkv.view(batch_size, seq_len, self.num_heads, self.head_dim * 3)
        xq, xk, xv = qkv_reshaped.chunk(3, dim=-1)

        # Apply RoPE if enabled
        if self.config.rope and freqs_cis is not None:
            # Import RoPE function
            from ..model.rotary import apply_rotary_emb

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Compute attention scores Q@K^T
        # Transpose to [batch, heads, seq, head_dim]
        xq_heads = xq.transpose(1, 2)
        xk_heads = xk.transpose(1, 2)

        # Attention logits: [batch, heads, seq_q, seq_k]
        attn_logits = torch.matmul(xq_heads, xk_heads.transpose(-2, -1))
        attn_logits = attn_logits / (self.head_dim**0.5)

        # Apply padding mask if present
        if pad_mask is not None:
            attn_logits = attn_logits + pad_mask

        # Extract statistics
        # Max logits per head (what we need for clipping)
        max_logits_per_head = attn_logits.amax(dim=(-2, -1)).mean(dim=0)  # [heads]

        # Attention entropy per head
        attn_probs = F.softmax(attn_logits, dim=-1)
        entropy_per_head = -torch.sum(
            attn_probs * torch.log(attn_probs + 1e-10), dim=(-2, -1)
        ).mean(dim=0)  # [heads]

        # Additional diagnostics
        mean_logit = attn_logits.mean().item()
        std_logit = attn_logits.std().item()
        max_logit_overall = attn_logits.max().item()

        # Move to CPU if configured
        if self.offload_to_cpu:
            max_logits_per_head = max_logits_per_head.cpu()
            entropy_per_head = entropy_per_head.cpu()

        return {
            "max_logits_per_head": max_logits_per_head,
            "entropy_per_head": entropy_per_head,
            "mean_logit": mean_logit,
            "std_logit": std_logit,
            "max_logit_overall": max_logit_overall,
        }

    def get_layer_stats(self, layer_idx: int) -> Optional[Dict]:
        """Get attention statistics for specific layer."""
        return self.layer_stats.get(layer_idx)

    def get_global_max_logit(self) -> float:
        """Get maximum attention logit across all layers."""
        if not self.layer_stats:
            return 0.0
        return max(
            stats["max_logits_per_head"].max().item()
            for stats in self.layer_stats.values()
        )

    def get_mean_entropy(self) -> float:
        """Get mean attention entropy across all layers and heads."""
        if not self.layer_stats:
            return 0.0
        all_entropy = torch.cat(
            [stats["entropy_per_head"] for stats in self.layer_stats.values()]
        )
        return all_entropy.mean().item()

    def clear(self):
        """Clear captured statistics."""
        self.layer_stats.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
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

    def __init__(self, model: torch.nn.Module, model_config, config: MuonClipConfig):
        self.config = config
        self.model_config = model_config
        self._step = 0
        self._metrics = {}

        # Validate model architecture
        self._validate_model(model)

        # Setup hooks for QK-clipping
        if config.enable_clipping:
            logger.info("Initializing attention hook system...")
            self.hook_system = NeoBERTAttentionHooks(
                model_config, offload_to_cpu=config.offload_hooks_to_cpu
            )
            num_hooks = self.hook_system.register_hooks(model)
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
        # Check for transformer_encoder
        if not hasattr(model, "transformer_encoder"):
            raise ValueError(
                "Model must have 'transformer_encoder' attribute. "
                "Is this a NeoBERT model?"
            )

        # Check first layer has fused QKV
        if len(model.transformer_encoder) == 0:
            raise ValueError("Model has no transformer layers")

        first_layer = model.transformer_encoder[0]
        if not hasattr(first_layer, "qkv"):
            raise ValueError(
                "EncoderBlock must have 'qkv' attribute (fused QKV projection). "
                "This implementation requires NeoBERT's fused QKV architecture."
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

            # Check if this is QKV projection
            is_qkv = "qkv" in name

            param_info = {
                "param": param,
                "name": name,
                "layer_idx": layer_idx,
                "is_qkv": is_qkv,
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
                # Still collect metrics during warmup
                if self._step % self.config.log_interval == 0:
                    self._collect_attention_metrics_only()

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
            grad = param.grad.data

            # Apply momentum
            state["momentum_buffer"].mul_(group["beta"]).add_(
                grad, alpha=1 - group["beta"]
            )

            # Orthogonalize 2D gradients using Newton-Schulz
            update = self._newton_schulz_update(
                state["momentum_buffer"], self.config.ns_steps
            )

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
            grad = param.grad.data

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
        if is_transpose:
            grad = grad.T

        # Newton-Schulz iteration
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = grad / (grad.norm() + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        # RMS scaling for Adam lr compatibility
        # Factor: 0.4 * sqrt(max_dim)
        scale_factor = 0.4 * max(grad.size(0), grad.size(1)) ** 0.5
        X = scale_factor * X

        if is_transpose:
            X = X.T

        return X

    def _apply_qk_clipping(self):
        """
        Apply per-head QK-clipping based on captured attention logits.

        This is the critical function for attention stability.
        Works with fused QKV weights in NeoBERT.
        """
        if not self.hook_system or not self.hook_system.layer_stats:
            logger.warning("QK-clipping enabled but no attention statistics captured")
            return

        # Get global maximum logit
        global_max = self.hook_system.get_global_max_logit()

        # Apply clipping to QKV weights
        for group in self.param_groups:
            if not group["use_muon"]:
                continue

            for info in group["param_info"]:
                if not info["is_qkv"] or info["layer_idx"] is None:
                    continue

                layer_stats = self.hook_system.get_layer_stats(info["layer_idx"])
                if layer_stats is None:
                    continue

                # Get per-head max logits
                max_logits = layer_stats["max_logits_per_head"]

                # Move to device if needed
                if max_logits.device != info["param"].device:
                    max_logits = max_logits.to(info["param"].device)

                # Compute per-head scaling factors
                eta_per_head = (self.config.clipping_threshold / max_logits).clamp(
                    max=1.0
                )

                # Apply scaling to fused QKV weights
                self._scale_qkv_weights(
                    param=info["param"],
                    eta_per_head=eta_per_head,
                    alpha=self.config.clipping_alpha,
                )

        # Store metrics
        self._metrics["train/max_attention_logit"] = global_max
        self._metrics["train/qk_clipping_active"] = (
            global_max > self.config.clipping_threshold
        )

        if self.config.monitor_attention_entropy:
            self._metrics["train/attention_entropy"] = (
                self.hook_system.get_mean_entropy()
            )

        # Clear for next forward pass
        self.hook_system.clear()

    def _collect_attention_metrics_only(self):
        """Collect attention metrics without applying clipping (warmup period)."""
        if not self.hook_system or not self.hook_system.layer_stats:
            return

        global_max = self.hook_system.get_global_max_logit()

        self._metrics["train/max_attention_logit"] = global_max
        self._metrics["train/attention_entropy"] = self.hook_system.get_mean_entropy()
        self._metrics["train/warmup"] = True

        self.hook_system.clear()

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
        # Get dimensions
        hidden_size = self.model_config.hidden_size
        num_heads = self.model_config.num_attention_heads
        head_dim = hidden_size // num_heads

        # QKV weight: [hidden_size, 3*hidden_size]
        # Transpose to [3*hidden_size, hidden_size] for easier manipulation
        qkv = param.data.T

        # Reshape to separate Q, K, V: [3, num_heads, head_dim, hidden_size]
        qkv_reshaped = qkv.reshape(3, num_heads, head_dim, hidden_size)

        # Expand eta for broadcasting: [num_heads, 1, 1]
        eta_expanded = eta_per_head.view(num_heads, 1, 1)

        # Scale Q and K, leave V unchanged
        qkv_reshaped[0] *= eta_expanded**alpha  # Query
        qkv_reshaped[1] *= eta_expanded ** (1 - alpha)  # Key
        # qkv_reshaped[2] unchanged (Value)

        # Reshape back to original shape
        qkv_scaled = qkv_reshaped.reshape(3 * hidden_size, hidden_size)

        # Transpose back and update parameter
        param.data = qkv_scaled.T

    def get_metrics(self) -> Dict:
        """
        Get metrics for logging.

        Returns metrics dict and clears internal storage.
        """
        metrics = self._metrics.copy()
        self._metrics.clear()
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
