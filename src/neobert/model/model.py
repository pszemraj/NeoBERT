# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import math
import warnings
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )

    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_ERROR: Optional[Exception] = None
except (ImportError, RuntimeError) as e:
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_ERROR = e
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    LIGER_AVAILABLE = True
except (ImportError, RuntimeError):
    LIGER_AVAILABLE = False
    LigerRopeFunction = None
    LigerSiLUMulFunction = None
    LigerRMSNorm = None


from .rmsnorm import RMSNorm
from .rotary import apply_rotary_emb, precompute_freqs_cis


class TorchSwiGLU(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


class LigerSwiGLU(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, bias: bool = False
    ):
        if not (LIGER_AVAILABLE and LigerSiLUMulFunction is not None):
            raise ImportError("Liger kernels not available for SwiGLU.")
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w2(x)
        return self.w3(LigerSiLUMulFunction.apply(gate, up))


def maybe_apply_liger_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, use_liger: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        use_liger
        and LIGER_AVAILABLE
        and LigerRopeFunction is not None
        and freqs_cis is not None
    ):
        cos = freqs_cis.real.to(dtype=xq.dtype, device=xq.device).unsqueeze(0)
        sin = freqs_cis.imag.to(dtype=xq.dtype, device=xq.device).unsqueeze(0)
        q = xq.transpose(1, 2).contiguous()
        k = xk.transpose(1, 2).contiguous()
        q, k = LigerRopeFunction.apply(q, k, cos, sin)
        return q.transpose(1, 2), k.transpose(1, 2)
    return apply_rotary_emb(xq, xk, freqs_cis) if freqs_cis is not None else (xq, xk)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if not FLASH_ATTN_AVAILABLE or flash_attn_func is None or flash_attn_varlen_func is None:
        raise ImportError(
            "Flash attention requires the flash-attn package with working CUDA kernels. "
            f"{FLASH_ATTN_ERROR if 'FLASH_ATTN_ERROR' in globals() else ''}"
        )

    if pad_mask is None:
        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

    mask = torch.isfinite(pad_mask)
    if bool(mask.all().item()):
        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

    seqlens = mask.sum(dim=1).to(dtype=torch.int32)
    if torch.any(seqlens == 0):
        raise ValueError(
            "Flash attention requires at least one valid token per sequence when padding is present."
        )
    cu_seqlens = torch.zeros(
        seqlens.size(0) + 1, dtype=torch.int32, device=q.device
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    max_seqlen = int(seqlens.max().item())

    q_unpadded = q[mask].contiguous()
    k_unpadded = k[mask].contiguous()
    v_unpadded = v[mask].contiguous()

    attn_unpadded = flash_attn_varlen_func(
        q_unpadded,
        k_unpadded,
        v_unpadded,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=False,
    )

    attn = torch.zeros_like(q)
    attn[mask] = attn_unpadded
    return attn


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        liger_kernels: bool = True,
        apple_ce: bool = False,
        apple_ce_impl: Optional[str] = None,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Core dims
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size

        # Dropout: accept legacy 'dropout_prob' as alias
        if "dropout_prob" in kwargs and "dropout" not in kwargs:
            dropout = kwargs["dropout_prob"]
        self.dropout = dropout

        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Positional length: accept HF-style 'max_position_embeddings'
        if (
            "max_position_embeddings" in kwargs
            and kwargs["max_position_embeddings"] is not None
        ):
            self.max_length = int(kwargs["max_position_embeddings"])
        else:
            self.max_length = max_length

        self.flash_attention = flash_attention
        self.liger_kernels = liger_kernels
        self.apple_ce = apple_ce
        self.apple_ce_impl = apple_ce_impl
        self.base_scale = base_scale
        self.ngpt = ngpt

        # Store any extra kwargs for reference
        self.kwargs = kwargs


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, bias=False
        )
        self.resid_dropout = nn.Dropout(config.dropout)

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                intermediate_size = int(2 * config.intermediate_size / 3)
                intermediate_size = multiple_of * (
                    (intermediate_size + multiple_of - 1) // multiple_of
                )
                if config.liger_kernels and LIGER_AVAILABLE and LigerSiLUMulFunction is not None:
                    self.ffn = LigerSwiGLU(
                        config.hidden_size,
                        intermediate_size,
                        config.hidden_size,
                        bias=False,
                    )
                else:
                    if config.liger_kernels and not LIGER_AVAILABLE:
                        warnings.warn(
                            "Liger kernels requested for SwiGLU but liger-kernel is not available; "
                            "falling back to PyTorch implementation."
                        )
                    self.ffn = TorchSwiGLU(
                        config.hidden_size,
                        intermediate_size,
                        config.hidden_size,
                        bias=False,
                    )
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                )
            case _:
                raise ValueError(f"Unsupported activation {config.hidden_act}")

        rms_factory = None
        if config.rms_norm:
            if config.liger_kernels and LIGER_AVAILABLE and LigerRMSNorm is not None:
                rms_factory = lambda: LigerRMSNorm(config.hidden_size, config.norm_eps)
            else:
                if config.liger_kernels and not LIGER_AVAILABLE:
                    warnings.warn(
                        "Liger kernels requested for RMSNorm but liger-kernel is not available; "
                        "falling back to PyTorch implementation."
                    )
                rms_factory = lambda: RMSNorm(config.hidden_size, config.norm_eps)
        self.attention_norm = (
            rms_factory() if rms_factory is not None else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            rms_factory() if rms_factory is not None else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor]
    ):
        x = x + self._att_block(self.attention_norm(x), pad_mask, freqs_cis)
        x = x + self._ff_block(self.ffn_norm(x))
        return x

    def _att_block(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor]
    ):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.dim_head * 3,
            )
            .chunk(3, axis=-1)
        )

        if self.config.rope:
            xq, xk = maybe_apply_liger_rope(xq, xk, freqs_cis, self.config.liger_kernels)

        pad_mask_2d: Optional[torch.Tensor] = None
        if pad_mask is not None:
            if pad_mask.dim() == 4:
                pad_mask_2d = pad_mask[:, 0, 0, :]
            elif pad_mask.dim() == 2:
                pad_mask_2d = pad_mask
            else:
                raise ValueError("Unsupported pad_mask shape for attention block.")

        if self.config.flash_attention:
            attn = flash_attention_forward(
                xq,
                xk,
                xv,
                pad_mask_2d,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
        else:
            attn_mask = None
            if pad_mask is not None:
                if pad_mask.dim() == 2:
                    attn_mask = (
                        pad_mask.unsqueeze(1)
                        .unsqueeze(1)
                        .to(dtype=xq.dtype, device=xq.device)
                    )
                else:
                    attn_mask = pad_mask.to(dtype=xq.dtype, device=xq.device)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.config.dropout if self.training else 0,
            ).transpose(1, 2)

        return self.resid_dropout(
            self.wo(
                attn.reshape(
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads * self.config.dim_head,
                )
            )
        )

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, bias=False
        )
        self.resid_dropout = nn.Dropout(config.dropout)

        self.c_fc = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling * torch.ones(config.hidden_size)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling * torch.ones(config.hidden_size)
        )

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(
            self.sqk_init_scaling * torch.ones(config.hidden_size)
        )

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(2 * config.intermediate_size)
        )

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor]
    ):
        x_attn = self._att_block(x, pad_mask, freqs_cis)

        lr = self.attn_alpha * (
            self.attn_alpha_init_value / self.attn_alpha_init_scaling
        )
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor], freqs_cis: Optional[torch.Tensor]
    ):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.dim_head * 3,
            )
            .chunk(3, axis=-1)
        )

        if self.config.rope:
            xq, xk = maybe_apply_liger_rope(xq, xk, freqs_cis, self.config.liger_kernels)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1,
            1,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (
            self.config.hidden_size / self.config.num_attention_heads
        ) ** 0.5

        pad_mask_2d: Optional[torch.Tensor] = None
        if pad_mask is not None:
            if pad_mask.dim() == 4:
                pad_mask_2d = pad_mask[:, 0, 0, :]
            elif pad_mask.dim() == 2:
                pad_mask_2d = pad_mask
            else:
                raise ValueError("Unsupported pad_mask shape for attention block.")

        if self.config.flash_attention:
            attn = flash_attention_forward(
                xq,
                xk,
                xv,
                pad_mask_2d,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
            )
        else:
            attn_mask = None
            if pad_mask is not None:
                if pad_mask.dim() == 2:
                    attn_mask = (
                        pad_mask.unsqueeze(1)
                        .unsqueeze(1)
                        .to(dtype=xq.dtype, device=xq.device)
                    )
                else:
                    attn_mask = pad_mask.to(dtype=xq.dtype, device=xq.device)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.config.dropout if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

        return self.resid_dropout(
            self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size))
        )

    def _ff_block(self, x: torch.Tensor):
        uv = self.c_fc(x)
        suv = self.suv * (
            (self.suv_init_value / self.suv_init_scaling)
            * (self.config.hidden_size**0.5)
        )
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(
                -self.config.decoder_init_range, self.config.decoder_init_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(
                -self.config.embedding_init_range, self.config.embedding_init_range
            )


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        if self.config.flash_attention and (
            not FLASH_ATTN_AVAILABLE or not torch.cuda.is_available()
        ):
            details = (
                f" ({FLASH_ATTN_ERROR})" if "FLASH_ATTN_ERROR" in globals() and FLASH_ATTN_ERROR else ""
            )
            warnings.warn(
                "Flash attention requested but flash-attn kernels are unavailable or CUDA is disabled."
                f"{details} Falling back to scaled_dot_product_attention."
            )
            self.config.flash_attention = False

        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads, config.max_length
            )
        else:
            self.positional_embedding = nn.Embedding(
                config.max_length + 1,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            )

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if config.rms_norm:
            if config.liger_kernels and LIGER_AVAILABLE and LigerRMSNorm is not None:
                self.layer_norm = LigerRMSNorm(config.hidden_size, config.norm_eps)
            else:
                if config.liger_kernels and not LIGER_AVAILABLE:
                    warnings.warn(
                        "Liger kernels requested for RMSNorm but liger-kernel is not available; "
                        "falling back to PyTorch implementation."
                    )
                self.layer_norm = RMSNorm(config.hidden_size, config.norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, src, pad_mask=None):
        additive_mask = None
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, (
                "NeoBERT expects an additive pad_mask"
            )
            additive_mask = pad_mask.to(device=src.device)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, additive_mask, freqs_cis)

        # Final normalization layer
        x = self.layer_norm(x)

        # Return the output of the last hidden layer
        return x


class NormNeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        if self.config.flash_attention and (
            not FLASH_ATTN_AVAILABLE or not torch.cuda.is_available()
        ):
            details = (
                f" ({FLASH_ATTN_ERROR})" if "FLASH_ATTN_ERROR" in globals() and FLASH_ATTN_ERROR else ""
            )
            warnings.warn(
                "Flash attention requested but flash-attn kernels are unavailable or CUDA is disabled."
                f"{details} Falling back to scaled_dot_product_attention."
            )
            self.config.flash_attention = False

        self.encoder = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads, config.max_length
            )
        else:
            self.positional_embedding = nn.Embedding(
                config.max_length + 1,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            )

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config))

        if config.rms_norm:
            if config.liger_kernels and LIGER_AVAILABLE and LigerRMSNorm is not None:
                self.layer_norm = LigerRMSNorm(config.hidden_size, config.norm_eps)
            else:
                if config.liger_kernels and not LIGER_AVAILABLE:
                    warnings.warn(
                        "Liger kernels requested for RMSNorm but liger-kernel is not available; "
                        "falling back to PyTorch implementation."
                    )
                self.layer_norm = RMSNorm(config.hidden_size, config.norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=config.base_scale / math.sqrt(2 * config.num_hidden_layers),
                )

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(
            self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32)
        )

    def forward(self, src, pad_mask=None):
        additive_mask = None
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, (
                "NeoBERT expects an additive pad_mask"
            )
            additive_mask = pad_mask.to(device=src.device)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, additive_mask, freqs_cis)

        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(self, src, pad_mask=None):
        hidden_representation = self.model.forward(src, pad_mask)
        logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, "logits": logits}

    def forward_hidden(self, src, pad_mask=None):
        return self.model.forward(src, pad_mask)


class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):
    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src, pad_mask=None):
        hidden_representation = self.model.forward(src, pad_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Convert HuggingFace attention mask (1s and 0s) to additive mask (-inf and 0)
        if attention_mask is not None:
            additive_mask = torch.where(attention_mask == 0, float("-inf"), float(0.0))
        else:
            additive_mask = None
        hidden_representation = self.model.forward(input_ids, additive_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs):
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [
                    (query + " " + kwargs["instructions"][query]).strip()
                    for query in queries
                ]
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                    for doc in corpus
                ]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _transform_func(tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]):
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        encodings = []
        for batch in tqdm(
            dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128
        ):
            input_ids = batch["input_ids"].to(device)

            pad_mask = batch["attention_mask"].to(device)
            additive_mask = torch.full_like(
                pad_mask, fill_value=float("-inf"), dtype=torch.float32, device=device
            )
            additive_mask = additive_mask.masked_fill(pad_mask.bool(), 0.0)

            outputs = self.model(input_ids, additive_mask)

            if self.pooling == "avg":
                outputs = outputs * pad_mask.unsqueeze(-1).expand(
                    -1, -1, outputs.shape[-1]
                )
                outputs = outputs.sum(dim=1) / pad_mask.to(device).sum(dim=1).unsqueeze(
                    -1
                )
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)
