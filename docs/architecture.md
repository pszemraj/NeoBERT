# NeoBERT Architecture

This document summarizes the implemented NeoBERT architecture and points to the source of truth in code.

## Overview

NeoBERT is a transformer encoder with modernized components:

- **Fused QKV attention** (single linear projection, split into Q/K/V)
- **RoPE positional embeddings** (optional)
- **RMSNorm or LayerNorm** (configurable)
- **SwiGLU or GELU** feed-forward (configurable)
- **Flash attention** via xformers when enabled
- **Optional nGPT-style normalization** (NormNeoBERT)

## Source of Truth

- Core model blocks: `src/neobert/model/model.py`
- Rotary embeddings: `src/neobert/model/rotary.py`
- RMSNorm: `src/neobert/model/rmsnorm.py`
- HF export model: `src/neobert/huggingface/modeling_neobert.py`

## Core Components (Implementation Summary)

### Embeddings

- Token embeddings only (no token-type embeddings).
- Position information is injected with RoPE when enabled.
- Padding uses `pad_token_id`.

### Attention

- Single `qkv` projection (`hidden_size -> 3 * hidden_size`) split into Q/K/V.
- RoPE applied to Q/K when `model.rope: true`.
- Attention backend:
  - `model.flash_attention: true` uses `xformers.ops.memory_efficient_attention`.
  - Otherwise uses PyTorch `scaled_dot_product_attention`.
- Attention masks are additive (0 = keep, -inf = mask).

### Feed-Forward Network

- `model.hidden_act: swiglu` uses the SwiGLU block (native PyTorch fallback if xformers is missing).
- `model.hidden_act: gelu` uses a standard 2-layer GELU MLP.

### Normalization

- `model.rms_norm: true` uses RMSNorm.
- `model.rms_norm: false` uses LayerNorm.

### NormNeoBERT (nGPT-style)

If `model.ngpt: true`, `NeoBERTLMHead` builds `NormNeoBERT`, which applies nGPT-style normalization and scaling inside the encoder block.

## Key Configuration Knobs

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 4096
  dropout_prob: 0.0
  rope: true
  rms_norm: true
  hidden_act: swiglu
  flash_attention: true
  ngpt: false
```

Notes:

- RoPE frequency scaling (`theta`) is currently fixed at 10,000 in `src/neobert/model/rotary.py`.
- Flash attention for training requires `xformers`. Exported HF models use `flash-attn` if installed.

## Differences from BERT (High Level)

| Feature | BERT | NeoBERT |
| --- | --- | --- |
| Position encoding | Learned | RoPE (optional) |
| Normalization | LayerNorm | RMSNorm or LayerNorm |
| Activation | GELU | SwiGLU or GELU |
| Attention backend | Standard | xformers (optional) |
| Token types | Yes | No |

## Next Steps

- Training workflows: [docs/training.md](training.md)
- Configuration reference: [docs/configuration.md](configuration.md)
- Evaluation guide: [docs/evaluation.md](evaluation.md)
