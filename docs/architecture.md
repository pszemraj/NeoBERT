# NeoBERT Architecture

This document summarizes the implemented architecture and runtime behavior.
Source of truth remains the code under `src/neobert/`.

## Overview

NeoBERT is a transformer encoder with:

- fused QKV projection,
- optional RoPE position encoding,
- RMSNorm or LayerNorm,
- SwiGLU or GELU feed-forward,
- backend-selectable attention (`sdpa` or `flash_attn_varlen` for packed training),
- optional nGPT-style normalized residual path (`ngpt=true`).

## Source Files

- Core model: `src/neobert/model/model.py`
- Attention dispatch and packed varlen helpers: `src/neobert/kernels/attention.py`
- Rotary embeddings: `src/neobert/model/rotary.py`
- RMSNorm backend wrappers: `src/neobert/kernels/backend.py`, `src/neobert/model/rmsnorm.py`
- HF export model: `src/neobert/huggingface/modeling_neobert.py`

## Embeddings and Positions

- Token embeddings use `pad_token_id` as the embedding padding index.
- With `rope: true`, Q/K receive rotary embeddings.
- With `rope: false`, learned positional embeddings are used.
- In learned-position mode, position IDs reserve `0` for padding and start real
  tokens at `1`.

## Attention Paths

### Unpacked path

- Uses PyTorch SDPA (`scaled_dot_product_attention`).
- Training API expects additive masks (`0` keep, `-inf` masked).

### Packed path

- For packed batches, model can use flash-attn varlen kernels when
  `attn_backend: flash_attn_varlen` and CUDA + flash-attn are available.
- Packed metadata is represented as `packed_seqlens` and converted to varlen
  flattening metadata (`flat_token_indices`, `cu_seqlens`, `max_seqlen`).
- Metadata is prepared once per forward pass and reused across all encoder
  layers to reduce host overhead.
- SDPA segmented fallback exists for correctness/testing when flash-attn is not
  used, but is slower.

## Feed-Forward

- `hidden_act: swiglu`: unpacked `w1/w2/w3` SwiGLU block.
- `hidden_act: gelu`: standard 2-layer GELU MLP.

## Normalization

- `rms_norm: true`: RMSNorm path.
- `rms_norm: false`: LayerNorm path.
- Kernel backend (`kernel_backend`) selects torch vs Liger primitives where
  available.

## nGPT Mode

When `ngpt: true`, `NormNeoBERT` is used:

- normalized residual interpolation,
- learned scaling parameters for attention/MLP branches,
- custom normalization dynamics relative to standard encoder blocks.

## HF Export Model Differences

- Exported HF model is intentionally standard/unpacked.
- It does not support packed-sequence inputs/metadata.
- Attention-mask normalization in HF path accepts bool/additive/binary forms and
  normalizes internally for compatibility.

## Key Config Knobs

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 4096
  hidden_act: swiglu
  rope: true
  rms_norm: true
  attn_backend: sdpa           # or flash_attn_varlen
  kernel_backend: auto         # auto | liger | torch
  ngpt: false
```

## Related Docs

- [Training](training.md)
- [Configuration](configuration.md)
- [Evaluation](evaluation.md)
