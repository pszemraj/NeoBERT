# NeoBERT Architecture

Config defaults live in [Configuration Reference](configuration.md).

## Overview

NeoBERT is a transformer encoder with:

- fused QKV projection,
- optional RoPE position encoding,
- RMSNorm or LayerNorm,
- SwiGLU or GELU feed-forward,
- backend-selectable attention (`sdpa` or `flash_attn_varlen` for packed training).

## Source Files

- Core encoder: `src/neobert/model/model.py`
- Sequence-classification heads: `src/neobert/model/classification.py`
- LM + embedding wrappers: `src/neobert/model/wrappers.py`
- Attention dispatch and packed varlen helpers: `src/neobert/kernels/attention.py`
- Shared model-shape and packed-sequence utilities: `src/neobert/modeling_utils.py`
- Rotary embeddings: `src/neobert/model/rotary.py`
- RMSNorm backend wrappers: `src/neobert/kernels/backend.py`, `src/neobert/model/rmsnorm.py`
- HF adapters and export model: `src/neobert/huggingface/adapters.py`, `src/neobert/huggingface/modeling_neobert.py`

## Embeddings and Positions

- Token embeddings use `model.pad_token_id` as the embedding padding index.
- With `rope: true`, Q/K receive rotary embeddings.
- With `rope: false`, learned positional embeddings are used.
- In learned-position mode, position IDs reserve `0` for padding and start real tokens at `1`.

## Attention Paths

### Unpacked path

- Uses PyTorch SDPA (`scaled_dot_product_attention`).
- Training API expects additive masks (`0` keep, `-inf` masked).

### Packed path

- For packed batches, the model can use flash-attn varlen kernels when `model.attn_backend: flash_attn_varlen` and CUDA plus flash-attn are available.
- Packed metadata is represented as `packed_seqlens` and converted to varlen flattening metadata (`flat_token_indices`, `cu_seqlens`, `max_seqlen`).
- Metadata is prepared once per forward pass and reused across all encoder layers to reduce host overhead.
- SDPA segmented fallback exists for correctness/testing when flash-attn is not used, but is slower.

## Feed-Forward

- `model.hidden_act: swiglu`: the encoder uses separate `w1/w2/w3` projections.
- `model.hidden_act: gelu`: standard 2-layer GELU MLP.

## Normalization

- `model.rms_norm: true`: RMSNorm path.
- `model.rms_norm: false`: LayerNorm path.
- `model.kernel_backend` selects torch or Liger primitives where available.

## HF Export Model Differences

- Exported HF model is intentionally standard/unpacked.
- It does not support packed-sequence inputs/metadata.
- The training-time HF classifier adapter accepts binary bool/integer/float masks and explicit nonpositive additive masks. An all-zero float mask follows the HF binary convention and masks every token; pass `None` or all ones for an unmasked batch.
- The training-time adapter rejects custom `position_ids` and nonzero `token_type_ids` because its backbone does not implement those semantics. The exported HF model supports explicit positions.

## Related Docs

- [Training](../guides/training.md)
- [Training Optimization](../guides/training-optimization.md)
- [Evaluation](../guides/evaluation.md)
