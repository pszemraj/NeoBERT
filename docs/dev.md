# Development Notes

This page tracks known limitations and TODOs that need follow-up work.

## Pretraining: `datacollator.pack_sequences`

Status: **not supported**. The pretraining trainer now errors fast if enabled.

Why: The current NeoBERT pretraining model uses standard attention with padded
sequences. Packing requires either:

- a block-diagonal attention mask that prevents cross-sequence attention, or
- FlashAttention varlen kernels (cu_seqlens + max_seqlen) with a packed layout.

TODO scope:
- Decide on the packed representation (block-diagonal mask vs. varlen kernels).
- If varlen: add collator support for `cu_seqlens`/`max_seqlen` and route through
  a model that supports FlashAttention varlen (similar to the HF export model).
- If block-diagonal: build a banded mask inside the collator and validate memory
  footprint for long sequences.
- Add end-to-end tests for packed batches with correct masking.

## Contrastive Training

Status: **not fully validated end-to-end**.

Notes / TODOs:
- Requires preprocessed datasets under `dataset.path` via
  `scripts/contrastive/preprocess.py`.
- Validate dataset schema consistency across all sources (query/corpus/negative).
- Run a short smoke training job to validate loss + checkpointing behavior.
