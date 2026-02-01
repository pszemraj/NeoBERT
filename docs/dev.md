# Development Notes

This page tracks known limitations and TODOs that need follow-up work.

## Pretraining: `datacollator.pack_sequences`

Status: **experimental**. Pretraining uses a block-diagonal attention mask.

Why: Packed sequences must prevent cross-sequence attention. Today we use a
block-diagonal mask on the standard attention path. For efficiency, we may
want to switch to FlashAttention varlen kernels in the future.

Options:
- block-diagonal attention mask (current implementation)
- FlashAttention varlen kernels (cu_seqlens + max_seqlen) with a packed layout

TODO scope:
- If varlen: add collator support for `cu_seqlens`/`max_seqlen` and route through
  a model that supports FlashAttention varlen (similar to the HF export model).
- Validate memory footprint for long sequences and document limits.
- Add end-to-end tests for packed batches with correct masking and training loss.

## Contrastive Training

Status: **not fully validated end-to-end**.

Notes / TODOs:
- Requires preprocessed datasets under `dataset.path` via
  `scripts/contrastive/preprocess.py`.
- Validate dataset schema consistency across all sources (query/corpus/negative).
- Run a short smoke training job to validate loss + checkpointing behavior.
