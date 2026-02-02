# Development Notes

This page tracks known limitations and TODOs that need follow-up work as well as lessons learned during development.

---

- [Development Notes](#development-notes)
  - [Pretraining: `datacollator.pack_sequences`](#pretraining-datacollatorpack_sequences)
    - [Next Step: FlashAttention varlen training path (cu\_seqlens)](#next-step-flashattention-varlen-training-path-cu_seqlens)
  - [Contrastive Training](#contrastive-training)
  - [Pretraining: Chunked / Fused Cross-Entropy](#pretraining-chunked--fused-cross-entropy)

---

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
  a training model that supports FlashAttention varlen. (The HF export model is
  intentionally vanilla Transformers and does **not** support packed inputs.)
- Validate memory footprint for long sequences and document limits.
- Add end-to-end tests for packed batches with correct masking and training loss.

### Next Step: FlashAttention varlen training path (cu_seqlens)

Goal: eliminate the O(seq²) block mask by using FlashAttention varlen kernels for
packed sequences during pretraining.

Current state:

- `src/neobert/collator/collator.py` builds a dense block-diagonal mask for packed
  sequences and converts it to additive 0/-inf (O(seq²) memory).
- `src/neobert/pretraining/trainer.py` uses `get_dataloader()` + `get_collator()`
  and passes `attention_mask` into `NeoBERTLMHead`.
- `src/neobert/model/model.py` supports xFormers (memory_efficient_attention) or
  SDPA; it expects additive masks and has **no** varlen path.
- `src/neobert/huggingface/modeling_neobert.py` intentionally **does not**
  support packed/varlen inputs; the exported HF model stays within vanilla
  Transformers expectations.
- `src/neobert/huggingface/modeling_neobert.py::DataCollatorWithPacking` raises
  if `pack_sequences=True`; use `src/neobert/collator/collator.py` for training.

Implementation plan:

- Collator: add a varlen packing mode in `src/neobert/collator/collator.py` that
  emits `input_ids`, `labels`, `position_ids`, `cu_seqlens` (int32), and
  `max_seqlen` **without** building a dense mask. Consider config like:
  `datacollator.pack_sequences_mode: "block" | "varlen"` (default "block").
- Trainer: when varlen mode is active:
  - pass `cu_seqlens` and `max_seqlen` into the model
  - **do not** create/expand `attention_mask`
  - ensure `position_ids` is used for RoPE (or compute from segments)
- Model:
  - add a varlen path to `src/neobert/model/model.py` using flash-attn.
  - if a separate module is needed, keep it training-only; do **not** route
    through the HF export model (packing is intentionally unsupported there).
  - require `flash_attn` on GPU; error clearly if missing.
  - keep additive-mask path unchanged for non-packed batches.

Constraints / gotchas:

- RoPE in packed mode needs correct `position_ids` per segment. The training
  collator will need to provide these; the training model must consume them.
- `flash_attn_varlen_func` expects q/k/v shaped `[total_tokens, nheads, head_dim]`
  with `cu_seqlens` and `max_seqlen`.
- Packed varlen only works on CUDA; make the failure mode explicit.

Tests to add:

- Unit test for varlen collator output (`cu_seqlens`, `max_seqlen`, `position_ids`).
- Parity test: varlen-packed output vs non-packed output on a tiny example
  (same weights, disable dropout).
- Training smoke test with `datacollator.pack_sequences_mode="varlen"` on a tiny
  dataset (GPU-only, likely skipped in CI).

## Contrastive Training

Status: **not fully validated end-to-end**.

Notes / TODOs:

- Requires preprocessed datasets under `dataset.path` via
  `scripts/contrastive/preprocess.py`.
- Validate dataset schema consistency across all sources (query/corpus/negative).
- Run a short smoke training job to validate loss + checkpointing behavior.

## Pretraining: Chunked / Fused Cross-Entropy

Goal: avoid materializing full logits `(B, S, V)` for long contexts.

Current state:

- `src/neobert/pretraining/trainer.py` computes full logits from
  `NeoBERTLMHead` and applies `CrossEntropyLoss(reduction="sum")`.
- There is a note in the trainer warning about full logits memory.

Implementation options:

- Chunked logits:
  - modify `NeoBERTLMHead` to optionally return hidden states only, or expose the
    decoder linear layer so trainer can compute logits in chunks.
  - split hidden states along sequence dimension, compute per-chunk logits and
    `CrossEntropyLoss(reduction="sum")`, then accumulate.
  - must preserve current per-token scaling semantics (loss sum over masked tokens).
- Fused kernels:
  - integrate `xentropy` / `liger-kernel` (if available) for fused CE to reduce
    memory and potentially improve speed.
  - keep a fallback path for CPU / missing dependency.

Constraints / gotchas:

- Needs to preserve the sum-reduction for `train_loss_fn` to keep existing
  gradient scaling logic unchanged.
- Keep dtype handling stable for bf16/fp16 (AMP).
- Ensure accuracy metrics (`_count_masked_correct`) still computed with logits;
  either compute logits in chunks a second time or compute `argmax` in chunks.

Tests to add:

- Unit test comparing chunked CE vs full CE on a tiny batch (same loss to ~1e-6).
- Integration test: trainer step with chunked CE enabled on a tiny config.
