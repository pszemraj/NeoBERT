# Development Notes

This page tracks the current engineering status and near-term TODOs for NeoBERT
training performance and stability work.

---

- [Current Status](#current-status)
  - [Pretraining packed path](#pretraining-packed-path)
  - [Compile and runtime behavior](#compile-and-runtime-behavior)
  - [Distributed runtime policy (FSDP2-first)](#distributed-runtime-policy-fsdp2-first)
  - [Checkpointing and serialization](#checkpointing-and-serialization)
- [Recently Landed](#recently-landed)
- [Lessons Learned](#lessons-learned)
- [Known Gaps](#known-gaps)
- [Next PR TODOs](#next-pr-todos)
  - [Contrastive Sweep TODOs](#contrastive-sweep-todos)
  - [Config Loader Architecture TODOs (Separate PR)](#config-loader-architecture-todos-separate-pr)
- [Longer-Term Backlog](#longer-term-backlog)

---

## Current Status

### Pretraining packed path

Status: **implemented and production-usable** for training.

- Packed pretraining uses FlashAttention varlen kernels via
  `model.attn_backend: flash_attn_varlen`.
- The training collator emits `packed_seqlens` metadata and avoids dense packed
  block masks.
- `src/neobert/model/model.py` routes packed batches through flash varlen on GPU
  and falls back to segmented SDPA where needed (correct but slower).
- HF export model (`src/neobert/huggingface/modeling_neobert.py`) intentionally
  remains standard/unpacked.

### Compile and runtime behavior

Status: **stabilized for current training configs**.

- Major `torch.compile` graph-break sources from scalar `.item()` extraction in
  packed attention were reduced.
- Compile path now avoids flowing extra Python metadata objects through hot
  attention calls (keeps compiled behavior closer to baseline).
- W&B-disabled runs now print key training metrics (including
  `train/tokens_per_sec`) directly to console for local validation.

### Distributed runtime policy (FSDP2-first)

Status: **enforced for pretraining**.

- NeoBERT pretraining is now explicitly **FSDP2-first**.
- FSDP v1 is rejected at runtime with a clear error.
- Reason: masked-logits-only objective needs decoder-weight unshard/reshard
  semantics that are safe with FSDP2. FSDP1 `summon_full_params` does not allow
  initiating forward/backward inside that context.
- For Accelerate launches, set FSDP config to `fsdp_version: 2`.
- Keep passing model + optimizer together to `accelerate.prepare(...)` in FSDP2
  paths (Accelerate requirement).

### Checkpointing and serialization

Status: **standardized**.

- Training checkpoints have been standardized on `safetensors`.
- This is compatible with the current pretraining flow and keeps export paths
  straightforward.
- **Breaking change (2026-02):** pretraining now writes a single checkpoint tree
  at `output_dir/checkpoints/<step>/` that contains both resumable Accelerate
  state and export assets (`model.safetensors`, `config.yaml`, tokenizer files).
  The previous parallel `model_checkpoints/` tree is no longer written.

## Recently Landed

Key landed changes from recent engineering cycles include:

- Packed varlen path hardening and backend resolution fixes.
- Packed metadata tensor-first normalization and compile-path fixes.
- Liger integration and kernel backend cleanup.
- Runtime import cleanup (remove `sys.path` hacks / normalize imports).
- Safetensors checkpoint standardization.
- Optional full packed-batch enforcement:
  `trainer.enforce_full_packed_batches`.
- Training-loop overhead reductions:
  - optional expensive accuracy logging,
  - worker env defaults,
  - reduced compile-unfriendly control flow.
- Packed flash metadata reuse across layers:
  compute once per forward and reuse in all encoder layers (eager path).
- Packed-batch stitching now re-pins CPU tensors before async H2D transfer so
  `non_blocking=True` can actually overlap copies with compute.

## Lessons Learned

This cycle surfaced several concrete process/perf lessons for this repo:

1. Throughput metric first, GPU-util second

- GPU utilization alone is not a reliable success metric.
- Primary metric for pretraining changes is warmup-trimmed
  `train/tokens_per_sec` on fixed configs.

1. Do not trust remote dashboards as the only source of truth

- Always ensure local console prints include `train/tokens_per_sec` when W&B is
  disabled.
- Keep raw run logs in `local-scratch/` for direct A/B inspection.

1. Pinned-memory assumptions can be invalidated silently

- `torch.cat` / `torch.split` on pinned CPU tensors return non-pinned tensors.
- Any trainer-side batch surgery must re-pin before async H2D copy, otherwise
  copy/compute overlap degrades.

1. Compile and eager paths should not diverge in structure unnecessarily

- Python objects moving through compiled hot paths can add guard/recompile churn.
- Keep compile path tensor-first and minimal; reserve richer caching for eager
  where safe.

1. Benchmark protocol must be strict

- Compare against a pinned baseline commit and identical config overrides.
- Use enough steps to clear warmup and compile ramp-up (short smoke runs are
  not enough to conclude regressions).
- Change one performance variable at a time.

## Known Gaps

These are the biggest remaining reasons we do not consistently hit target
throughput/tokens-per-second:

1. Input-side CPU work still creates bubbles in streaming + packing mode.
2. Trainer-side packed batch stitching is still a complexity/perf risk surface.
3. Packing logic still has Python-heavy sections in the collator hot path.
4. We do not yet have first-class in-trainer timing attribution (data wait vs
   forward/backward/optimizer), so regressions can hide.

## Next PR TODOs

Priority order for next performance PR:

1. Add trainer timing attribution (required first)

- Add step-time timers for: dataloader wait, collate, forward, backward,
  optimizer step, logging/checkpoint sections.
- Log p50/p95 and moving averages every N steps.
- Keep overhead negligible and disable by default for clean production runs.

1. Add a deterministic perf harness for A/B

- Single script/config overlay to run short, reproducible throughput probes.
- Standard report: warmup-trimmed `tokens/sec`, `steps/sec`, and timing
  breakdown from (1).
- Use this harness for all optimizer/backend comparisons.

1. Remove trainer-side packed-batch surgery from hot path

- Replace `to_target_batch_size` behavior for packed mode with a fixed-output
  packing iterator that emits full microbatches directly.
- Keep `input_ids`/`labels`/`packed_seqlens` shapes stable without runtime
  stitch/merge logic in the training loop.

1. Move packing upstream out of collator hot path

- Implement dataset-side/stateful iterable packing to reduce per-step Python
  list operations and collator surgery.
- Keep collator simple: mask + MLM corruption + tensorization.

1. Tensorize remaining collator packing operations

- Remove remaining `.tolist()` and per-token Python loops in common paths.
- Keep behavior identical with regression tests.

1. Reduce train-loop overhead in perf mode

- Gate expensive metrics and low-value per-step bookkeeping behind explicit
  config flags.
- Keep debug richness available, but make fast mode lean by default.

1. Continue compile/recompile stabilization

- Collect recompile reasons under the harness above.
- Remove avoidable dynamic guards and static-attribute churn in hot modules.

### Contrastive Sweep TODOs

These are explicitly tracked for sweep-readiness follow-up work:

1. Unify checkpoint layout across tasks

- Pretraining and contrastive now write one canonical tree at
  `checkpoints/<step>/`.
- GLUE now also writes to the canonical `checkpoints/<step>/` tree.
- Legacy `model_checkpoints/<step>/` loading support remains for backwards
  compatibility with older runs.

1. Tighten sweep-time observability parity

- Ensure comparable train/eval metrics are emitted across pretraining,
  contrastive, and GLUE for easier sweep analysis.
- Keep metric naming and logging cadence aligned across task trainers.

### Config Loader Architecture TODOs (Separate PR)

These are intentionally deferred because they are higher-risk refactors that
touch most training/eval entrypoints and config mutation assumptions.

1. Move config dataclasses to immutable/frozen mode end-to-end

- Freeze dataclasses and remove in-place mutation patterns in loaders/trainers.
- Introduce explicit rebuild/replace helpers where updates are required.

1. Rework overrides onto immutable rebuild semantics

- Apply dot-path overrides by rebuilding the object tree bottom-up instead of
  mutating fields.
- Keep strict type-casting and fail-fast behavior unchanged.

1. Centralize config construction around a single strict path

- YAML load -> variable resolution -> dataclass hydration -> dot overrides ->
  validation should be the only supported flow.
- Remove duplicate ad-hoc config mutation/parsing paths across scripts.

1. Expand cross-field validation coverage into a dedicated validator module

- Keep one authoritative validation surface for bounds, coupled constraints,
  and sweep-time comparability requirements.

1. Rationalize serializable config snapshots

- Ensure one canonical `to_dict` representation for logging/checkpoint metadata
  without task-specific drift.

## Longer-Term Backlog

1. Chunked/fused cross-entropy for long contexts

- Avoid full `(B, S, V)` logits materialization in the default path.
- Maintain exact sum-reduction semantics used by current gradient scaling.

1. Contrastive training end-to-end hardening

- Validate schema consistency and full smoke runs for preprocess + train +
  checkpoint + resume.

1. Optional packed metadata ABI cleanup

- Evaluate whether collator should emit a richer packed metadata object
  (precomputed/reusable forms) when that gives measurable net wins without
  overcomplicating APIs.
