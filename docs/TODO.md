# Deferred Work

Register of improvements that were consciously deferred rather than forgotten. Each entry records why it was deferred and what finishing it requires, so it can be picked up without re-deriving the context. Remove entries when they land.

## Optimizer

### Batch fused-QKV orthogonalization

`_orthogonalize_fused_qkv_update` (`src/neobert/optimizer/muon_clip.py`) runs Newton-Schulz/Polar-Express on the split Q/K/V matrices as three sequential 2D calls, costing roughly `3 x ns_steps x 3` small square GEMM launches per fused parameter per optimizer step (~45 launches at the default `ns_steps=5`, versus ~15 if the three same-shape matrices were stacked into one `[3, hidden, hidden]` batch). Splitting per projection is the intended correctness behavior and must not change; this item is purely about doing the same math with batched kernels.

Deferred because it is an overhead-only win (identical FLOPs, fewer launches) that requires modifying numerically sensitive shared code, which was out of scope for a correctness-fix branch. Completing it requires:

- teaching `_newton_schulz_update` and `_polar_express_update` to accept 3D batched input: `transpose(-2, -1)`/`.mT` instead of `.T`, and per-matrix norms (`dim=(-2, -1), keepdim=True`) instead of one scalar `torch.linalg.norm` that would mix Q/K/V magnitudes,
- stacking the split matrices in `_orthogonalize_fused_qkv_update` and applying `_normalize_muon_update` per matrix (all three share one shape, so the scale is common),
- verifying against `tests/test_muonclip_unit.py` reference implementations and the manual FSDP2 golden tests (`tests/manual/test_muonclip_fsdp2_golden.py`), plus a wall-clock benchmark demonstrating the win in eager mode.

### Split the nGPT fused FFN before orthogonalization

`_uses_fused_qkv_muon_split` (`src/neobert/optimizer/muon_clip.py`) special-cases only `proj_type == "qkv"`, so the attention `qkv.weight` is split into Q/K/V and each is orthogonalized separately (the correctness fix from commit `08e668a`). The nGPT block's `c_fc` weight fuses the SwiGLU gate and up projections into one `(2*intermediate, hidden)` matrix (`NormEncoderBlock`, split only at forward time via `torch.chunk`), but it is never tagged, so Muon runs Newton-Schulz/Polar-Express over the whole fused matrix and mixes the two projections' singular subspaces - exactly the error the QKV split avoids, for the FFN instead. Dormant today: every shipped config sets `ngpt: false`, and no test exercises `ngpt=true` with MuonClip. Completing it requires tagging `c_fc.weight` with a fused-FFN `proj_type`, adding a 2-way split path (mirroring `_orthogonalize_fused_qkv_update`), and adding an `ngpt=true` + Muon golden test - there is currently no nGPT+Muon numerical reference to validate against, which is why this is deferred rather than done blind in a correctness branch.

## Resume

### Name-keyed optimizer-state transplant

The `optimizer_param_names.json` manifest fails fast when optimizer parameter order or state semantics drift (see [Training](guides/training.md)). If the repo later needs optimizer resume across intentional parameter-registration refactors, replace the fail-fast check with a true name-keyed optimizer-state transplant: load saved per-parameter state by manifest name instead of group position, then validate semantics as today. Until that need exists, fail-fast is the correct behavior.

## Streaming

### Exact streaming resume via a stateful-dataloader boundary

Trainer-level streaming resume is approximate (skip-based): on resume it re-advances the stream by the consumed batch count instead of restoring a cursor. Checkpointing the raw `train_dataset` cursor is unsafe because the dataset is consumed through an Accelerate-prepared `DataLoader` whose adapter (`DataLoaderShard`/`DataLoaderDispatcher`) iterates one batch ahead (and dispatch mode can prefetch `num_processes` batches) before yielding the batch the trainer optimizes - so the raw cursor at checkpoint time is ahead of the last trained batch, and trusting it would silently drop prefetched-but-untrained examples (regression: `tests/training/test_streaming_shuffle.py::TestStreamingRetryHelpers::test_prepared_dataloader_advances_raw_cursor_past_trained_batch`).

Deferred because the correct boundary is the prepared dataloader, not the dataset, and wiring that up is a focused piece of work of its own. Completing it requires:

- enabling Accelerate's `use_stateful_dataloader` (backed by torchdata `StatefulDataLoader`) via `DataLoaderConfiguration`, so the prepared loader accounts for its own prefetch/lookahead state,
- checkpointing the prepared/stateful dataloader's `state_dict()` rather than the raw dataset while continuing to restore the trainer's existing rank-local `stored_batch` packed-fragment buffer,
- an integration regression through `accelerator.prepare_data_loader()`: consume one yielded batch, save, rebuild, load, and assert resume yields the next untrained batch (not the batch after the loader's lookahead),
- confirming behavior under both `num_workers=0` and `num_workers>0`, and with an active HF shuffle buffer (whose contents are not serialized).

### Optional snapshot cadence for retry resume

`RetryingStreamingDataset.__iter__` (`src/neobert/streaming.py`) calls `dataset.state_dict()` after every yielded example to guarantee exactly-once retry recovery. The payload is small (HF serializes cursor counters, never shuffle-buffer contents), so this is currently cheap; if profiling of a deep `.map()`/`.filter()` pipeline ever shows the per-yield snapshot mattering, add an opt-in snapshot-every-N-examples knob. That trades the exactly-once guarantee for "may re-yield up to N-1 examples on retry," so it must stay opt-in and documented. Snapshot-on-failure is not an alternative: nested iterable state advances past the failed example before the exception surfaces.
