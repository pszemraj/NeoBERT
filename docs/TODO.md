# Deferred Work

Register of improvements that were consciously deferred rather than forgotten. Each entry records why it was deferred and what finishing it requires, so it can be picked up without re-deriving the context. Remove entries when they land.

## Optimizer

### Batch fused-QKV orthogonalization

`_orthogonalize_fused_qkv_update` (`src/neobert/optimizer/muon_clip.py`) runs Newton-Schulz/Polar-Express on the split Q/K/V matrices as three sequential 2D calls, costing roughly `3 x ns_steps x 3` small square GEMM launches per fused parameter per optimizer step (~45 launches at the default `ns_steps=5`, versus ~15 if the three same-shape matrices were stacked into one `[3, hidden, hidden]` batch). Splitting per projection is the intended correctness behavior and must not change; this item is purely about doing the same math with batched kernels.

Deferred because it is an overhead-only win (identical FLOPs, fewer launches) that requires modifying numerically sensitive shared code, which was out of scope for a correctness-fix branch. Completing it requires:

- teaching `_newton_schulz_update` and `_polar_express_update` to accept 3D batched input: `transpose(-2, -1)`/`.mT` instead of `.T`, and per-matrix norms (`dim=(-2, -1), keepdim=True`) instead of one scalar `torch.linalg.norm` that would mix Q/K/V magnitudes,
- stacking the split matrices in `_orthogonalize_fused_qkv_update` and applying `_normalize_muon_update` per matrix (all three share one shape, so the scale is common),
- verifying against `tests/test_muonclip_unit.py` reference implementations and the manual FSDP2 golden tests (`tests/manual/test_muonclip_fsdp2_golden.py`), plus a wall-clock benchmark demonstrating the win in eager mode.

### Split MuonClip clipping intent from runtime toggle

`get_optimizer` (`src/neobert/optimizer/optimizer.py`) mutates `config.enable_clipping = False` under FSDP2 because hooks are created before `accelerator.prepare()`. A dedicated post-prepare runtime toggle would preserve the configured intent separately from the runtime decision. Noted in a factory-side comment; tracked here so it survives comment churn.

## Resume

### Name-keyed optimizer-state transplant

The `optimizer_param_names.json` manifest fails fast when optimizer parameter order or state semantics drift (see [Training](guides/training.md)). If the repo later needs optimizer resume across intentional parameter-registration refactors, replace the fail-fast check with a true name-keyed optimizer-state transplant: load saved per-parameter state by manifest name instead of group position, then validate semantics as today. Until that need exists, fail-fast is the correct behavior.

## Streaming

### Optional snapshot cadence for retry resume

`RetryingStreamingDataset.__iter__` (`src/neobert/streaming.py`) calls `dataset.state_dict()` after every yielded example to guarantee exactly-once retry recovery. The payload is small (HF serializes cursor counters, never shuffle-buffer contents), so this is currently cheap; if profiling of a deep `.map()`/`.filter()` pipeline ever shows the per-yield snapshot mattering, add an opt-in snapshot-every-N-examples knob. That trades the exactly-once guarantee for "may re-yield up to N-1 examples on retry," so it must stay opt-in and documented. Snapshot-on-failure is not an alternative: nested iterable state advances past the failed example before the exception surfaces.
