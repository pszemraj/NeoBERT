# Training Optimization

This guide covers optimizer selection, MuonClip defaults, throughput tuning, and training metrics that matter when comparing runs.

Field names and defaults live in [Configuration Reference](../reference/configuration.md).

## MuonClip Defaults

NeoBERT ships MuonClip with:

- `param_policy=hidden_2d`
- `norm_factor=neobert`
- `nesterov=true`
- `orthogonalization=polar_express`

These are repo defaults, not attempts to mirror reference Muon exactly.

### Parameter routing

`hidden_2d` is the shipped default.

- Muon applies to hidden transformer matrices.
- Embeddings, output/unembedding weights, biases, and norm parameters stay on Adam-style fallback groups.
- This matches the usual Muon guidance and keeps FSDP2 owner-compute costs bounded.

`all_2d` remains available for explicit compatibility or ablation runs.

### Normalization modes

| `norm_factor`      | Behavior | Typical use |
| ------------------ | -------- | ----------- |
| `neobert`          | `0.4 * sqrt(max(d_out, d_in))` | repo default for encoder pretraining |
| `muon_reference`   | `sqrt(max(1, d_out / d_in))` | reference Muon parity / ablations |
| `spectral`         | `sqrt(d_out / d_in)` | experimental scaling |
| `match_rms_adamw`  | `0.2 * sqrt(max(d_out, d_in))` | reduced legacy-style scale |
| `none`             | no post-orthogonalization scaling | debugging only |

Why `neobert` is the default:

- this encoder setup has trained better with the symmetric `neobert` scale than with reference Muon scaling,
- the name reflects repo intent rather than compatibility baggage,
- `muon_reference` is still available when exact reference behavior matters.

### Orthogonalization

- `polar_express` is the shipped default for CUDA throughput
- `newton_schulz` remains available for reference-style runs and debugging

### Fused QKV handling

NeoBERT uses fused `qkv.weight` matrices, but Muon does not treat them as one big matrix.

- the interleaved fused matrix is split into Q, K, and V projections,
- Muon orthogonalizes and normalizes each projection separately,
- the result is packed back into the model's fused row layout before the update is applied.

## Clipping

Two clipping systems exist and they are separate:

- `trainer.gradient_clipping`: clips final accumulated gradients
- `optimizer.muon_config.enable_clipping`: MuonClip QK activation clipping

QK clipping is auto-disabled for sharded FSDP2 Muon runs because the current owner-compute path does not support the activation-hook capture flow.

## Gradient Accumulation and Logged Norms

- `trainer.gradient_accumulation_steps` counts microbatches per optimizer update
- pretraining rescales gradients by the global masked-token count after accumulation
- `train/grad_norm` is logged after accumulation and masked-token rescaling, but before `trainer.gradient_clipping`
- `train/weight_norm` is logged after the optimizer update

For packed pretraining runs, compare `train/tokens_per_sec` and token counts, not only `steps/sec`.

## Packed Training

Enable packing with:

```yaml
datacollator:
  pack_sequences: true
```

Recommended for throughput:

- `model.attn_backend: flash_attn_varlen`
- flash-attn installed via `pip install -e .[flash]`

Useful control:

- `trainer.enforce_full_packed_batches: true` improves token-throughput stability by buffering undersized packed outputs, usually at some cost to step rate

`attn_backend: sdpa` still works for packed runs but uses the slower segmented fallback path.

## Dataloader and Streaming Throughput

Primary throughput knobs:

- `dataset.num_workers`
- `dataset.pin_memory`
- `dataset.persistent_workers`
- `dataset.prefetch_factor`
- `dataset.streaming_read_retries`
- `dataset.streaming_read_retry_backoff_seconds`
- `dataset.streaming_read_retry_max_backoff_seconds`

NeoBERT keeps pinned CPU staging enabled on CUDA. When Accelerate owns device placement, loaders preserve `pin_memory=True` so unpacked batches stay pinned end to end; packed/manual-transfer paths instead re-pin the final CPU batch immediately before the non-blocking device copy.

For hub-backed streaming datasets:

- Hugging Face iterable streams are detected as streaming datasets before DataLoader construction, adapted to PyTorch's iterable API when needed, and not given map-style options such as DataLoader-level `shuffle`,
- NeoBERT retries transient read failures while inspecting schemas and during long-running stream iteration,
- retry recovery resumes from the last yielded example when the underlying HF iterable dataset supports `state_dict()/load_state_dict()`, and the retry wrapper remains visible to checkpoint/save-state resume paths and streaming eval-budget checks,
- shuffled streams can still perturb exact in-buffer order after a retry because HF refill semantics do not preserve the old shuffle buffer contents.

## Contrastive Objective Details

Contrastive CE is still accumulated as a summed loss for logging, but the training loop divides by local query count before backward so gradient scale does not change just because a task uses a larger local batch size. `train/loss` remains a per-sample mean in metrics.

`contrastive.pooling` is active in the trainer. Use `avg` for masked mean pooling, `cls` for the first token, or `max` for masked max pooling.

When `contrastive.pretraining_prob > 0`, `model.dropout_prob` must be greater than zero so the SimCSE-style anti-forgetting branch has stochastic two-view corruption.

## Distributed Muon

The maintained distributed Muon path is:

- Accelerate FSDP2
- 1D row-sharded DTensor mesh
- owner-compute update path

Do not combine MuonClip with tensor parallelism, context parallelism, or other multi-axis DTensor layouts.

Before long multi-rank runs, use the commands in [Manual Validation Scripts](../../tests/manual/README.md).

## Choosing a Mode

Use `neobert` when:

- you want the repo's recommended training behavior,
- you care about reproducing current branch baselines,
- you are running normal single-GPU or FSDP2 encoder pretraining.

Use `muon_reference` when:

- you want reference-Muon semantics for comparison,
- you are comparing against PyTorch/OpenAI/Keller-style Muon implementations,
- you are running a focused optimizer ablation rather than the repo default.
