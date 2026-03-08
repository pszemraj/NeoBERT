# Troubleshooting Guide

Common runtime and performance issues when training/evaluating NeoBERT.

## Training Issues

### Unknown config keys or type errors

- Config loading is strict; unknown keys raise.
- Fix YAML field names/types against [configuration.md](configuration.md).

### Packed training is slow

Symptoms:

- lower-than-expected `tokens/sec`
- high CPU usage, GPU bubbles

Checklist:

1. use `attn_backend: flash_attn_varlen` for packed runs,
2. ensure flash-attn is installed,
3. tune dataloader knobs (`dataset.num_workers`, `pin_memory`,
   `persistent_workers`, `prefetch_factor`),
4. compare `tokens/sec` (not only `steps/sec`) when
   `enforce_full_packed_batches=true`.

### Pretraining OOM from logits memory

Symptoms:

- high VRAM usage during MLM loss
- OOM when sequence length / batch size increases

Checklist:

1. keep `trainer.masked_logits_only_loss: true` (project default),
2. keep `trainer.mixed_precision: bf16` (or `no` if bf16 unsupported),
3. use `gradient_checkpointing: true` for additional memory headroom.

### `bf16` CUDA GEMM/runtime failures

Symptoms:

- PyTorch raises a CUDA/bf16 GEMM error soon after startup
- bf16 matmuls fail on one PyTorch build but succeed on another
- flash-attn may need to be disabled manually by setting `mixed_precision: no`

What happens:

1. this is usually an environment/runtime issue rather than a NeoBERT config
   issue,
2. NeoBERT does not override PyTorch BLAS library selection at startup,
3. if bf16 is broken on the current stack, the failing PyTorch operation will
   still fail until you change the environment or disable bf16.

Actions:

1. pin a known-good PyTorch build for the affected host/GPU combination,
2. verify CUDA/driver/PyTorch compatibility and rebuild extension wheels after
   version changes,
3. set `trainer.mixed_precision: no` if that environment cannot run bf16
   reliably,
4. if you disable bf16, keep `attn_backend: sdpa` for supported execution.

### `torch.compile` warnings/recompiles

Typical warnings:

- symbolic shape guard churn,
- recompile-limit messages.

Actions:

- keep compile static unless needed (`trainer.torch_compile_dynamic: false` or unset),
- reduce dynamic control flow and per-step Python-side variability,
- use `TORCH_LOGS="recompiles"` to inspect root causes.

### Streaming resume is slow or not exact

- Streaming resume is best-effort: trainer restores state and skips consumed
  batches from stream start/current epoch position.
- For large consumed-step counts, startup can take a while due to stream
  advancement.
- With shuffled streams, exact sample continuity is not guaranteed.
- If you need deterministic continuation, pre-tokenize to disk and run with
  `dataset.streaming: false`.

### Streaming eval budget error

- If streaming eval has no explicit budget, trainer raises:
  set `trainer.eval_max_batches` or `dataset.eval_samples`.
- Use fixed values across sweep runs for comparable metrics.

## Evaluation Issues

### GLUE backend errors

- GLUE wrappers use SDPA path; packed flash varlen is training-oriented.
- Ensure GLUE configs point to valid pretrained checkpoints unless intentionally
  using `allow_random_weights: true`.

### MTEB task filtering is not what you expected

- Config-based selection uses `mteb_task_type`.
- CLI override `run_mteb.py --task_types` is also supported.
- `--task_types` accepts categories (`classification`, `clustering`,
  `pair_classification`, `reranking`, `retrieval`, `sts`, `all`) and/or
  explicit task names (comma-separated).

## Export Issues

### Export fails with missing tokenizer files

- Check checkpoint has `tokenizer/` directory with special tokens map and vocab
  files.

### `ngpt` checkpoint export failure

- HF export path currently does not support `ngpt: true` checkpoints.

### Packed input mismatch at inference

- Exported HF model does not support packed metadata inputs.
- Use standard HF batches + attention masks.

## Checkpointing Notes

- Training checkpoints are safetensors-first (`model.safetensors`).
- Portable checkpoint loading recursively strips known runtime wrappers such as
  `_orig_mod.` and `module.` from generic compiled/distributed saves.
- Warning about removing shared tensors during save can be expected when tied
  weights are de-duplicated; validate by reloading checkpoint and running a
  forward pass.

## Environment and Dependency Notes

- Flash-attn path: install optional extra `pip install -e .[flash]`.
- Liger kernels are used when available under `kernel_backend: auto` on CUDA.
- Keep PyTorch/CUDA versions aligned with your installed extension wheels.

## Minimal Diagnostics

```bash
# Validate exported model
python scripts/export-hf/validate.py /path/to/exported/model

# Tiny pretraining smoke test
python scripts/pretraining/pretrain.py tests/configs/pretraining/test_tiny_pretrain.yaml

# Focused tests for packed attention path
pytest tests/kernels/test_attention.py tests/test_model_forward.py -q
```
