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

1. ensure `trainer.masked_logits_only_loss: true` for masked-logits-only loss,
2. keep `trainer.mixed_precision: bf16` (or `no` if bf16 unsupported),
3. use `gradient_checkpointing: true` for additional memory headroom.

### `torch.compile` warnings/recompiles

Typical warnings:

- symbolic shape guard churn,
- recompile-limit messages.

Actions:

- keep compile static unless needed (`trainer.torch_compile_dynamic: false` or unset),
- reduce dynamic control flow and per-step Python-side variability,
- use `TORCH_LOGS="recompiles"` to inspect root causes.

### Streaming resume caveat

- Accelerator checkpoint resume works, but exact data position/order is not
  preserved for streaming datasets.
- For reproducible resume, prefer pre-tokenized non-streaming datasets.

## Evaluation Issues

### GLUE backend errors

- GLUE wrappers use SDPA path; packed flash varlen is training-oriented.
- Ensure GLUE configs point to valid pretrained checkpoints unless intentionally
  using `allow_random_weights: true`.

### MTEB task filtering seems ignored

- Set `mteb_task_type` in config.
- `run_mteb.py --task_types` is currently parsed but not wired into task
  selection logic.

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
pytest tests/kernels/test_attention.py tests/model/test_model_forward.py -q
```
