# Troubleshooting Guide

Common runtime and performance issues when training/evaluating NeoBERT.

## Training Issues

### Unknown config keys or type errors

- Config loading is strict; unknown keys raise.
- Fix YAML field names/types against [Configuration Reference](../reference/configuration.md).

### Packed training is slow

Symptoms:

- lower-than-expected `tokens/sec`
- high CPU usage, GPU bubbles

Checklist:

1. use `model.attn_backend: flash_attn_varlen` for packed runs,
2. ensure flash-attn is installed,
3. tune dataloader knobs (`dataset.num_workers`, `dataset.pin_memory`, `dataset.persistent_workers`, `dataset.prefetch_factor`); pinning-path behavior is described in [Training Optimization](training-optimization.md#dataloader-and-streaming-throughput),
4. compare `tokens/sec` rather than only `steps/sec`; single-process runs can benchmark `trainer.enforce_full_packed_batches: true`, while distributed runs must leave it false.

### Pretraining OOM from logits memory

Symptoms:

- high VRAM usage during MLM loss
- OOM when sequence length / batch size increases

Checklist:

1. keep `trainer.masked_logits_only_loss: true` (project default),
2. keep `trainer.mixed_precision: bf16` (or `no` if bf16 unsupported),
3. use `trainer.gradient_checkpointing: true` for additional memory headroom.

### `bf16` CUDA GEMM/runtime failures

Symptoms:

- PyTorch raises a CUDA/bf16 GEMM error soon after startup
- bf16 matmuls fail on one PyTorch build but succeed on another

What happens:

1. this is usually an environment/runtime issue rather than a NeoBERT config issue,
2. NeoBERT does not override PyTorch BLAS library selection at startup,
3. if bf16 is broken on the current stack, the failing PyTorch operation will still fail until you change the environment or disable bf16.

Actions:

1. pin a known-good PyTorch build for the affected host/GPU combination,
2. verify CUDA/driver/PyTorch compatibility and rebuild extension wheels after version changes,
3. set `trainer.mixed_precision: no` if that environment cannot run bf16 reliably,
4. if you disable bf16, keep `model.attn_backend: sdpa` for supported execution.

### MuonClip QK clipping rejected under FSDP2

- Set `optimizer.muon_config.enable_clipping: false` for FSDP2 Muon, or use an unsharded run for QK clipping. See [Clipping](training-optimization.md#clipping).

### Full packed batches rejected in distributed training

- Leave `trainer.enforce_full_packed_batches: false` for distributed runs. Rank-local packing can produce different row counts, so buffering until each rank independently has a full batch can desynchronize model calls and resume cursors. Variable local packed microbatches are supported and normalized by the global masked-token count; see [Packed Training](training-optimization.md#packed-training).

### Accelerate launch warnings about mixed precision or dynamo

- `accelerate launch` warnings about its default `--mixed_precision` or `--dynamo_backend` describe omitted launcher flags, not a NeoBERT override; the trainer still constructs `Accelerator(...)` from `trainer.mixed_precision` and the repo's compile settings.
- Pass matching launcher flags (`--mixed_precision bf16`, `--dynamo_backend no`) to keep startup output aligned with the runtime policy; see [Distributed launch policy](training.md#distributed-launch-policy).

### `torch.compile` warnings/recompiles

Typical warnings:

- symbolic shape guard churn,
- recompile-limit messages.

Actions:

- keep compile static unless needed (`trainer.torch_compile_dynamic: false` or unset),
- reduce dynamic control flow and per-step Python-side variability,
- use `TORCH_LOGS="recompiles"` to inspect root causes.

### Streaming resume is slow or not exact

- Streaming resume replays consumed batches, so late checkpoints can take time to reach their saved position and shuffled streams do not preserve exact sample order. Use a non-streaming dataset for deterministic continuation. See [Checkpointing and Resume](training.md#checkpointing-and-resume) and [streaming recovery](training-optimization.md#dataloader-and-streaming-throughput).

### Streaming eval budget error

- If streaming eval has no explicit budget, trainer raises: set `trainer.eval_max_batches` or `dataset.eval_samples`.
- Use fixed values across sweep runs for comparable metrics.

### Resume refuses to load optimizer state

- `optimizer_param_names.json is missing`: the checkpoint predates the optimizer resume manifest, so parameter order and state semantics cannot be verified. Start a new run or continue from model weights only; this repo does not carry checkpoint back-compat before a stable release.
- `outdated manifest schema`: the manifest was written before state-semantics tracking. Start a current run from the portable model weights or start fresh.
- `Optimizer state semantics changed`: the optimizer's update rule changed since the checkpoint was written. If a MuonClip selector (`norm_factor`, `param_policy`, `orthogonalization`, `nesterov`, or `clipping`) drifted, re-pin the checkpoint's value explicitly in the launch config. If the implementation or clipping reduction contract changed, the old optimizer buffers are not safely resumable; start a new run or continue from portable model weights only.
- `Optimizer parameter order changed`: the model's parameter registration order differs from the checkpoint; PyTorch optimizer state is positional, so a silent load would hand buffers to the wrong parameters.

### Resume fails because the Accelerate state directory is missing

- Full resume requires `<step>/accelerate/`. A checkpoint written before that layout may still contain portable `model.safetensors`, but it cannot restore optimizer, scheduler, random, or custom trainer state. Start a current run from the portable model weights or start fresh.

### Contrastive job exits with status 143 after SIGTERM

- This is the expected preemption path. The trainer synchronizes the termination request at the next completed optimizer update, writes a complete checkpoint under `checkpoints/<step>/`, and exits with `128 + SIGTERM`. Resume from `latest`; do not treat the nonzero status as an incomplete save unless the completion marker is absent.

## Evaluation Issues

### GLUE backend errors

- GLUE uses SDPA classifier wrappers and requires pretrained weights unless `glue.allow_random_weights: true` or `model.from_hub: true`. See [GLUE](evaluation.md#glue).

### MTEB task filtering is not what you expected

- Task selection and CLI override behavior are described in [MTEB](evaluation.md#mteb).

## Export Issues

### Export fails with missing tokenizer files

- Check checkpoint has `tokenizer/` directory with special tokens map and vocab files.

### Packed input mismatch at inference

- Exported HF model does not support packed metadata inputs.
- Use standard HF batches + attention masks.
