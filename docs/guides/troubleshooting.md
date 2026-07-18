# Troubleshooting Guide

Common runtime and performance issues when training/evaluating NeoBERT.

## Training Issues

### Unknown config keys or type errors

- Config loading is strict; unknown keys raise.
- Fix YAML field names/types against the [YAML configuration reference](../reference/config_reference.yaml).

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

- Use the supported topology and configuration combinations in [Clipping](training-optimization.md#clipping).

### Full packed batches rejected in distributed training

- Leave `trainer.enforce_full_packed_batches: false`; distributed packed-batch behavior is described in [Packed Training](training-optimization.md#packed-training).

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

- See [Checkpointing and Resume](training.md#checkpointing-and-resume) for cross-restart behavior and [streaming recovery](training-optimization.md#dataloader-and-streaming-throughput) for in-process retry behavior. Use a non-streaming dataset when deterministic continuation is required.

### Streaming eval budget error

- If streaming eval has no explicit budget, trainer raises: set `trainer.eval_max_batches` or `dataset.eval_samples`.
- Use fixed values across sweep runs for comparable metrics.

### Resume refuses to load optimizer state

- `optimizer_param_names.json is missing`: the checkpoint predates the optimizer resume manifest, so parameter order and state semantics cannot be verified. Start a new run or continue from model weights only; this repo does not carry checkpoint back-compat before a stable release.
- `outdated manifest schema`: the manifest was written before state-semantics tracking. Start a current run from the portable model weights or start fresh.
- `Optimizer state semantics changed`: the saved manifest and the optimizer reconstructed from checkpoint-owned configuration no longer describe the same update rule. Launch-time optimizer overrides cannot repair this because resume restores optimizer selectors from `config.yaml` before validation. Use the code revision that produced the checkpoint, start a new run, or continue from portable model weights without optimizer state.
- `Optimizer parameter order changed`: the model's parameter registration order differs from the checkpoint; PyTorch optimizer state is positional, so a silent load would hand buffers to the wrong parameters.

### Checkpoint config reports unknown configuration keys

Pre-stable checkpoint configs are intentionally not schema-compatible with current code. Recreate the configuration against [the current reference](../reference/config_reference.yaml), or use the historical code revision that produced the checkpoint; portable model weights can still be used when their architecture matches.

### Resume fails because the Accelerate state directory is missing

- Full resume requires `<step>/accelerate/`. A checkpoint written before that layout may still contain portable `model.safetensors`, but it cannot restore optimizer, scheduler, random, or custom trainer state. Start a current run from the portable model weights or start fresh.

### Training job exits with status 143 after SIGTERM

- This is the expected preemption path for pretraining and contrastive jobs. Resume from `latest`; [checkpoint validation](training.md#checkpointing-and-resume) rejects missing or damaged artifacts.

## Evaluation Issues

### GLUE loads the wrong weights or produces flat metrics

- Verify `glue.pretrained_checkpoint_dir`, `glue.pretrained_checkpoint`, and `glue.pretrained_model_path`, then confirm pretrained weights were loaded unless random weights are intentional.
- GLUE uses its SDPA classifier path; packed flash attention is a pretraining optimization. See [Important GLUE behavior](evaluation.md#important-glue-behavior) for the runtime contract.

### Evaluation runs out of memory

- Reduce the evaluation batch size or sequence length.

### MTEB task filtering is not what you expected

- Task selection and CLI override behavior are described in [MTEB](evaluation.md#mteb).

## Export Issues

### Export fails with missing tokenizer files

- Check the required [export inputs](export.md#supported-inputs).

### Packed input mismatch at inference

- Use the input contract in [HF Export Model Differences](../reference/architecture.md#hf-export-model-differences).
