# Troubleshooting Guide

Common issues and their solutions when training and using NeoBERT.

## Table of Contents

- [Training Issues](#training-issues)
- [Export Issues](#export-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)

## Training Issues

### Configuration & CLI Issues

- **Config validation errors**: verify YAML indentation, required fields, and value types. Unknown keys are reported explicitly during config load; fix/remove those keys. `--debug` (pretraining only) prints vocab/tokenizer diagnostics, not a full config dump.
- **Import errors**: ensure your virtual environment has `pip install -e .[dev]` applied and that you are inside the project root before invoking scripts.
- **Slow data loading**: increase `dataset.num_workers`, place datasets on faster storage, or enable streaming mode for giant corpora.

### Attention Backend Issues During GLUE Evaluation

- **Symptom**: Launching GLUE evaluation with `attn_backend: flash_attn_varlen` produces runtime errors or crashes.
- **Cause**: GLUE tasks use variable-length batches that are incompatible with packed-sequence attention.
- **Solution**:
  1. When using the provided GLUE scripts/configs, no action is needed; SDPA is forced automatically.
  2. If you author custom launchers, set `model.attn_backend: sdpa` (or pass `--model.attn_backend sdpa`) before evaluation.

### Model Checkpoint Corruption

**Problem**: Model checkpoints appear to train well (good metrics) but fail during inference after export.

**Symptom**: MLM always predicts the same token (e.g., "1") despite showing 65%+ training accuracy.

**Solution**: Ensure the model is unwrapped from the accelerator before saving:

```python
# ❌ Wrong - saves accelerator wrapper
save_model_safetensors(model, "checkpoint_dir")

# ✅ Correct - saves actual model weights
save_model_safetensors(accelerator.unwrap_model(model), "checkpoint_dir")
```

This issue was fixed in `trainer.py`.

### OOM During Training

**Problem**: Out of memory errors during training.

**Solutions**:

1. Enable gradient checkpointing: `--trainer.gradient_checkpointing true`
2. Reduce batch size: `--trainer.per_device_train_batch_size 16`
3. Use gradient accumulation: `--trainer.gradient_accumulation_steps 4`
4. Enable mixed precision: `--trainer.mixed_precision bf16`

## Export Issues

### HuggingFace Model Fails to Load

**Problem**: `AttributeError: 'NoneType' object has no attribute 'bool()'` or similar mask-handling errors when loading an exported model.

**Solution**: Re-export with the latest `scripts/export-hf/export.py`, which bundles a modeling file that accepts `None`, bool, or additive attention masks. If you manually copied files, ensure `model.py`/`rotary.py` match the current repo versions.

### Missing Decoder Weights

**Problem**: MLM head weights not found after export.

**Solution**: Check that the export script correctly maps weights:

- Training format: `model.decoder.weight`, `model.decoder.bias`
- These are mapped to top-level `decoder.*` weights in the HF export

### Packed Sequences Not Supported

**Problem**: Errors or OOMs when passing packed inputs (`cu_seqlens` /
block-diagonal masks) to an exported model.

**Solution**: Exported HF models are vanilla Transformers and expect standard
attention masks on unpacked batches. Packing is a training-only feature; unpack
inputs before export/inference.

## Inference Issues

### MLM Always Predicts Same Token

**Problem**: Model consistently predicts "1" or another single token for all masked positions.

**Causes and Solutions**:

1. **Metaspace Tokenizer Issue**: The tokenizer adds extra space tokens before `[MASK]`

   ```python
   # Remove extra space tokens (ID 454) before mask tokens
   cleaned_ids = []
   for i, token_id in enumerate(input_ids):
       if token_id == 454 and i < len(input_ids) - 1 and input_ids[i + 1] == mask_token_id:
           continue
       cleaned_ids.append(token_id)
   ```

2. **Checkpoint Corruption**: See [Model Checkpoint Corruption](#model-checkpoint-corruption) above

3. **Attention Mask Issue**: See [HuggingFace Model Fails to Load](#huggingface-model-fails-to-load) above

### Poor MLM Performance

**Problem**: Model predictions are random or nonsensical.

**Checks**:

1. Verify training completed successfully with good metrics
2. Check that decoder weights are non-zero and have reasonable statistics
3. Ensure proper tokenizer is being used (should match training)
4. Verify model is in eval mode: `model.eval()`

## Performance Issues

### Slow Inference

**Solutions**:

1. Use batch processing for multiple inputs
2. Enable CUDA if available: `model.cuda()`
3. Use half precision: `model.half()` or `torch.autocast`
4. Install the appropriate attention backend:
   - Packed-sequence training: `pip install flash-attn` (for `attn_backend: flash_attn_varlen`)
   - Liger kernel primitives: `pip install liger-kernel` (auto-used on CUDA when `kernel_backend: auto`)
   - Exported HF models: rely on PyTorch SDPA (flash/mem-efficient kernels are selected by PyTorch when available)

### High Memory Usage

**Solutions**:

1. Use gradient checkpointing during fine-tuning
2. Clear cache between batches: `torch.cuda.empty_cache()`
3. Reduce max sequence length if possible
4. Use quantization for deployment

## Environment Issues

### Attention Backends

NeoBERT uses different backends depending on the code path:

- **Training** uses SDPA by default (`model.attn_backend: sdpa`). For packed sequences,
  set `model.attn_backend: flash_attn_varlen` (requires `flash-attn`).
- **Kernel primitives** (`model.kernel_backend: auto`): Liger kernel is used on CUDA for
  RMSNorm, SwiGLU, and CrossEntropy when available; falls back to torch on CPU or when
  `liger-kernel` is not installed.
- **Exported HF models** use PyTorch `scaled_dot_product_attention`; kernel selection
  (flash/mem-efficient/math) is handled by PyTorch.

**If packed-sequence training backend is missing**:

```bash
pip install flash-attn --no-build-isolation
```

**If you want faster HF inference**: use a recent PyTorch build with SDPA/flash kernels.
No extra dependency is required by default.

## Validation Tools

Use these scripts to diagnose issues:

```bash
# Validate exported model
python scripts/export-hf/validate.py /path/to/exported/model

# Test MLM predictions
python scripts/export-hf/mlm_predict.py /path/to/model --text "Test [MASK] sentence"

# Run comprehensive tests
python tests/run_tests.py
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/chandar-lab/NeoBERT/issues)
2. Review the test outputs for clues
3. Enable debug mode: pass `--debug` or set `debug: true` in your config
4. Open a new issue with:
   - Error message and stack trace
   - Config file used
   - Environment details (PyTorch version, GPU, etc.)
   - Steps to reproduce
