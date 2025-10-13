# Troubleshooting Guide

Common issues and their solutions when training and using NeoBERT.

## Table of Contents

- [Training Issues](#training-issues)
- [Export Issues](#export-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)

## Training Issues

### Configuration & CLI Issues

- **Config validation errors**: verify YAML indentation, required fields, and value types. Run with `--debug` to print the resolved config and validation warnings.
- **Import errors**: ensure your virtual environment has `pip install -e .[dev]` applied and that you are inside the project root before invoking scripts.
- **Slow data loading**: increase `trainer.dataloader_num_workers`, place datasets on faster storage, or enable streaming mode for giant corpora.

### Flash Attention Issues During GLUE Evaluation

- **Symptom**: Launching GLUE evaluation with Flash Attention enabled produces runtime errors or crashes.
- **Cause**: GLUE tasks use variable-length batches that are currently incompatible with Flash Attention's alignment requirements.
- **Solution**:
  1. When using the provided GLUE scripts/configs, no action is needed—Flash Attention is automatically disabled for you.
  2. If you author custom launchers, set `model.flash_attention: false` (or pass `--model.flash_attention false`) before evaluation.
  3. Restart the run after toggling the setting; mixed Flash Attention/eager runs in the same process can leave partially initialized CUDA kernels.

### Model Checkpoint Corruption

**Problem**: Model checkpoints appear to train well (good metrics) but fail during inference after export.

**Symptom**: MLM always predicts the same token (e.g., "1") despite showing 65%+ training accuracy.

**Solution**: Ensure the model is unwrapped from the accelerator before saving:

```python
# ❌ Wrong - saves accelerator wrapper
torch.save(model.state_dict(), "checkpoint.pt")

# ✅ Correct - saves actual model weights
torch.save(accelerator.unwrap_model(model).state_dict(), "checkpoint.pt")
```

This issue was fixed in `trainer.py` and `trainer_phase_2.py`.

### OOM During Training

**Problem**: Out of memory errors during training.

**Solutions**:

1. Enable gradient checkpointing: `--model.gradient_checkpointing true`
2. Reduce batch size: `--trainer.per_device_train_batch_size 16`
3. Use gradient accumulation: `--trainer.gradient_accumulation_steps 4`
4. Enable mixed precision: `--trainer.fp16 true` or `--trainer.bf16 true`

## Export Issues

### HuggingFace Model Fails to Load

**Problem**: `AttributeError: 'NoneType' object has no attribute 'bool()'` when loading exported model.

**Solution**: The HF modeling file needs to handle None attention masks properly:

```python
# In modeling_neobert.py
attn_mask=attention_mask.bool() if attention_mask is not None else None
```

### Missing Decoder Weights

**Problem**: MLM head weights not found after export.

**Solution**: Check that the export script correctly maps weights:

- Training format: `decoder.weight`, `decoder.bias`
- These should be preserved at the top level in HF format

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
4. Install Flash Attention: `pip install flash-attn`

### High Memory Usage

**Solutions**:

1. Use gradient checkpointing during fine-tuning
2. Clear cache between batches: `torch.cuda.empty_cache()`
3. Reduce max sequence length if possible
4. Use quantization for deployment

## Environment Issues

### Flash Attention Not Available

**Problem**: Flash Attention fails to install or isn't detected.

**Solution**:

```bash
# Build from source for latest GPUs
pip install flash-attn --no-build-isolation
```

### XFormers Compatibility

**Problem**: XFormers not working with your PyTorch version.

**Solution**:

```bash
# Build from source
pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git@main
```

## Validation Tools

Use these scripts to diagnose issues:

```bash
# Validate exported model
python scripts/export-hf/validate.py /path/to/exported/model

# Test MLM predictions
python scratch/mlm_predict.py /path/to/model --text "Test [MASK] sentence"

# Run comprehensive tests
python tests/run_tests.py
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/chandar-lab/NeoBERT/issues)
2. Review the test outputs for clues
3. Enable debug mode: `export NEOBERT_DEBUG=1`
4. Open a new issue with:
   - Error message and stack trace
   - Config file used
   - Environment details (PyTorch version, GPU, etc.)
   - Steps to reproduce
