# Test Configuration Files

Test configs live here and are optimized for quick runs and regression coverage.

## Directory Structure

- `pretraining/`
  - `test_tiny_pretrain.yaml` - Minimal CPU-friendly pretraining
  - `test_streaming*.yaml` - Streaming dataset coverage
  - `test_gpu*.yaml` - GPU-oriented smoke tests
  - `test_pretokenized.yaml` / `test_tiny_pretrain_tokenized.yaml` - Pretokenized pipeline
  - `test_smollm2_*.yaml` - SmolLM2 dataset variants

- `evaluation/`
  - `test_tiny_glue.yaml` / `test_tiny_glue_random.yaml` - Minimal GLUE runs

- `contrastive/`
  - `test_tiny_contrastive.yaml` - Minimal contrastive fine-tuning

## Usage

```bash
# Pretraining smoke test
python scripts/pretraining/pretrain.py --config tests/configs/pretraining/test_tiny_pretrain.yaml

# GLUE evaluation smoke test
python scripts/evaluation/run_glue.py --config tests/configs/evaluation/test_tiny_glue.yaml

# Contrastive fine-tuning smoke test
python scripts/contrastive/finetune.py --config tests/configs/contrastive/test_tiny_contrastive.yaml
```

## Notes

- These configs are intentionally small and may use CPU-friendly settings.
- For full training runs, use configs under `configs/`.
