# Configuration Files

Production-ready YAML configurations for NeoBERT training, evaluation, and contrastive learning live here. Lightweight smoke-test variants are under `tests/configs/`.

> [!TIP]
> For schema details, validation rules, and override examples, read [docs/configuration.md](/docs/configuration.md). End-to-end recipes live in [docs/training.md](/docs/training.md) and [docs/evaluation.md](/docs/evaluation.md).

## Production Configurations

### Pretraining

- `pretrain_neobert.yaml` – Standard 768×12 recipe
- `pretrain_streaming.yaml` – Streaming dataset pipeline for large-scale runs
- `pretrain_gpu_small.yaml` – Compact GPU-friendly config (SwiGLU)
- `pretrain_smollm2_custom_tokenizer.yaml` – 250M recipe with SmolLM2 tokenizer (32k, 1024 context)
- `pretrain_neobert100m_smollm2data_muonclip.yaml` – 100M MuonClip pretraining with SmolLM2 Stage-4 data

### Fine-tuning & Evaluation

- `evaluate_neobert.yaml` – GLUE/MTEB evaluation template
- `contrastive_neobert.yaml` – SimCSE-style contrastive training

### Custom Tokenizer

- `train_small_custom_tokenizer.yaml` – Example tokenizer training config
- `pretrain_smollm2_custom_tokenizer.yaml` – Reused for custom tokenizer pretraining

## Test Configurations

Tiny configs meant for smoke tests live in `tests/configs/`:

- `tests/configs/pretraining/test_tiny_pretrain.yaml`
- `tests/configs/evaluation/test_tiny_glue.yaml`
- `tests/configs/contrastive/test_tiny_contrastive.yaml`

## Related Documentation

- [Configuration Guide](/docs/configuration.md)
- [Training Guide](/docs/training.md)
- [Evaluation Guide](/docs/evaluation.md)
- [Testing Guide](/docs/testing.md)
