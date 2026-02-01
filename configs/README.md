# Configuration Files

Production-ready YAML configurations for NeoBERT training and evaluation live here. Tiny smoke-test variants are under `tests/configs/`.

> [!TIP]
> For schema details and overrides, read [docs/configuration.md](../docs/configuration.md). End-to-end recipes live in [docs/training.md](../docs/training.md) and [docs/evaluation.md](../docs/evaluation.md).

## Directory Layout

```
configs/
├── pretraining/
├── glue/
├── contrastive/
└── README.md
```

## Pretraining (`configs/pretraining/`)

- `pretrain_neobert.yaml` – Standard 768×12 recipe
- `pretrain_streaming.yaml` – Streaming dataset example
- `pretrain_gpu_small.yaml` – Smaller GPU-friendly config
- `pretrain_neobert100m_smollm2data.yaml` – 100M SmolLM2 dataset variant
- `pretrain_neobert250m_smollm2data.yaml` – 250M SmolLM2 dataset variant
- `pretrain_neobert100m_smollm2data_muonclip.yaml` – MuonClip variant
- `train_small_custom_tokenizer.yaml` – Custom tokenizer training example

## GLUE (`configs/glue/`)

Task-specific GLUE configs:

- `cola.yaml`, `sst2.yaml`, `mrpc.yaml`, `stsb.yaml`, `qqp.yaml`, `mnli.yaml`, `qnli.yaml`, `rte.yaml`, `wnli.yaml`
- Generated configs land under `configs/glue/generated/`

## Contrastive (`configs/contrastive/`)

- `contrastive_neobert.yaml` – SimCSE-style contrastive fine-tuning

## Test Configs

Tiny configs meant for smoke tests live in `tests/configs/` (see [tests/configs/README.md](../tests/configs/README.md)).
