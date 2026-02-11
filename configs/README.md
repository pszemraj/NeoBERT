# Configuration Files

Repository configs for training and evaluation workflows.
Field definitions/defaults are documented in
[docs/configuration.md](../docs/configuration.md).

- Production/experiment configs live under `configs/`.
- Tiny smoke-test configs live under `tests/configs/`.

## Layout

```text
configs/
  pretraining/
  glue/
  contrastive/
```

## Config Authoring Policy

- Keep semantics/default definitions in
  [docs/configuration.md](../docs/configuration.md).
- Keep runtime behavior in [docs/training.md](../docs/training.md) and
  [docs/evaluation.md](../docs/evaluation.md).
- Use this directory for runnable recipes, not schema documentation.

Commonly pinned fields in repo configs (for transparency in experiment YAMLs):

- `wandb.enabled`
- `tokenizer.truncation`
- pretraining: `datacollator.mask_all`, `trainer.masked_logits_only_loss`,
  `trainer.log_train_accuracy`, `trainer.enforce_full_packed_batches`
- GLUE: `glue.num_workers`, `glue.preprocessing_num_proc`

## Pretraining Configs

- `pretrain_neobert.yaml` - baseline recipe
- `pretrain_streaming.yaml` - streaming data example
- `pretrain_gpu_small.yaml` - smaller GPU-friendly run
- `pretrain_smoke.yaml` - short smoke config
- `pretrain_neobert100m_smollm2data.yaml` - 100M SmolLM2 variant
- `pretrain_neobert250m_smollm2data.yaml` - 250M SmolLM2 variant
- `pretrain_neobert100m_smollm2data_muonclip.yaml` - MuonClip variant
- `train_small_custom_tokenizer.yaml` - custom tokenizer recipe

## GLUE Configs

Task configs under `configs/glue/`:
- `cola.yaml`, `sst2.yaml`, `mrpc.yaml`, `stsb.yaml`, `qqp.yaml`,
  `mnli.yaml`, `qnli.yaml`, `rte.yaml`, `wnli.yaml`

Generated sweep-derived configs typically go under `configs/glue/generated/`.

## Contrastive Configs

- `contrastive/contrastive_neobert.yaml`

## Related Docs

- [Configuration reference](../docs/configuration.md)
- [Training guide](../docs/training.md)
- [Evaluation guide](../docs/evaluation.md)
