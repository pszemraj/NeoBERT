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

## Shared Explicit Defaults

Pretraining and GLUE configs now include a small set of explicit defaults that
users commonly ask about (instead of relying on implicit schema defaults):

- `tokenizer.truncation: true`
- `wandb.enabled: true|false` (explicit per config)
- No deprecated `trainer.report_to` fields

Pretraining configs also make these defaults explicit:

- `datacollator.mask_all: false` (standard sampled-token 80/10/10 corruption)
- `trainer.masked_logits_only_loss: true`
- `trainer.log_train_accuracy: true`
- `trainer.log_grad_norm: true`
- `trainer.log_weight_norms: true`
- `trainer.enforce_full_packed_batches: true`

GLUE configs also make these defaults explicit:

- `glue.num_workers: 4`
- `glue.preprocessing_num_proc: 4`
- `trainer.save_steps` is set to a positive value (even when `save_strategy: "no"`)
  so configs pass strict validation

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
