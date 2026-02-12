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
- Keep this README structural; avoid copying field semantics here.

## Directory Intent

- `configs/pretraining/` - production/experiment pretraining recipes.
- `configs/glue/` - per-task GLUE recipes and generated sweep configs.
- `configs/contrastive/` - contrastive fine-tuning recipes.

## Related Docs

- [Configuration reference](../docs/configuration.md)
- [Training guide](../docs/training.md)
- [Evaluation guide](../docs/evaluation.md)
- [Tiny test configs](../tests/configs/README.md)
