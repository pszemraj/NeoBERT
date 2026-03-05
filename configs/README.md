# Configuration Files

Repository configs for training and evaluation workflows.

Field definitions and defaults are in [docs/configuration.md](../docs/configuration.md).

## Layout

- `configs/pretraining/` - production/experiment pretraining recipes.
- `configs/glue/` - per-task GLUE recipes and generated sweep configs.
- `configs/contrastive/` - contrastive fine-tuning recipes.

Tiny smoke-test configs live under [Test Config Files](../tests/configs/README.md).

## Related Docs

- [Configuration reference](../docs/configuration.md)
- [Training guide](../docs/training.md)
- [Evaluation guide](../docs/evaluation.md)
- [Tiny test configs](../tests/configs/README.md)
