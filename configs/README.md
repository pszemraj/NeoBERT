# Configuration Files

Repository configs for training and evaluation workflows.

Field definitions and defaults are in [docs/reference/configuration.md](../docs/reference/configuration.md).

## Layout

- `configs/pretraining/` - runnable pretraining recipes; most use Hub datasets or tokenizers.
- `configs/glue/` - checkpoint-specific templates plus generated configs under `generated/`; generate configs for the checkpoint you intend to evaluate.
- `configs/contrastive/` - contrastive fine-tuning recipe that expects preprocessed data at `dataset.path`.

Small test fixtures live under [Test Config Files](../tests/configs/README.md); they are not all standalone jobs.

## Related Docs

- [Training guide](../docs/guides/training.md)
- [Training optimization](../docs/guides/training-optimization.md)
- [Evaluation guide](../docs/guides/evaluation.md)
