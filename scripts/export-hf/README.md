# Hugging Face Export Scripts

Scripts for exporting and validating NeoBERT checkpoints in HF format.

## Scripts

- `export.py` - convert a training checkpoint to HF-compatible model folder
- `validate.py` - run structural + forward-pass validation on exported folder
- `mlm_predict.py` - masked-token inference for a local export or Hub model

## Related Docs

- [Export](../../docs/guides/export.md)
- [Troubleshooting](../../docs/guides/troubleshooting.md)
