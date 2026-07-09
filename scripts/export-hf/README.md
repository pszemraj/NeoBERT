# Hugging Face Export Scripts

Scripts for exporting and validating NeoBERT checkpoints in HF format.

## Scripts

- `export.py` - convert a training checkpoint to HF-compatible model folder
- `validate.py` - run structural + forward-pass validation on exported folder
- `mlm_predict.py` - quick local masked-token inference sanity check for exports

## Related Docs

- [Export](../../docs/guides/export.md)
- [Troubleshooting](../../docs/guides/troubleshooting.md)
