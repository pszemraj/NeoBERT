# Hugging Face Export Scripts

Scripts for exporting and validating NeoBERT checkpoints in HF format.

## Scripts

- `export.py` - convert a training checkpoint to HF-compatible model folder
- `validate.py` - run structural + forward-pass validation on exported folder

## Typical Flow

```bash
python scripts/export-hf/export.py outputs/<run>/model_checkpoints/<step>
python scripts/export-hf/validate.py outputs/<run>/hf/<export_name>
```

## Notes

- Input checkpoint can be native safetensors or DeepSpeed ZeRO state.
- Export writes both `model.safetensors` and `pytorch_model.bin`.
- Exported model is standard/unpacked HF path (no packed metadata inputs).

## Related Docs

- [docs/export.md](../../docs/export.md)
- [docs/troubleshooting.md](../../docs/troubleshooting.md)
