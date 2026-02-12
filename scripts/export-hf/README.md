# Hugging Face Export Scripts

Scripts for exporting and validating NeoBERT checkpoints in HF format.

## Scripts

- `export.py` - convert a training checkpoint to HF-compatible model folder
- `validate.py` - run structural + forward-pass validation on exported folder

## Typical Flow

```bash
python scripts/export-hf/export.py outputs/<run>/checkpoints/<step>
python scripts/export-hf/validate.py outputs/<run>/hf/<export_name>
```

## Notes

- Detailed export constraints and compatibility notes live in
  [docs/export.md](../../docs/export.md).
- Legacy checkpoints that still contain a decoder bias require explicit opt-in:
  use `--allow-decoder-bias-drop` with `export.py`.
- Keep this README focused on script entry points.

## Related Docs

- [docs/export.md](../../docs/export.md)
- [docs/troubleshooting.md](../../docs/troubleshooting.md)
