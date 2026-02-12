# Hugging Face Export Guide

Export NeoBERT training checkpoints to a Hugging Face-compatible folder.
This is the canonical reference for export behavior and constraints.
`scripts/export-hf/README.md` is intentionally command/script oriented.

## Supported Inputs

Point export to a checkpoint directory containing:

- `config.yaml`
- either `model.safetensors` (native) or DeepSpeed ZeRO checkpoint state
- `tokenizer/` directory (required)
- `tokenizer_info.json` (recommended; validated when present)

## Export Command

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step>
```

Optional output override:

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step> \
  --output outputs/<run>/hf/my_export
```

Optional legacy PyTorch checkpoint file:

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step> \
  --include-pytorch-bin
```

Legacy checkpoints with a decoder bias must opt in to dropping that bias:

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step> \
  --allow-decoder-bias-drop
```

## Export Output

Generated folder contains:

- `config.json`
- `model.safetensors`
- `modeling_neobert.py`
- `rotary.py`
- tokenizer assets (`tokenizer.json`, `special_tokens_map.json`, etc.)
- `README.md`

`pytorch_model.bin` is only written when `--include-pytorch-bin` is passed.

## Validation

```bash
python scripts/export-hf/validate.py outputs/<run>/hf/<export_name>
```

Validator checks file presence, model/tokenizer loading, MLM forward pass, and
basic output sanity.
It now also checks attention-mask parity (no-mask vs all-ones and
int/bool/additive equivalence) to catch exported-model mask regressions.

## Mapping Notes

- Export supports `hidden_act: swiglu|gelu`.
- `ngpt: true` checkpoints are not supported by HF export path.
- Export expects unpacked SwiGLU weights (`w1/w2/w3`).
- Export target LM head is biasless. If a checkpoint includes `decoder.bias`,
  export fails by default unless `--allow-decoder-bias-drop` is set.
- `attn_backend` is converted to HF `flash_attention` flag for config parity,
  but exported HF model remains standard/unpacked.

## Constraints

- Packed inputs/metadata are training-only and not supported in exported HF
  model.
- Exported model expects normal HF batches and attention masks.

## Troubleshooting

- Missing tokenizer: ensure checkpoint has `tokenizer/`.
- Config mismatch: ensure `config.yaml` and checkpoint weights match dimensions.
- Missing weights: verify checkpoint folder contains expected model files.

## Related Docs

- [scripts/export-hf/README.md](../scripts/export-hf/README.md)
- [Troubleshooting](troubleshooting.md)
- [Evaluation](evaluation.md)
