# Hugging Face Export Guide

Export NeoBERT training checkpoints to a Hugging Face-compatible folder. Helper scripts live under [Hugging Face Export Scripts](../../scripts/export-hf/README.md).

## Supported Inputs

Point export to a checkpoint directory containing:

- `config.yaml`
- either `model.safetensors` (native) or DeepSpeed ZeRO checkpoint state
- `tokenizer/` directory (required)
- `tokenizer_info.json` (recommended; validated when present)

DeepSpeed ZeRO checkpoint conversion is legacy-only and requires the optional `neobert[legacy-checkpoints]` install extra.

## Export Command

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step>
```

Without `--output`, export writes `outputs/<run>/hf/<run>_<step>/`.

Optional output override:

```bash
python scripts/export-hf/export.py \
  outputs/<run>/checkpoints/<step> \
  --output outputs/<run>/hf/my_export
```

Optional `pytorch_model.bin` export:

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

Validator checks file presence, model/tokenizer loading, MLM forward pass, basic output sanity, and attention-mask parity across no-mask, all-ones, integer, boolean, and additive forms.

Run a masked-token prediction against a local export or Hub model:

```bash
python scripts/export-hf/mlm_predict.py \
  outputs/<run>/hf/<run>_<step> \
  --text "NeoBERT is a [MASK] encoder."
```

## Mapping Notes

- Export supports `model.hidden_act: swiglu|gelu`.
- Export expects unpacked SwiGLU weights (`w1/w2/w3`).
- Export target LM head is biasless. If a checkpoint includes `decoder.bias`, export fails by default unless `--allow-decoder-bias-drop` is set.
- Exported HF models use their standalone standard attention implementation; training-only attention backend settings are not serialized.

## Constraints

The [architecture support matrix](../reference/architecture.md#ngpt-mode) covers nGPT task limitations. Exported models use ordinary Hugging Face batches and attention masks; packed training metadata is not supported.

## Troubleshooting

- Missing tokenizer: ensure checkpoint has `tokenizer/`.
- Config mismatch: ensure `config.yaml` and checkpoint weights match dimensions.
- Missing weights: verify checkpoint folder contains expected model files.

## Related Docs

- [Troubleshooting](troubleshooting.md)
- [Evaluation](evaluation.md)
