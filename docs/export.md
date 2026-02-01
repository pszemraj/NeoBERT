# Hugging Face Export Guide

This guide covers exporting NeoBERT checkpoints to Hugging Face format.

> [!NOTE]
> Script details live in [scripts/export-hf/README.md](../scripts/export-hf/README.md).

## Prerequisites

Your checkpoint directory must contain:

- `state_dict.pt`
- `config.yaml`
- `tokenizer/` (with `special_tokens_map.json`, vocab, etc.)

Install dependencies:

```bash
pip install transformers safetensors pyyaml
```

## Export

```bash
# Export a specific checkpoint
python scripts/export-hf/export.py outputs/neobert_pretrain/model_checkpoints/100000
```

By default, the export lands in:

```
outputs/neobert_pretrain/hf/neobert_pretrain_100000/
```

You can override the destination:

```bash
python scripts/export-hf/export.py \
  outputs/neobert_pretrain/model_checkpoints/100000 \
  --output my_exported_model
```

## Output Files

The exporter writes:

- `config.json`
- `model.safetensors`
- `pytorch_model.bin`
- `model.py` (HF modeling file)
- `rotary.py`
- tokenizer assets (`tokenizer.json`, `vocab.txt`, `special_tokens_map.json`, ...)
- `README.md` (auto-generated)

## Config Mapping

The export script maps NeoBERT config fields to HF config fields, including:

| NeoBERT                   | HF                    | Notes               |
| ------------------------- | --------------------- | ------------------- |
| `hidden_size`             | `hidden_size`         | Model dimension     |
| `num_hidden_layers`       | `num_hidden_layers`   | Layers              |
| `num_attention_heads`     | `num_attention_heads` | Heads               |
| `intermediate_size`       | `intermediate_size`   | FFN size            |
| `max_position_embeddings` | `max_length`          | Max sequence length |
| `norm_eps`                | `norm_eps`            | Norm epsilon        |
| `vocab_size`              | `vocab_size`          | Vocab size          |
| `pad_token_id`            | `pad_token_id`        | Padding token       |

## Validation

```bash
python scripts/export-hf/validate.py outputs/neobert_pretrain/hf/neobert_pretrain_100000
```

The validator checks file presence, model loading, tokenizer loading, MLM head, end-to-end encode, and a cosine-similarity sanity check.

## Troubleshooting

- **Missing tokenizer files**: ensure the checkpoint has a complete `tokenizer/` directory.
- **Missing required config fields**: confirm `config.yaml` includes the `model` section with required keys.
- **Flash-attn warnings**: exported HF models use `flash-attn` if installed; install it if you want faster inference.

## Next Steps

- Evaluation guide: [docs/evaluation.md](evaluation.md)
- Training guide: [docs/training.md](training.md)
- Configuration reference: [docs/configuration.md](configuration.md)
