# Evaluation Guide

This guide covers evaluating NeoBERT on GLUE and MTEB.

> [!NOTE]
> Script-level notes live in [scripts/evaluation/README.md](../scripts/evaluation/README.md).

## GLUE Benchmark

### Run a single task

```bash
python scripts/evaluation/run_glue.py --config configs/glue/cola.yaml
```

### Run the full suite

```bash
bash scripts/evaluation/glue/run_all_glue.sh
```

### Config essentials

GLUE configs live in `configs/glue/` and include both **model** and **glue** sections. The GLUE trainer reads a few extra fields from the raw `model` block, so keep these in place:

```yaml
task: glue

model:
  pretrained_checkpoint_dir: ./outputs/neobert_pretrain
  pretrained_checkpoint: 100000  # or "latest"
  pretrained_config_path: ./outputs/neobert_pretrain/model_checkpoints/100000/config.yaml
  # Optional when testing with random weights:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  dropout_prob: 0.1
  vocab_size: 30522
  max_position_embeddings: 512
  hidden_act: swiglu

glue:
  task_name: cola
  num_labels: 2
  max_seq_length: 512
  allow_random_weights: false
```

### Random-weights sanity checks

For smoke tests, set **either** of the following:

```yaml
model:
  allow_random_weights: true
# or
glue:
  allow_random_weights: true
```

### Flash Attention behavior

GLUE runs always force eager attention (Flash Attention is disabled) to avoid variable-length alignment issues.

### Summarize results

```bash
python scripts/evaluation/glue/summarize_glue.py outputs/glue/neobert-100m
```

### Generate configs from a sweep

```bash
bash scripts/evaluation/glue/build_configs.sh outputs/my_sweep neobert/glue \
  --config-output-dir configs/glue/generated \
  --tasks cola,qnli
```

## MTEB Benchmark

### Run evaluation

```bash
python scripts/evaluation/run_mteb.py \
  --config outputs/<pretrain_run>/model_checkpoints/<step>/config.yaml \
  --model_name_or_path outputs/<pretrain_run> \
  --task_types retrieval,sts
```

Notes:

- Requires the `mteb` package.
- If `use_deepspeed: true`, the script loads weights via DeepSpeed utilities.
- Outputs land under `outputs/<pretrain_run>/mteb/<step>/<max_length>/`.
- The MTEB runner currently reads `tokenizer.name`; if you trained with a local tokenizer, set `tokenizer.name` to that path in the config.

## Troubleshooting

- Flash attention errors on GLUE: expected (forced off).
- OOM: lower `trainer.per_device_train_batch_size` and/or enable `trainer.gradient_checkpointing`.
- Random or flat metrics: verify `pretrained_config_path` and checkpoint paths.

## Next Steps

- Configuration reference: [docs/configuration.md](configuration.md)
- Training workflows: [docs/training.md](training.md)
- Export guide: [docs/export.md](export.md)
