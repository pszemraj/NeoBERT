# Evaluation Guide

NeoBERT evaluation currently focuses on GLUE and MTEB.
This is the canonical reference for evaluation behavior and caveats.

## GLUE

### Run one task

```bash
python scripts/evaluation/run_glue.py configs/glue/cola.yaml
```

### Run quick/full suites

```bash
bash scripts/evaluation/glue/run_quick_glue.sh configs/glue
bash scripts/evaluation/glue/run_all_glue.sh configs/glue
```

### Important GLUE behavior

- GLUE always runs with SDPA attention in classifier wrappers.
- Pretrained local checkpoints are required unless either
  `glue.allow_random_weights: true` or `model.from_hub: true`.
- GLUE resumable checkpoints are written to `trainer.output_dir/checkpoints/<step>/`
  and model export snapshots to `trainer.output_dir/model_checkpoints/<step>/`.
- Results are stored under `trainer.output_dir` as JSON metrics.

### Summarize GLUE outputs

```bash
python scripts/evaluation/glue/summarize_glue.py outputs/glue/<run>
```

### Build generated GLUE configs from sweeps

```bash
bash scripts/evaluation/glue/build_configs.sh outputs/my_sweep my-tag \
  --config-output-dir configs/glue/generated \
  --tasks cola,qnli
```

## MTEB

### Run MTEB

```bash
python scripts/evaluation/run_mteb.py \
  configs/pretraining/pretrain_neobert.yaml \
  --model_name_or_path outputs/<pretrain_run>
```

### Important MTEB behavior

- Runner loads checkpoints from `<model_name_or_path>/checkpoints/`.
- Task family selection is read from config field `mteb_task_type`.
- `--task_types` can override config selection at launch time.
  Accepts categories (`classification`, `retrieval`, `sts`, `all`) and/or
  explicit task names (comma-separated).
- Output path is currently derived from run dir + checkpoint + max length:
  `outputs/<run>/mteb/<ckpt>/<max_length>/`.
- If using a local tokenizer, point `tokenizer.name` to that path.

## Common Evaluation Pitfalls

1. Wrong checkpoint path

- verify `glue.pretrained_checkpoint_dir`, `glue.pretrained_checkpoint`, and
  `glue.pretrained_model_path` in GLUE configs.

1. Flat/random GLUE metrics

- confirm pretrained weights were actually loaded (or intentionally set
  `allow_random_weights: true`).

1. OOM during eval

- reduce eval batch size and/or sequence length.

1. Attention backend confusion

- GLUE path is SDPA-oriented; packed flash varlen is a training optimization.

## Related Docs

- [Configuration](configuration.md)
- [Training](training.md)
- [Troubleshooting](troubleshooting.md)
