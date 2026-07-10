# Evaluation Guide

NeoBERT evaluation currently focuses on GLUE and MTEB.

## GLUE

The YAML files directly under `configs/glue/` are checkpoint-specific templates and reference example artifacts that are not included in the repository. Generate configs from a local pretraining run, or override the model with a real Hub identifier.

### Generate configs from a pretraining run

```bash
python scripts/evaluation/glue/build_glue_configs.py \
  --checkpoint-dir outputs/<pretrain-run> \
  --checkpoint-step latest
```

Generated configs are written beneath `configs/glue/generated/` by default. Validate a config before a long launch:

```bash
python scripts/evaluation/glue/validate_glue_config.py \
  configs/glue/generated/<run>/cola.yaml
```

### Run one task

```bash
python scripts/evaluation/run_glue.py configs/glue/generated/<run>/cola.yaml

# Evaluate/fine-tune a Hub sequence classifier instead of local weights
python scripts/evaluation/run_glue.py configs/glue/cola.yaml \
  --model_name_or_path <hub-model-id>
```

### Run quick or full suites

```bash
python scripts/evaluation/glue/run_glue_suite.py \
  configs/glue/generated/<run> --suite quick
python scripts/evaluation/glue/run_glue_suite.py \
  configs/glue/generated/<run> --suite all
```

The quick suite runs RTE, MRPC, STS-B, and CoLA and stops at the first failure. The full suite attempts every registered GLUE task, then exits nonzero if any task failed. Logs default to `logs/<config-directory-name>/<task>.log`; use `--log-dir` to override that location, `--model-name-or-path` to override every task's model, and `--dry-run` to validate configs and print commands without training.

### Important GLUE behavior

- GLUE always runs with SDPA attention in classifier wrappers; non-SDPA `model.attn_backend` requests are normalized away with a warning.
- Pretrained local checkpoints are required unless either `glue.allow_random_weights: true` or `model.from_hub: true`.
- When checkpoint saving is enabled, GLUE writes checkpoints to `trainer.output_dir/checkpoints/<step>/`.
- Legacy `model_checkpoints/<step>/` paths are still accepted when loading older artifacts.
- Results are stored under `trainer.output_dir` as JSON metrics.

### Summarize GLUE outputs

```bash
python scripts/evaluation/glue/summarize_glue.py outputs/glue/<run>
```

## MTEB

### Run MTEB

```bash
python scripts/evaluation/run_mteb.py \
  outputs/<pretrain-run>/checkpoints/<step>/config.yaml \
  --model_name_or_path outputs/<pretrain-run> \
  --pretrained_checkpoint <step>
```

### Important MTEB behavior

- Runner loads checkpoints from `<model_name_or_path>/checkpoints/`.
- Task family selection is read from config field `mteb_task_type`.
- `--task_types` can override config selection at launch time. It accepts `classification`, `clustering`, `pair_classification`, `reranking`, `retrieval`, `sts`, `summarization`, `all`, or explicit task names as a comma-separated list.
- Omitting `--task_types` preserves `mteb_task_type`; explicitly passing `--task_types all` always selects the full registry.
- Task split metadata comes from the shared registry. MSMARCO runs and aggregates its `dev` score; other registered tasks currently use `test`.
- `mteb_pooling` defaults to mask-aware average pooling (`avg`; `mean` is accepted as an alias). Set it to `cls` for first-token pooling.
- Output defaults to `outputs/<run>/mteb/<ckpt>/<max_length>/`; pass `--output_folder` to choose another result directory.
- If using a local tokenizer, point `tokenizer.name` to that path.
- `NeoBERTForMTEB.encode()` honors `num_workers` and `pin_memory` overrides; on CUDA it keeps loader-side pinned staging enabled for overlapped host-to-device copies.

### Aggregate MTEB results

Point the aggregator at the directory containing checkpoint result folders. It writes `<model-name>_avg_table.json` with separate `scores` and `coverage` objects for each model/checkpoint. The default is fail-closed: every concrete result must be present on its configured split, and all CQADupstack variants must exist before that logical task contributes. Use `--allow-partial` only for exploratory subsets; missing categories are `null`, coverage is marked incomplete, and missing task/split pairs are listed instead of silently shrinking the denominator.

```bash
python scripts/evaluation/avg_mteb.py \
  --result_folder outputs/<pretrain-run>/mteb \
  --model_name <model-name>
```

## Pseudo-Perplexity Utility

`scripts/evaluation/pseudo_perplexity.py` can load NeoBERT checkpoints from the current portable step layout (`checkpoints/<step>/model.safetensors`) and falls back to legacy DeepSpeed ZeRO conversion only when portable weights are absent. That legacy fallback requires the optional `neobert[legacy-checkpoints]` extra. When `checkpoint_path` points at a checkpoint root, `--checkpoint latest` first honors a legacy DeepSpeed `latest` file when present; otherwise it resolves to the newest loadable numbered step. If `checkpoint_path` already points at a specific step directory, pass the matching `--checkpoint` tag; explicit missing non-`latest` tags fail fast instead of silently loading the direct path.

Select exactly one model source: use `--hub_model <model-id>` for a Hub masked LM, or use `--config_path <training-config> --checkpoint_path <run-or-checkpoint-path>` for a NeoBERT checkpoint. Hub models retain their learned position embeddings and reject `--max_length` values beyond the model's configured position limit.

The default dataset is `wikipedia` (`20220301.en`, `train`). Override it with `--dataset_name`, `--dataset_config`, and `--dataset_split`, or pass `--data_path` for a dataset saved with `Dataset.save_to_disk()`. The utility applies one inclusive `--min_chars`/`--max_chars` filter, then deterministic shuffling and optional `--num_dataset_shards`/`--dataset_shard_index` sharding.

Results default to `results/pseudo_perplexity/<model>/<checkpoint>/...csv`; `--output_path` changes the root. Reruns read existing sample IDs from the CSV and skip completed examples. Use `--device`, `--bf16`, and `--compile` to control execution.

```bash
python scripts/evaluation/pseudo_perplexity.py \
  --config_path outputs/my-run/checkpoints/<step>/config.yaml \
  --checkpoint_path outputs/my-run \
  --checkpoint <step> \
  --max_length 512

python scripts/evaluation/pseudo_perplexity.py \
  --hub_model <hub-model-id> \
  --max_length 512
```

## Common Evaluation Pitfalls

- Wrong checkpoint path: verify `glue.pretrained_checkpoint_dir`, `glue.pretrained_checkpoint`, and `glue.pretrained_model_path`.
- Flat or random GLUE metrics: confirm pretrained weights were loaded or intentionally set `glue.allow_random_weights: true`.
- Evaluation OOM: reduce evaluation batch size or sequence length.
- Attention backend confusion: GLUE uses SDPA; packed flash varlen is a training optimization.

## Related Docs

- [Configuration](../reference/configuration.md)
- [Training](training.md)
- [Training optimization](training-optimization.md)
- [Troubleshooting](troubleshooting.md)
