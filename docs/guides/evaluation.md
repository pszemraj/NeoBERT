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
- `glue.max_seq_length` controls task tokenization. Checkpoint tokenizer defaults do not override it; learned-position checkpoints reject lengths beyond their trained position table, while RoPE checkpoints can extend context without resizing weights.
- A fresh run requires pretrained local weights unless either `glue.allow_random_weights: true` or `model.from_hub: true`. `trainer.resume_from_checkpoint` instead uses the selected GLUE step's bundled config, tokenizer, model state, loop cursor, and metric-selection state; the original pretraining checkpoint may be moved or pruned.
- When checkpoint saving is enabled, GLUE writes checkpoints to `trainer.output_dir/checkpoints/<step>/`.
- GLUE resume preserves existing result files, restores the checkpoint's seed and batch/accumulation geometry, and rejects a different distributed world size.
- Hub classifiers that require `tokenizer.trust_remote_code: true` may run only with checkpoint saving disabled; their external modeling code is not portable enough for the self-contained GLUE resume contract.
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
- Output defaults to the MTEB result cache rooted at `outputs/<run>/mteb/<ckpt>/<max_length>/`; task files live below its `results/` directory. Pass `--output_folder` to choose another cache root.
- Local checkpoint evaluation requires `config.yaml` and `tokenizer/` inside the resolved step. Model architecture and token-to-ID identity come from those artifacts; launch configuration controls only evaluation settings such as task selection, pooling, batch size, and requested context length.
- MTEB owns evaluation batching: `mteb_batch_size` sizes the `DataLoader` passed to `NeoBERTForMTEB.encode()`, which tokenizes each supplied text batch and uses pinned token staging for non-blocking CUDA transfers.

### Aggregate MTEB results

Point the aggregator at the directory containing checkpoint result folders. It writes `<model-name>_avg_table.json` with separate `scores` and `coverage` objects for each model/checkpoint. The default is fail-closed: every concrete result must be present on its configured split, and all CQADupstack variants must exist before that logical task contributes. Use `--allow-partial` only for exploratory subsets; missing categories are `null`, coverage is marked incomplete, and missing task/split pairs are listed instead of silently shrinking the denominator.

```bash
python scripts/evaluation/avg_mteb.py \
  --result_folder outputs/<pretrain-run>/mteb \
  --model_name <model-name>
```

## Pseudo-Perplexity Utility

`scripts/evaluation/pseudo_perplexity.py` can load NeoBERT checkpoints from the current portable step layout (`checkpoints/<step>/model.safetensors`) and falls back to legacy DeepSpeed ZeRO conversion only when portable weights are absent. That legacy fallback requires the optional `neobert[legacy-checkpoints]` extra. `--checkpoint_path` accepts a run root, its `checkpoints/` directory, or a concrete step. `--checkpoint latest` resolves once to a concrete tag; the concrete tag, rather than the reusable word `latest`, becomes the output directory identity.

Select exactly one model source: use `--hub_model <model-id>` for a Hub masked LM, or `--checkpoint_path <run-or-checkpoint-path>` for a NeoBERT checkpoint. Local evaluation requires checkpoint-local `config.yaml` and `tokenizer/`, validates vocabulary and pad-token identity, and does not accept a separate launch config that could silently disagree. Non-RoPE models are constructed with their trained positional-table size and reject evaluation beyond it; shorter evaluation contexts do not resize learned embeddings. Hub sources accept `--revision` and retain their learned position embeddings.

The default dataset is `wikipedia` (`20220301.en`, `train`). Override it with `--dataset_name`, `--dataset_config`, and `--dataset_split`, or pass `--data_path` for a dataset saved with `Dataset.save_to_disk()`. The utility applies one inclusive `--min_chars`/`--max_chars` filter, then deterministic shuffling and optional `--num_dataset_shards`/`--dataset_shard_index` sharding.

Results default to `results/pseudo_perplexity/<model>/<checkpoint>/...csv`; `--output_path` changes the root. Each CSV has a manifest recording the concrete model/config/tokenizer source, dataset selection and sharding, length filters, seed, batch size, context length, and precision. Reruns skip completed sample IDs only when that manifest matches exactly; stale or mixed-checkpoint CSVs fail closed. Evaluation uses bf16 autocast by default; pass `--no-bf16` for fp32, and use `--device` and `--compile` to control placement and compilation.

```bash
python scripts/evaluation/pseudo_perplexity.py \
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

- [YAML configuration reference](../reference/config_reference.yaml)
- [Training](training.md)
- [Training optimization](training-optimization.md)
- [Troubleshooting](troubleshooting.md)
