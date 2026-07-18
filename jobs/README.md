# Job Scripts

Shell launcher examples for common workflows.

These wrap [scripts](../scripts/README.md) and [configs](../configs/README.md).

## Files

- `example_pretrain.sh` - small-model network-backed pretraining example; set `RUN_FULL=1` to continue into the production examples
- `example_evaluate.sh` - sequential single-task, quick-suite, and full-suite examples; its shipped GLUE configs require the checkpoint paths they reference

## Example Launches

Activate the Python environment containing your NeoBERT installation using your preferred environment manager, then run the launchers from the repository root:

```bash
bash jobs/example_pretrain.sh
RUN_FULL=1 bash jobs/example_pretrain.sh
bash jobs/example_evaluate.sh
```

## Notes

- Training outputs and checkpoints use each config's `trainer.output_dir`; GLUE suite logs default to `logs/<config-directory-name>/`.
- `example_pretrain.sh` uses the selected config's full `trainer.max_steps`; use the root README's local pytest command for a bounded setup smoke.
- For long runs on clusters, copy these scripts and adapt resource flags, paths, and environment setup.

## Related Docs

- [Training](../docs/guides/training.md)
- [Evaluation](../docs/guides/evaluation.md)
