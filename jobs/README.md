# Job Scripts

Shell launcher examples for common workflows.

These wrap [scripts](../scripts/README.md) and [configs](../configs/README.md).

## Files

- `example_pretrain.sh` - pretraining launcher; set `RUN_FULL=1` for the long run
- `example_evaluate.sh` - evaluation launcher

## Example Launches

```bash
./jobs/example_pretrain.sh
RUN_FULL=1 ./jobs/example_pretrain.sh
./jobs/example_evaluate.sh
```

## Notes

- Checkpoints and logs are written under each run's `trainer.output_dir`.
- For long runs on clusters, copy these scripts and adapt resource flags, paths, and environment setup.

## Related Docs

- [Scripts](../scripts/README.md)
- [Training](../docs/guides/training.md)
- [Evaluation](../docs/guides/evaluation.md)
