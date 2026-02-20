# Job Scripts

`jobs/` contains shell launcher examples for common workflows.

```text
jobs/
  example_pretrain.sh
  example_evaluate.sh
  experiment_dion2_robustness.sh
```

These are convenience wrappers around scripts in `scripts/` and configs in
`configs/`.
Behavior semantics remain documented in `docs/training.md` and
`docs/evaluation.md`.

## Environment

Examples assume Python environment has NeoBERT installed.

```bash
python -c "import torch, neobert; print(torch.__version__)"
```

## Example Launches

```bash
# Tiny pretraining smoke test
./jobs/example_pretrain.sh

# Full variant (if script supports RUN_FULL gate)
RUN_FULL=1 ./jobs/example_pretrain.sh

# Evaluation example
./jobs/example_evaluate.sh

# Dion2 robustness experiment runner (W&B-enabled)
./jobs/experiment_dion2_robustness.sh

# 2-GPU W&B experiment with explicit FSDP2 enablement
WANDB_PROJECT=neobert-dion2-robustness \
FSDP2_CONFIG=<accelerate_fsdp2_config.yaml> \
RUN_FSDP2=1 \
./jobs/experiment_dion2_robustness.sh
```

## Notes

- Checkpoints and logs are written under each run's `trainer.output_dir`.
- For long runs on clusters, copy these scripts and adapt resource flags,
  paths, and environment setup.

## Related Docs

- [docs/README.md](../docs/README.md)
- [scripts/README.md](../scripts/README.md)
- [training.md](../docs/training.md)
- [evaluation.md](../docs/evaluation.md)
