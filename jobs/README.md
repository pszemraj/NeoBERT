# Job Scripts

`jobs/` contains shell launcher examples for common workflows.

```text
jobs/
  example_pretrain.sh
  example_evaluate.sh
  pretrain_fp8_fsdp2_4gpu.sh
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

# FP8 + FSDP2 pretraining on 4 GPUs (Accelerate launch)
./jobs/pretrain_fp8_fsdp2_4gpu.sh

# Optional overrides
TRAIN_CONFIG=configs/pretraining/pretrain_neobert_fp8_4gpu.yaml \
ACCELERATE_CONFIG=configs/accelerate/fsdp2_fp8_4gpu.yaml \
WANDB_MODE=offline \
./jobs/pretrain_fp8_fsdp2_4gpu.sh --trainer.max_steps 2000
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
