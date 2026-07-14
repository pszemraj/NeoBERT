# NeoBERT

> [!IMPORTANT]
> This repository builds on the original [chandar-lab/NeoBERT](https://github.com/chandar-lab/NeoBERT) codebase and is focused on active experimentation and training-system iteration.

## Description

NeoBERT is an encoder architecture for masked-language-model pretraining, embedding extraction, and downstream evaluation (GLUE/MTEB).

This repo adds:

- configurable attention backends (`sdpa`, `flash_attn_varlen` for packed training),
- optional Liger kernel dispatch (`model.kernel_backend: auto|liger|torch`),
- safetensors-first checkpointing,
- end-to-end training/eval/export scripts with config-driven workflows.

Pretraining supports masked-logits-only training and a full-logits ablation path; see [Training](docs/guides/training.md#mlm-loss-path-selection).

Original paper: <https://arxiv.org/abs/2502.19587>

## Install

NeoBERT requires Python 3.10 or newer and PyTorch 2.6 or newer.

```bash
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT
pip install -e .[dev]
```

Optional extras:

```bash
pip install -U -q packaging wheel ninja
# Packed flash-attn training backend
pip install -e .[flash] --no-build-isolation
# Legacy DeepSpeed ZeRO checkpoint conversion only
pip install -e .[legacy-checkpoints]
```

See [Troubleshooting](docs/guides/troubleshooting.md) for environment issues.

## Verify Setup

```bash
# Deterministic one-step local pretraining smoke
pytest -q tests/training/test_pretrain_pipeline.py::TestPretrainPipeline::test_pretraining_one_step_local_smoke
```

See [Testing](docs/guides/testing.md) for the full suite and targeted test commands.

## Documentation

Use the [documentation index](docs/README.md) for training, evaluation, export, configuration, architecture, testing, and troubleshooting.

Directory READMEs under `configs/`, `scripts/`, `tests/`, and `jobs/` focus on entrypoints and quick usage.

## Repository Layout

- `src/neobert/` - core model/trainer/config/runtime code
- `configs/` - example configs for pretraining/eval/contrastive
- `scripts/` - CLI entry points
- `jobs/` - shell launcher examples
- `tests/` - regression tests and tiny configs
- `docs/` - user and developer documentation

## Citation

```bibtex
@misc{breton2025neobertnextgenerationbert,
      title={NeoBERT: A Next-Generation BERT},
      author={Lola Le Breton and Quentin Fournier and Mariam El Mezouar and John X. Morris and Sarath Chandar},
      year={2025},
      eprint={2502.19587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19587},
}
```

## License

MIT License.
