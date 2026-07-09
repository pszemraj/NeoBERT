# NeoBERT

> [!IMPORTANT]
> This repository builds on the original [chandar-lab/NeoBERT](https://github.com/chandar-lab/NeoBERT) codebase and is focused on active experimentation and training-system iteration.

## Description

NeoBERT is an encoder architecture for masked-language-model pretraining, embedding extraction, and downstream evaluation (GLUE/MTEB).

This repo adds:

- configurable attention backends (`sdpa`, `flash_attn_varlen` for packed training),
- optional Liger kernel dispatch (`kernel_backend: auto|liger|torch`),
- safetensors-first checkpointing,
- end-to-end training/eval/export scripts with config-driven workflows.

Pretraining supports masked-logits-only training and a full-logits ablation path; see [Training](docs/guides/training.md#mlm-loss-path-selection).

Original paper: <https://arxiv.org/abs/2502.19587>

## Install

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

See [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md) for environment issues.

## Verify Setup

```bash
# Tiny pretraining smoke test
python scripts/pretraining/pretrain.py \
  tests/configs/pretraining/test_tiny_pretrain.yaml

# Full test suite
pytest -q
```

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
      author={Lola Le Breton and Quentin Fournier and Mariam El Mezouar and Sarath Chandar},
      year={2025},
      eprint={2502.19587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19587},
}
```

## License

MIT License.
