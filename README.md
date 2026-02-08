# NeoBERT

> [!IMPORTANT]
> This repository is a fork of [chandar-lab/NeoBERT](https://github.com/chandar-lab/NeoBERT) focused on active experimentation and training-system iteration.

## Description

NeoBERT is an encoder architecture for masked-language-model pretraining,
embedding extraction, and downstream evaluation (GLUE/MTEB).

This fork adds:

- configurable attention backends (`sdpa`, `flash_attn_varlen` for packed training),
- optional Liger kernel dispatch (`kernel_backend: auto|liger|torch`),
- safetensors-first checkpointing,
- end-to-end training/eval/export scripts with config-driven workflows.

Pretraining loss path is selected with one explicit flag:
`trainer.masked_logits_only_loss` (`true` = masked-logits-only path,
`false` = original full-logits CE path).

Paper (original): <https://arxiv.org/abs/2502.19587>

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
```

See [docs/troubleshooting.md](docs/troubleshooting.md) for environment issues.

## Verify Setup

```bash
# Tiny pretraining smoke test
python scripts/pretraining/pretrain.py \
  tests/configs/pretraining/test_tiny_pretrain.yaml

# Full test suite
python tests/run_tests.py
```

## Quick Commands

| Task      | Command                                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------------------------- |
| Pretrain  | `python scripts/pretraining/pretrain.py configs/pretraining/pretrain_neobert.yaml`                                   |
| GLUE eval | `python scripts/evaluation/run_glue.py configs/glue/cola.yaml`                                                       |
| MTEB eval | `python scripts/evaluation/run_mteb.py configs/pretraining/pretrain_neobert.yaml --model_name_or_path outputs/<run>` |
| Export HF | `python scripts/export-hf/export.py outputs/<run>/model_checkpoints/<step>`                                          |
| Tests     | `python tests/run_tests.py`                                                                                          |

## Documentation

- [docs/README.md](docs/README.md)
- [Training Guide](docs/training.md)
- [Configuration Reference](docs/configuration.md)
- [Evaluation Guide](docs/evaluation.md)
- [Export Guide](docs/export.md)
- [Architecture](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Testing](docs/testing.md)
- [Dev Notes](docs/dev.md)

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
