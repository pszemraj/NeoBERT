# NeoBERT

> [!IMPORTANT]
> This is a fork of the [original chandar-lab/NeoBERT](https://github.com/chandar-lab/NeoBERT), refactored to support experimentation. ⚠️ WIP/active development⚠️

---

- [NeoBERT](#neobert)
  - [Description](#description)
  - [Get started](#get-started)
    - [Install](#install)
    - [Verify your setup](#verify-your-setup)
    - [Quick commands](#quick-commands)
    - [Next steps](#next-steps)
  - [How to use](#how-to-use)
    - [For Text Embeddings](#for-text-embeddings)
    - [For Masked Language Modeling](#for-masked-language-modeling)
  - [Documentation](#documentation)
  - [Features](#features)
  - [License](#license)
  - [Citation](#citation)
  - [Training and Development](#training-and-development)
    - [Repository Structure](#repository-structure)

---

## Description

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions.

- Paper (_original_): [paper](https://arxiv.org/abs/2502.19587)
- Model (_original_): [huggingface](https://huggingface.co/chandar-lab/NeoBERT)
- Documentation (_this repo_): [docs/](/docs/README.md)

## Get started

### Install

```bash
git clone https://github.com/pszemraj/NeoBERT.git
cd NeoBERT
# activate virtual environment (if not already active)
pip install -e .[dev]  # drop [dev] if you only need runtime deps
```

See [docs/troubleshooting.md](/docs/troubleshooting.md) for help with common installation issues.

<!-- > [!TIP]
> For faster training on supported GPUs, add `flash-attn` (and optionally `xformers`) with `pip install flash-attn --no-build-isolation`. -->

### Verify your setup

```bash
# 5-minute smoke test (tiny model, CPU-friendly)
python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml

# Optional: run the full regression suite
python tests/run_tests.py
```

### Quick commands

| Task           | Command                                                                         |
| -------------- | ------------------------------------------------------------------------------- |
| Pretrain       | `python scripts/pretraining/pretrain.py --config configs/pretrain_neobert.yaml` |
| GLUE eval      | `python scripts/evaluation/run_glue.py --config configs/glue/{task}.yaml`       |
| Summarize GLUE | `python scripts/evaluation/glue/summarize_glue.py {results_path}`               |
| Run tests      | `python tests/run_tests.py`                                                     |

### Next steps

- Train or fine-tune: see [/docs/training.md](/docs/training.md)
- Evaluate on GLUE or MTEB: see [/docs/evaluation.md](/docs/evaluation.md)
- Tune configs and overrides: see [/docs/configuration.md](/docs/configuration.md)
- Export checkpoints to Hugging Face: see [/docs/export.md](/docs/export.md)
- Troubleshoot common issues: see [/docs/troubleshooting.md](/docs/troubleshooting.md)

## How to use

Load [the official model](https://huggingface.co/chandar-lab/NeoBERT) using Hugging Face Transformers and use it for text embeddings or fill-mask predictions[^1].

[^1]: Encoder models are usually meant to be [fine-tuned for specific tasks](https://github.com/huggingface/transformers/tree/81b4f9882c8a46c8274084503d297874bb372260/examples/pytorch) rather than used directly after pretraining. this is analogous to instruction tuning for decoder-only models: the base model is a good starting point, but fine-tuning is necessary for practical applications.

<details>
<summary><b>Click to Expand:</b> MLM & Embedding Examples</summary>

### For Text Embeddings

```python
from transformers import AutoModel, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Tokenize input text
text = "NeoBERT is the most efficient model of its kind!"
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
print(embedding.shape)
```

### For Masked Language Modeling

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

# Fill in masked tokens
text = "The quick brown [MASK] jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
print(tokenizer.decode(predicted_token_id))
```

</details>

## Documentation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pszemraj/NeoBERT)

For detailed guides and documentation, see the **[Documentation](/docs/README.md)**:

- [Training Guide](/docs/training.md) - Pretraining, contrastive learning, and monitoring runs
- [Evaluation Guide](/docs/evaluation.md) - GLUE, MTEB, and result analysis
- [Configuration System](/docs/configuration.md) - YAML hierarchy and CLI overrides
- [Export Guide](/docs/export.md) - Convert checkpoints to Hugging Face format
- [Architecture Details](/docs/architecture.md) - Model internals
- [Testing Guide](/docs/testing.md) - Regression suite and coverage
- [Troubleshooting](/docs/troubleshooting.md) - Common failure modes and fixes

## Features

| **Feature**             | **NeoBERT**    |
| ----------------------- | -------------- |
| `Depth-to-width`        | 28 × 768       |
| `Parameter count`       | 250M           |
| `Activation`            | SwiGLU         |
| `Positional embeddings` | RoPE           |
| `Normalization`         | Pre-RMSNorm    |
| `Data Source`           | RefinedWeb     |
| `Data Size`             | 2.8 TB         |
| `Tokenizer`             | google/bert    |
| `Context length`        | 4,096          |
| `MLM Masking Rate`      | 20%            |
| `Optimizer`             | AdamW          |
| `Scheduler`             | CosineDecay    |
| `Training Tokens`       | 2.1 T          |
| `Efficiency`            | FlashAttention |

## License

Model weights and code repository are licensed under the permissive MIT license.

## Citation

If you use this model in your research, please cite:

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

## Training and Development

This repository includes the complete training and evaluation codebase for NeoBERT, featuring:

### Repository Structure

- **`configs/`** - YAML configuration files for training, evaluation, and contrastive learning
- **`scripts/`** - CLI entry points for pretraining, evaluation, contrastive learning, and exporting
- **`jobs/`** - Example shell launchers for clusters or batch systems
- **`tests/`** - Automated regression suite and tiny configs
- **`src/neobert/`** - Core model, trainer, and utilities

Additional guidance lives in:

- [`docs/training.md`](/docs/training.md) for full training workflows
- [`docs/evaluation.md`](/docs/evaluation.md) for benchmark recipes
- [`docs/testing.md`](/docs/testing.md) for extending the test suite
- [`docs/export.md`](/docs/export.md) for Hugging Face conversion
- [`docs/troubleshooting.md`](/docs/troubleshooting.md) for debugging tips

---
