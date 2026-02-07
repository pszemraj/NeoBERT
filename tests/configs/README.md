# Test Config Files

Small, fast configs used by unit/integration tests and smoke runs.

## Layout

- `pretraining/`
  - `test_tiny_pretrain.yaml`
  - `test_tiny_pretrain_tokenized.yaml`
  - `test_gpu_pretrain.yaml`
  - `test_gpu_small.yaml`
  - `test_streaming.yaml`
  - `test_streaming_pretrain.yaml`
  - `test_streaming_gpu.yaml`
  - `test_pretokenized.yaml`
  - `test_simple_wiki.yaml`
  - `test_smollm2_streaming.yaml`
  - `test_smollm2_200steps.yaml`
- `evaluation/`
  - `test_tiny_glue.yaml`
  - `test_tiny_glue_random.yaml`
- `contrastive/`
  - `test_tiny_contrastive.yaml`

## Example Usage

```bash
python scripts/pretraining/pretrain.py tests/configs/pretraining/test_tiny_pretrain.yaml
python scripts/evaluation/run_glue.py tests/configs/evaluation/test_tiny_glue.yaml
python scripts/contrastive/finetune.py tests/configs/contrastive/test_tiny_contrastive.yaml
```

## Notes

- These are intentionally tiny and not representative of production throughput.
- For production experiments, use configs under `configs/`.
