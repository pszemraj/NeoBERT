# Manual Validation Scripts

Opt-in validation and benchmark scripts under `tests/manual/`.

## Commands

```bash
conda run -s --name neobert pytest -q tests/manual/test_muonclip_integration.py -s
conda run -s --name neobert python tests/manual/test_muonclip_training.py
conda run -s --name neobert python tests/manual/validate_muonclip.py
conda run -s --name neobert torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_fsdp2_golden.py
```

## Notes

- `test_muonclip_fsdp2_golden.py` requires CUDA and 2 ranks; it validates same-batch
  full-step parity plus same-world-size optimizer resume for the FSDP2 Muon path.
- That golden test validates the raw FSDP2 owner-compute path directly; it does
  not replace a dedicated Accelerate `save_state` / `load_state` smoke test for
  the production checkpoint integration.
- `test_muonclip_training.py` can run for multiple minutes and may download datasets.
- These scripts are for exploratory/perf validation, not fast CI regression checks.
- `tests/manual/` is excluded from default `pytest -q` discovery.
