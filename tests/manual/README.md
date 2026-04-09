# Manual Validation Scripts

Opt-in validation and benchmark scripts under `tests/manual/`.

## Commands

```bash
conda run -s --name neobert pytest -q tests/manual/test_muonclip_integration.py -s
conda run -s --name neobert python tests/manual/test_muonclip_training.py
conda run -s --name neobert python tests/manual/validate_muonclip.py
conda run -s --name neobert torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_fsdp2_golden.py
conda run -s --name neobert torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_accelerate_fsdp2_resume.py
```

## Notes

- Distributed launch policy, topology, and gradient/norm logging behavior are in
  [docs/guides/training.md](../../docs/guides/training.md#distributed-topology)
  and
  [docs/guides/training-optimization.md](../../docs/guides/training-optimization.md#gradient-accumulation-and-logged-norms).
- `test_muonclip_fsdp2_golden.py` requires CUDA and 2 ranks; it validates the
  raw FSDP2 owner-compute update path and sharded DCP optimizer-state round-trip.
- `test_muonclip_accelerate_fsdp2_resume.py` validates the shipped Accelerate
  `prepare(...) -> save_state(...) -> load_state(...)` resume path. The smoke
  uses a synthetic pre-batched dataloader and opts into `even_batches=False`
  internally for recent Accelerate releases.
- Raw local-shard `optimizer.state_dict()` round-trips are not a supported
  FSDP2 Muon resume surface.
- `test_muonclip_training.py` can run for multiple minutes and may download datasets.
- These scripts are for exploratory/perf validation, not fast CI regression checks.
- `tests/manual/` is excluded from default `pytest -q` discovery.
