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

- `test_muonclip_fsdp2_golden.py` requires CUDA and 2 ranks; it validates same-batch
  full-step parity plus same-world-size optimizer resume for the FSDP2 Muon path.
- That golden test validates the raw FSDP2 owner-compute update path directly,
  but it saves and restores sharded optimizer state through the supported DCP
  (`get_optimizer_state_dict` / `set_optimizer_state_dict`) flow.
- Raw local-shard `optimizer.state_dict()` round-trips are intentionally not
  treated as a supported FSDP2 Muon resume surface; production checkpointing
  should go through Accelerate `save_state` / `load_state` or the underlying
  DCP helpers.
- `test_muonclip_accelerate_fsdp2_resume.py` covers that production checkpoint
  plumbing explicitly: prepared model/optimizer/scheduler/dataloader, one saved
  step, fresh-object restore through `Accelerator.load_state`, then continuation
  parity on the next step. The smoke uses a synthetic pre-batched dataloader,
  so it opts into Accelerate `even_batches=False` for compatibility with recent
  Accelerate releases.
- `test_muonclip_training.py` can run for multiple minutes and may download datasets.
- These scripts are for exploratory/perf validation, not fast CI regression checks.
- `tests/manual/` is excluded from default `pytest -q` discovery.
