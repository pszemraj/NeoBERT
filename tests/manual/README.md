# Manual Distributed Validation

Opt-in distributed validation scripts under `tests/manual/`.

## Commands

```bash
torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_fsdp2_golden.py
torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_accelerate_fsdp2_resume.py
```

## Notes

- Distributed launch policy, topology, and gradient/norm logging behavior are in [docs/guides/training.md](../../docs/guides/training.md#distributed-topology) and [docs/guides/training-optimization.md](../../docs/guides/training-optimization.md#gradient-accumulation-and-logged-norms).
- `test_muonclip_fsdp2_golden.py` requires CUDA and 2 ranks; it validates the raw FSDP2 owner-compute update path and sharded DCP optimizer-state round-trip.
- `test_muonclip_accelerate_fsdp2_resume.py` validates the shipped Accelerate resume path using the production step-checkpoint layout (`accelerate/` resume state beside portable `model.safetensors`). The smoke uses a synthetic pre-batched dataloader and opts into `even_batches=False` internally for recent Accelerate releases.
- Raw local-shard `optimizer.state_dict()` round-trips are not a supported FSDP2 Muon resume surface.
- Local MuonClip configuration, hooks, optimizer steps, stability, and orthogonalization are covered by `tests/test_muonclip_unit.py`.
- These scripts validate distributed execution paths that cannot be covered by fast CPU regression tests.
