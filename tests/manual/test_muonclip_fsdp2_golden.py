"""Manual golden test for MuonClip FSDP2 owner-compute correctness.

Run on a CUDA machine with 2 ranks:
`torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_fsdp2_golden.py`
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import MuonClipConfig, MuonClipOptimizer


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_get_rank())))


def _init_dist() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _get_local_rank()
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def _row_counts(dtensor: DTensor) -> list[int]:
    local_rows = torch.tensor(
        [int(dtensor.to_local().shape[0])],
        device=dtensor.to_local().device,
        dtype=torch.int64,
    )
    group = dtensor.device_mesh.get_group()
    world_size = dist.get_world_size(group)
    gathered = [torch.zeros_like(local_rows) for _ in range(world_size)]
    dist.all_gather(gathered, local_rows, group=group)
    return [int(x.item()) for x in gathered]


def _set_dtensor_grad_from_full(param: DTensor, full_grad: torch.Tensor) -> None:
    local = param.to_local()
    group = param.device_mesh.get_group()
    rank = dist.get_rank(group)
    counts = _row_counts(param)
    start = sum(counts[:rank])
    end = start + int(local.shape[0])
    local_grad = full_grad[start:end].to(device=local.device, dtype=local.dtype)
    param.grad = DTensor.from_local(
        local_grad.contiguous(),
        device_mesh=param.device_mesh,
        placements=param.placements,
        run_check=False,
        shape=param.shape,
    )


def _set_dtensor_param_from_full(param: DTensor, full_value: torch.Tensor) -> None:
    local = param.to_local()
    group = param.device_mesh.get_group()
    rank = dist.get_rank(group)
    counts = _row_counts(param)
    start = sum(counts[:rank])
    end = start + int(local.shape[0])
    local_value = full_value[start:end].to(device=local.device, dtype=local.dtype)
    with torch.no_grad():
        local.copy_(local_value)


def _build_sharded_model(
    config: NeoBERTConfig,
    device: torch.device,
    init_state: dict[str, torch.Tensor],
) -> NeoBERT:
    model = NeoBERT(config).to(device)
    model.load_state_dict(init_state)
    for layer in model.transformer_encoder:
        fully_shard(layer)
    fully_shard(model)
    return model


def _target_param(model: NeoBERT) -> torch.nn.Parameter | DTensor:
    return model.transformer_encoder[0].qkv.weight


def main() -> None:
    rank = _get_rank()
    if not torch.cuda.is_available():
        if rank == 0:
            print("CUDA is required for this manual FSDP2 golden test.")
        return

    rank, world_size, device = _init_dist()
    if world_size != 2:
        raise RuntimeError(
            f"This manual test expects world_size=2, got world_size={world_size}."
        )

    ckpt_root: Path | None = None
    try:
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        config = NeoBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=257,
            max_length=64,
            attn_backend="sdpa",
            hidden_act="gelu",
            rope=False,
        )
        init_model = NeoBERT(config).to(device)
        init_state = {
            key: value.detach().clone()
            for key, value in init_model.state_dict().items()
        }
        del init_model

        muon_cfg = MuonClipConfig(
            lr=1e-3,
            muon_beta=0.95,
            muon_decay=0.0,
            adam_decay=0.0,
            enable_clipping=False,
            orthogonalization="polar_express",
            norm_factor="spectral",
        )

        # Rank-0 local baseline (full tensor update).
        if rank == 0:
            baseline_model = NeoBERT(config).to(device)
            baseline_model.load_state_dict(init_state)
            baseline_optimizer = MuonClipOptimizer(baseline_model, config, muon_cfg)
            baseline_target = _target_param(baseline_model)
        else:
            baseline_model = None
            baseline_optimizer = None
            baseline_target = None

        # Distributed FSDP2 model and optimizer.
        dist_model = _build_sharded_model(config, device, init_state)
        dist_optimizer = MuonClipOptimizer(dist_model, config, muon_cfg)
        dist_target = _target_param(dist_model)
        if not isinstance(dist_target, DTensor):
            raise RuntimeError("Expected DTensor parameter under FSDP2 sharding.")

        # Step 1: baseline-vs-distributed equivalence.
        full_grad_1 = torch.empty(tuple(dist_target.shape), device=device)
        if rank == 0:
            full_grad_1.normal_()
        dist.broadcast(full_grad_1, src=0)

        dist_optimizer.zero_grad(set_to_none=True)
        _set_dtensor_grad_from_full(dist_target, full_grad_1)
        dist_optimizer.step()
        dist_step1_full = dist_target.full_tensor().detach().clone()

        baseline_step1_full = torch.empty_like(dist_step1_full)
        if rank == 0:
            assert baseline_optimizer is not None
            assert baseline_target is not None
            baseline_optimizer.zero_grad(set_to_none=True)
            baseline_target.grad = full_grad_1.clone()
            baseline_optimizer.step()
            baseline_step1_full.copy_(baseline_target.detach())
        dist.broadcast(baseline_step1_full, src=0)

        step1_diff = (dist_step1_full - baseline_step1_full).abs().max()
        step1_diff_global = step1_diff.clone()
        dist.all_reduce(step1_diff_global, op=dist.ReduceOp.MAX)
        if float(step1_diff_global.item()) > 5e-5:
            raise AssertionError(
                "FSDP2 Muon step mismatch vs local baseline: "
                f"max_diff={float(step1_diff_global.item()):.6e}"
            )

        # Checkpoint/save-load path for same-world-size resume.
        root_obj: list[str | None] = [None]
        if rank == 0:
            ckpt_root = Path(tempfile.mkdtemp(prefix="muonclip_fsdp2_"))
            root_obj[0] = str(ckpt_root)
        dist.broadcast_object_list(root_obj, src=0)
        ckpt_root = Path(root_obj[0]) if root_obj[0] is not None else None
        if ckpt_root is None:
            raise RuntimeError("Failed to resolve checkpoint directory path.")

        ckpt_path = ckpt_root / "optimizer_state.pt"
        if rank == 0:
            torch.save(dist_optimizer.state_dict(), ckpt_path)
        dist.barrier()
        loaded_optim_state = torch.load(ckpt_path, map_location=device)

        # Continuation path A: no reload.
        full_grad_2 = torch.empty(tuple(dist_target.shape), device=device)
        if rank == 0:
            full_grad_2.normal_()
        dist.broadcast(full_grad_2, src=0)

        dist_optimizer.zero_grad(set_to_none=True)
        _set_dtensor_grad_from_full(dist_target, full_grad_2)
        dist_optimizer.step()
        final_no_reload = dist_target.full_tensor().detach().clone()

        # Continuation path B: reload model+optimizer at step-1 state.
        resume_model = _build_sharded_model(config, device, init_state)
        resume_target = _target_param(resume_model)
        if not isinstance(resume_target, DTensor):
            raise RuntimeError("Expected DTensor parameter under FSDP2 sharding.")
        _set_dtensor_param_from_full(resume_target, dist_step1_full)

        resume_optimizer = MuonClipOptimizer(resume_model, config, muon_cfg)
        resume_optimizer.load_state_dict(loaded_optim_state)
        resume_optimizer.zero_grad(set_to_none=True)
        _set_dtensor_grad_from_full(resume_target, full_grad_2)
        resume_optimizer.step()
        final_reloaded = resume_target.full_tensor().detach().clone()

        resume_diff = (final_no_reload - final_reloaded).abs().max()
        resume_diff_global = resume_diff.clone()
        dist.all_reduce(resume_diff_global, op=dist.ReduceOp.MAX)
        if float(resume_diff_global.item()) > 5e-5:
            raise AssertionError(
                "FSDP2 Muon resume mismatch after optimizer load_state_dict: "
                f"max_diff={float(resume_diff_global.item()):.6e}"
            )

        if rank == 0:
            print("PASS: FSDP2 Muon golden step and resume checks succeeded.")
    finally:
        dist.barrier()
        if rank == 0 and ckpt_root is not None and ckpt_root.exists():
            shutil.rmtree(ckpt_root, ignore_errors=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
