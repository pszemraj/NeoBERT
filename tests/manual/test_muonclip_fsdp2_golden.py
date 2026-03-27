"""Manual golden test for MuonClip FSDP2 owner-compute correctness.

This script uses a real forward/backward/step with the same batch on:

- a rank-0 local baseline model, and
- a 2-rank FSDP2-sharded model.

It compares all trainable parameters after one optimizer step, then verifies
same-world-size optimizer resume equivalence on the next step using the
supported distributed-checkpoint optimizer-state APIs.

Run on a CUDA machine with 2 ranks:
`torchrun --standalone --nproc_per_node=2 tests/manual/test_muonclip_fsdp2_golden.py`
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import MuonClipConfig, MuonClipOptimizer

_GOLDEN_MAX_DIFF = 5e-5
# Polar Express intentionally runs orthogonalization in a fast CUDA work dtype
# (typically bf16), so allow small end-to-end drift in the manual parity check.


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_get_rank())))


def _init_dist() -> tuple[int, int, torch.device]:
    local_rank = _get_local_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, device


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


def _sample_input_ids(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
    if dist.get_rank() == 0:
        input_ids.random_(0, vocab_size)
    dist.broadcast(input_ids, src=0)
    return input_ids


def _run_step(
    model: NeoBERT,
    optimizer: MuonClipOptimizer,
    input_ids: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _full_parameter_state(model: NeoBERT) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            tensor = param.full_tensor().detach().cpu()
        else:
            tensor = param.detach().cpu()
        state[name] = tensor.clone()
    return state


def _load_parameter_state(
    model: NeoBERT,
    state: dict[str, torch.Tensor],
) -> None:
    for name, param in model.named_parameters():
        full_value = state[name]
        if isinstance(param, DTensor):
            _set_dtensor_param_from_full(param, full_value)
        else:
            with torch.no_grad():
                param.copy_(full_value.to(device=param.device, dtype=param.dtype))


def _max_state_diff(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, str]:
    max_diff = 0.0
    max_name = ""
    for name, ref_tensor in reference.items():
        candidate_tensor = candidate[name]
        diff = float((candidate_tensor - ref_tensor).abs().max().item())
        if diff > max_diff:
            max_diff = diff
            max_name = name
    return max_diff, max_name


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
            hidden_size=15,
            num_hidden_layers=2,
            num_attention_heads=3,
            intermediate_size=60,
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

        for norm_factor in ("legacy_compat", "spectral"):
            muon_cfg = MuonClipConfig(
                lr=1e-3,
                muon_beta=0.95,
                muon_decay=0.0,
                adam_decay=0.0,
                enable_clipping=False,
                orthogonalization="polar_express",
                norm_factor=norm_factor,
                param_policy="transformer_only",
            )

            # Rank-0 local baseline (full tensor update).
            if rank == 0:
                baseline_model = NeoBERT(config).to(device)
                baseline_model.load_state_dict(init_state)
                baseline_optimizer = MuonClipOptimizer(baseline_model, config, muon_cfg)
            else:
                baseline_model = None
                baseline_optimizer = None

            # Distributed FSDP2 model and optimizer.
            dist_model = _build_sharded_model(config, device, init_state)
            dist_optimizer = MuonClipOptimizer(dist_model, config, muon_cfg)
            dist_target = _target_param(dist_model)
            if not isinstance(dist_target, DTensor):
                raise RuntimeError("Expected DTensor parameter under FSDP2 sharding.")
            if int(dist_target.shape[0]) % world_size == 0:
                raise RuntimeError(
                    "Golden test expected uneven row sharding for qkv.weight, but "
                    f"shape={tuple(dist_target.shape)} divides evenly across world_size={world_size}."
                )

            # Step 1: same-batch local baseline vs distributed FSDP2 equivalence.
            input_ids_1 = _sample_input_ids(
                batch_size=3,
                seq_len=11,
                vocab_size=config.vocab_size,
                device=device,
            )
            dist_loss_1 = _run_step(dist_model, dist_optimizer, input_ids_1)
            dist_step1_state = _full_parameter_state(dist_model)

            baseline_step1_state: dict[str, torch.Tensor] | None = None
            baseline_loss_1 = None
            if rank == 0:
                assert baseline_optimizer is not None
                assert baseline_model is not None
                baseline_loss_1 = _run_step(
                    baseline_model, baseline_optimizer, input_ids_1
                )
                baseline_step1_state = _full_parameter_state(baseline_model)

            if rank == 0:
                assert baseline_step1_state is not None
                assert baseline_loss_1 is not None
                if abs(dist_loss_1 - baseline_loss_1) > 1e-6:
                    raise AssertionError(
                        "FSDP2 Muon forward loss mismatch vs local baseline: "
                        f"norm_factor={norm_factor} "
                        f"dist_loss={dist_loss_1:.6e} "
                        f"baseline_loss={baseline_loss_1:.6e}"
                    )

                step1_max_diff, step1_name = _max_state_diff(
                    baseline_step1_state, dist_step1_state
                )
                if step1_max_diff > _GOLDEN_MAX_DIFF:
                    raise AssertionError(
                        "FSDP2 Muon step mismatch vs local baseline: "
                        f"norm_factor={norm_factor} "
                        f"param={step1_name} max_diff={step1_max_diff:.6e}"
                    )
            dist.barrier()

            root_obj: list[str | None] = [None]
            if rank == 0:
                ckpt_root = Path(
                    tempfile.mkdtemp(prefix=f"muonclip_fsdp2_{norm_factor}_")
                )
                root_obj[0] = str(ckpt_root)
            dist.broadcast_object_list(root_obj, src=0)
            ckpt_root = Path(root_obj[0]) if root_obj[0] is not None else None
            if ckpt_root is None:
                raise RuntimeError("Failed to resolve checkpoint directory path.")

            # Save/load the sharded optimizer state through DCP so DTensor topology
            # is preserved across resume. Raw ``optimizer.state_dict()`` tensors are
            # intentionally unsupported for FSDP2 Muon restores.
            ckpt_path = ckpt_root / "optimizer_dcp"
            dist_cp.save(
                state_dict={
                    "optimizer": get_optimizer_state_dict(dist_model, dist_optimizer)
                },
                storage_writer=dist_cp.FileSystemWriter(str(ckpt_path)),
            )
            dist.barrier()

            input_ids_2 = _sample_input_ids(
                batch_size=2,
                seq_len=9,
                vocab_size=config.vocab_size,
                device=device,
            )
            _ = _run_step(dist_model, dist_optimizer, input_ids_2)
            final_no_reload = _full_parameter_state(dist_model)

            resume_model = _build_sharded_model(config, device, init_state)
            resume_optimizer = MuonClipOptimizer(resume_model, config, muon_cfg)
            _load_parameter_state(resume_model, dist_step1_state)
            loaded_optim_state = {
                # DCP save/load must use the same FQN-keyed optimizer schema on both
                # sides of the checkpoint round-trip.
                "optimizer": get_optimizer_state_dict(resume_model, resume_optimizer)
            }
            dist_cp.load(
                loaded_optim_state,
                checkpoint_id=str(ckpt_path),
                storage_reader=dist_cp.FileSystemReader(str(ckpt_path)),
            )
            set_optimizer_state_dict(
                resume_model,
                resume_optimizer,
                loaded_optim_state["optimizer"],
            )
            _ = _run_step(resume_model, resume_optimizer, input_ids_2)
            final_reloaded = _full_parameter_state(resume_model)

            if rank == 0:
                resume_max_diff, resume_name = _max_state_diff(
                    final_no_reload, final_reloaded
                )
                if resume_max_diff > _GOLDEN_MAX_DIFF:
                    raise AssertionError(
                        "FSDP2 Muon resume mismatch after DCP optimizer restore: "
                        f"norm_factor={norm_factor} "
                        f"param={resume_name} max_diff={resume_max_diff:.6e}"
                    )
                shutil.rmtree(ckpt_root, ignore_errors=True)
                ckpt_root = None
            dist.barrier()

        if rank == 0:
            print("PASS: FSDP2 Muon golden step and resume checks succeeded.")
    finally:
        if dist.is_initialized():
            dist.barrier()
        if rank == 0 and ckpt_root is not None and ckpt_root.exists():
            shutil.rmtree(ckpt_root, ignore_errors=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
