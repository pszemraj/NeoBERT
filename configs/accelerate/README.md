# Accelerate Launcher Configs

This directory contains launch-time `accelerate` configs for distributed runs.

## Files

- `fsdp2_2x5090.yaml`: 2-GPU local-machine FSDP2 config tuned for NeoBERT.
  - `mixed_precision: bf16`
  - `num_processes: 2`
  - FSDP2 transformer auto-wrap targets `EncoderBlock,NormEncoderBlock`
  - Sharded state dict enabled (`SHARDED_STATE_DICT`)

## Usage

```bash
accelerate launch --config_file configs/accelerate/fsdp2_2x5090.yaml \
  scripts/pretraining/pretrain.py \
  configs/pretraining/pretrain_neobert100m_smollm2data_dion2.yaml
```

For job scripts:

```bash
FSDP2_CONFIG=configs/accelerate/fsdp2_2x5090.yaml \
RUN_FSDP2=1 \
./jobs/experiment_dion2_robustness.sh
```
