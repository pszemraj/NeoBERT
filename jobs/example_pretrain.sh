#!/bin/bash
# Example pretraining script using new configuration system

# Basic pretraining with config file
python scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml

# Pretraining with command-line overrides
python scripts/pretraining/pretrain.py \
    --config configs/pretraining/pretrain_neobert.yaml \
    --trainer.per_device_train_batch_size 32 \
    --trainer.max_steps 500000 \
    --optimizer.lr 2e-4 \
    --wandb.project my-neobert-project \
    --wandb.name neobert-test-run

# Small test run for CPU
python scripts/pretraining/pretrain.py \
    --config tests/configs/pretraining/test_tiny_pretrain.yaml \
    --trainer.per_device_train_batch_size 2 \
    --trainer.max_steps 10 \
    --trainer.save_steps 5 \
    --trainer.eval_steps 5 \
    --trainer.logging_steps 1 \
    --wandb.mode disabled \
    --debug
