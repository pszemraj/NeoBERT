#!/bin/bash
# Example evaluation scripts using new configuration system

# GLUE evaluation - single task example
python scripts/evaluation/run_glue.py \
    --config configs/evaluate_neobert.yaml \
    --task glue \
    --dataset.name cola \
    --trainer.output_dir ./output/glue/cola

# GLUE evaluation - all tasks
# bash scripts/evaluation/glue/run_all_glue.sh

# MTEB evaluation - all tasks
python scripts/evaluation/run_mteb_new.py \
    --config configs/evaluate_neobert.yaml \
    --task mteb \
    --mteb_task_type all \
    --trainer.output_dir ./output/pretrain

# MTEB evaluation - only STS tasks
python scripts/evaluation/run_mteb_new.py \
    --config configs/evaluate_neobert.yaml \
    --task mteb \
    --mteb_task_type sts \
    --mteb_batch_size 64 \
    --pretrained_checkpoint 100000 \
    --trainer.output_dir ./output/pretrain
