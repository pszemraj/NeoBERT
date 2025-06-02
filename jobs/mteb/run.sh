#!/bin/bash
#SBATCH --job-name=mteb-neobert
#SBATCH --output=logs/%x_%j_output.txt
#SBATCH --error=logs/%x_%j_error.txt
#SBATCH --time=2-00:00:00
#SBATCH --partition=dgxh100long
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gpus-per-task=1               # number of gpus per node
#SBATCH --cpus-per-task=6                # number of cpus per gpu
#SBATCH --mem=32G                       # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end
#SBATCH --switches=1@01:00:00           # number of leaf switches (InfiniBand Island) with time limit for the constraint


###################
model=neobert-opt-dgxh100-refinedweb
ckpt=1025000
overwrite_results=false
instructions_query=$HOME/neo-bert/conf/mteb_task_to_instruction_query.json
instructions_corpus=$HOME/neo-bert/conf/mteb_task_to_instruction_corpus.json
max_len=1024
tasks_to_run=all
# run="subset" or "all" for
  # subset: a subset of 15 tasks
  # all: the full MTEB-en benchmark
  # TASK_NAME: for a specific task
###################

cd $HOME/neo-bert/
conda activate .venv

srun --kill-on-bad-exit=1 python /home/mila/l/lola.lebreton/neo-bert/scripts/evaluation/run_mteb.py \
    mteb.batch_size=64 \
    mteb.output_folder=$SCRATCH/logs/$model/mteb/$ckpt/$max_len \
    model.pretrained_checkpoint_dir=$SCRATCH/logs/$model \
    model.pretrained_checkpoint=$ckpt \
    mteb.overwrite_results=$overwrite_results \
    mteb.tasks=$tasks_to_run \
    mteb.max_length=$max_len \
    mteb.tasks_to_instructions_query=$instructions_query \
    mteb.tasks_to_instructions_corpus=$instructions_corpus \
