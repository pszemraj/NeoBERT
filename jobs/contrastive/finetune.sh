#!/bin/bash
#SBATCH --job-name=neobert-base-dgxh100-1024
#SBATCH --output=logs/txt/%x_finetuning_output.txt
#SBATCH --error=logs/txt/%x_finetuning_error.txt
#SBATCH --time=03:00:00
#SBATCH --partition=short-unkillable    # ask for high-priority job
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gres=gpu:4	                # number and type of gpus
#SBATCH --constraint="[ampere|hopper]&80gb"  # gpu type - either a100 or h100
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem=128G                      # memory per node
# #SBATCH --gpus-per-task=1             # number of gpus per task

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Maximum number of threads in the OpenMP parallel region (defaults to 1)
# (called by `torch.distributed.run`, called by `accelerate launch`)
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

ckpt=1000000

# Define the command to run on each node
cmd=(
    accelerate launch \
    --config_file=$HOME/neo-bert/conf/accelerate_deepspeed_zero3.yaml \
    --machine_rank=\$SLURM_NODEID \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE)) \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE)) \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --gradient_clipping=1.0 \
    $HOME/neo-bert/scripts/contrastive/finetune.py \
    wandb.name=$SLURM_JOB_NAME"_ft" \
    wandb.dir=$SCRATCH/logs/$SLURM_JOB_NAME/finetuning/$ckpt \
    trainer.dir=$SCRATCH/logs/$SLURM_JOB_NAME/finetuning/$ckpt \
    hydra.run.dir=$SCRATCH/logs/$SLURM_JOB_NAME/finetuning/$ckpt/hydra \
    model.ckpt_dir=$SCRATCH/logs/$SLURM_JOB_NAME/model_checkpoints \
    model.ckpt=$ckpt \
    dataloader.target_bsz=32 \
    datasets.alpha=0.1 \
)

# Load modules and python environment
module load python/3.10 cuda/12.3.2

cd $HOME/neo-bert
source .venv/bin/activate
pip install -e .

# Load datasets
srun --kill-on-bad-exit=1 bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"

