#!/bin/bash
#SBATCH --job-name=bert-ablations-00
#SBATCH --output=logs/txt/%x_output.txt
#SBATCH --error=logs/txt/%x_error.txt
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gpus-per-task=2               # number of gpus per node
#SBATCH --cpus-per-task=32              # number of cpus per node
#SBATCH --mem-per-gpu=64G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end
#SBATCH --switches=1@01:00:00           # number of leaf switches (InfiniBand Island) with time limit for the constraint

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Maximum number of threads in the OpenMP parallel region (defaults to 1)
# (called by `torch.distributed.run`, called by `accelerate launch`)
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))


export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# Define the command to run on each node
cmd=(
    accelerate launch \
    --config_file=$HOME/neo-bert/conf/accelerate_deepspeed_zero2.yaml \
    --machine_rank=\$SLURM_NODEID \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE)) \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_processes=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_ON_NODE)) \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --gradient_clipping=1.0 \
    $HOME/neo-bert/scripts/pretraining/pretrain.py \
    wandb.name=$SLURM_JOB_NAME \
    wandb.mode=offline \
    wandb.dir=$SCRATCH/logs/$SLURM_JOB_NAME/wandb \
    trainer.dir=$SCRATCH/logs/$SLURM_JOB_NAME \
    hydra.run.dir=$SCRATCH/logs/$SLURM_JOB_NAME/hydra \
    dataset=wikibook \
    tokenizer=google \
    tokenizer.max_length=512 \
    model=[bert,120M] \
    datacollator=mlm_15 \
    optimizer=adam \
    scheduler=linear_decay \
    trainer.gradient_accumulation_steps=1 \
    dataloader.train.batch_size=128 \
    dataset.path_to_disk=tokenized_datasets/wikibook_google_512 \
)

# Load modules and python environment
source activate base
conda activate .venv

# Launch cmd on each node
srun --kill-on-bad-exit=1 bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"