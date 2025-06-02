#!/bin/bash
#SBATCH --job-name=neobert-long
#SBATCH --output=logs/txt/%x_%j_output.txt
#SBATCH --error=logs/txt/%x_%j_error.txt
#SBATCH --time=30-00:00:00
#SBATCH --partition=dgxh100long    # ask for high-priority job
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --gpus-per-task=8               # number of gpus per node
#SBATCH --cpus-per-task=64              # number of cpus per node
#SBATCH --mem-per-gpu=128G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end
#SBATCH --switches=1@01:00:00           # number of leaf switches (InfiniBand Island) with time limit for the constraint
# #SBATCH --constraint=80gb               # constraints

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Maximum number of threads in the OpenMP parallel region (defaults to 1)
# (called by `torch.distributed.run`, called by `accelerate launch`)
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

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
    dataset=refinedweb \
    dataset.path_to_disk=$HOME/neo-bert/tokenized_datasets/falcon-refinedweb_google_4096 \
    tokenizer=google \
    model=[neobert,250M-opt] \
    datacollator=mlm_20 \
    optimizer=adamw \
    scheduler=linear_decay \
    +scheduler.constant_steps=50000 \
    scheduler.decay_steps=0 \
    trainer.gradient_accumulation_steps=8 \
    trainer.max_steps=50000 \
    trainer.model.save_steps=2000 \
    dataloader.train.batch_size=8 \
    tokenizer.max_length=4096
    +dataset.min_length=1024 \
    +model.pretrained_checkpoint_dir=logs/neobert/model_checkpoints \
    +model.pretrained_checkpoint=1000000 \
)

# Load modules and python environment
source activate base
conda activate .venv

# Launch cmd on each node
srun --kill-on-bad-exit=1 bash -c "$(for a in "${cmd[@]}" ; do echo -n \"$a\" "" ; done)"