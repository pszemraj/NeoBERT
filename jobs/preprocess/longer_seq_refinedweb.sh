#!/bin/bash
#SBATCH --job-name=filter-longer-seq-refinedweb
#SBATCH --output=logs/%x_output.txt
#SBATCH --error=logs/%x_error.txt
#SBATCH --time=7-00:00
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --cpus-per-task=80               # number of cpus per node
#SBATCH --mem=250G                      # memory per node

# Load python environment
source activate base
conda activate .venv

# Launch the tokenization
python $HOME/neo-bert/scripts/pretraining/longer_seq.py \
    wandb.mode=disabled \
    hydra.run.dir=logs/$SLURM_JOB_NAME/hydra \
    dataset.path_to_disk=/home/users/lebreton/neo-bert/tokenized_datasets/falcon-refinedweb_google_4096 \
    tokenizer.max_length=4096 \
    +dataset.min_length=1024 \