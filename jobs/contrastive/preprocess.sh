#!/bin/bash
#SBATCH --job-name=preprocess-contrastive
#SBATCH --output=logs/txt/%x_output.txt
#SBATCH --error=logs/txt/%x_error.txt
#SBATCH --time=7-00:00
#SBATCH --partition=dgxh100long         # ask for high-priority job
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --cpus-per-task=48               # number of cpus per node
#SBATCH --mem=500G                       # memory per node

# Load modules and python environment
module load python/3.10 cuda/12.3.2
source $HOME/neo-bert/.venv/bin/activate

# Load datasets
python $HOME/neo-bert/scripts/contrastive/preprocess.py \
    datasets.from_disk=true