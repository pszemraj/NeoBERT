#!/bin/bash
#SBATCH --job-name=preprocess-wikibook-llama
#SBATCH --output=logs/%x_output.txt
#SBATCH --error=logs/%x_error.txt
#SBATCH --time=7-00:00
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --cpus-per-task=32              # number of cpus per node
#SBATCH --mem=128G                      # memory per node

# Load python environment
source activate base
conda activate .venv

# Launch the tokenization
python $HOME/neo-bert/scripts/pretraining/preprocess.py \
    wandb.mode=disabled \
    trainer.dir=$SCRATCH/logs/$SLURM_JOB_NAME \
    hydra.run.dir=$SCRATCH/logs/$SLURM_JOB_NAME/hydra \
    tokenizer.pretrained_model_name_or_path=meta-llama/Llama-2-7b-hf \
    tokenizer.max_length=512 \
    dataset.name=wikibook \
    dataset.column=text \
    dataset.path_to_disk=$SCRATCH/wikibook_llama \