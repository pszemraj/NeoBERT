#!/bin/bash
#SBATCH --job-name=sequence_lengths_short  # Job name
#SBATCH --output=logs/txt/%x_%j_output.txt     # Standard output and error log
#SBATCH --error=logs/txt/%x_%j_error.txt       # Error log
#SBATCH --partition=dgxh100long            # GPU partition
#SBATCH --gres=gpu:4               # Request 8 GPUs
#SBATCH --ntasks=8                 # Number of tasks (one per GPU)
#SBATCH --cpus-per-task=4          # CPUs per task (adjust as needed)
#SBATCH --time=10-00:00:00            # Time limit
#SBATCH --mem=256G                  # Memory allocation
# #SBATCH --dependency=afterok:19340

# Load modules and python environment
module load python/3.10 cuda/12.3.2
source $HOME/neo-bert/.venv/bin/activate

# Launch tasks in parallel, each on a different GPU
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python /home/users/lebreton/neo-bert/scripts/evaluation/pseudo_perplexity.py \
        --model_name "chandar-lab/NeoBERT" \
        --from_hub \
        --data_name wikipedia \
        --output_path logs/ \
        --batch_size 8 \
        --dataset_shard $i &
done

# Wait for all background processes to finish
wait