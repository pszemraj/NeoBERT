#!/bin/bash

###################
meta_task=glue

# Model
model=neobert

# Pretrained checkpoint
ckpt=1000000
num_ckpt=1

transfer_from_task=true
overwrite=false

mixed_precision=fp16
early_stopping=15
###################

if [[ ! -z "$1" ]]; then meta_task=$1; fi
if [[ ! -z "$2" ]]; then model=$2; fi
if [[ ! -z "$3" ]]; then ckpt=$3; fi
if [[ ! -z "$4" ]]; then transfer_from_task=$4; fi
if [[ ! -z "$5" ]]; then num_ckpt=$5; fi
if [[ ! -z "$6" ]]; then overwrite=$6; fi

if [[ $model =~ bert-ablations* ]] ; then
    max_length=512
else
    max_length=1024
fi
from_hub=false
fi

# Function to print usage
print_usage() {
    echo "Usage: $0 [options] <meta_task> <batch_size> <learning_rate> <seed> <model> <ckpt> <run_dir>"
    echo
    echo "Arguments:"
    echo "  meta_task           The name of the meta task (e.g., glue, super_glue)"
    echo "  model               Name of the model to fine-tune"
    echo "  ckpt                Checkpoint to start from (e.g., 1000000)"
    echo "  transfer_from_task  Whether or not to transfer from matching task (ex: if true, will start finetuning on mnli from snli checkpoint)."
    echo "  num_ckpt            Number of previous checkpoints to merge to the starting checkpoint (e.g., 3, 5)."
    echo "  overwrite           Whether or not to overwrite existing results."
    echo
    echo "Options:"
    echo "  -h, --help       Show this help message and exit"
}

# Check for help option or missing arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Initialize grid search arguments
D=()

if [[ $meta_task=="glue" ]]; then
    for task in 'stsb' 'mrpc' 'rte' 'qnli' 'mnli' 'cola' 'sst2' 'qqp' 'mnli' ; do
        for batch_size in 16 32 4 8 2 ; do
            for lr in  6e-6 1e-5 2e-5 3e-5 8e-6 5e-6 ; do
                for wd in 0.01 0.00001 ; do
                    for seed in 0; do
                        D+=("$task $batch_size $lr $seed $wd")
                    done
                done
            done
        done
    done
else
    for dataset in 'axb' 'axg' 'boolq' 'cb' 'copa' 'multirc' 'record' 'rte' 'wic' 'wsc' ; do
        for batch_size in 4 8 16 32 64; do
            for lr in 6e-6 1e-5 2e-5 3e-5; do
                for seed in 0; do
                    D+=("$dataset $batch_size $lr $seed")
                done
            done
        done
    done
fi

# Loop through the argument combinations and submit jobs
for i in "${!D[@]}"; do
    set -- ${D[$i]}

    task=$1
    batch_size=$2
    lr=$3
    seed=$4
    wd=$5
    
    run_dir=$SCRATCH/logs/$model/glue/$ckpt/$mixed_precision/$early_stopping/$task/$batch_size/$lr/$wd/$seed/$transfer_from_task/$num_ckpt

    # Submit job if results do not exist or overwrite is allowed
    if [[ ! -f "$run_dir/all_results.json" || "$overwrite" = true ]]; then
        echo $meta_task $task $model $ckpt $batch_size $lr $seed $transfer_from_task $num_ckpt $from_hub $max_length $mixed_precision $early_stopping $wd
        sbatch --job-name=glue-$model $HOME/neo-bert/jobs/glue/train.sh $meta_task $task $model $ckpt $batch_size $lr $seed $transfer_from_task $num_ckpt $from_hub $max_length $run_dir $mixed_precision $early_stopping $wd
    fi
done
