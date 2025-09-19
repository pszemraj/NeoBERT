#!/bin/bash
# Run full GLUE evaluation suite on NeoBERT-100m checkpoint

set -e  # Exit on error

# List of GLUE tasks in order of size (smallest first for quick feedback)
TASKS=(
    "wnli"    # ~600 train examples
    "rte"     # ~2.5k train examples  
    "mrpc"    # ~3.7k train examples
    "stsb"    # ~5.7k train examples
    "cola"    # ~8.5k train examples
    "sst2"    # ~67k train examples
    "qnli"    # ~105k train examples
    "qqp"     # ~364k train examples
    "mnli"    # ~393k train examples
)

echo "=========================================="
echo "NeoBERT-100m Full GLUE Evaluation"
echo "Checkpoint: 100k steps (~2.0 CE loss)"
echo "=========================================="
echo ""

# Create output summary file
SUMMARY_FILE="outputs/glue/neobert-100m/glue_summary.txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"
echo "GLUE Evaluation Results - $(date)" > "$SUMMARY_FILE"
echo "Model: NeoBERT-100m (100k steps)" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

# Run each task
for task in "${TASKS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running $task..."
    echo "----------------------------------------"
    
    # Run evaluation
    if python scripts/evaluation/run_glue.py --config configs/glue/${task}.yaml; then
        echo "✅ $task completed successfully"
        
        # Extract final metrics from results file
        RESULTS_FILE="outputs/glue/neobert-100m/${task}/all_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            echo "" >> "$SUMMARY_FILE"
            echo "$task:" >> "$SUMMARY_FILE"
            cat "$RESULTS_FILE" >> "$SUMMARY_FILE"
        fi
    else
        echo "❌ $task failed"
        echo "" >> "$SUMMARY_FILE"
        echo "$task: FAILED" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "=========================================="
echo "All GLUE tasks completed!"
echo "Results saved to: $SUMMARY_FILE"
echo "=========================================="

# Display summary
echo ""
echo "Summary of results:"
cat "$SUMMARY_FILE"