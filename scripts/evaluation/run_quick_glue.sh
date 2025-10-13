#!/bin/bash
# Quick GLUE evaluation on small tasks only
# These tasks train quickly and give a good indication of model performance

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Small GLUE tasks only (in order of dataset size)
SMALL_TASKS=(
    "rte"
    "mrpc"
    "stsb"
    "cola"
)

CONFIG_DIR="${1:-configs/glue}"
echo "Using config directory: ${CONFIG_DIR}"
echo ""

echo -e "${YELLOW}=========================================="
echo "NeoBERT-100m Quick GLUE Evaluation"
echo "Running small tasks only for quick validation"
echo "==========================================${NC}"
echo ""

# Create output directory
mkdir -p outputs/glue/neobert-100m
mkdir -p logs

# Track results
PASSED=()
FAILED=()

# Run each small task
for task in "${SMALL_TASKS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${task}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Config not found for $task at $CONFIG_PATH. Skipping."
        continue
    fi

    echo -e "${YELLOW}Running $task...${NC}"
    
    # Run the evaluation
    if python scripts/evaluation/run_glue.py --config "$CONFIG_PATH" 2>&1 | tee logs/glue_${task}_quick.log; then
        echo -e "${GREEN}✓ $task completed successfully${NC}"
        PASSED+=("$task")
        
        # Show the results immediately
        RESULTS_FILE="outputs/glue/neobert-100m/${task}/all_results.json"
        if [ -f "$RESULTS_FILE" ]; then
            echo "Results:"
            python -m json.tool "$RESULTS_FILE" | head -20
        fi
    else
        echo "✗ $task failed"
        FAILED+=("$task")
    fi
    echo "---"
done

# Summary
echo ""
echo -e "${YELLOW}===== Quick GLUE Evaluation Summary =====${NC}"
echo -e "${GREEN}Passed (${#PASSED[@]}): ${PASSED[@]}${NC}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed (${#FAILED[@]}): ${FAILED[@]}"
fi

echo ""
echo "Results saved to:"
for task in "${SMALL_TASKS[@]}"; do
echo "  - outputs/glue/neobert-100m/$task/"
done

# Check if any model checkpoints were incorrectly saved
echo ""
echo "Checking for unwanted model checkpoints..."
if find outputs/glue/neobert-100m -name "state_dict.pt" -o -name "*.safetensors" 2>/dev/null | grep -q .; then
    echo "WARNING: Found model checkpoints that shouldn't be saved!"
    find outputs/glue/neobert-100m -name "state_dict.pt" -o -name "*.safetensors"
else
    echo -e "${GREEN}✓ No model checkpoints found (correct behavior)${NC}"
fi
