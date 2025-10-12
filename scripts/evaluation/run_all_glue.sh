#!/bin/bash
# Run all GLUE evaluations for NeoBERT
# Usage: bash scripts/evaluation/run_all_glue.sh [config_dir]
#        config_dir defaults to configs/glue

CONFIG_DIR="${1:-configs/glue}"
echo "Using config directory: ${CONFIG_DIR}"
echo "---"

# Optional model path override (future use)
MODEL_PATH_OVERRIDE="$2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting GLUE evaluation suite${NC}"
if [ -n "$MODEL_PATH_OVERRIDE" ]; then
    echo "Model override: $MODEL_PATH_OVERRIDE"
fi
echo "---"

# All GLUE tasks
TASKS=("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")

# Track results
PASSED=()
FAILED=()

# Run each task
for task in "${TASKS[@]}"; do
    echo -e "${YELLOW}Running $task...${NC}"
    
    # Build command
    CONFIG_PATH="${CONFIG_DIR}/${task}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Config not found for $task at $CONFIG_PATH. Skipping."
        continue
    fi
    CMD="python scripts/evaluation/run_glue.py --config ${CONFIG_PATH}"
    
    # Add model override if provided
    if [ -n "$MODEL_PATH_OVERRIDE" ]; then
        CMD="$CMD --model_name_or_path $MODEL_PATH_OVERRIDE"
    fi
    
    # Run the evaluation
    $CMD 2>&1 | tee logs/glue_${task}.log
    
    # Check if successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ $task completed successfully${NC}"
        PASSED+=("$task")
    else
        echo -e "${RED}✗ $task failed${NC}"
        FAILED+=("$task")
    fi
    echo "---"
done

# Summary
echo -e "${YELLOW}===== GLUE Evaluation Summary =====${NC}"
echo -e "${GREEN}Passed (${#PASSED[@]}): ${PASSED[@]}${NC}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${RED}Failed (${#FAILED[@]}): ${FAILED[@]}${NC}"
else
    echo -e "${GREEN}All tasks completed successfully!${NC}"
fi

# Print results location
echo ""
echo "Results saved to:"
for task in "${TASKS[@]}"; do
    echo "  - outputs/glue/neobert-100m/$task/"
done
echo ""
echo "Logs saved to:"
for task in "${TASKS[@]}"; do
    echo "  - logs/glue_${task}.log"
done
