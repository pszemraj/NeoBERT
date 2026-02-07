#!/bin/bash
# Quick GLUE evaluation on small tasks only
# These tasks train quickly and give a good indication of model performance

set -e  # Exit on error

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
RED='\033[0;31m'

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

CONFIG_BASENAME="$(basename "${CONFIG_DIR}")"
LOG_DIR="logs/${CONFIG_BASENAME}"
mkdir -p "${LOG_DIR}"

echo -e "${YELLOW}=========================================="
echo "NeoBERT Quick GLUE Evaluation"
echo "Running small tasks only for quick validation"
echo "==========================================${NC}"
echo ""

# Track results
PASSED=()
FAILED=()
declare -A OUTPUT_DIRS
declare -A LOG_PATHS

# Run each small task
for task in "${SMALL_TASKS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${task}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Config not found for $task at $CONFIG_PATH. Skipping."
        OUTPUT_DIRS["$task"]="(config missing)"
        LOG_PATHS["$task"]="(config missing)"
        continue
    fi

    echo -e "${YELLOW}Running $task...${NC}"

    OUTPUT_PATH="$(python - "$CONFIG_PATH" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
text = config_path.read_text()
try:
    import yaml  # type: ignore
except ImportError:
    match = re.search(r'^\s*output_dir:\s*["\']?(.*?)["\']?\s*$', text, flags=re.MULTILINE)
    print(match.group(1) if match else '', end='')
else:
    data = yaml.safe_load(text)
    trainer = data.get('trainer', {}) if isinstance(data, dict) else {}
    print(trainer.get('output_dir', '') or '', end='')
PY
)"
    if [ -z "${OUTPUT_PATH}" ]; then
        OUTPUT_PATH="(output_dir not found in config)"
    fi
    OUTPUT_DIRS["$task"]="${OUTPUT_PATH}"

    LOG_PATH="${LOG_DIR}/${task}_quick.log"
    LOG_PATHS["$task"]="${LOG_PATH}"
    
    # Run the evaluation
    cmd=(
        python "${SCRIPT_DIR}/../run_glue.py"
        "${CONFIG_PATH}"
    )

    if ( set -o pipefail; "${cmd[@]}" 2>&1 | tee "${LOG_PATH}" ); then
        echo -e "${GREEN}✓ $task completed successfully${NC}"
        PASSED+=("$task")
        
        # Show the results immediately
        if [ -d "${OUTPUT_PATH}" ]; then
            RESULTS_FILE="${OUTPUT_PATH}/all_results.json"
        else
            RESULTS_FILE=""
        fi
        if [ -n "$RESULTS_FILE" ] && [ -f "$RESULTS_FILE" ]; then
            echo "Results:"
            python -m json.tool "$RESULTS_FILE" | head -20
        else
            echo -e "${YELLOW}No all_results.json found at ${OUTPUT_PATH}${NC}"
        fi
    else
        echo -e "${RED}✗ $task failed${NC}"
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
    output="${OUTPUT_DIRS[$task]}"
    if [ -z "$output" ]; then
        output="(not available)"
    fi
    echo "  - ${output}"
done
echo ""
echo "Logs saved to:"
for task in "${SMALL_TASKS[@]}"; do
    log_path="${LOG_PATHS[$task]}"
    if [ -z "$log_path" ]; then
        log_path="(not available)"
    fi
    echo "  - ${log_path}"
done

# Check if any model checkpoints were incorrectly saved
echo ""
echo "Checking for unwanted model checkpoints..."
CHECK_DIRS=()
for task in "${SMALL_TASKS[@]}"; do
    output="${OUTPUT_DIRS[$task]}"
    if [ -d "$output" ]; then
        CHECK_DIRS+=("$output")
    fi
done
if [ ${#CHECK_DIRS[@]} -gt 0 ] && find "${CHECK_DIRS[@]}" -name "*.safetensors" 2>/dev/null | grep -q .; then
    echo "WARNING: Found model checkpoints that shouldn't be saved!"
    find "${CHECK_DIRS[@]}" -name "*.safetensors"
else
    echo -e "${GREEN}✓ No model checkpoints found (correct behavior)${NC}"
fi
