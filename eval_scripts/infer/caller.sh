#!/bin/bash
# =====================================================================
# Async Inference Launcher
# =====================================================================
# Usage:
#   1. Fill in the placeholder paths marked with {YOUR_...} below
#   2. Adjust GPU / concurrency settings as needed
#   3. Run:  bash caller.sh
# =====================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "${SCRIPT_DIR}/caller_async.py" \
    --model {YOUR_MODEL_PATH} \
    --input {YOUR_INPUT_JSONL} \
    --output {YOUR_OUTPUT_JSONL} \
    --hyperparam mimo \
    --prompt-field prompt \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --max-num-seqs 512 \
    --concurrent-per-endpoint 32 \
    --max-model-len 20000 \
    --max-tokens 16384 \
    --n 1 \
    --gpu-memory-utilization 0.9 \
    --max-retry-rounds 50 \
    --prompt-file {YOUR_PROMPT_FILE}

