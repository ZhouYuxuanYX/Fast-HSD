#!/usr/bin/env bash
# EAGLE-3 + lossless verification (Li et al., 2025).
# Single GPU is enough.
#
# Usage:
#   bash examples/eagle3_baseline.sh [BENCH] [SEED]

set -euo pipefail
BENCH="${1:-math}"
SEED="${2:-0}"

fast-hsd-eval \
    --benchmark "$BENCH" --use-eagle3 \
    --method baseline \
    --target-model "meta-llama/Llama-3.1-8B-Instruct" \
    --draft-model  "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "eagle3_baseline_${BENCH}_seed${SEED}"
