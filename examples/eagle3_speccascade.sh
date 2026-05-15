#!/usr/bin/env bash
# EAGLE-3 + SpecCascade truncation-based verification.
# Paper sweeps min-p threshold in {0.1, 0.3, 0.5, 0.7, 0.9}.
#
# Usage:
#   bash examples/eagle3_speccascade.sh [BENCH] [P_BASE] [SEED]

set -euo pipefail
BENCH="${1:-math}"
P_BASE="${2:-0.5}"
SEED="${3:-0}"

fast-hsd-eval \
    --benchmark "$BENCH" --use-eagle3 \
    --method speccascade --param "$P_BASE" \
    --target-model "meta-llama/Llama-3.1-8B-Instruct" \
    --draft-model  "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "eagle3_speccascade_${P_BASE}_${BENCH}_seed${SEED}"
