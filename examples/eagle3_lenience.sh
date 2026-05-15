#!/usr/bin/env bash
# EAGLE-3 + lenience-based collaborative verification.
# Paper sweeps lenience in {0.2, 0.4, 0.6, 0.8}.
#
# Usage:
#   bash examples/eagle3_lenience.sh [BENCH] [LENIENCE] [SEED]

set -euo pipefail
BENCH="${1:-math}"
LENIENCE="${2:-0.4}"
SEED="${3:-0}"

fast-hsd-eval \
    --benchmark "$BENCH" --use-eagle3 \
    --method lenience --param "$LENIENCE" \
    --target-model "meta-llama/Llama-3.1-8B-Instruct" \
    --draft-model  "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "eagle3_lenience_${LENIENCE}_${BENCH}_seed${SEED}"
