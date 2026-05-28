#!/usr/bin/env bash
# EAGLE-3 + Medusa-style typical acceptance.
# Paper sweeps eta cutoff in {0.05, 0.10, 0.15, 0.20, 0.25}.
#
# Usage:
#   bash examples/eagle3_typical_sampling.sh [BENCH] [ETA_CUTOFF] [SEED]

set -euo pipefail
BENCH="${1:-math}"
ETA_CUTOFF="${2:-0.10}"
SEED="${3:-0}"

fast-hsd-eval \
    --benchmark "$BENCH" --use-eagle3 \
    --method typical_sampling --param "$ETA_CUTOFF" \
    --target-model "meta-llama/Llama-3.1-8B-Instruct" \
    --draft-model  "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
    --temperature 0.7 \
    --seed "$SEED"
