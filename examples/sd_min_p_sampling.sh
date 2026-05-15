#!/usr/bin/env bash
# Min-p truncation sampling on the DRAFT + lossless SD verification.
# This is the "true" min-p baseline against which SpecCascade should be
# compared (paper §4.1, Figure 1).
#
# Method:   draft samples from min-p truncated q; verification is standard SD.
# Legal min_p: [0, 1]. Higher => stricter truncation.
#              Paper sweeps {0.1, 0.3, 0.5, 0.7, 0.9}.
#
# Usage:
#   bash examples/sd_min_p_sampling.sh [BENCH] [MIN_P] [SEED]

set -euo pipefail
BENCH="${1:-math}"
MIN_P="${2:-0.5}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method min_p_sampling --param "$MIN_P" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "min_p_${MIN_P}_${BENCH}_seed${SEED}"
