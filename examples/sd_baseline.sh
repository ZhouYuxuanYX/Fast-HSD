#!/usr/bin/env bash
# Lossless speculative decoding baseline (Leviathan et al., 2023).
#
# Method: standard SD acceptance rule. No lossy relaxation.
# Equivalent to running with --method lenience --param 1.0.
#
# Usage:
#   bash examples/sd_baseline.sh [BENCH] [SEED]
# where BENCH is one of {math, mbppplus, include, bfcl}.

set -euo pipefail
BENCH="${1:-math}"
SEED="${2:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method baseline \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED"
