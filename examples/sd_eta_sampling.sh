#!/usr/bin/env bash
# Eta truncation sampling on the DRAFT + lossless SD verification.
# The "true" eta baseline against which Medusa typical-acceptance should be
# compared.
#
# Method:   draft samples from eta-truncated q; verification is standard SD.
# Legal eta: > 0. Paper sweeps {0.05, 0.10, 0.15, 0.20, 0.25}.
#
# Usage:
#   bash examples/sd_eta_sampling.sh [BENCH] [ETA] [SEED]

set -euo pipefail
BENCH="${1:-math}"
ETA="${2:-0.10}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method eta_sampling --param "$ETA" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED"
