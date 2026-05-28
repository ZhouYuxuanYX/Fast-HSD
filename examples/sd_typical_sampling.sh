#!/usr/bin/env bash
# Medusa-style typical acceptance: truncation-based verification with the
# eta-sampling allowed set of the TARGET distribution (paper §3.2).
#
# Method:   accept x iff x is in the eta-truncated allowed set of p.
# Legal eta_cutoff: > 0. Paper sweeps {0.05, 0.10, 0.15, 0.20, 0.25}.
#
# Usage:
#   bash examples/sd_typical_sampling.sh [BENCH] [ETA_CUTOFF] [SEED]

set -euo pipefail
BENCH="${1:-math}"
ETA_CUTOFF="${2:-0.10}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method typical_sampling --param "$ETA_CUTOFF" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED"
