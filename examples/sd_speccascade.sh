#!/usr/bin/env bash
# SpecCascade truncation-based verification (Narasimhan et al., 2024).
#
# Method:   accept x iff p(x) >= p_base * max(p). I.e., x is in the min-p
#           allowed set of the TARGET distribution.
# Legal p_base: [0, 1]. The paper sweeps {0.1, 0.3, 0.5, 0.7, 0.9}.
#
# Usage:
#   bash examples/sd_speccascade.sh [BENCH] [P_BASE] [SEED]

set -euo pipefail
BENCH="${1:-math}"
P_BASE="${2:-0.5}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method speccascade --param "$P_BASE" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED"
