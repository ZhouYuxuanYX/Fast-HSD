#!/usr/bin/env bash
# Collaborative Decoding via Speculation (CoS) - weighted-ensemble form.
#
# Method:   yield distribution is the convex mixture lam*p + (1-lam)*q.
# Legal lam: [0, 1]. lam=1 recovers lossless SD; lam=0 yields pure draft.
#           The paper sweeps {0.2, 0.4, 0.6, 0.8}.
#
# Usage:
#   bash examples/sd_cos.sh [BENCH] [COS_LAMBDA] [SEED]

set -euo pipefail
BENCH="${1:-math}"
LAMBDA="${2:-0.4}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method cos --param "$LAMBDA" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "cos_${LAMBDA}_${BENCH}_seed${SEED}"
