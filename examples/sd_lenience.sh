#!/usr/bin/env bash
# Lenience-based collaborative verification (paper Eq. 6).
#
# Method:   accept draft token with probability  min(1, p(x) / (l * q(x))).
# Legal l:  (0, 1]. Lower l => more permissive acceptance => more speed,
#           less fidelity. The paper sweeps {0.2, 0.4, 0.6, 0.8}.
#
# Usage:
#   bash examples/sd_lenience.sh [BENCH] [LENIENCE] [SEED]

set -euo pipefail
BENCH="${1:-math}"
LENIENCE="${2:-0.4}"
SEED="${3:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

fast-hsd-eval \
    --benchmark "$BENCH" \
    --method lenience --param "$LENIENCE" \
    --target-model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8" \
    --draft-model  "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8" \
    --temperature 0.7 \
    --seed "$SEED" \
    --name "lenience_${LENIENCE}_${BENCH}_seed${SEED}"
