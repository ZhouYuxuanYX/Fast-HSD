#!/usr/bin/env bash
# Reproduce the EAGLE-3 + Llama-3.1-8B table from the paper.
# Single A6000 GPU is enough.
#
# Usage:
#   bash examples/reproduce_eagle3.sh [SEED]

set -euo pipefail

SEED="${1:-0}"

TARGET="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

declare -A SWEEPS
SWEEPS[lenience]="0.2 0.4 0.6 0.8"
SWEEPS[speccascade]="0.1 0.3 0.5 0.7 0.9"
SWEEPS[typical_sampling]="0.05 0.10 0.15 0.20 0.25"
SWEEPS[min_p_sampling]="0.1 0.3 0.5 0.7 0.9"
SWEEPS[eta_sampling]="0.05 0.10 0.15 0.20 0.25"

for BENCH in math mbppplus include bfcl; do
  fast-hsd-eval \
    --benchmark "$BENCH" --use-eagle3 \
    --method baseline \
    --target-model "$TARGET" --draft-model "$DRAFT" \
    --temperature 0.7 --seed "$SEED" \
    --name "eagle_baseline_seed${SEED}"

  for METHOD in "${!SWEEPS[@]}"; do
    for VAL in ${SWEEPS[$METHOD]}; do
      fast-hsd-eval \
        --benchmark "$BENCH" --use-eagle3 \
        --method "$METHOD" --param "$VAL" \
        --target-model "$TARGET" --draft-model "$DRAFT" \
        --temperature 0.7 --seed "$SEED" \
        --name "eagle_${METHOD}_${VAL}_seed${SEED}"
    done
  done
done
