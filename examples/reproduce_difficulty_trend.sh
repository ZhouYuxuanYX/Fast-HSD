#!/usr/bin/env bash
# Reproduce Figure 1 (accuracy gap widens with task difficulty).
# Runs SpecCascade at min-p=0.5 against the true min-p baseline on GSM8K, MATH,
# and AIME.
#
# Usage:
#   bash examples/reproduce_difficulty_trend.sh [SEED]

set -euo pipefail

SEED="${1:-0}"
TARGET="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"

for BENCH in math; do
  # True baseline: target with min-p sampling, no SD verification.
  fast-hsd-eval \
    --benchmark "$BENCH" \
    --method min_p_sampling --param 0.5 \
    --target-model "$TARGET" --draft-model "$DRAFT" \
    --temperature 0.7 --seed "$SEED" \
    --name "trueminp_${BENCH}_seed${SEED}"

  # Lossy SpecCascade: same allowed set, but the verification gate is the
  # paper's pitfall.
  fast-hsd-eval \
    --benchmark "$BENCH" \
    --method speccascade --param 0.5 \
    --target-model "$TARGET" --draft-model "$DRAFT" \
    --temperature 0.7 --seed "$SEED" \
    --name "speccascade_${BENCH}_seed${SEED}"
done
