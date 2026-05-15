#!/usr/bin/env bash
# Reproduce the main Qwen2.5-72B / 0.5B speculative-decoding table from the paper.
# Sweeps every method × every hyperparameter value across MATH, MBPP+, INCLUDE, BFCL.
#
# Usage:
#   bash examples/reproduce_table_main_results.sh [SEED]
#
# Expects two A100 GPUs on CUDA_VISIBLE_DEVICES=0,1 (per the paper setup).

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"
SEED="${1:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TARGET="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"

# Method sweeps as defined in configs/methods/*.json.
declare -A SWEEPS
SWEEPS[lenience]="0.2 0.4 0.6 0.8"
SWEEPS[cos]="0.2 0.4 0.6 0.8"
SWEEPS[speccascade]="0.1 0.3 0.5 0.7 0.9"
SWEEPS[min_p_sampling]="0.1 0.3 0.5 0.7 0.9"
SWEEPS[eta_sampling]="0.05 0.10 0.15 0.20 0.25"
SWEEPS[typical_sampling]="0.05 0.10 0.15 0.20 0.25"

for BENCH in math mbppplus include bfcl; do
  # Lossless baseline.
  fast-hsd-eval \
    --benchmark "$BENCH" \
    --method baseline \
    --target-model "$TARGET" --draft-model "$DRAFT" \
    --temperature 0.7 --seed "$SEED" \
    --name "baseline_seed${SEED}"

  # Lossy methods.
  for METHOD in "${!SWEEPS[@]}"; do
    for VAL in ${SWEEPS[$METHOD]}; do
      fast-hsd-eval \
        --benchmark "$BENCH" \
        --method "$METHOD" --param "$VAL" \
        --target-model "$TARGET" --draft-model "$DRAFT" \
        --temperature 0.7 --seed "$SEED" \
        --name "${METHOD}_${VAL}_seed${SEED}"
    done
  done
done

echo "Done. Results under outputs/{math,mbppplus,include,bfcl}/. Aggregate with:"
echo "  python scripts/results_analysis.py outputs/<bench>/*.jsonl"
