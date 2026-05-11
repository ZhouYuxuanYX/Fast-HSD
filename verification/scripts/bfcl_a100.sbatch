#!/bin/bash
#SBATCH -J bfcl_lenience1.0_s42
#SBATCH -A gts-rg1002
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=6
#SBATCH -t 3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# ---- Env ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/home/hcoda1/8/twang730/r-rg1002-0/envs/spd_clean

# ---- Config ----
PROJECT_ROOT="/storage/home/hcoda1/8/twang730/scratch/spd_410"
SRC_DIR="${PROJECT_ROOT}/src/bfcl"
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
TARGET_MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
DATASET="bfcl"
CATEGORY="parallel_multiple"
DATA_DIR="${PROJECT_ROOT}/src/bfcl/bfcl_data"
GAMMA=10

TASK_METHOD="lenience"
TASK_PARAM_NAME="lenience"
TASK_VALUE="1.0"
TASK_SEED="42"

# ---- Run info ----
RUNSTAMP="$(date +%Y%m%d-%H%M%S)"
DATE_DIR="$(date +%m%d)"
RUN_NAME="${DATASET}_${TASK_METHOD}_${TASK_VALUE}_seed${TASK_SEED}_${RUNSTAMP}"

RUN_DIR="${PROJECT_ROOT}/results/${DATE_DIR}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/exp.txt"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"

echo "GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null)"
echo "Method: ${TASK_METHOD} | ${TASK_PARAM_NAME}=${TASK_VALUE} | Seed: ${TASK_SEED}"
echo "Category: ${CATEGORY}"
echo "Run dir: ${RUN_DIR}"
echo "============================================="

# Build command array
CMD=(
    python "${SRC_DIR}/eval_bfcl.py"
    --name "${RUN_NAME}"
    --target-model "${TARGET_MODEL}"
    --draft-model "${DRAFT_MODEL}"
    --speculative
    --gamma "${GAMMA}"
    --seed "${TASK_SEED}"
    --category "${CATEGORY}"
    "--${TASK_PARAM_NAME}" "${TASK_VALUE}"
)

# Append --data-dir if set
if [[ -n "${DATA_DIR}" ]]; then
    CMD+=(--data-dir "${DATA_DIR}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[$(date)] Job ${SLURM_JOB_ID} failed (exit ${EXIT_CODE})"
else
    echo "[$(date)] Job ${SLURM_JOB_ID} finished"
fi
echo "[$(date)] Artifacts: ${RUN_DIR}"
