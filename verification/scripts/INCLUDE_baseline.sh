#!/bin/bash

# ---- Env ----
# Initialize conda for shell interaction

# ---- Repo ----
# Base directory for the project
PROJECT_ROOT="."
# Directory containing the python script
SRC_DIR="${PROJECT_ROOT}/src/INCLUDE"

# ---- Models ----
DRAFT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
TARGET_MODEL="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
DATASET_DIR="PATH_TO_INCLUDE_DATASET"

# ---- Shared args ----
GAMMA=10
DATASET="include"

# Parameter arrays
seeds=(5 10)
min_p_param=(0.1 0.3 0.5 0.7 0.9)
eta_param=(0.05 0.1 0.15 0.2 0.25)

# Ensure output directories exist
mkdir -p "${PROJECT_ROOT}/results"

cd "${SRC_DIR}" || exit

# Function to run a single experiment
run_experiment() {
    local TASK_ID=$1
    local GPU_START=$2
    local GPU_END=$((GPU_START + 1))
    
    # Determine parameters based on TASK_ID
    local PARAM_FLAG=""
    local SEED_VALUE=""
    local METHOD=""
    local PARAM_VALUE=""

    if [ $TASK_ID -lt 10 ]; then
        # Tasks 0-9: min_p experiments (5 params * 2 seeds)
        local p_idx=$((TASK_ID / 2))
        local s_idx=$((TASK_ID % 2))
        
        local current_min_p=${min_p_param[$p_idx]}
        local current_seed=${seeds[$s_idx]}
        
        METHOD="min_p_baseline"
        PARAM_VALUE="${current_min_p}"
        PARAM_FLAG="--min_p ${current_min_p}"
        SEED_VALUE="${current_seed}"
        
    else
        # Tasks 10-19: eta experiments (5 params * 2 seeds)
        local local_id=$((TASK_ID - 10))
        local p_idx=$((local_id / 2))
        local s_idx=$((local_id % 2))
        
        local current_eta=${eta_param[$p_idx]}
        local current_seed=${seeds[$s_idx]}
        
        METHOD="eta_baseline"
        PARAM_VALUE="${current_eta}"
        PARAM_FLAG="--eta_cutoff ${current_eta}"
        SEED_VALUE="${current_seed}"
    fi

    # ---- Unique name per run ----
    local RUNSTAMP="$(date +%Y%m%d-%H%M%S)"
    local DATE_DIR="$(date +%m%d)"
    local RUN_NAME="${DATASET}_${METHOD}_${PARAM_VALUE}_seed${SEED_VALUE}_${RUNSTAMP}"

    # Create output directories for this specific run
    # Note: adjusting path to be project-relative or similar to previous script
    local LOG_DIR="${PROJECT_ROOT}/results/${DATE_DIR}/${RUN_NAME}/logs"
    mkdir -p "${LOG_DIR}"

    echo "[$(date)] Starting Task ${TASK_ID} on GPUs ${GPU_START},${GPU_END} | Method: ${METHOD} | Value: ${PARAM_VALUE} | Seed: ${SEED_VALUE}"

    # Execute with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} python run_include_qwen2.5_simple.py \
        --name "${RUN_NAME}" \
        --target-model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --dataset-dir "${DATASET_DIR}" \
        --speculative \
        --gamma ${GAMMA} \
        --seed ${SEED_VALUE} \
        ${PARAM_FLAG} \
        > "${LOG_DIR}/exp.txt" 2>&1
    
    echo "[$(date)] Finished Task ${TASK_ID}"
}

# Master loop to manage parallelism
MAX_CONCURRENT_JOBS=4
job_count=0

for TASK_ID in {0..19}; do
    # Calculate GPU index (0, 2, 4, 6)
    # job_slot will be 0, 1, 2, 3
    job_slot=$((job_count % MAX_CONCURRENT_JOBS))
    gpu_start=$((job_slot * 2))
    
    run_experiment $TASK_ID $gpu_start &
    
    job_count=$((job_count + 1))
    
    # Wait if we hit the max jobs limit (every 4 tasks)
    if [[ $((job_count % MAX_CONCURRENT_JOBS)) -eq 0 ]]; then
        wait
    fi
done

wait
echo "[$(date)] All tasks completed!"
