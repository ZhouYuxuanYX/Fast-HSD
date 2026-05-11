#!/bin/bash

# ---- Env ----
# Initialize conda for shell interaction
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate YOUR_ENV

# ---- Config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config_gsm8k.json}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# Control which sweeps to run (overrides config if set):
#   ENABLED_SWEEPS="all"
#   ENABLED_SWEEPS="lenience cos_lambda eta_spd"
#   ENABLED_SWEEPS="lenience,cos_lambda,eta_spd"
ENABLED_SWEEPS="${ENABLED_SWEEPS:-__FROM_CONFIG__}"
CUDA_DEVICE_SET="${CUDA_DEVICE_SET:-0,1,2,3,4,5,6,7}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-1}"
GPU_GROUP_SIZE="${GPU_GROUP_SIZE:-2}"
SAMPLE_SHARDS="${SAMPLE_SHARDS:-1}"

eval "$(python - "${CONFIG_PATH}" "${ENABLED_SWEEPS}" <<'PY'
import json
import shlex
import sys

config_path = sys.argv[1]
enabled_override = sys.argv[2]

with open(config_path, "r") as f:
    cfg = json.load(f)

def q(value):
    return shlex.quote(str(value))

project_root = cfg.get("project_root", ".")
src_dir = cfg.get("src_dir", f"{project_root}/src/gsm8k")
draft_model = cfg.get("draft_model", "")
target_model = cfg.get("target_model", "")
dataset = cfg.get("dataset", "gsm8k")
task_timeout_sec = cfg.get("task_timeout_sec", 10800)
gamma = cfg.get("gamma", 10)
max_concurrent_jobs = cfg.get("max_concurrent_jobs", 4)
num_samples = cfg.get("num_samples", 500)

enabled_sweeps = cfg.get("enabled_sweeps", "all") if enabled_override == "__FROM_CONFIG__" else enabled_override
enabled_norm = str(enabled_sweeps).replace(",", " ").split()

def is_enabled(key: str) -> bool:
    if str(enabled_sweeps) == "all":
        return True
    return key in enabled_norm

sweeps = cfg.get("sweeps", {})
available_keys = list(sweeps.keys())

tasks = []
for key, spec in sweeps.items():
    if not is_enabled(key):
        continue

    method = spec.get("method", f"{key}_baseline")
    param_name = spec.get("param", key)
    values = spec.get("values", [])
    seeds = spec.get("seeds", [])

    if len(values) != len(seeds):
        raise ValueError(
            f"Sweep '{key}' has mismatched lengths: values={len(values)} seeds={len(seeds)}"
        )

    for value, seed in zip(values, seeds):
        tasks.append((method, param_name, value, seed))

print(f"PROJECT_ROOT={q(project_root)}")
print(f"SRC_DIR={q(src_dir)}")
print(f"DRAFT_MODEL={q(draft_model)}")
print(f"TARGET_MODEL={q(target_model)}")
print(f"DATASET={q(dataset)}")
print(f"TASK_TIMEOUT_SEC={q(task_timeout_sec)}")
print(f"GAMMA={q(gamma)}")
print(f"NUM_SAMPLES={q(num_samples)}")
print(f"MAX_CONCURRENT_JOBS={q(max_concurrent_jobs)}")
print(f"ENABLED_SWEEPS={q(enabled_sweeps)}")
print(f"AVAILABLE_SWEEP_KEYS={q(' '.join(available_keys))}")

print("declare -a TASK_METHODS=()")
print("declare -a TASK_PARAM_NAMES=()")
print("declare -a TASK_VALUES=()")
print("declare -a TASK_SEEDS=()")

for method, param_name, value, seed in tasks:
    print(f"TASK_METHODS+=({q(method)})")
    print(f"TASK_PARAM_NAMES+=({q(param_name)})")
    print(f"TASK_VALUES+=({q(value)})")
    print(f"TASK_SEEDS+=({q(seed)})")

print(f"TOTAL_TASKS={len(tasks)}")
PY
)"

if [[ -z "${DRAFT_MODEL}" || -z "${TARGET_MODEL}" ]]; then
    echo "[ERROR] draft_model and target_model must be set in config: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ${TOTAL_TASKS} -eq 0 ]]; then
    echo "[ERROR] No tasks selected. ENABLED_SWEEPS='${ENABLED_SWEEPS}', available: ${AVAILABLE_SWEEP_KEYS}" >&2
    exit 1
fi

if [[ ${MAX_EXPERIMENTS} -lt 1 ]]; then
    echo "[ERROR] MAX_EXPERIMENTS must be >= 1, got: ${MAX_EXPERIMENTS}" >&2
    exit 1
fi

if [[ ${TOTAL_TASKS} -gt ${MAX_EXPERIMENTS} ]]; then
    echo "Limiting tasks from ${TOTAL_TASKS} to ${MAX_EXPERIMENTS}"
    TOTAL_TASKS=${MAX_EXPERIMENTS}
fi

IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_DEVICE_SET}"

if [[ ${#CUDA_DEVICES[@]} -eq 0 ]]; then
    echo "[ERROR] CUDA_DEVICE_SET must list at least one GPU" >&2
    exit 1
fi

if [[ ${GPU_GROUP_SIZE} -lt 1 ]]; then
    echo "[ERROR] GPU_GROUP_SIZE must be >= 1, got: ${GPU_GROUP_SIZE}" >&2
    exit 1
fi

AVAILABLE_SLOTS=$(( ${#CUDA_DEVICES[@]} / GPU_GROUP_SIZE ))
if [[ ${AVAILABLE_SLOTS} -lt 1 ]]; then
    echo "[ERROR] Need at least ${GPU_GROUP_SIZE} GPUs in CUDA_DEVICE_SET, got: ${#CUDA_DEVICES[@]}" >&2
    exit 1
fi

if [[ ${MAX_CONCURRENT_JOBS} -gt ${AVAILABLE_SLOTS} ]]; then
    echo "Reducing MAX_CONCURRENT_JOBS from ${MAX_CONCURRENT_JOBS} to ${AVAILABLE_SLOTS} based on CUDA_DEVICE_SET/GPU_GROUP_SIZE"
    MAX_CONCURRENT_JOBS=${AVAILABLE_SLOTS}
fi

if [[ ${SAMPLE_SHARDS} -lt 1 ]]; then
    echo "[ERROR] SAMPLE_SHARDS must be >= 1, got: ${SAMPLE_SHARDS}" >&2
    exit 1
fi

declare -a JOB_TASK_IDS=()
declare -a JOB_SAMPLE_STARTS=()
declare -a JOB_SAMPLE_ENDS=()

if [[ ${SAMPLE_SHARDS} -gt 1 ]]; then
    if [[ ${TOTAL_TASKS} -ne 1 ]]; then
        echo "[ERROR] SAMPLE_SHARDS>1 currently supports exactly one selected task, got: ${TOTAL_TASKS}" >&2
        exit 1
    fi

    SHARD_SIZE=$(( (NUM_SAMPLES + SAMPLE_SHARDS - 1) / SAMPLE_SHARDS ))
    for ((SHARD_ID=0; SHARD_ID<SAMPLE_SHARDS; SHARD_ID++)); do
        SAMPLE_START=$(( SHARD_ID * SHARD_SIZE ))
        SAMPLE_END=$(( SAMPLE_START + SHARD_SIZE ))
        if [[ ${SAMPLE_START} -ge ${NUM_SAMPLES} ]]; then
            break
        fi
        if [[ ${SAMPLE_END} -gt ${NUM_SAMPLES} ]]; then
            SAMPLE_END=${NUM_SAMPLES}
        fi
        JOB_TASK_IDS+=(0)
        JOB_SAMPLE_STARTS+=("${SAMPLE_START}")
        JOB_SAMPLE_ENDS+=("${SAMPLE_END}")
    done
else
    for ((TASK_ID=0; TASK_ID<TOTAL_TASKS; TASK_ID++)); do
        JOB_TASK_IDS+=("${TASK_ID}")
        JOB_SAMPLE_STARTS+=(0)
        JOB_SAMPLE_ENDS+=("${NUM_SAMPLES}")
    done
fi

TOTAL_JOBS=${#JOB_TASK_IDS[@]}

if [[ ${TOTAL_JOBS} -eq 0 ]]; then
    echo "[ERROR] No jobs were created after shard expansion" >&2
    exit 1
fi

if [[ ${MAX_CONCURRENT_JOBS} -gt ${TOTAL_JOBS} ]]; then
    MAX_CONCURRENT_JOBS=${TOTAL_JOBS}
fi

echo "Using config: ${CONFIG_PATH}"
echo "Enabled sweeps: ${ENABLED_SWEEPS}"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Expanded jobs: ${TOTAL_JOBS}"
echo "CUDA_VISIBLE_DEVICES pool: ${CUDA_DEVICE_SET}"
echo "GPU group size: ${GPU_GROUP_SIZE}"
echo "Max concurrent jobs: ${MAX_CONCURRENT_JOBS}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Sample shards: ${SAMPLE_SHARDS}"
echo "Results root: ${PROJECT_ROOT}/results"

mkdir -p "${PROJECT_ROOT}/results"
cd "${SRC_DIR}" || exit

run_experiment() {
    local JOB_ID=$1
    local SLOT_ID=$2
    local GPU_OFFSET=$((SLOT_ID * GPU_GROUP_SIZE))
    local SLOT_GPUS=()

    for ((IDX=0; IDX<GPU_GROUP_SIZE; IDX++)); do
        SLOT_GPUS+=("${CUDA_DEVICES[$((GPU_OFFSET + IDX))]}")
    done

    local GPU_SET
    GPU_SET=$(IFS=','; echo "${SLOT_GPUS[*]}")

    local TASK_ID="${JOB_TASK_IDS[$JOB_ID]}"
    local SAMPLE_START="${JOB_SAMPLE_STARTS[$JOB_ID]}"
    local SAMPLE_END="${JOB_SAMPLE_ENDS[$JOB_ID]}"

    local METHOD="${TASK_METHODS[$TASK_ID]}"
    local PARAM_NAME="${TASK_PARAM_NAMES[$TASK_ID]}"
    local PARAM_VALUE="${TASK_VALUES[$TASK_ID]}"
    local SEED_VALUE="${TASK_SEEDS[$TASK_ID]}"

    local RUNSTAMP="$(date +%Y%m%d-%H%M%S)"
    local DATE_DIR="$(date +%m%d)"
    local SHARD_LABEL="samples${SAMPLE_START}-$((SAMPLE_END - 1))"
    local RUN_NAME="${DATASET}_${METHOD}_${PARAM_VALUE}_seed${SEED_VALUE}_${SHARD_LABEL}_${RUNSTAMP}"

    local RUN_DIR="${PROJECT_ROOT}/results/${DATE_DIR}/${RUN_NAME}"
    local LOG_DIR="${RUN_DIR}/logs"
    local LOG_FILE="${LOG_DIR}/exp.txt"
    mkdir -p "${LOG_DIR}"

    echo "[$(date)] Starting Job ${JOB_ID} on GPUs ${GPU_SET} | Method: ${METHOD} | Value: ${PARAM_VALUE} | Seed: ${SEED_VALUE} | Samples: ${SAMPLE_START}-${SAMPLE_END}"
    echo "[$(date)] Run directory: ${RUN_DIR}"
    echo "[$(date)] Log file: ${LOG_FILE}"

    local CMD=(
        python eval_gsm8k.py
        --name "${RUN_NAME}"
        --target-model "${TARGET_MODEL}"
        --draft-model "${DRAFT_MODEL}"
        --speculative
        --gamma "${GAMMA}"
        --seed "${SEED_VALUE}"
        --num_samples "$((SAMPLE_END - SAMPLE_START))"
        --sample_start "${SAMPLE_START}"
        --sample_end "${SAMPLE_END}"
        "--${PARAM_NAME}" "${PARAM_VALUE}"
    )

    CUDA_VISIBLE_DEVICES=${GPU_SET} timeout --signal=TERM --kill-after=120 ${TASK_TIMEOUT_SEC} "${CMD[@]}" \
        > "${LOG_FILE}" 2>&1

    local EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 124 || ${EXIT_CODE} -eq 137 ]]; then
        echo "[$(date)] Job ${JOB_ID} timed out after ${TASK_TIMEOUT_SEC}s on GPUs ${GPU_SET}"
    elif [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "[$(date)] Job ${JOB_ID} failed with exit code ${EXIT_CODE} on GPUs ${GPU_SET}"
    else
        echo "[$(date)] Finished Job ${JOB_ID}"
    fi
    echo "[$(date)] Job ${JOB_ID} artifacts: ${RUN_DIR}"
}

# Master loop to manage parallelism (dynamic scheduling)
# By default 8 GPUs total, 2 GPUs per run => 4 concurrent jobs
declare -a SLOT_PIDS
NEXT_TASK=0

for ((SLOT_ID=0; SLOT_ID<MAX_CONCURRENT_JOBS && NEXT_TASK<TOTAL_JOBS; SLOT_ID++)); do
    run_experiment ${NEXT_TASK} ${SLOT_ID} &
    SLOT_PIDS[${SLOT_ID}]=$!
    NEXT_TASK=$((NEXT_TASK + 1))
done

while (( NEXT_TASK < TOTAL_JOBS )); do
    wait -n
    for ((SLOT_ID=0; SLOT_ID<MAX_CONCURRENT_JOBS && NEXT_TASK<TOTAL_JOBS; SLOT_ID++)); do
        PID=${SLOT_PIDS[${SLOT_ID}]}
        if [[ -n "${PID}" ]] && ! kill -0 ${PID} 2>/dev/null; then
            run_experiment ${NEXT_TASK} ${SLOT_ID} &
            SLOT_PIDS[${SLOT_ID}]=$!
            NEXT_TASK=$((NEXT_TASK + 1))
        fi
    done
done

wait
echo "[$(date)] All tasks completed!"
echo "[$(date)] All results saved under: ${PROJECT_ROOT}/results"
