#!/bin/bash

# ---- Env ----
# Initialize conda for shell interaction
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate YOUR_ENV

# ---- Config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config_math.json}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# Control which sweeps to run (overrides config if set):
#   ENABLED_SWEEPS="all"
#   ENABLED_SWEEPS="lenience cos_lambda eta_spd"
#   ENABLED_SWEEPS="lenience,cos_lambda,eta_spd"
ENABLED_SWEEPS="${ENABLED_SWEEPS:-__FROM_CONFIG__}"

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
src_dir = cfg.get("src_dir", f"{project_root}/src/MATH")
draft_model = cfg.get("draft_model", "")
target_model = cfg.get("target_model", "")
dataset = cfg.get("dataset", "math")
task_timeout_sec = cfg.get("task_timeout_sec", 10800)
gamma = cfg.get("gamma", 10)
max_concurrent_jobs = cfg.get("max_concurrent_jobs", 4)

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

echo "Using config: ${CONFIG_PATH}"
echo "Enabled sweeps: ${ENABLED_SWEEPS}"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Results root: ${PROJECT_ROOT}/results"

mkdir -p "${PROJECT_ROOT}/results"
cd "${SRC_DIR}" || exit

run_experiment() {
    local TASK_ID=$1
    local SLOT_ID=$2
    local GPU_START=$((SLOT_ID * 2))
    local GPU_END=$((GPU_START + 1))

    local METHOD="${TASK_METHODS[$TASK_ID]}"
    local PARAM_NAME="${TASK_PARAM_NAMES[$TASK_ID]}"
    local PARAM_VALUE="${TASK_VALUES[$TASK_ID]}"
    local SEED_VALUE="${TASK_SEEDS[$TASK_ID]}"

    local RUNSTAMP="$(date +%Y%m%d-%H%M%S)"
    local DATE_DIR="$(date +%m%d)"
    local RUN_NAME="${DATASET}_${METHOD}_${PARAM_VALUE}_seed${SEED_VALUE}_${RUNSTAMP}"

    local RUN_DIR="${PROJECT_ROOT}/results/${DATE_DIR}/${RUN_NAME}"
    local LOG_DIR="${RUN_DIR}/logs"
    local LOG_FILE="${LOG_DIR}/exp.txt"
    mkdir -p "${LOG_DIR}"

    echo "[$(date)] Starting Task ${TASK_ID} on GPUs ${GPU_START},${GPU_END} | Method: ${METHOD} | Value: ${PARAM_VALUE} | Seed: ${SEED_VALUE}"
    echo "[$(date)] Run directory: ${RUN_DIR}"
    echo "[$(date)] Log file: ${LOG_FILE}"

    local CMD=(
        python eval_math.py
        --name "${RUN_NAME}"
        --target-model "${TARGET_MODEL}"
        --draft-model "${DRAFT_MODEL}"
        --speculative
        --gamma "${GAMMA}"
        --seed "${SEED_VALUE}"
        "--${PARAM_NAME}" "${PARAM_VALUE}"
    )

    CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} timeout --signal=TERM --kill-after=120 ${TASK_TIMEOUT_SEC} "${CMD[@]}" \
        > "${LOG_FILE}" 2>&1

    local EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 124 || ${EXIT_CODE} -eq 137 ]]; then
        echo "[$(date)] Task ${TASK_ID} timed out after ${TASK_TIMEOUT_SEC}s on GPUs ${GPU_START},${GPU_END}"
    elif [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "[$(date)] Task ${TASK_ID} failed with exit code ${EXIT_CODE} on GPUs ${GPU_START},${GPU_END}"
    else
        echo "[$(date)] Finished Task ${TASK_ID}"
    fi
    echo "[$(date)] Task ${TASK_ID} artifacts: ${RUN_DIR}"
}

# Master loop to manage parallelism (dynamic scheduling)
# By default 8 GPUs total, 2 GPUs per run => 4 concurrent jobs
declare -a SLOT_PIDS
NEXT_TASK=0

for ((SLOT_ID=0; SLOT_ID<MAX_CONCURRENT_JOBS && NEXT_TASK<TOTAL_TASKS; SLOT_ID++)); do
    run_experiment ${NEXT_TASK} ${SLOT_ID} &
    SLOT_PIDS[${SLOT_ID}]=$!
    NEXT_TASK=$((NEXT_TASK + 1))
done

while (( NEXT_TASK < TOTAL_TASKS )); do
    wait -n
    for ((SLOT_ID=0; SLOT_ID<MAX_CONCURRENT_JOBS && NEXT_TASK<TOTAL_TASKS; SLOT_ID++)); do
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
