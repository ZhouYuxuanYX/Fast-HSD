#!/bin/bash
#SBATCH -J mbppplus_sweep
#SBATCH -A gts-rg1002
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH -t 2:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err


# ---- Env ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/home/hcoda1/8/twang730/r-rg1002-0/envs/spd_clean

# ---- Config ----
# Accept config path as first positional arg: sbatch mbppplus.sbatch /path/to/config.json
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${CONFIG_PATH:-${SCRIPT_DIR}/config_pace.json}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENABLED_SWEEPS="${ENABLED_SWEEPS:-__FROM_CONFIG__}"

# Resolve this array task's parameters from config
eval "$(python - "${CONFIG_PATH}" "${ENABLED_SWEEPS}" "${TASK_ID}" <<'PY'
import json, shlex, sys

config_path, enabled_override, task_id = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(config_path) as f:
    cfg = json.load(f)

def q(v):
    return shlex.quote(str(v))

project_root = cfg.get("project_root", "/storage/home/hcoda1/8/twang730/scratch/spd_410")
src_dir      = cfg.get("src_dir", f"{project_root}/src/mbppplus")
draft_model  = cfg.get("draft_model", "")
target_model = cfg.get("target_model", "")
dataset      = cfg.get("dataset", "mbppplus")
gamma        = cfg.get("gamma", 10)

enabled_sweeps = cfg.get("enabled_sweeps", "all") if enabled_override == "__FROM_CONFIG__" else enabled_override
enabled_norm = str(enabled_sweeps).replace(",", " ").split()

def is_enabled(key):
    return str(enabled_sweeps) == "all" or key in enabled_norm

tasks = []
for key, spec in cfg.get("sweeps", {}).items():
    if not is_enabled(key):
        continue
    method     = spec.get("method", f"{key}_baseline")
    param_name = spec.get("param", key)
    for value, seed in zip(spec.get("values", []), spec.get("seeds", [])):
        tasks.append((method, param_name, value, seed))

if task_id >= len(tasks):
    print(f'echo "[ERROR] SLURM_ARRAY_TASK_ID={task_id} >= total tasks ({len(tasks)})"; exit 1')
    sys.exit(0)

method, param_name, value, seed = tasks[task_id]

print(f"PROJECT_ROOT={q(project_root)}")
print(f"SRC_DIR={q(src_dir)}")
print(f"DRAFT_MODEL={q(draft_model)}")
print(f"TARGET_MODEL={q(target_model)}")
print(f"DATASET={q(dataset)}")
print(f"GAMMA={q(gamma)}")
print(f"TASK_METHOD={q(method)}")
print(f"TASK_PARAM_NAME={q(param_name)}")
print(f"TASK_VALUE={q(value)}")
print(f"TASK_SEED={q(seed)}")
print(f"TOTAL_TASKS={len(tasks)}")
PY
)"

if [[ -z "${DRAFT_MODEL}" || -z "${TARGET_MODEL}" ]]; then
    echo "[ERROR] draft_model and target_model must be set in config: ${CONFIG_PATH}" >&2
    exit 1
fi

# ---- Run info ----
RUNSTAMP="$(date +%Y%m%d-%H%M%S)"
DATE_DIR="$(date +%m%d)"
RUN_NAME="${DATASET}_${TASK_METHOD}_${TASK_VALUE}_seed${TASK_SEED}_${RUNSTAMP}"

RUN_DIR="${PROJECT_ROOT}/results/${DATE_DIR}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/exp.txt"
mkdir -p "${LOG_DIR}"

echo "============================================="
echo "SLURM Array Job: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID} / ${TOTAL_TASKS}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null)"
echo "Method: ${TASK_METHOD} | ${TASK_PARAM_NAME}=${TASK_VALUE} | Seed: ${TASK_SEED}"
echo "Run dir: ${RUN_DIR}"
echo "============================================="

cd "${SRC_DIR}" || exit

python eval_mbppplus.py \
    --name "${RUN_NAME}" \
    --results-dir "${RUN_DIR}" \
    --target-model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --speculative \
    --gamma "${GAMMA}" \
    --seed "${TASK_SEED}" \
    "--${TASK_PARAM_NAME}" "${TASK_VALUE}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} failed (exit ${EXIT_CODE})"
else
    echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished"
fi
echo "[$(date)] Artifacts: ${RUN_DIR}"
