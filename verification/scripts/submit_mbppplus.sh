#!/bin/bash
# submit_mbppplus.sh — Compute task count from config and submit the job array
#
# Usage:
#   ./submit_mbppplus.sh                    # use config defaults
#   ENABLED_SWEEPS="min_p" ./submit_mbppplus.sh   # override sweep selection
#   MAX_PARALLEL=2 ./submit_mbppplus.sh     # limit concurrent array tasks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config.json}"
ENABLED_SWEEPS="${ENABLED_SWEEPS:-__FROM_CONFIG__}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# Count total tasks from config
TOTAL_TASKS=$(python - "${CONFIG_PATH}" "${ENABLED_SWEEPS}" <<'PY'
import json, sys
config_path, enabled_override = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)
enabled_sweeps = cfg.get("enabled_sweeps", "all") if enabled_override == "__FROM_CONFIG__" else enabled_override
enabled_norm = str(enabled_sweeps).replace(",", " ").split()
def is_enabled(key):
    return str(enabled_sweeps) == "all" or key in enabled_norm
n = 0
for key, spec in cfg.get("sweeps", {}).items():
    if is_enabled(key):
        n += len(spec.get("values", []))
print(n)
PY
)

if [[ "${TOTAL_TASKS}" -eq 0 ]]; then
    echo "[ERROR] No tasks found. Check enabled_sweeps in ${CONFIG_PATH}" >&2
    exit 1
fi

LAST_IDX=$((TOTAL_TASKS - 1))

echo "Config: ${CONFIG_PATH}"
echo "Enabled sweeps: ${ENABLED_SWEEPS}"
echo "Total tasks: ${TOTAL_TASKS} (array 0-${LAST_IDX})"
echo "Max parallel: ${MAX_PARALLEL}"
echo ""

# List all tasks
python - "${CONFIG_PATH}" "${ENABLED_SWEEPS}" <<'PY'
import json, sys
config_path, enabled_override = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)
enabled_sweeps = cfg.get("enabled_sweeps", "all") if enabled_override == "__FROM_CONFIG__" else enabled_override
enabled_norm = str(enabled_sweeps).replace(",", " ").split()
def is_enabled(key):
    return str(enabled_sweeps) == "all" or key in enabled_norm
i = 0
for key, spec in cfg.get("sweeps", {}).items():
    if not is_enabled(key):
        continue
    for v, s in zip(spec.get("values", []), spec.get("seeds", [])):
        print(f"  [{i}] {spec['param']}={v}  seed={s}  ({spec['method']})")
        i += 1
PY

echo ""
echo "Submitting: sbatch --array=0-${LAST_IDX}%${MAX_PARALLEL} ${SCRIPT_DIR}/mbppplus.sbatch"

sbatch \
    --array="0-${LAST_IDX}%${MAX_PARALLEL}" \
    --export="ALL,ENABLED_SWEEPS=${ENABLED_SWEEPS}" \
    "${SCRIPT_DIR}/mbppplus.sbatch"
