#!/usr/bin/env bash
# E155 -- off-side gate probe.
#
#   usage: research/e155_gate_probe.sh [PROMPT_ID]
#
# The audit-off legs ran with neither audit variable set, so they cannot say
# WHICH variable gates the instrument. This leg sets the output path and leaves
# `MLX_E155_RECALL_AUDIT` unset. The audit creates its file at install time,
# before the first round, so the file must still be absent afterwards.
#
# Writes research/e155-artifacts/probe.json.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

id="${1:-beagle_a}"
audit_dir="${E155_AUDIT_DIR:-.mlxfast-private/e155}"
mkdir -p "${audit_dir}" research/e155-artifacts
path="${PWD}/${audit_dir}/${id}-gate-probe.jsonl"
rm -f "${path}"

echo "=== e155: off-side gate probe ${id} (path set, gate unset) ==="
env MLX_E155_RECALL_AUDIT_PATH="${path}" \
    E128_FORCE=1 \
    E128_TOKENS="${E128_TOKENS:-512}" \
    E128_DEPTH="${E128_DEPTH:-8}" \
    E128_RUNS_DIR="${E128_RUNS_DIR:-runs-e155-probe}" \
    research/e128_session.sh "${id}"
rc=$?

created=false
lines=0
if [[ -e "${path}" ]]; then
  created=true
  lines="$(wc -l < "${path}" | tr -d ' ')"
fi

cat > research/e155-artifacts/probe.json <<JSON
{
  "prompt": "${id}",
  "gate_variable": "MLX_E155_RECALL_AUDIT",
  "gate_variable_set": false,
  "path_variable": "MLX_E155_RECALL_AUDIT_PATH",
  "path_variable_set": true,
  "audit_path": "${path}",
  "audit_file_created": ${created},
  "audit_rows_written": ${lines},
  "leg_exit_code": ${rc}
}
JSON

cat research/e155-artifacts/probe.json
[[ "${created}" == "false" && ${rc} -eq 0 ]]
