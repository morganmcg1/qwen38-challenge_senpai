#!/usr/bin/env bash
# Run the E59 submission gates and record each verdict.
#
#   research/e59_gates.sh
#
# Writes research/e59-artifacts/e59-gates.json. Every gate runs even when an
# earlier one fails, so one pass produces the complete picture instead of the
# first failure only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

base_sha="${E59_BASE_SHA:-989596895b7c8f889443dac0c87e024a428e6e9e}"
out_dir="research/e59-artifacts"
log_dir="${out_dir}/gate-logs"
mkdir -p "${log_dir}"

names=()
commands=()
codes=()

run_gate() {
  local name="$1"; shift
  local log="${log_dir}/${name}.log"
  echo "=== gate ${name}: $* ==="
  "$@" > "${log}" 2>&1
  local rc=$?
  tail -n 20 "${log}"
  echo "--- ${name} exit ${rc} ---"
  names+=("${name}")
  commands+=("$*")
  codes+=("${rc}")
}

run_gate assignment_scope senpai/validate-assignment-scope.sh "${base_sha}" \
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h" \
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
run_gate editable_budget senpai/check-editable-budget.sh "${base_sha}"
run_gate twin_audit python3 research/twin_audit.py
run_gate scored_surface research/scored-surface-gate.sh
run_gate ranked_score_boundary senpai/verify-ranked-score-boundary.sh

NAMES="${names[*]}" CODES="${codes[*]}" python3 - "${out_dir}" "${commands[@]}" <<'PY'
import json, os, pathlib, sys

out_dir = pathlib.Path(sys.argv[1])
commands = sys.argv[2:]
names = os.environ["NAMES"].split()
codes = [int(c) for c in os.environ["CODES"].split()]
gates = [
    {"gate": n, "command": c, "exit_code": rc, "passed": rc == 0}
    for n, c, rc in zip(names, commands, codes)
]
payload = {"gates": gates, "all_passed": all(g["passed"] for g in gates)}
path = out_dir / "e59-gates.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
print("wrote %s  all_passed=%s" % (path, payload["all_passed"]))
sys.exit(0 if payload["all_passed"] else 1)
PY
