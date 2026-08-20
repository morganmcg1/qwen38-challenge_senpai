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

base_sha="${E59_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
out_dir="research/e59-artifacts"
log_dir="${out_dir}/gate-logs"
mkdir -p "${log_dir}"

# The scored-surface gate reads two objects that exist only in the organizer
# repository: the scored submission commit and the promoted frontier tip. Our
# `origin` fork answers `not our ref` for both, and this checkout has no
# `upstream` remote, so fetch them by URL into refs we own. Fetch only; nothing
# here merges organizer work into the experiment.
organizer_url="${E59_ORGANIZER_URL:-https://github.com/Layr-Labs/qwen-3.8-mtp-challenge}"
scored_commit="2b0c36a078b7660c9215adee933336ff46da25af"
GIT_TERMINAL_PROMPT=0 git fetch "${organizer_url}" \
  "+${scored_commit}:refs/e59/scored-commit" "+main:refs/e59/frontier-main" \
  > "${log_dir}/organizer-fetch.log" 2>&1
export SCORED_GATE_FRONTIER_REF="${SCORED_GATE_FRONTIER_REF:-refs/e59/frontier-main}"

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

NAMES="${names[*]}" CODES="${codes[*]}" python3 - "${out_dir}" "${log_dir}" "${commands[@]}" <<'PY'
import json, os, pathlib, sys

out_dir = pathlib.Path(sys.argv[1])
log_dir = pathlib.Path(sys.argv[2])
commands = sys.argv[3:]
names = os.environ["NAMES"].split()
codes = [int(c) for c in os.environ["CODES"].split()]


def failure_note(name):
    log = log_dir / f"{name}.log"
    if not log.exists():
        return ""
    lines = [l.strip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
    for line in lines:
        if line.startswith("FAIL"):
            return line
    return lines[-1] if lines else ""


gates = [
    {"gate": n, "command": c, "exit_code": rc, "passed": rc == 0,
     "note": "" if rc == 0 else failure_note(n)}
    for n, c, rc in zip(names, commands, codes)
]
payload = {"gates": gates, "all_passed": all(g["passed"] for g in gates)}
path = out_dir / "e59-gates.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
print("wrote %s  all_passed=%s" % (path, payload["all_passed"]))
sys.exit(0 if payload["all_passed"] else 1)
PY
