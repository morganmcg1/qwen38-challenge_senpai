#!/usr/bin/env bash
# E107 rung 0/1: one isolated-dose session of the affine-2 draft-readout arms.
#
# The session is self-counterbalancing: every block runs the arms forward and
# then backward in one palindrome, so monotone thermal drift cancels to first
# order inside each block. No thermal gate applies to an isolated Metal
# harness, so entry and exit GPU temperature are recorded and the result is
# marked ungated.
#
# usage: research/e107_iso_leg.sh TAG [extra harness args...]
set -uo pipefail

tag="${1:?usage: e107_iso_leg.sh TAG [args...]}"
shift || true

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

out="research/out/${tag}"
mkdir -p "${out}"

bin="/tmp/e107/e107_affine2_ab"
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${bin}" research/e107_affine2_ab.m || exit 1

gpu_temp() {
  local macmon
  macmon="$(command -v macmon || true)"
  [ -n "${macmon}" ] || { echo ""; return; }
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
}

entry_c="$(gpu_temp)"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${bin}" --out "${out}/arms.json" "$@" \
  >"${out}/harness.out" 2>"${out}/harness.err"
rc=$?

exit_c="$(gpu_temp)"
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "tag=${tag}"
  echo "rc=${rc}"
  echo "started_utc=${started}"
  echo "finished_utc=${finished}"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "order=palindrome_within_block"
  echo "host_model=$(sysctl -n hw.model)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "os_build=$(sw_vers -buildVersion)"
  echo "metal_version=$(xcrun metal --version 2>&1 | head -1)"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "worktree_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "args=$*"
} > "${out}/meta.txt"

tail -40 "${out}/harness.err"
cat "${out}/meta.txt"
exit "${rc}"
