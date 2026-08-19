#!/usr/bin/env bash
# E59 rung 2: prove both row-block routes are bit-identical to the unchanged
# base at every width the scored session can reach, and prove the check has
# power by requiring two coverage-defect controls and one lane control to fail.
#
#   research/e59_parity.sh
#
# The wrapper exists because run_job takes an argv list with no environment
# field, and the parity driver reads the shared local run lock directory from
# the environment.
#
# Reachability note. The parity sweep covers widths 1..12, but the scored
# session verifies at most one primary token plus eight drafts, so M <= 9.
# `ceil_only` carries an unreachable `case 10:` on purpose, so it MUST diverge
# at M=10 (the base falls through to `qmv_fast_impl` there) and must not move
# any width at or below 9. That is a routing check, not a correctness failure.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QMV_PARITY_DIR="${repo_root}/.mlxfast-private/e59-parity"
base_sha="${E59_BASE_SHA:-989596895b7c8f889443dac0c87e024a428e6e9e}"
py="${MLXFAST_PYTHON_BIN:-python3}"

arms=(
  shipped
  m5_rb2
  m5_rbx
  ceil_only
  m5_rb2_lane_perturb
  m5_rb2_coverage_drop
  m5_rbx_coverage_drop
)

specs=()
for arm in "${arms[@]}"; do
  if [[ "${arm}" == shipped ]]; then
    specs+=("${arm}=${base_sha}")      # unpatched: no `+patch` suffix
  else
    specs+=("${arm}=${base_sha}+e59:${arm}")
  fi
done

date -u "+e59_parity: === %Y-%m-%dT%H:%M:%SZ digesting ${#specs[@]} arms ==="
research/run-qmv-parity.sh "${specs[@]}"

out="research/e59-artifacts"
mkdir -p "${out}"
verdicts="${out}/parity-verdicts.txt"
: > "${verdicts}"

compare() {
  local label="$1"; shift
  local expect="$1"; shift
  {
    echo "### ${label}  (expected: ${expect})"
    "${py}" research/qmv_parity_compare.py "$@"
    echo
  } | tee -a "${verdicts}"
}

d="${MLXFAST_QMV_PARITY_DIR}"
compare "CANDIDATE base vs m5_rb2" "BIT-IDENTICAL at every width" \
  "${d}/shipped.json" "${d}/m5_rb2.json"
compare "CANDIDATE base vs m5_rbx" "BIT-IDENTICAL at every width" \
  "${d}/shipped.json" "${d}/m5_rbx.json"
compare "CONTROL ARM base vs ceil_only" "identical at M<=9, DIVERGES at M=10" \
  "${d}/shipped.json" "${d}/ceil_only.json"
compare "POSITIVE CONTROL m5_rb2 vs lanes 3<->4" "DIVERGES at M=5 only" \
  "${d}/m5_rb2.json" "${d}/m5_rb2_lane_perturb.json"
compare "POSITIVE CONTROL m5_rb2 vs one row block" "DIVERGES at M=5 only" \
  "${d}/m5_rb2.json" "${d}/m5_rb2_coverage_drop.json"
compare "POSITIVE CONTROL m5_rbx vs one x-group" "DIVERGES at M=5 only" \
  "${d}/m5_rbx.json" "${d}/m5_rbx_coverage_drop.json"

"${py}" research/e59_parity_verdict.py --parity-dir "${d}" \
  --out "${out}/e59-parity.json"

date -u "+e59_parity: === %Y-%m-%dT%H:%M:%SZ parity complete ==="
echo "e59_parity: verdicts written to ${verdicts}"
