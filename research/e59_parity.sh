#!/usr/bin/env bash
# E59 rung 4: prove both timed routes are bit-identical to the unchanged base
# at every width the scored session can reach, and prove the check has power by
# requiring three defect controls to fail.
#
#   research/e59_parity.sh
#
# The wrapper exists because run_job takes an argv list with no environment
# field, and the parity driver reads the shared local run lock directory from
# the environment.
#
# Why `t55` needs this gate more than the row-block routes do. `m5_rbx` holds
# NA fixed at the value the base already compiles, so its answer is exact by
# construction. `t55` replaces the two accumulator groups {3, 2} with one group
# of 5, which changes which input rows share an accumulator. E55 measured that
# change to be benign at `case 9`. One call site is evidence, not a law, so
# `t55` must clear this gate BEFORE it is timed.
#
# Reachability note. The parity sweep covers widths 1..12, but the scored
# session verifies at most one primary token plus eight drafts, so M <= 9.
#
# Base-move note. E55 shipped `<T,9,5>`, so NA == 5 now exists at M=9 in the
# base itself. `perturb_lanes` is gated on `NA == 5`, so a lane control built on
# `t55` must diverge at M=5 AND at M=9. On the pre-E55 base the same control
# diverged at M=5 alone. The M=9 divergence is independent confirmation that
# E55's shipped M=9 cell really is an NA=5 mapping.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QMV_PARITY_DIR="${repo_root}/.mlxfast-private/e59-parity-rung4"
base_sha="${E59_BASE_SHA:-$(git -C "${repo_root}" rev-parse HEAD)}"
py="${MLXFAST_PYTHON_BIN:-python3}"

arms=(
  shipped
  t55
  m5_rbx
  t55_lane_perturb
  t55_row_drop
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
verdicts="${out}/parity-verdicts-rung4.txt"
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
compare "CANDIDATE base vs t55" "BIT-IDENTICAL at every width" \
  "${d}/shipped.json" "${d}/t55.json"
compare "CANDIDATE base vs m5_rbx" "BIT-IDENTICAL at every width" \
  "${d}/shipped.json" "${d}/m5_rbx.json"
compare "POSITIVE CONTROL t55 vs lanes 3<->4" "DIVERGES at M=5 and M=9" \
  "${d}/t55.json" "${d}/t55_lane_perturb.json"
compare "POSITIVE CONTROL t55 vs four written rows" "DIVERGES at M=5 only" \
  "${d}/t55.json" "${d}/t55_row_drop.json"
compare "POSITIVE CONTROL m5_rbx vs one x-group" "DIVERGES at M=5 only" \
  "${d}/m5_rbx.json" "${d}/m5_rbx_coverage_drop.json"

"${py}" research/e59_parity_verdict.py --parity-dir "${d}" \
  --out "${out}/e59-parity-rung4.json"

date -u "+e59_parity: === %Y-%m-%dT%H:%M:%SZ parity complete ==="
echo "e59_parity: verdicts written to ${verdicts}"
