#!/usr/bin/env bash
# E54 bit-parity gate: prove every timed NA=5 arm is bit-identical to its own
# shipped-IPG partner, and prove the check has power by requiring the
# lane-perturbation arm to diverge.
#
#   research/e54_parity.sh
#
# The wrapper exists for two reasons. run_job takes an argv list with no
# environment field, and the parity driver reads the shared local run lock
# directory from the environment. The driver also compares every arm against
# its FIRST argument, but E54's meaningful comparisons are within-pair, so the
# within-pair orderings are replayed here from the persisted digests.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QMV_PARITY_DIR="${repo_root}/.mlxfast-private/e54-parity"
base_sha="${E49_BASE_SHA:-a35bb006fd47785dc916241df63ec8780bda8e5c}"
py="${MLXFAST_PYTHON_BIN:-python3}"

arms=(
  iso_m5_ipg3 iso_m5_ipg5 iso_m5_ipg5_lane_perturb
  iso_m7_ipg4 iso_m7_ipg5
  iso_m8_ipg4 iso_m8_ipg5
  shipped e27_full
)

specs=()
for arm in "${arms[@]}"; do
  if [[ "${arm}" == shipped ]]; then
    specs+=("${arm}=${base_sha}")      # unpatched: no `+patch` suffix
  else
    specs+=("${arm}=${base_sha}+e54:${arm}")
  fi
done

date -u "+e54_parity: === %Y-%m-%dT%H:%M:%SZ digesting ${#specs[@]} arms ==="
research/run-qmv-parity.sh "${specs[@]}"

# Within-pair verdicts. Cross-pair rows differ by construction, because each
# isolated arm keeps a different `case` in the wide tier, so only these
# orderings carry a correctness claim.
#
# Only the verdict text is written under research/. The per-arm digest files
# stay in .mlxfast-private, which is gitignored: they are half a megabyte of
# hashes, and the comparator can be re-run over them at any time with no GPU.
out="research/e54-artifacts"
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
compare "P1 <T,5,3> vs <T,5,5>" BIT-IDENTICAL \
  "${d}/iso_m5_ipg3.json" "${d}/iso_m5_ipg5.json"
compare "P2 <T,7,4> vs <T,7,5>" BIT-IDENTICAL \
  "${d}/iso_m7_ipg4.json" "${d}/iso_m7_ipg5.json"
compare "P3 <T,8,4> vs <T,8,5>" BIT-IDENTICAL \
  "${d}/iso_m8_ipg4.json" "${d}/iso_m8_ipg5.json"
compare "P4 shipped vs e27_full" BIT-IDENTICAL \
  "${d}/shipped.json" "${d}/e27_full.json"
compare "POSITIVE CONTROL <T,5,5> vs lane 3<->4" "DIVERGES at M=5 only" \
  "${d}/iso_m5_ipg5.json" "${d}/iso_m5_ipg5_lane_perturb.json"

"${py}" research/e54_routing.py --parity-dir "${d}" \
  --out "${out}/parity-routing.md"

date -u "+e54_parity: === %Y-%m-%dT%H:%M:%SZ parity complete ==="
echo "e54_parity: within-pair verdicts written to ${verdicts}"
