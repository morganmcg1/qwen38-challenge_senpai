#!/usr/bin/env bash
# E78: run the campaign worker content assertion around ONE timed leg.
#
#   research/e78_leg_assert.sh PHASE LEG_INDEX
#
# PHASE is `before` or `after`. The script reads this arm's dispatch SET out of
# the twin, builds the matching --require / --forbid table, and calls
# senpai/rebuild-and-assert-worker.sh with --no-build so the assertion never
# itself changes the binary it is certifying.
#
# The `after` phase also compares its own worker_mtime and worker_sha256 with
# the values the `before` phase recorded for the same leg, and fails when either
# moved. A worker that changed under a running leg invalidates that leg
# (ledger 202(H)).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

phase="${1:?usage: research/e78_leg_assert.sh PHASE LEG_INDEX}"
leg="${2:?usage: research/e78_leg_assert.sh PHASE LEG_INDEX}"
state_dir="${E78_LEG_ASSERT_STATE:-${repo_root}/.mlxfast-private/e78/leg-assert}"
twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

fail() { echo "e78-leg-assert: $*" >&2; exit 1; }

case "${phase}" in before|after) ;; *) fail "unknown phase '${phase}'" ;; esac
[[ -r "${twin}" ]] || fail "cannot read ${twin}"
mkdir -p "${state_dir}"

read_ipgs() {
  sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, $1, \([0-9]\{1,\}\), true>.*/\1/p" \
    "${twin}" | sort -u | tr '\n' ' ' | sed 's/ $//'
}

args=()
record=""
for m in 5 6 9; do
  ipgs="$(read_ipgs "${m}")"
  [[ -n "${ipgs}" ]] || fail "no M=${m} wide dispatch found in ${twin}"
  for ipg in ${ipgs}; do
    args+=(--require "<T, ${m}, ${ipg}, true>")
  done
  # Forbid every other member of the legal set at this width, so a stale binary
  # from another arm cannot pass.
  for other in 3 4 5 6; do
    [[ " ${ipgs} " == *" ${other} "* ]] && continue
    args+=(--forbid "<T, ${m}, ${other}, true>")
  done
  record+="leg${leg}_${phase}_arm_m${m}_ipgs=${ipgs// /,}"$'\n'
done

out="$(senpai/rebuild-and-assert-worker.sh --no-build "${args[@]}" 2>&1)"
rc=$?
printf '%s\n' "${out}" >&2

mtime="$(printf '%s\n' "${out}" | sed -n 's/^worker_mtime //p')"
sha="$(printf '%s\n' "${out}" | sed -n 's/^worker_sha256 //p')"

printf '%s' "${record}"
echo "leg${leg}_${phase}_worker_mtime=${mtime}"
echo "leg${leg}_${phase}_worker_sha256=${sha}"
echo "leg${leg}_${phase}_rebuild_assert_exit=${rc}"
echo "leg${leg}_${phase}_metallib_source_fingerprint=$(
  tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"

stamp="${state_dir}/leg-${leg}.before"
if [[ "${phase}" == "before" ]]; then
  printf '%s\n%s\n' "${mtime}" "${sha}" > "${stamp}"
else
  [[ -r "${stamp}" ]] || fail "leg ${leg}: no 'before' record at ${stamp}"
  before_mtime="$(sed -n 1p "${stamp}")"
  before_sha="$(sed -n 2p "${stamp}")"
  echo "leg${leg}_worker_unchanged_across_leg=$(
    [[ "${before_mtime}" == "${mtime}" && "${before_sha}" == "${sha}" ]] \
      && echo true || echo false)"
  if [[ "${before_mtime}" != "${mtime}" || "${before_sha}" != "${sha}" ]]; then
    echo "e78-leg-assert: leg ${leg}: the worker CHANGED during the leg" >&2
    echo "e78-leg-assert:   before mtime=${before_mtime} sha256=${before_sha}" >&2
    echo "e78-leg-assert:   after  mtime=${mtime} sha256=${sha}" >&2
    rc=1
  fi
fi

exit "${rc}"
