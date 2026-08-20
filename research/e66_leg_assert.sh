#!/usr/bin/env bash
# E66 rung 1: run the campaign worker content assertion around ONE timed leg.
#
#   research/e66_leg_assert.sh PHASE LEG_INDEX
#
# PHASE is `before` or `after`. The script reads the arm out of the twin, builds
# the assignment's exact --require / --forbid table for that arm, and calls
# senpai/rebuild-and-assert-worker.sh with --no-build so the assertion never
# itself changes the binary it is certifying.
#
# It writes key=value lines to stdout for meta.txt and its verdict to stderr.
# The `after` phase also compares its own worker_mtime and worker_sha256 with
# the values the `before` phase recorded for the same leg, and fails when either
# moved. A worker that changed under a running leg invalidates that leg
# (ledger 202(H)).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

phase="${1:?usage: research/e66_leg_assert.sh PHASE LEG_INDEX}"
leg="${2:?usage: research/e66_leg_assert.sh PHASE LEG_INDEX}"
state_dir="${E66_LEG_ASSERT_STATE:-${repo_root}/.mlxfast-private/e66/leg-assert}"
twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

fail() { echo "e66-leg-assert: $*" >&2; exit 1; }

case "${phase}" in before|after) ;; *) fail "unknown phase '${phase}'" ;; esac
[[ -r "${twin}" ]] || fail "cannot read ${twin}"
mkdir -p "${state_dir}"

read_na() {
  local m="$1" na
  na="$(sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, ${m}, \([0-9]\{1,\}\), true>.*/\1/p" "${twin}")"
  [[ -n "${na}" ]] || fail "could not read the M=${m} NA from ${twin}"
  echo "${na}"
}

na5="$(read_na 5)" || exit 1
na6="$(read_na 6)" || exit 1

# The assignment's table names exactly two candidate forms per cell: M=5 is
# either <T,5,3> or <T,5,5>, M=6 is either <T,6,3> or <T,6,6>. Require what this
# arm holds and forbid the other member of each pair.
other5=$((na5 == 3 ? 5 : 3))
other6=$((na6 == 3 ? 6 : 3))
[[ "${na5}" == "3" || "${na5}" == "5" ]] || fail "M=5 NA=${na5} is not an E66 arm"
[[ "${na6}" == "3" || "${na6}" == "6" ]] || fail "M=6 NA=${na6} is not an E66 arm"

out="$(senpai/rebuild-and-assert-worker.sh --no-build \
  --require "<T, 5, ${na5}, true>" \
  --require "<T, 6, ${na6}, true>" \
  --forbid  "<T, 5, ${other5}, true>" \
  --forbid  "<T, 6, ${other6}, true>" 2>&1)"
rc=$?
printf '%s\n' "${out}" >&2

mtime="$(printf '%s\n' "${out}" | sed -n 's/^worker_mtime //p')"
sha="$(printf '%s\n' "${out}" | sed -n 's/^worker_sha256 //p')"

echo "leg${leg}_${phase}_arm_m5_na=${na5}"
echo "leg${leg}_${phase}_arm_m6_na=${na6}"
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
    echo "e66-leg-assert: leg ${leg}: the worker CHANGED during the leg" >&2
    echo "e66-leg-assert:   before mtime=${before_mtime} sha256=${before_sha}" >&2
    echo "e66-leg-assert:   after  mtime=${mtime} sha256=${sha}" >&2
    rc=1
  fi
fi

exit "${rc}"
