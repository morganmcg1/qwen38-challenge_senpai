#!/usr/bin/env bash
# E192 stage 0 session: build the diagnostic worker once, then run the two
# in-process arm-switch legs that decide the account behind FINDING 495.
#
#   usage: research/e192_stage0_session.sh TAG_PREFIX TOKENS
#
# Apply `research/e192_stage0_patch.sh apply` and commit it before running this
# script; the leg wrapper records `base_sha` and the worker digest, so the
# measured build must be reachable from a commit.
#
#   leg 1  MLX_E192_BARRIER=alt   in-flight account vs allocator account
#   leg 2  MLX_E192_TAPE=alt      relocation account vs removal account
#
# The worker digest is asserted before the first leg and compared after the
# last one, so a leg that timed another build is visible.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e192_stage0_session.sh TAG_PREFIX TOKENS}"
tokens="${2:?usage: e192_stage0_session.sh TAG_PREFIX TOKENS}"

senpai/rebuild-and-assert-worker.sh \
  --require "clear_release_cpu_us=" \
  --require "e192_tape_suppressed_arm=" \
  --require "clear_barrier_us=" \
  --require "MLX_E192_BARRIER" || exit 1

worker_before="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"

run_leg() {
  local tag="$1"; shift
  echo "== e192 stage 0 leg ${tag}: $* =="
  ( export "$@" && research/e90_leg.sh "${tag}" "${tokens}" )
  local status=$?
  {
    echo "experiment=e192-persistent-rollback-slot"
    echo "e192_stage=0"
    echo "e192_arm_env=$*"
    echo "e192_instrument=clear-timer+cpu-clock+barrier-arm+tape-arm"
    echo "e192_worker_sha256_session=${worker_before}"
  } >> "research/out/${tag}/meta.txt"
  return "${status}"
}

run_leg "${prefix}-barrier-alt" MLX_E192_BARRIER=alt MLX_E192_TAPE=on
barrier_status=$?

run_leg "${prefix}-tape-alt" MLX_E192_BARRIER=0 MLX_E192_TAPE=alt
tape_status=$?

worker_after="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"
echo "worker_sha256_before ${worker_before}"
echo "worker_sha256_after  ${worker_after}"
if [[ "${worker_before}" != "${worker_after}" ]]; then
  echo "e192_stage0_session: FATAL worker changed between legs" >&2
  exit 1
fi

echo "e192_stage0_session: barrier leg exit ${barrier_status}," \
     "tape leg exit ${tape_status}"
exit $(( barrier_status | tape_status ))
