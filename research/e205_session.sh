#!/usr/bin/env bash
# E205 stage-A session: price a rejection round at the round endpoint.
#
#   usage: research/e205_session.sh TAG_PREFIX TOKENS
#
# Apply `research/e205_trunc_patch.sh apply` and commit it before running this
# script; the leg wrapper records `base_sha` and the worker digest, so the
# measured build must be reachable from a commit.
#
# THREE LEGS, ONE J EACH, ARMS ALTERNATING BY ROUND PARITY INSIDE THE PROCESS
# (RULE 388). One leg cannot carry all three j values and still leave each of
# them enough paired rounds: a 256-token leg runs ~40-55 rounds, so a
# three-way split would leave ~6 truncated rounds per j, well short of the
# power needed at the 0.2 ms/round composition bar. Parity alternation gives
# each j about half the leg's rounds against its own in-process baseline.
#
# The j order is 4, 1, 2 on purpose. Every premium is decided WITHIN its leg,
# so leg-level drift cannot create one; the non-monotone order additionally
# stops a monotone session drift from imitating a j-scaling law in the free
# cross-leg observation.
#
# The worker digest is asserted before the first leg and compared after the
# last one, so a leg that timed another build is visible.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e205_session.sh TAG_PREFIX TOKENS}"
tokens="${2:?usage: e205_session.sh TAG_PREFIX TOKENS}"

senpai/rebuild-and-assert-worker.sh --no-build \
  --require "e205_trunc_applied=" \
  --require "repair_prefetch_us=" \
  --require "clear_release_cpu_us=" \
  --require "repair_replay_us=" \
  --require "MLX_E205_TRUNCATION" || exit 1

worker_before="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"

status=0

run_leg() {
  local tag="$1"; shift
  echo "== e205 stage A leg ${tag}: $* =="
  ( export "$@" && research/e90_leg.sh "${tag}" "${tokens}" )
  local leg_status=$?
  {
    echo "experiment=e205-rejection-round-cost"
    echo "e205_stage=A"
    echo "e205_arm_env=$*"
    echo "e205_instrument=forced-truncation+repair-timers+cpu-clock"
    echo "e205_worker_sha256_session=${worker_before}"
  } >> "research/out/${tag}/meta.txt"
  status=$(( status | leg_status ))
  return "${leg_status}"
}

run_leg "${prefix}-alt4" MLX_E205_TRUNCATION=alt4
run_leg "${prefix}-alt1" MLX_E205_TRUNCATION=alt1
run_leg "${prefix}-alt2" MLX_E205_TRUNCATION=alt2

worker_after="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"
echo "worker_sha256_before ${worker_before}"
echo "worker_sha256_after  ${worker_after}"
if [[ "${worker_before}" != "${worker_after}" ]]; then
  echo "e205_session: FATAL worker changed between legs" >&2
  exit 1
fi

echo "e205_session: exit ${status}"
exit "${status}"
