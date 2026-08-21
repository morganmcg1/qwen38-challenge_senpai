#!/usr/bin/env bash
# E101 rung 4: the fused arm C row top-32 against the shipped argPartition
# chain, counterbalanced inside ONE session.
#
#   usage: research/e101_abba.sh [TOKENS] [LABEL]
#
# The two arms are the SAME worker binary under one environment variable, so
# no leg rebuilds and no leg commits. `MLX_E101_ROW_TOP32=0` restores the
# argPartition selection and its five index-arithmetic buffers bit for bit.
#
# The order is b k b k b k b. The four base legs sit at positions 1, 3, 5, 7
# and the three kernel legs at 2, 4, 6, so both arms have mean position 4 and
# any monotone linear thermal drift cancels exactly to first order. The four
# base legs also give the session drift slope and the session null.
#
# The arm witness is `d_chain_us` in the trace. The fused arm encodes 2 MLX
# ops where the base arm encodes the argPartition plus five elementary index
# ops, so a leg whose chain-build time does not move with its flag did not
# run the arm its tag claims. The census pair e101c-off / e101c-on is the
# kernel-level execution proof behind that witness.
#
# Timing runs with the cool gate OFF, which `research/e79_trace_leg.sh`
# records verbatim as cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false. These legs are counterbalanced local
# evidence, never a gated or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-r4}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e101_abba: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker_start="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

failures=0
position=0
for arm in off on off on off on off; do
  position=$((position + 1))
  if [[ "${arm}" == "on" ]]; then
    flag=1
    suffix=k
  else
    flag=0
    suffix=b
  fi
  tag="e101${label}p${position}${suffix}"
  echo "=== ${tag}: arm=${arm} MLX_E101_ROW_TOP32=${flag} tokens=${tokens} ==="

  MLX_E101_ROW_TOP32="${flag}" research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?

  worker_now="$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  {
    echo "experiment=e101-row-top32-abba"
    echo "e101_arm=${arm}"
    echo "e101_position=${position}"
    echo "MLX_E101_ROW_TOP32=${flag}"
    echo "session_commit=${session_commit}"
    echo "worker_sha256_session_start=${worker_start}"
    echo "worker_sha256_after_leg=${worker_now}"
  } >> "research/out/${tag}/meta.txt"

  if [[ "${worker_now}" != "${worker_start}" ]]; then
    echo "e101_abba: ${tag} ran a worker that moved during the session" >&2
    status=7
  fi
  if ((status != 0)); then
    echo "e101_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
  echo "status=${status}" >> "research/out/${tag}/meta.txt"
done

echo "e101_abba: ${failures} failed legs"
exit "${failures}"
