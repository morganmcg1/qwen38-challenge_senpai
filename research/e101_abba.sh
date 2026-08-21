#!/usr/bin/env bash
# E101 rung 4: the fused arm C row top-32 against the shipped argPartition
# chain, counterbalanced inside ONE session.
#
#   usage: research/e101_abba.sh [TOKENS] [LABEL] [MODE]
#
#   MODE=prod      9 production legs, order b k b k b k b k b.
#   MODE=synchead  7 legs with MLX_QWEN_MTP_TRACE_SYNC_HEAD=1, same order.
#   MODE=both      the synchead block first, then the prod block (default).
#
# The two arms are the SAME worker binary under one environment variable, so
# no leg rebuilds and no leg commits. `MLX_E101_ROW_TOP32=0` restores the
# argPartition selection and its five index-arithmetic buffers bit for bit.
#
# Base legs sit at the odd positions and kernel legs at the even ones, so both
# arms have the same mean position and any monotone linear thermal drift
# cancels exactly to first order. The base legs also give the session drift
# slope and the session null.
#
# WHY THE SYNCHEAD BLOCK EXISTS. `d_chain_us` is the interval between two
# `DispatchTime` reads that bracket only lazy MLX graph construction
# (Qwen36MTPBlockSession.swift:1403-1412), so it is host encode time and holds
# no GPU execution. In production the head chain runs asynchronously under the
# verify-build window, where its cost is inseparable from host build cost.
# `MLX_QWEN_MTP_TRACE_SYNC_HEAD=1` drains the chain before that window, which
# moves head-chain GPU execution into `d_submit2_us`. That counter is the
# direct production-dispatch measurement of the GPU time the head chain
# spends, and the arm C selection kernels are inside it.
#
# The census pair e101c-off / e101c-on remains the kernel-level execution
# proof. Timing runs with the cool gate OFF, which `research/e79_trace_leg.sh`
# records verbatim as cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false. These legs are counterbalanced local
# evidence, never a gated or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-r5}"
mode="${3:-both}"

case "${mode}" in
  prod|synchead|both) ;;
  *) echo "e101_abba: unknown mode ${mode}" >&2; exit 2 ;;
esac

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e101_abba: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker_start="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
failures=0

run_block() {
  local block_label="$1" sync_head="$2"
  shift 2
  local position=0 arm flag suffix tag status worker_now
  for arm in "$@"; do
    position=$((position + 1))
    if [[ "${arm}" == "on" ]]; then
      flag=1
      suffix=k
    else
      flag=0
      suffix=b
    fi
    tag="e101${block_label}p${position}${suffix}"
    echo "=== ${tag}: arm=${arm} MLX_E101_ROW_TOP32=${flag}" \
      "sync_head=${sync_head} tokens=${tokens} ==="

    if ((sync_head)); then
      MLX_E101_ROW_TOP32="${flag}" \
        research/e79_trace_leg.sh "${tag}" "${tokens}" --sync-head
    else
      MLX_E101_ROW_TOP32="${flag}" \
        research/e79_trace_leg.sh "${tag}" "${tokens}"
    fi
    status=$?

    worker_now="$(
      shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
        | awk '{print $1}')"
    {
      echo "experiment=e101-row-top32-abba"
      echo "e101_arm=${arm}"
      echo "e101_position=${position}"
      echo "e101_block=${block_label}"
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
}

if [[ "${mode}" == "synchead" || "${mode}" == "both" ]]; then
  run_block "${label}s" 1 off on off on off on off
fi
if [[ "${mode}" == "prod" || "${mode}" == "both" ]]; then
  run_block "${label}" 0 off on off on off on off on off
fi

echo "e101_abba: ${failures} failed legs"
exit "${failures}"
