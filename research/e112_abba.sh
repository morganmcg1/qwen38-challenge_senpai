#!/usr/bin/env bash
# E112 rung 1: keep or delete the kL=1025 SDPA compile warm, counterbalanced
# inside ONE session.
#
#   usage: research/e112_abba.sh [TOKENS] [LABEL] [ORDER...]
#
#   TOKENS  decode tokens per leg (default 512). 512 is the only representative
#           window: the seed is 512 tokens, so a shorter leg never walks the
#           key length to 1024 and can never exercise the boundary this warm
#           protects.
#   ORDER   leg arms, default `off on off on off on off`.
#
# ARMS. Both arms are the SAME worker binary under one environment variable.
#   off  MLX_E112_SKIP_1025_WARM=0, the shipped base: the warm compiles the
#        blocks=128 `sdpa_vector_2pass` pipelines at kL=1025.
#   on   MLX_E112_SKIP_1025_WARM=1, the candidate: that family is not warmed.
#
# The warm is UNTIMED. `QwenRuntimeMTPDriver.swift:84` sends `mtp_decode_warm`
# and only line 94 starts the decode clock, so this arm cannot move wall clock
# out of the scored window. Its one causal path into the timed window is the
# pipeline that `sdpa_vector_2pass` selects through Metal function constant 26
# (`blocks`), which is part of `hash_name`, so blocks=64 and blocks=128 are
# separate pipeline objects. `blocks` jumps 64 -> 128 at N > 1024 only when the
# GPU architecture string ends in `s`; local is applegpu_g16s and the ranked M5
# is applegpu_g17s, so the boundary is live on both hosts.
#
# Base legs sit at the odd positions and candidate legs at the even ones, so
# both arms have the same mean position and monotone thermal drift cancels to
# first order. Timing runs with the cool gate OFF, which
# `research/e79_trace_leg.sh` records verbatim as
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false. These
# legs are counterbalanced local evidence, never a gated or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-r1}"
(($# > 2)) && order=("${@:3}") || order=(off on off on off on off)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e112_abba: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker_start="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
failures=0
position=0

for arm in "${order[@]}"; do
  position=$((position + 1))
  case "${arm}" in
    on) flag=1; suffix=k ;;
    off) flag=0; suffix=b ;;
    *) echo "e112_abba: unknown arm ${arm}" >&2; exit 2 ;;
  esac
  tag="e112${label}p${position}${suffix}"
  echo "=== ${tag}: arm=${arm} MLX_E112_SKIP_1025_WARM=${flag}" \
    "tokens=${tokens} ==="

  MLX_E112_SKIP_1025_WARM="${flag}" \
    research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?

  worker_now="$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
      | awk '{print $1}')"
  witness="$(
    grep -c '^mtp-trace: e112 ' "research/out/${tag}/trace.txt" 2>/dev/null \
      || echo 0)"
  witness_flag="$(
    sed -n 's/^mtp-trace: e112 skip_1025_warm=\([01]\).*/\1/p' \
      "research/out/${tag}/trace.txt" 2>/dev/null | sort -u | tr '\n' ',')"
  {
    echo "experiment=e112-kl1025-warm-abba"
    echo "e112_arm=${arm}"
    echo "e112_position=${position}"
    echo "MLX_E112_SKIP_1025_WARM=${flag}"
    echo "e112_warm_witness_lines=${witness}"
    echo "e112_warm_witness_flag=${witness_flag}"
    echo "session_commit=${session_commit}"
    echo "worker_sha256_session_start=${worker_start}"
    echo "worker_sha256_after_leg=${worker_now}"
  } >> "research/out/${tag}/meta.txt"

  if [[ "${worker_now}" != "${worker_start}" ]]; then
    echo "e112_abba: ${tag} ran a worker that moved during the session" >&2
    status=7
  fi
  # The arm is worthless without proof the worker actually read the switch.
  if [[ "${witness_flag}" != "${flag}," ]]; then
    echo "e112_abba: ${tag} witnessed skip_1025_warm=${witness_flag}" \
      "but the arm asked for ${flag}" >&2
    status=8
  fi
  if ((status != 0)); then
    echo "e112_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
  echo "status=${status}" >> "research/out/${tag}/meta.txt"
  # A broken first leg means every later leg is wasted GPU time.
  if ((failures == 1 && position == 1)); then
    echo "e112_abba: first leg failed; aborting the session" >&2
    break
  fi
done

echo "e112_abba: ${failures} failed legs"
exit "${failures}"
