#!/usr/bin/env bash
# E147 rung A positive control (Rule 101): break the pipelined schedule and
# prove the exactness leg reports it.
#
#   usage: research/e147_rungA_control.sh [MODE ...]      default: all three
#
# A check that has never failed is not a check. Each mode builds a deliberately
# broken worker, runs one 512-token exactness leg against the same
# base-provenance reference rows the ABBA session used, and records whether the
# leg caught it. The tree is restored and the real candidate worker is rebuilt
# before the script exits, whatever the outcome. No broken kernel is ever
# committed and none of these legs is timed.
#
# THE THREE MODES, AND WHY THREE.
#
# `barrier0` suppresses the loop-top barrier on the first iteration only. That
# is the barrier the prologue depends on, because the prologue stages tile 0
# outside the loop. This is the control the brief asked for. It DID NOT FIRE:
# the broken build still matched bit-exactly over 512 tokens at
# 2026-08-23T06:18Z. That is a real result and it is kept in the table rather
# than dropped. The likely reason is that the pipelined iteration issues the
# next tile's device load and dequantize pass BETWEEN the suppressed barrier
# and the mma, so hundreds of cycles of memory latency sit inside the race
# window and hide it. A race that does not manifest on one host on one prompt
# is not evidence of correctness, which is exactly why the other modes exist.
#
# `nobarrier` removes the loop-top barrier on every iteration. It widens the
# same race from the first tile to every tile.
#
# `wronghalf` makes the mma read the staging half the loader is currently
# filling instead of the half the loader finished. This is deterministic rather
# than racy, and it is the failure mode double buffering actually introduces:
# an off-by-one in the alternation. If the exactness leg catches this, the leg
# can detect a wrong staging buffer, which is the property the gate needs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

header=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
twin=Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
prompt_id="${E147_CONTROL_PROMPT:-beagle_a}"
report=research/e147-rungA-control.txt

modes=("$@")
((${#modes[@]})) || modes=(barrier0 nobarrier wronghalf)

needle_for() {
  case "$1" in
    barrier0)  echo 'if (k > 0) { threadgroup_barrier(mem_flags::mem_threadgroup); }' ;;
    nobarrier) echo '/* e147 control: loop-top barrier deliberately removed */' ;;
    wronghalf) echo 'mma_op.mma(Xs + (cur ^ 1) * Xs_tile, Ws + (cur ^ 1) * Ws_tile);' ;;
  esac
}

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e147_control: scored surface is dirty; refusing to start" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

restore() {
  git checkout "${session_commit}" -- "${header}" "${twin}"
}

finish() {
  restore
  echo "=== e147_control: rebuilding the real candidate worker ==="
  senpai/rebuild-and-assert-worker.sh \
    --require 'mma_op.mma(Xs + cur * Xs_tile, Ws + cur * Ws_tile);' \
    --forbid 'mma_op.mma(Xs + (cur ^ 1) * Xs_tile, Ws + (cur ^ 1) * Ws_tile);'
}
trap finish EXIT

{
  echo "e147_control_session_commit=${session_commit}"
  echo "e147_control_prompt=${prompt_id}"
} > "${report}"

any_caught=0
for mode in "${modes[@]}"; do
  echo "=== e147_control ${mode}: editing the kernel ==="
  python3 research/e147_break_kernel.py "${header}" "${mode}" || exit 1
  research/e147_port_twin.py "${header}" "${twin}" "${session_commit}" || exit 1

  if ! senpai/rebuild-and-assert-worker.sh --require "$(needle_for "${mode}")"; then
    echo "e147_control ${mode}: the broken worker did not build" >&2
    exit 3
  fi

  out=".mlxfast-private/e147/runs/control-${mode}/${prompt_id}"
  echo "=== e147_control ${mode}: one 512-token exactness leg ==="
  E128_FORCE=1 \
  E128_NO_TRACE=1 \
  E128_TOKENS=512 \
  E128_DEPTH=8 \
  E128_ROOT=".mlxfast-private/e147" \
  E128_GOLDENS_DIR="${E147_GOLDENS_DIR:-.mlxfast-private/e128/goldens}" \
  E128_RUNS_DIR="runs/control-${mode}" \
    research/e128_session.sh "${prompt_id}"
  leg_status=$?

  caught=1
  matched="none"
  divergences="none"
  if [[ -s "${out}/report.json" ]]; then
    matched="$(jq -r '.all_tokens_matched' "${out}/report.json")"
    divergences="$(jq -r '.residual_divergence_count' "${out}/report.json")"
    if [[ "${matched}" == "true" && "${divergences}" == "0" ]]; then
      caught=0
    fi
  fi
  ((caught)) && any_caught=1

  {
    echo "e147_control_${mode}_leg_exit=${leg_status}"
    echo "e147_control_${mode}_all_tokens_matched=${matched}"
    echo "e147_control_${mode}_residual_divergence_count=${divergences}"
    echo "e147_control_${mode}_caught=${caught}"
  } >> "${report}"

  restore
done

echo "e147_rungA_positive_control_failed=$([[ ${any_caught} -eq 1 ]] \
  && echo 1.0 || echo 0.0)" >> "${report}"
cat "${report}"

((any_caught)) || {
  echo "e147_control: no mode was caught; the exactness gate is unproven" >&2
  exit 1; }
echo "e147_control: PASS, at least one broken schedule was caught"
exit 0
