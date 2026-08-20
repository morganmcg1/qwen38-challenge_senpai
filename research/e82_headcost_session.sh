#!/usr/bin/env bash
# E82: price the proposal-head step of each distinct head COST profile.
#
#   usage: research/e82_headcost_session.sh PREFIX
#
# The six E82 acceptance arms collapse to three cost profiles, because the
# readout path and the trunk precision are what the head step pays for:
#
#   pinned       bf16 trunk + bf16 fc, no shipped draft_lm_head. The runtime
#                derives a compact affine-4 trim of the exact lm_head at warm
#                and reads it through ONE fused select dispatch.
#   master-bf16  the same bf16 trunk + fc, plus the declared 2-bit
#                draft_lm_head, so the readout takes the coarse+rerank path.
#                kamciosz has this profile too.
#   declared     affine-4 trunk + fc + bf16 islands, same coarse+rerank
#                readout. soup-q4 and qat-q4 have this profile too.
#
# `qat-q4` runs as well, even though its byte split equals `declared`. E79's
# gated legs put both measured heads on one line, 179 MB of head tensor per
# millisecond, which says the step is bandwidth bound and the readout's extra
# op boundaries are free. Two arms cannot separate a bytes law from a
# trunk-precision law, so master-bf16 is the falsifying point (same 4-bit
# readout, 2.35x the bytes) and qat-q4 is the null point (same bytes, better
# acceptance, so it must land on declared's time).
#
# Every leg uses --sync-head so the head chain is drained before the verify
# window and its GPU time lands in draft_build_us instead of hiding in the
# trailing asyncEval. The leg order is a palindrome, so monotone thermal drift
# cancels to first order across the four heads.
#
# These legs are UNGATED on purpose (program.md permits an ungated,
# ABBA-counterbalanced local timed arm). Entry and exit GPU temperature are
# recorded per leg by e79_trace_leg.sh, and every leg carries
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false. The
# result is directional causal evidence inside this session, never a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e82_headcost_session.sh PREFIX}"
cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"

declare -a order=(pinned master-bf16 qat-q4 declared
                  declared qat-q4 master-bf16 pinned)
declare -a rep=(1 1 1 1 2 2 2 2)

dir_for() {
  case "$1" in
    pinned) echo "${cache}/mtp-head" ;;
    declared) echo "${cache}/mtp-head-declared-run" ;;
    qat-q4) echo "${cache}/e82/built/e82-qat-q4-run" ;;
    master-bf16) echo "${cache}/e82/built/e82-master-bf16-run" ;;
    *) echo "e82_headcost_session.sh: unknown head $1" >&2; exit 2 ;;
  esac
}

for i in "${!order[@]}"; do
  head="${order[$i]}"
  E79_HEAD_DIR="$(dir_for "${head}")" \
    research/e79_trace_leg.sh "${prefix}-${head}-${rep[$i]}" 512 --sync-head
  status=$?
  echo "leg ${prefix}-${head}-${rep[$i]} exit=${status}"
  ((status == 0)) || exit "${status}"
done
