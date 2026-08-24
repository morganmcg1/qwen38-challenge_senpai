#!/usr/bin/env bash
# E174 screen: first-hand per-fill and per-kernel-record coefficients.
#
#   usage: research/e174_screen.sh [TOKENS] [LABEL]
#
# FOUR ARMS, one worker, one thermal session, no rebuild between legs. Two
# independent contrasts, four gated legs each, which is E135/F41's design so
# the `f - r` number is directly comparable with Thorfinn's.
#
#   s  shipped sumtable                     127 epilogues, 130 fills
#   o  sumtable + MLX_E174_XSUMS_SIDECAR=off  0 epilogues, 257 fills
#   r  MLX_E120_QMV_ARM=replica               0 fills, sums recomputed in the QMV
#   f  MLX_E120_QMV_ARM=fill_noconsume      257 fills, sums recomputed in the QMV
#
#   o - s   replace 127 epilogues with 127 fills and their host kernel records.
#           E174's own mechanism, sign flipped, at 127/130 of its scale.
#   f - r   add 257 fills and their host kernel records with the consumer held
#           fixed (`consume:` false on both arms). This is the coefficient the
#           advisor asked for: it prices one dispatch AND one kernel record
#           together, and `o - s` cannot separate those from the epilogue.
#
# All four arms are bit-exact: same tokens, same schedule. The report refuses to
# price a contrast whose arms disagree on `effective_mean_draft_len` or
# `accepted_draft_rate`, and it reports the serial leg as a null control -
# `wants` and `routable` both refuse at M = 1, so the serial leg must not move.
#
# PALINDROME `s o r f f r o s`. Every arm has mean position 4.5, so a monotone
# thermal or clock drift in leg index cancels to first order.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-s1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e174_screen: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

apply_arm() {
  case "$1" in
    s) : ;;
    o) export MLX_E174_XSUMS_SIDECAR=off ;;
    r) export MLX_E120_QMV_ARM=replica ;;
    f) export MLX_E120_QMV_ARM=fill_noconsume ;;
    *) echo "e174_screen: unknown arm $1" >&2; exit 2 ;;
  esac
}

clear_arm() { unset MLX_E174_XSUMS_SIDECAR MLX_E120_QMV_ARM; }

# PHASE 0: warmup, ungated, shipped arm, full length (Rule 137).
warm_tag="e174${label}warm"
echo "=== warmup ${warm_tag}: arm=s tokens=${tokens} ungated ==="
research/e79_trace_leg.sh "${warm_tag}" "${tokens}" --no-trace
warm_status=$?
echo "e174_arm=s" >> "research/out/${warm_tag}/meta.txt"
echo "e174_leg_role=warmup" >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e174_screen: the warmup leg exited ${warm_status}" >&2
  exit 5
fi

# PHASE 1: prove every arm reaches the worker, from the run's own census
# counters. `s` and `o` are already witnessed by research/e174_census.sh; this
# repeats them inside the session so one record carries all four.
for arm in s o r f; do
  tag="e174${label}w${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=32 traced ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" 32
  status=$?
  clear_arm
  echo "e174_arm=${arm}" >> "research/out/${tag}/meta.txt"
  echo "e174_leg_role=witness" >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e174_screen: witness ${tag} exited ${status}" >&2
    exit 5
  fi
done

if ! python3 research/e174_screen.py witness --label "${label}"; then
  echo "e174_screen: an arm did not reach the worker; refusing to time" >&2
  exit 3
fi

# PHASE 2: the counterbalanced gated session.
failures=0
position=0
for arm in s o r f f r o s; do
  position=$((position + 1))
  tag="e174${label}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} gated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
  status=$?
  clear_arm
  {
    echo "e174_arm=${arm}"
    echo "e174_position=${position}"
    echo "e174_leg_role=timed"
    echo "e174_session_commit=${session_commit}"
    echo "e174_session_worker_sha256=${session_worker}"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e174_screen: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e174_screen: ${failures} failed legs"
python3 research/e174_screen.py report --label "${label}"
exit $((failures > 0))
