#!/usr/bin/env bash
# E179 screen: does the launch-config cache move candidate MTP time?
#
#   usage: research/e179_screen.sh [TOKENS] [LABEL]
#
# TWO ARMS, one worker, one thermal session, no rebuild between legs.
#
#   c  shipped: the cache serves every repeated launch geometry
#   n  MLX_E179_CFG_CACHE_ARM=off: the incumbent MLXFastKernel launcher
#
# Both arms run the same kernels with the same bindings, the same grid and the
# same dispatch, so they must agree token for token. The report refuses to
# price the contrast when the arms disagree on `effective_mean_draft_len` or
# `accepted_draft_rate`, and it reports the serial leg as a null control: the
# routed replica path never takes M = 1, so the serial leg must not move.
#
# PALINDROME `c n n c c n n c`. Every arm has mean position 4.5, so a monotone
# thermal or clock drift in leg index cancels to first order.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-s1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e179_screen: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

apply_arm() {
  case "$1" in
    c) : ;;
    n) export MLX_E179_CFG_CACHE_ARM=off ;;
    *) echo "e179_screen: unknown arm $1" >&2; exit 2 ;;
  esac
}

clear_arm() { unset MLX_E179_CFG_CACHE_ARM; }

publish_leg() {
  python3 research/e179_screen.py leg --label "${label}" --tag "$1" \
    || echo "e179_screen: W&B publish failed for $1; evidence is on disk" >&2
}

# PHASE 0: warmup, ungated, shipped arm, full length (Rule 137).
warm_tag="e179${label}warm"
echo "=== warmup ${warm_tag}: arm=c tokens=${tokens} ungated ==="
research/e79_trace_leg.sh "${warm_tag}" "${tokens}" --no-trace
warm_status=$?
echo "e179_arm=c" >> "research/out/${warm_tag}/meta.txt"
echo "e179_leg_role=warmup" >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e179_screen: the warmup leg exited ${warm_status}" >&2
  exit 5
fi
publish_leg "${warm_tag}"

# PHASE 1: prove both arms reach the worker and produce identical tokens, from
# the run's own counters. `c` must show a warm hit rate and a frozen geometry
# set; `n` must show zero of both.
for arm in c n; do
  tag="e179${label}w${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=32 traced ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" 32
  status=$?
  clear_arm
  echo "e179_arm=${arm}" >> "research/out/${tag}/meta.txt"
  echo "e179_leg_role=witness" >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e179_screen: witness ${tag} exited ${status}" >&2
    exit 5
  fi
  publish_leg "${tag}"
done

if ! python3 research/e179_screen.py witness --label "${label}"; then
  echo "e179_screen: the arms are not witnessed; refusing to time" >&2
  exit 3
fi

# PHASE 2: the counterbalanced gated session.
failures=0
position=0
for arm in c n n c c n n c; do
  position=$((position + 1))
  tag="e179${label}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} gated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
  status=$?
  clear_arm
  {
    echo "e179_arm=${arm}"
    echo "e179_position=${position}"
    echo "e179_leg_role=timed"
    echo "e179_session_commit=${session_commit}"
    echo "e179_session_worker_sha256=${session_worker}"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e179_screen: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
  publish_leg "${tag}"
done

echo "e179_screen: ${failures} failed legs"
python3 research/e179_screen.py report --label "${label}"
exit $((failures > 0))
