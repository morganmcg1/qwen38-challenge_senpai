#!/usr/bin/env bash
# E160: price the SwiGLU producer fusion against the shipped composition.
#
#   usage: research/e160_abba.sh [TOKENS] [LABEL] [REP]
#
# THREE ARMS, one worker, one thermal session, no rebuild between legs. They
# differ only in `MLX_E160_SWIGLU_ARM`:
#
#   off      MLX's compiled `silu(a) * b`, then the standalone chunk-sum fill
#   replica  the candidate kernel with no epilogue, then the standalone fill
#   fuse     the candidate kernel with the epilogue, and no standalone fill
#
# All three write the same activation bytes, so they emit the same tokens and
# the same draft schedule. The contrast is a pure cost measurement and the
# report refuses to price it if `effective_mean_draft_len` or
# `accepted_draft_rate` moves between arms (RULE 179).
#
# PALINDROME. off replica fuse fuse replica off gives every arm mean position
# 3.5, so a monotone thermal or clock drift in leg index cancels to first
# order.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-fuse}"
rep="${3:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e160_abba: the scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

apply_arm() { export MLX_E160_SWIGLU_ARM="$1"; }
clear_arm() { unset MLX_E160_SWIGLU_ARM; }

anti_arm() {
  case "$1" in off) echo fuse ;; replica) echo off ;; fuse) echo replica ;; esac
}

# PHASE 0: warmup, ungated, at full length, on the shipped candidate arm.
warm_tag="e160${label}warm${rep}"
echo "=== warmup ${warm_tag}: arm=fuse tokens=${tokens} ungated ==="
apply_arm fuse
research/e79_trace_leg.sh "${warm_tag}" "${tokens}"
warm_status=$?
clear_arm
{
  echo "e160_arm=fuse"
  echo "e160_leg_role=warmup"
} >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e160_abba: the warmup leg exited ${warm_status}" >&2
  exit 5
fi

# PHASE 1: prove every arm reaches the worker, each with a control that must
# fail. The warmup leg is the `fuse` witness, so only two short legs are new.
for arm in fuse off replica; do
  if [[ "${arm}" == "fuse" ]]; then
    out="research/out/${warm_tag}"
  else
    tag="e160${label}w${rep}${arm}"
    out="research/out/${tag}"
    echo "=== witness ${tag}: arm=${arm} tokens=48 ==="
    apply_arm "${arm}"
    research/e79_trace_leg.sh "${tag}" 48
    status=$?
    clear_arm
    echo "e160_arm=${arm}" >> "${out}/meta.txt"
    if ((status != 0)); then
      echo "e160_abba: witness ${tag} exited ${status}" >&2
      exit 5
    fi
  fi
  echo "--- arm witness for ${arm} ---"
  python3 research/e160_fuse.py witness "${out}/trace.txt" --want "${arm}" \
    | tee "${out}/e160-arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e160_abba: the ${arm} leg did not run the ${arm} arm" >&2
    exit 3
  fi
  anti="$(anti_arm "${arm}")"
  echo "--- control: the same check must FAIL against ${anti} ---"
  python3 research/e160_fuse.py witness "${out}/trace.txt" --want "${anti}" \
    | tee "${out}/e160-arm-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e160_abba: the ${arm} leg also passed the ${anti} check" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the counterbalanced gated session.
failures=0
position=0
for arm in off replica fuse fuse replica off; do
  position=$((position + 1))
  tag="e160${label}k${rep}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} gated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
  status=$?
  clear_arm
  out="research/out/${tag}"
  {
    echo "e160_arm=${arm}"
    echo "e160_rep=${rep}"
    echo "e160_position=${position}"
    echo "e160_session_commit=${session_commit}"
    echo "e160_session_worker_sha256=${session_worker}"
  } >> "${out}/meta.txt"
  if ((status != 0)); then
    echo "e160_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e160_abba: ${failures} failed legs"
python3 research/e160_fuse.py report --label "${label}"
exit $((failures > 0))
