#!/usr/bin/env bash
# E193 Stage 1: does the E165 round-start head-chain prefetch still win on the
# CURRENT tree?
#
#   usage: research/e193_screen.sh [TOKENS] [PHASES]
#
# PHASES is `all` (default) or `check` (witness + exactness gates only, then
# stop before the timed session).
#
# ONE BINARY, TWO ARMS. The prefetch machinery on this base is byte-identical
# to receipt D's snapshot 376a43f3; only the gate polarity differs
# (`MLX_E165_HEAD_PREFETCH == "1"` here, `!= "0"` there). So the screen needs
# no Swift edit and no rebuild between legs, and the two arms share a worker
# digest by construction.
#
# UNGATED, under the three standing conditions: the timed legs are
# ABBA-counterbalanced inside one session, every leg records entry and exit GPU
# temperature, and `cool_gate_passed_real_gate=false` /
# `gate_qualified_for_timing=false` are preserved verbatim in each meta.txt.
# This is directional causal evidence within the session; it is not a
# gate-qualified result and never an official score.
#
# EIGHT TIMED LEGS, not four. The predicted effect is ~-0.3% against a
# per-leg spread of the same order, so two legs per arm cannot separate it from
# drift; the palindrome off on on off off on on off puts both arms at mean
# position 4.5 and gives n=4 per arm. Ungated legs cost ~3 min each, so the
# whole session is far cheaper than four gated legs would be.
#
# Tags are `e165e193k1p<pos><arm>` on purpose: research/e165_prefetch.py
# already loads, gates and prices exactly that shape.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-256}"
phases="${2:-all}"
label="e193"
rep="1"

if [[ "${phases}" != "all" && "${phases}" != "check" ]]; then
  echo "e193_screen: PHASES must be 'all' or 'check', got '${phases}'" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e193_screen: the scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e193_screen: commit=${session_commit} worker=${session_worker:0:16}"

apply_arm() {
  if [[ "$1" == "off" ]]; then
    export MLX_E165_HEAD_PREFETCH=0
  else
    export MLX_E165_HEAD_PREFETCH=1
  fi
}
clear_arm() { unset MLX_E165_HEAD_PREFETCH; }
anti_arm() { [[ "$1" == "off" ]] && echo on || echo off; }

# PHASE 1: prove each arm reaches the worker, and that the witness FAILS
# against the opposite arm. Both legs are traced and ungated; they also warm
# the caches and supply the row dumps the exactness gate compares.
#
# macOS ships bash 3.2, which has no associative arrays. The witness output
# directory is derived from the arm instead of looked up.
witness_dir() { echo "research/out/e165${label}w${rep}$1"; }

for arm in on off; do
  tag="e165${label}w${rep}${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=${tokens} ungated traced ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  clear_arm
  {
    echo "e165_arm=${arm}"
    echo "e165_leg_role=witness"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e193_screen: witness ${tag} exited ${status}" >&2
    exit 5
  fi
done

for arm in on off; do
  out="$(witness_dir "${arm}")"
  echo "--- arm witness for ${arm} ---"
  python3 research/e193_gates.py witness "${out}/trace.txt" --want "${arm}" \
    | tee "${out}/e193-arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e193_screen: the ${arm} leg did not run the ${arm} arm" >&2
    exit 3
  fi
  anti="$(anti_arm "${arm}")"
  echo "--- control: the same check must FAIL against ${anti} ---"
  python3 research/e193_gates.py witness "${out}/trace.txt" --want "${anti}" \
    | tee "${out}/e193-arm-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e193_screen: the ${arm} leg also passed the ${anti} check" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the exactness gate. Every declared row, both arms, exact hexfloats,
# with the positive control that proves the comparison can fail.
echo "--- exactness: every declared row, both arms, as exact hexfloats ---"
python3 research/e193_gates.py rows \
  "$(witness_dir off)/trace.txt" "$(witness_dir on)/trace.txt" \
  | tee "$(witness_dir on)/e193-rows.txt"
if [[ "${PIPESTATUS[0]}" != "0" ]]; then
  echo "e193_screen: the two arms did not declare identical rows" >&2
  exit 6
fi
python3 research/e193_gates.py rows \
  "$(witness_dir off)/trace.txt" "$(witness_dir on)/trace.txt" \
  --positive-control | tee "$(witness_dir on)/e193-rows-control.txt"
if [[ "${PIPESTATUS[0]}" == "0" ]]; then
  echo "e193_screen: the exactness comparison cannot fail; it is not a gate" >&2
  exit 7
fi
echo "control ok: the exactness comparison fails when a value is perturbed"

if [[ "${phases}" == "check" ]]; then
  echo "e193_screen: pre-flight complete; stopping before the timed session"
  exit 0
fi

# PHASE 3: the counterbalanced ungated session.
failures=0
position=0
for arm in off on on off off on on off; do
  position=$((position + 1))
  tag="e165${label}k${rep}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} ungated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
  status=$?
  clear_arm
  out="research/out/${tag}"
  {
    echo "e165_arm=${arm}"
    echo "e165_rep=${rep}"
    echo "e165_position=${position}"
    echo "e165_leg_role=timed"
    echo "e165_session_commit=${session_commit}"
    echo "e165_session_worker_sha256=${session_worker}"
  } >> "${out}/meta.txt"
  if ((status != 0)); then
    echo "e193_screen: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

post_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
if [[ "${post_worker}" != "${session_worker}" ]]; then
  echo "e193_screen: the worker digest changed during the session" >&2
  failures=$((failures + 1))
fi

echo "e193_screen: ${failures} failed legs"
python3 research/e165_prefetch.py report --label "${label}"
exit $((failures > 0))
