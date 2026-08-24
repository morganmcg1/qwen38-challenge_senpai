#!/usr/bin/env bash
# E165: price the cross-round head-chain prefetch against the shipped order.
#
#   usage: research/e165_abba.sh [TOKENS] [LABEL] [REP]
#
# TWO ARMS, one worker, one thermal session, no rebuild between legs. They
# differ only in `MLX_E165_HEAD_PREFETCH`:
#
#   off   the shipped order. The next round's head flush and first draft step
#         are built and submitted at the TOP of that round, so the GPU is idle
#         from the round's blocking eval until then.
#   on    the same work, issued at the earliest point in the previous round at
#         which the committed row is known, so the device runs it during the
#         host tail and the protocol turnaround.
#
# Both arms compute the same head forward on the same inputs, so they emit the
# same tokens, propose the same drafts and take the same schedule. The report
# refuses to price the contrast if `effective_mean_draft_len` or
# `accepted_draft_rate` moves between arms (RULE 179).
#
# PALINDROME. off on on off off on on off gives both arms mean position 4.5, so
# a monotone thermal or clock drift in leg index cancels to first order.
#
# GATED. Every timed leg runs the real 40 C gate.
#
# NATURAL SCHEDULE. No leg here pins the draft depth: the mechanism is a
# per-round fixed cost and it has to pay at the shipped operating point.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-pf}"
rep="${3:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e165_abba: the scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

apply_arm() {
  if [[ "$1" == "off" ]]; then
    export MLX_E165_HEAD_PREFETCH=0
  else
    export MLX_E165_HEAD_PREFETCH=1
  fi
}
clear_arm() { unset MLX_E165_HEAD_PREFETCH; }
anti_arm() { [[ "$1" == "off" ]] && echo on || echo off; }

# PHASE 0: warmup, ungated, at full length, on the candidate arm. It is also
# the candidate's witness leg and the source of the exact row dump the
# exactness gate compares.
warm_tag="e165${label}warm${rep}"
echo "=== warmup ${warm_tag}: arm=on tokens=${tokens} ungated ==="
apply_arm on
research/e79_trace_leg.sh "${warm_tag}" "${tokens}"
warm_status=$?
clear_arm
{
  echo "e165_arm=on"
  echo "e165_leg_role=warmup"
} >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e165_abba: the warmup leg exited ${warm_status}" >&2
  exit 5
fi

# PHASE 1: prove both arms reach the worker, each with a control that must
# fail. The warmup leg is the `on` witness, so only the `off` leg is new — and
# it runs at full length because the exactness gate compares its row dump with
# the warmup leg's, position by position, as exact hexfloats.
off_tag="e165${label}w${rep}off"
echo "=== witness ${off_tag}: arm=off tokens=${tokens} ungated ==="
apply_arm off
research/e79_trace_leg.sh "${off_tag}" "${tokens}"
off_status=$?
clear_arm
{
  echo "e165_arm=off"
  echo "e165_leg_role=witness"
} >> "research/out/${off_tag}/meta.txt"
if ((off_status != 0)); then
  echo "e165_abba: witness ${off_tag} exited ${off_status}" >&2
  exit 5
fi

for arm in on off; do
  if [[ "${arm}" == "on" ]]; then out="research/out/${warm_tag}"; else
    out="research/out/${off_tag}"; fi
  echo "--- arm witness for ${arm} ---"
  python3 research/e165_prefetch.py witness "${out}/trace.txt" --want "${arm}" \
    | tee "${out}/e165-arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e165_abba: the ${arm} leg did not run the ${arm} arm" >&2
    exit 3
  fi
  anti="$(anti_arm "${arm}")"
  echo "--- control: the same check must FAIL against ${anti} ---"
  python3 research/e165_prefetch.py witness "${out}/trace.txt" --want "${anti}" \
    | tee "${out}/e165-arm-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e165_abba: the ${arm} leg also passed the ${anti} check" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the exactness gate, with the control that proves it can fail.
echo "--- exactness: every declared row, both arms, as exact hexfloats ---"
python3 research/e165_prefetch.py rows \
  "research/out/${off_tag}/trace.txt" "research/out/${warm_tag}/trace.txt" \
  | tee "research/out/${warm_tag}/e165-rows.txt"
if [[ "${PIPESTATUS[0]}" != "0" ]]; then
  echo "e165_abba: the two arms did not declare identical rows" >&2
  exit 6
fi
python3 research/e165_prefetch.py rows \
  "research/out/${off_tag}/trace.txt" "research/out/${warm_tag}/trace.txt" \
  --positive-control | tee "research/out/${warm_tag}/e165-rows-control.txt"
if [[ "${PIPESTATUS[0]}" == "0" ]]; then
  echo "e165_abba: the exactness comparison cannot fail; it is not a gate" >&2
  exit 7
fi
echo "control ok: the exactness comparison fails when a value is perturbed"

# PHASE 3: the counterbalanced gated session.
failures=0
position=0
for arm in off on on off off on on off; do
  position=$((position + 1))
  tag="e165${label}k${rep}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} gated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
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
    echo "e165_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

post_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
if [[ "${post_worker}" != "${session_worker}" ]]; then
  echo "e165_abba: the worker digest changed during the session" >&2
  failures=$((failures + 1))
fi

echo "e165_abba: ${failures} failed legs"
python3 research/e165_prefetch.py report --label "${label}"
exit $((failures > 0))
