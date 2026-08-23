#!/usr/bin/env bash
# T29-A: the advisor base configuration timed against the composition that
# ranked receipt `1db9d63e` shipped, plus the one cell nobody has built.
#
#   usage: research/e135_composition_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# THREE ARMS, one worker, no rebuild between legs:
#
#   base        wide  + onepass67 + p25 + width 2 to MLX   (advisor base)
#   composed    tight + shipped   + p15 + width 2 routed   (what 1db9d63e shipped)
#   composed67  tight + onepass67 + p15 + width 2 routed   (never built)
#
# WHY THE THIRD ARM. `Grid`'s own doc comment states that under `wide` the
# one-pass table pays twice the launch count of the shipped table at M = 6, 7
# and 8 for the same work, and that under `tight` both tables launch the same
# count. F194 measured "drop onepass67" under WIDE, where the table is
# penalised, and `1db9d63e` shipped TIGHT, where it is not. `composed` against
# `composed67` is that interaction, and it is the required metric
# `e135_onepass_gain_under_tight_pct`.
#
# ORDER  base composed composed67 composed67 composed base, one palindrome per
# replicate. Every arm has mean position 3.5, so a monotone drift in leg index
# cancels exactly in all three pairwise contrasts.
#
# GATED. F29 asks for the real 40 C gate, so each leg runs with `--cool-gate`
# and `cool_gate_passed_real_gate=true`. This session is gate-qualified.
#
# E87 IS HELD OFF in every arm. `1db9d63e` did not carry the probe-select port,
# so `MLX_E87_SELECT=0` keeps it out of this contrast on all three arms.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-c1}"
first="${4:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_composition_abba: scored surface is dirty; refusing to time" \
       "over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

# One place that defines an arm, so the witness leg and the timed leg cannot
# drift apart. bash 3.2 has no associative arrays.
apply_arm() {
  export MLX_E87_SELECT=0
  case "$1" in
    base)
      export MLX_E120_QMV_GRID=wide
      export MLX_E120_QMV_TABLE=onepass67
      export MLX_E135_PROBE_ARM=p25
      export MLX_E120_QMV_WIDTH2=0
      ;;
    composed)
      export MLX_E120_QMV_GRID=tight
      export MLX_E120_QMV_TABLE=shipped
      export MLX_E135_PROBE_ARM=p15
      export MLX_E120_QMV_WIDTH2=1
      ;;
    composed67)
      export MLX_E120_QMV_GRID=tight
      export MLX_E120_QMV_TABLE=onepass67
      export MLX_E135_PROBE_ARM=p15
      export MLX_E120_QMV_WIDTH2=1
      ;;
    *)
      echo "e135_composition_abba: unknown arm $1" >&2
      exit 2
      ;;
  esac
}

clear_arm() {
  unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E135_PROBE_ARM \
        MLX_E120_QMV_WIDTH2 MLX_E87_SELECT
}

# The arm each witness leg is checked AGAINST, which it must fail. Rule 101.
anti_arm() {
  case "$1" in
    base) echo composed ;;
    composed) echo composed67 ;;
    composed67) echo composed ;;
  esac
}

# PHASE 1: prove all four selectors reach the worker on every arm.
#
# `composed` against `composed67` differs on the table alone, so that anti-arm
# is the tightest available control: it fails on one axis only, which shows the
# check discriminates rather than merely rejecting everything.
witness_tokens="${E135_WITNESS_TOKENS:-16}"
for arm in base composed composed67; do
  tag="e135${label}w${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  apply_arm "${arm}"
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG
  clear_arm
  {
    echo "e135_arm=${arm}"
    echo "e135_leg_exit=${status}"
  } >> "${out}/meta.txt"

  python3 research/e135_composition_arm_check.py "${out}/pipelines.json" \
    --arm "${arm}" | tee "${out}/arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_composition_abba: ${arm} leg did not run the ${arm} arm;" \
         "not timing" >&2
    exit 3
  fi

  anti="$(anti_arm "${arm}")"
  echo "--- Rule 101 control: this leg must FAIL the ${anti} check ---"
  python3 research/e135_composition_arm_check.py "${out}/pipelines.json" \
    --arm "${anti}" | tee "${out}/arm-check-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e135_composition_abba: the ${arm} leg passed the ${anti} check," \
         "so the witness cannot fail and proves nothing" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the counterbalanced gated session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in base composed composed67 composed67 composed base; do
    position=$((position + 1))
    tag="e135${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    apply_arm "${arm}"
    research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
    status=$?
    clear_arm
    {
      echo "e135_arm=${arm}"
      echo "e135_replicate=${rep}"
      echo "e135_position=${position}"
      echo "e135_leg_index=$(( (rep - first) * 6 + position ))"
      echo "e135_session_commit=${session_commit}"
      echo "e135_session_worker_sha256=${session_worker}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e135_composition_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e135_composition_abba: ${failures} failed legs"
python3 research/e135_composition_report.py --label "${label}"
exit $((failures > 0))
