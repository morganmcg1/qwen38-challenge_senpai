#!/usr/bin/env bash
# F34 sessions 1 and 2, merged into one counterbalanced palindrome.
#
#   usage: research/e135_f34_abba.sh [TOKENS] [LABEL] [REPLICATE]
#
# THREE ARMS, one worker, one thermal session, no rebuild between legs:
#
#   c67ship   tight + onepass67  + p15 + width 2 + depth arm ship
#   c67pb6    tight + onepass67  + p15 + width 2 + depth arm pb6   (the payload)
#   c678pb6   tight + onepass678 + p15 + width 2 + depth arm pb6
#
# WHY ONE SESSION INSTEAD OF TWO. F34 asked for two 4-leg ABBA sessions that
# share the arm `c67pb6`. The palindrome
#
#   c67ship c67pb6 c678pb6 c678pb6 c67pb6 c67ship
#
# gives every arm mean position 3.5, so a monotone drift in leg index cancels
# in BOTH contrasts, and each contrast still has n = 2 per arm, exactly as the
# two separate sessions would. It costs six gated legs instead of eight and one
# warmup instead of two, and both contrasts are then measured inside one
# thermal session against one binary, which the split design cannot promise.
#
#   c67ship  vs c67pb6    e135_pb6_under_tight_pct
#   c67pb6   vs c678pb6   e135_onepass678_local_pct
#   c67ship  vs c678pb6   the whole move, reported for completeness
#
# WARMUP. Rule 137. Leg 0 is an untimed 512-token leg on `c67pb6` with the
# per-round trace ON. It equilibrates the GPU before the gated session AND it
# is the source of the realised verify-width histogram under the pb6 schedule,
# which F34 asks for. The timed legs run `--no-trace`, because the trace costs
# about 9 % of candidate time on this host.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-f34}"
rep="${3:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_f34_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

# One place that defines an arm, so the witness leg and the timed leg cannot
# drift apart. bash 3.2 has no associative arrays.
apply_arm() {
  export MLX_E120_QMV_GRID=tight
  export MLX_E135_PROBE_ARM=p15
  export MLX_E120_QMV_WIDTH2=1
  case "$1" in
    c67ship)
      export MLX_E120_QMV_TABLE=onepass67
      export MLX_E134_DEPTH_PRICE_ARM=ship
      ;;
    c67pb6)
      export MLX_E120_QMV_TABLE=onepass67
      export MLX_E134_DEPTH_PRICE_ARM=pb6
      ;;
    c678pb6)
      export MLX_E120_QMV_TABLE=onepass678
      export MLX_E134_DEPTH_PRICE_ARM=pb6
      ;;
    *)
      echo "e135_f34_abba: unknown arm $1" >&2
      exit 2
      ;;
  esac
}

clear_arm() {
  unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E135_PROBE_ARM \
        MLX_E120_QMV_WIDTH2 MLX_E134_DEPTH_PRICE_ARM
}

# The QMV arm each witness leg is checked against, which it must fail.
# Rule 101. `composed` is the shipped table on the same grid and probe, so it
# fails on the table axis alone and shows the check discriminates.
qmv_arm() {
  case "$1" in
    c67ship|c67pb6) echo composed67 ;;
    c678pb6) echo composed678 ;;
  esac
}

qmv_anti_arm() {
  case "$1" in
    c67ship|c67pb6) echo composed678 ;;
    c678pb6) echo composed67 ;;
  esac
}

depth_arm() {
  case "$1" in
    c67ship) echo ship ;;
    *) echo pb6 ;;
  esac
}

depth_anti_arm() {
  case "$1" in
    c67ship) echo pb6 ;;
    *) echo ship ;;
  esac
}

# PHASE 0: the warmup and histogram leg (Rule 137).
warm_tag="e135${label}warm"
echo "=== warmup ${warm_tag}: arm=c67pb6 tokens=${tokens} traced, ungated ==="
apply_arm c67pb6
research/e79_trace_leg.sh "${warm_tag}" "${tokens}"
warm_status=$?
clear_arm
echo "e135_arm=c67pb6" >> "research/out/${warm_tag}/meta.txt"
echo "e135_leg_role=warmup" >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e135_f34_abba: the warmup leg exited ${warm_status}; not timing" >&2
  exit 5
fi

# PHASE 1: prove every selector reaches the worker on every arm.
witness_tokens="${E135_WITNESS_TOKENS:-16}"
for arm in c67ship c67pb6 c678pb6; do
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

  want="$(qmv_arm "${arm}")"
  python3 research/e135_composition_arm_check.py "${out}/pipelines.json" \
    --arm "${want}" | tee "${out}/arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_f34_abba: the ${arm} leg did not run the ${want} QMV arm;" \
         "not timing" >&2
    exit 3
  fi

  anti="$(qmv_anti_arm "${arm}")"
  echo "--- Rule 101 control: this leg must FAIL the ${anti} check ---"
  python3 research/e135_composition_arm_check.py "${out}/pipelines.json" \
    --arm "${anti}" | tee "${out}/arm-check-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e135_f34_abba: the ${arm} leg passed the ${anti} check, so the" \
         "witness cannot fail and proves nothing" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the counterbalanced gated session.
#
# The depth arm is witnessed per TIMED leg rather than on the 16-token witness
# leg, because the schedule signature needs enough rounds to be readable.
failures=0
position=0
for arm in c67ship c67pb6 c678pb6 c678pb6 c67pb6 c67ship; do
  position=$((position + 1))
  tag="e135${label}k${rep}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
  status=$?
  clear_arm
  out="research/out/${tag}"
  {
    echo "e135_arm=${arm}"
    echo "e135_replicate=${rep}"
    echo "e135_position=${position}"
    echo "e135_leg_index=${position}"
    echo "e135_session_commit=${session_commit}"
    echo "e135_session_worker_sha256=${session_worker}"
  } >> "${out}/meta.txt"
  if ((status != 0)); then
    echo "e135_f34_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
    continue
  fi

  want="$(depth_arm "${arm}")"
  anti="$(depth_anti_arm "${arm}")"
  echo "--- Rule 114 witness: the depth arm off this leg's own score.json ---"
  python3 research/e135_arm_check.py "${out}/score.json" --want "${want}" \
    | tee "${out}/depth-arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_f34_abba: ${tag} did not run the ${want} depth arm" >&2
    failures=$((failures + 1))
  fi
  echo "--- Rule 101 control: the same check must FAIL against ${anti} ---"
  python3 research/e135_arm_check.py "${out}/score.json" --want "${anti}" \
    | tee "${out}/depth-arm-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e135_f34_abba: ${tag} passed the ${anti} check too, so the depth" \
         "witness proves nothing" >&2
    failures=$((failures + 1))
  fi
done

echo "e135_f34_abba: ${failures} failed legs"
python3 research/e135_f34_report.py --label "${label}"
exit $((failures > 0))
