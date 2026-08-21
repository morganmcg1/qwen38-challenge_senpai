#!/usr/bin/env bash
# E99 rungs 6 and 7: replicate the threshold plateau and price the clamp depth.
#
#   usage: research/e99_rung67.sh debug   TOKENS CAP
#          research/e99_rung67.sh session TOKENS CAP
#
# Six arms, one build, one session. Arms B, C and D vary only the threshold at
# the shipped clamp depth, which is rung 6. Arms C, E and F vary only the clamp
# depth at the shipped threshold, which is rung 7. Arm A is the shared OFF
# baseline and its own spread is the session null.
#
#   A  gate off                     baseline and null
#   B  t = 8.2500   depth 3         rung 6 low plateau point
#   C  shipped default              rung 6 middle point AND rung 7 depth 3
#   D  t = 11.5625  depth 3         rung 6 high plateau point
#   E  t = 9.4375   depth 2         rung 7, M = 3, still G = 1
#   F  t = 9.4375   depth 4         rung 7 FALSIFICATION arm, M = 5, G = 2
#
# Arm C exports no gate variable at all, so its trace witnesses exactly what a
# ranked worker would run.
#
# The session runs the arm order forward, backward, backward, forward. Every
# arm then occupies positions summing to 50 out of 1..24, so a linear session
# drift cancels EXACTLY rather than to first order, which three replicates of
# six arms cannot do at any ordering.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: e99_rung67.sh debug|session TOKENS CAP}"
tokens="${2:?missing TOKENS}"
cap="${3:?missing CAP}"
failures=0

arm_spec() {
  case "$1" in
    A) echo "off 9.4375 3" ;;
    B) echo "g1 8.2500 3" ;;
    C) echo "default 9.4375 3" ;;
    D) echo "g1 11.5625 3" ;;
    E) echo "g1 9.4375 2" ;;
    F) echo "g1 9.4375 4" ;;
    *) echo "e99_rung67: unknown arm $1" >&2; return 1 ;;
  esac
}

run_leg() {
  local tag="$1" arm="$2" block="$3"
  local gate threshold depth
  read -r gate threshold depth <<<"$(arm_spec "${arm}")" || return 1
  echo "=== ${tag}: arm=${arm} gate=${gate} t=${threshold} d=${depth}" \
       "cap=${cap} tokens=${tokens} ==="
  rm -rf "research/out/${tag}"
  local status
  if [[ "${gate}" == "default" ]]; then
    env -u MLX_QWEN_MTP_MARGIN_GATE -u MLX_QWEN_MTP_MARGIN_GATE_T \
        -u MLX_QWEN_MTP_MARGIN_GATE_D \
      MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  else
    MLX_QWEN_MTP_MARGIN_GATE="${gate}" \
    MLX_QWEN_MTP_MARGIN_GATE_T="${threshold}" \
    MLX_QWEN_MTP_MARGIN_GATE_D="${depth}" \
    MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  fi
  local trace="research/out/${tag}/trace.txt"
  {
    echo "experiment=e99-rung67"
    echo "e99_cap=${cap}"
    echo "e99_arm=${arm}"
    echo "e99_block=${block}"
    echo "e99_gate=${gate}"
    echo "e99_gate_threshold=${threshold}"
    echo "e99_gate_depth=${depth}"
    echo "e99_fired_rounds=$(grep -c ' fire=1 ' "${trace}" 2>/dev/null || true)"
    echo "e99_traced_gd=$(
      grep -o ' gd=[0-9]*' "${trace}" 2>/dev/null | sort -u | tr -d ' ' \
        | paste -sd, -)"
    echo "e99_traced_gt=$(
      grep -o ' gt=[0-9.]*' "${trace}" 2>/dev/null | sort -u | tr -d ' ' \
        | paste -sd, -)"
    echo "e99_traced_gate=$(
      grep -o 'gate=[a-z0-9]*' "${trace}" 2>/dev/null | sort -u | paste -sd, -)"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e99_rung67: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
}

case "${mode}" in
  debug)
    # Prove the depth override reaches the worker before an 85-minute session.
    run_leg "e99r67dbgE" E 0
    run_leg "e99r67dbgF" F 0
    ;;
  session)
    for spec in "1 A B C D E F" "2 F E D C B A" \
                "3 F E D C B A" "4 A B C D E F"; do
      # shellcheck disable=SC2086
      set -- ${spec}
      block="$1"
      shift
      for arm in "$@"; do
        run_leg "e99r67c${cap}${arm}${block}" "${arm}" "${block}"
      done
    done
    ;;
  *)
    echo "e99_rung67: unknown mode ${mode}" >&2
    exit 2
    ;;
esac

echo "e99_rung67: ${failures} failed legs"
exit "${failures}"
