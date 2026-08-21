#!/usr/bin/env bash
# E94: one arm against `ship`, ABBA-counterbalanced inside one session.
#
#   usage: research/e94_abba.sh ARM CAPS [TOKENS] [LABEL]
#
#   ARM    the arm under test, for example `snap4` or `m5fit`.
#   CAPS   comma-separated offered caps, for example `5,8`.
#   LABEL  tag prefix, `r2` for rung 2 and `r3` for rung 3.
#
# Each cap runs ship, ARM, ARM, ship in that order, so the arm effect is
# orthogonal to monotone thermal drift to first order and the two same-arm
# pairs give the session null. Every leg rebuilds and witnesses its own arm.
#
# Rung 2 was run as `research/e94_abba.sh snap4 4,5,8 512 r2`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm_under_test="${1:?usage: e94_abba.sh ARM CAPS [TOKENS] [LABEL]}"
caps="${2:?usage: e94_abba.sh ARM CAPS [TOKENS] [LABEL]}"
tokens="${3:-512}"
label="${4:-r3}"

export E94_EXPERIMENT="e94-rung${label#r}"

failures=0
IFS=',' read -r -a cap_list <<< "${caps}"
for cap in "${cap_list[@]}"; do
  position=0
  for arm in ship "${arm_under_test}" "${arm_under_test}" ship; do
    position=$((position + 1))
    tag="e94${label}c${cap}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} cap=${cap} tokens=${tokens} ==="
    research/e94_run_leg.sh "${arm}" "${tag}" "${cap}" "${tokens}"
    status=$?
    echo "e94_position=${position}" >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e94_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e94_abba: ${failures} failed legs"
exit "${failures}"
