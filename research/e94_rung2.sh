#!/usr/bin/env bash
# E94 rung 2: `snap4` against `ship`, ABBA-counterbalanced inside one session.
#
#   usage: research/e94_rung2.sh CAPS [TOKENS]
#
#   CAPS  comma-separated offered caps, for example `5,8`.
#
# Each cap runs ship, snap4, snap4, ship in that order, so the arm effect is
# orthogonal to monotone thermal drift to first order and the two same-arm
# pairs give the session null. Every leg rebuilds and witnesses its own arm.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

caps="${1:?usage: e94_rung2.sh CAPS [TOKENS]}"
tokens="${2:-512}"

failures=0
IFS=',' read -r -a cap_list <<< "${caps}"
for cap in "${cap_list[@]}"; do
  position=0
  for arm in ship snap4 snap4 ship; do
    position=$((position + 1))
    tag="e94r2c${cap}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} cap=${cap} tokens=${tokens} ==="
    research/e94_run_leg.sh "${arm}" "${tag}" "${cap}" "${tokens}"
    status=$?
    echo "e94_position=${position}" >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e94_rung2: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e94_rung2: ${failures} failed legs"
exit "${failures}"
