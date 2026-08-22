#!/usr/bin/env bash
# E130 rung 10a: the counterbalanced wired-residency session.
#
#   usage: research/e130_rung10a_session.sh PREFIX TOKENS
#
# Runs the three arms as a palindrome inside one session:
#
#   none  s64  s512  s512  s64  none
#
# Every arm is measured twice, once in each half, so a monotone thermal or
# power drift over the session cancels to first order in the arm means. The
# palindrome is the counterbalancing that the standing ungated-timing mode
# requires; MLXFAST_LOCAL_COOL_GATE=0 is set by the leg script.
#
# A leg that fails does not stop the session. A missing arm is better than a
# half-counterbalanced one, and the reader reports which legs are present.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e130_rung10a_session.sh PREFIX TOKENS}"
tokens="${2:?usage: e130_rung10a_session.sh PREFIX TOKENS}"

order=(none s64 s512 s512 s64 none)
index=0
for arm in "${order[@]}"; do
  index=$((index + 1))
  tag="${prefix}-${index}-${arm}"
  echo "############ leg ${index}/6  arm=${arm}  tag=${tag} ############"
  research/e130_rung10a_leg.sh "${tag}" "${arm}" "${tokens}"
  echo "############ leg ${index}/6 exit=$? ############"
done

echo "=== session complete: ${prefix} ==="
for arm_index in 1 2 3 4 5 6; do
  tag="${prefix}-${arm_index}-${order[$((arm_index - 1))]}"
  echo "--- ${tag} ---"
  grep -E "^(arm|exit|gpu_temp_entry_c|gpu_temp_exit_c|wired_residency_active)=" \
    "research/out/${tag}/meta.txt" 2>/dev/null || echo "missing"
done
