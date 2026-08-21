#!/usr/bin/env bash
# E96 rung 3a: price the fused residual + RMSNorm family the same way.
#
#   usage: research/e96_rung3a.sh [TOKENS] [FORCE_DRAFTS]
#
# The brief models this family at 1,187.1 us per round over 127 dispatches,
# 20.81 MB and 17.5 GB/s. Rung 1 showed that the census-style model overstated
# the GDN recurrent step by 9.66x, so the same three-column protocol runs here:
# the repeat slope is the measurement, the removal arm is a bracket, and the
# isolated-buffer census supplies the modelled line.
#
# The order is a palindrome about the session midpoint, so monotone thermal
# drift cancels to first order. `clone` is the same GDN control the rung 1 and
# rung 1c sessions used, which ties all three sessions to one scale.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-128}"
drafts="${2:-4}"
failures=0

run_leg() {
  local slot="$1" mode="$2" reps="$3"
  if [[ "${mode}" == "clone" ]]; then
    MLX_E96_REPEAT=1 \
      research/e96_direct_leg.sh "e96r3a-${slot}-clone" "${tokens}" \
        clone 4 "${drafts}"
  else
    MLX_E96_NORM="${mode}" MLX_E96_NORM_REPEAT="${reps}" \
      research/e96_direct_leg.sh "e96r3a-${slot}-${mode}${reps}" "${tokens}" \
        vendor 4 "${drafts}"
  fi
  local status=$?
  failures=$((failures + (status != 0)))
  echo "e96_rung3a: leg ${slot} norm=${mode} R=${reps} exit=${status}"
}

for spec in a1:clone:1 a2:rep:1 a3:rep:2 a4:rep:4 a5:rep:8 a6:off:1 \
            b6:off:1 b5:rep:8 b4:rep:4 b3:rep:2 b2:rep:1 b1:clone:1
do
  IFS=: read -r slot mode reps <<<"${spec}"
  run_leg "${slot}" "${mode}" "${reps}"
done

echo "e96_rung3a: ${failures} failed legs"
exit "${failures}"
