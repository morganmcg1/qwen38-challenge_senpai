#!/usr/bin/env bash
# E96 rung 1c: is the repeat slope a throughput floor or a serial latency?
#
#   usage: research/e96_rung1c.sh [TOKENS] [FORCE_DRAFTS]
#
# Rung 1 measured 839.6 us per recurrent step per round from the slope of the
# independent repeat arm. Those repetitions are mutually independent, so the
# GPU may overlap them and the slope may report a throughput floor that is
# smaller than the latency the round actually pays. The chained arm removes
# that freedom: repetition r + 1 addresses its loads and stores through a value
# that repetition r computed, so the repetitions must serialise.
#
# The two arms are interleaved inside one counterbalanced session, so both
# slopes come from the same thermal trajectory and the ratio is not confounded
# by drift. The order is a palindrome about the session midpoint.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-128}"
drafts="${2:-4}"
failures=0

for spec in a1:clone:1 \
            a2:rep:1 a3:repchain:1 a4:rep:2 a5:repchain:2 a6:rep:4 \
            a7:repchain:4 a8:rep:8 a9:repchain:8 a10:rep:16 a11:repchain:16 \
            b11:repchain:16 b10:rep:16 b9:repchain:8 b8:rep:8 b7:repchain:4 \
            b6:rep:4 b5:repchain:2 b4:rep:2 b3:repchain:1 b2:rep:1 \
            b1:clone:1
do
  IFS=: read -r slot mode reps <<<"${spec}"
  MLX_E96_REPEAT="${reps}" \
    research/e96_direct_leg.sh "e96r1c-${slot}-${mode}${reps}" "${tokens}" \
      "${mode}" 4 "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_rung1c: leg ${slot} mode=${mode} R=${reps} exit=${status}"
done

echo "e96_rung1c: ${failures} failed legs"
exit "${failures}"
