#!/usr/bin/env bash
# E92 rung 1: the read-bandwidth residency sweep in both size orders.
#
#   usage: research/e92_rung1.sh [TOKENS] [REPS]
#
# The reversed leg exists to separate "smallest size" from "first entry in the
# sweep". If the small point stays slow when it runs last, the cost is a
# per-reduction floor; if it becomes fast, the forward leg measured a
# first-entry effect and not a size effect.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-64}"
reps="${2:-7}"
failures=0

research/e92_bw_leg.sh e92bw1r "${tokens}" 16,64,157,330,428,1024 "${reps}"
failures=$((failures + $?))
research/e92_bw_leg.sh e92bw2r "${tokens}" 1024,428,330,157,64,16 "${reps}"
failures=$((failures + $?))

echo "e92_rung1: ${failures} failed legs"
exit "${failures}"
