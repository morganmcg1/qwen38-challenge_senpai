#!/usr/bin/env bash
# E92 rung 2: the whole width sweep as one ABBA-counterbalanced session.
#
#   usage: research/e92_rung2.sh [TOKENS]
#
# Order in time is ascending width 1..9, then descending width 9..1, so each
# width holds two samples placed symmetrically about the session midpoint and
# monotone thermal drift cancels to first order. The production-form control
# legs follow in the same shape at widths 1, 6 and 9.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
failures=0

run() {
  research/e92_width_session.sh "$@"
  status=$?
  ((status == 0)) || failures=$((failures + status))
}

run a "${tokens}" 0,1,2,3,4,5,6,7,8
run b "${tokens}" 8,7,6,5,4,3,2,1,0
run a "${tokens}" 0,5,8 --production
run b "${tokens}" 8,5,0 --production

echo "e92_rung2: ${failures} failed legs"
exit "${failures}"
