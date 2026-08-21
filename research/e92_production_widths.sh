#!/usr/bin/env bash
# E92 rung 2, production form: the marginal verify cost of every width.
#
#   usage: research/e92_production_widths.sh [TOKENS]
#
# The sync-head sweep attributes the head chain away from the verify window.
# This session runs the shipped form at every width so the marginal round cost
# curve that a depth policy actually pays is measured directly. Order in time
# is ascending width 1..9 then descending 9..1, so each width holds two samples
# placed symmetrically about the session midpoint.
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

run c "${tokens}" 0,1,2,3,4,5,6,7,8 --production
run d "${tokens}" 8,7,6,5,4,3,2,1,0 --production

echo "e92_production_widths: ${failures} failed legs"
exit "${failures}"
