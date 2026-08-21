#!/usr/bin/env bash
# E94 rung 1: the offered-cap sweep on the shipped arm, counterbalanced.
#
#   usage: research/e94_rung1.sh [TOKENS]
#
# Ascending caps 4..8 then descending 8..4, so each cap holds two samples
# placed symmetrically about the session midpoint and monotone thermal drift
# cancels to first order.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
failures=0

research/e94_cap_session.sh a 4,5,6,7,8 "${tokens}"
failures=$((failures + $?))
research/e94_cap_session.sh b 8,7,6,5,4 "${tokens}"
failures=$((failures + $?))

echo "e94_rung1: ${failures} failed legs"
exit "${failures}"
