#!/usr/bin/env bash
# E149 terminal gate chain, in the order the assignment specifies.
# Every step runs even if an earlier one fails, so one pass reports every
# failure rather than only the first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

BUDGET_BASE=770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf

declare -a NAMES=()
declare -a CODES=()

step() {
    local name="$1"; shift
    echo
    echo "=================================================================="
    echo "GATE $name"
    echo "=================================================================="
    "$@"
    local rc=$?
    NAMES+=("$name")
    CODES+=("$rc")
    echo "GATE $name exit $rc"
}

step rebuild-and-assert-worker senpai/rebuild-and-assert-worker.sh
step swift-test swift test --force-resolved-versions
step twin-audit python3 research/twin_audit.py
step validate-assignment-scope senpai/validate-assignment-scope.sh
step check-editable-budget senpai/check-editable-budget.sh "$BUDGET_BASE"
step verify-ranked-score-boundary senpai/verify-ranked-score-boundary.sh
step e129-entry-point-census python3 research/e129_entry_point_census.py --table shipped

echo
echo "=================================================================="
echo "E149 GATE CHAIN SUMMARY"
echo "=================================================================="
fail=0
for i in "${!NAMES[@]}"; do
    printf '  %-34s exit %s\n' "${NAMES[$i]}" "${CODES[$i]}"
    [ "${CODES[$i]}" -ne 0 ] && fail=1
done
echo "overall: $([ $fail -eq 0 ] && echo PASS || echo NEEDS-REVIEW)"
exit 0
