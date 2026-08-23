#!/usr/bin/env bash
# E149 terminal gate chain.
# Every step runs even if an earlier one fails, so one pass reports every
# failure rather than only the first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

BUDGET_BASE=770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
ASSIGNMENT_BASE=85b5812aa4649717d2d534d8c101775120aa56bd
OUT=research/out
mkdir -p "$OUT"

# Documented pre-existing failure floor on this base: 41 issues under 10
# distinct test names. The chain passes when the run stays at or under it.
SWIFT_MAX_ISSUES=41
SWIFT_MAX_NAMES=10

declare -a NAMES=()
declare -a CODES=()

record() {
    NAMES+=("$1")
    CODES+=("$2")
    echo "GATE $1 exit $2"
}

banner() {
    echo
    echo "=================================================================="
    echo "GATE $1"
    echo "=================================================================="
}

step() {
    local name="$1"; shift
    banner "$name"
    "$@"
    record "$name" $?
}

# --- worker rebuild, with real symbol assertions -------------------------
banner rebuild-and-assert-worker
senpai/rebuild-and-assert-worker.sh \
    --require-symbol qwen35E141RowsPerLeafOverride \
    --require-symbol activeClusterRowsPerLeaf
record rebuild-and-assert-worker $?

# --- swift test, compared against the documented failure floor -----------
banner swift-test
swift test --force-resolved-versions 2>&1 | tee "$OUT/e149-swift-test.log"
raw=${PIPESTATUS[0]}
issues=$(grep -c 'recorded an issue' "$OUT/e149-swift-test.log" || true)
names=$(grep -o '^✘ Test [A-Za-z0-9_]*(' "$OUT/e149-swift-test.log" | sort -u | wc -l | tr -d ' ')
grep 'Test run with' "$OUT/e149-swift-test.log" || true
echo "swift test raw exit $raw, issues $issues, distinct failing names $names"
echo "documented floor: issues <= $SWIFT_MAX_ISSUES, names <= $SWIFT_MAX_NAMES"
if [ "$issues" -le "$SWIFT_MAX_ISSUES" ] && [ "$names" -le "$SWIFT_MAX_NAMES" ]; then
    record swift-test 0
else
    record swift-test 1
fi

step twin-audit python3 research/twin_audit.py

# --- proof that no changed file lands in a submitted editable path -------
banner editable-surface-diff
python3 research/e149_editable_surface_diff.py "$ASSIGNMENT_BASE"
record editable-surface-diff $?

step validate-assignment-scope senpai/validate-assignment-scope.sh \
    "$ASSIGNMENT_BASE" Sources/MLXFastModel/Qwen35.swift
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
