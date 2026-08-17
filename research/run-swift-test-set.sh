#!/bin/bash
# Capture the full Swift test failure SET at one commit.
#
# Counts are not comparable across bases (the organizer's archive-restore churn
# adds and removes whole test files), so a base-vs-candidate comparison must
# diff sorted failing test identifiers, not totals.
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: run-swift-test-set.sh TAG}"
OUT=".mlxfast-private/swift-tests/${TAG}"
mkdir -p "$OUT"

HEAD_SHA="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
echo "run-swift-test-set: tag=${TAG} head=${HEAD_SHA} dirty=${DIRTY}" | tee "$OUT/identity.txt"

swift test --force-resolved-versions -c debug >"$OUT/raw.log" 2>&1
RC=$?
echo "swift_test_exit=${RC}" | tee -a "$OUT/identity.txt"

grep -oE '(Test|Suite) [A-Za-z0-9_()."'"'"' :-]+ (failed|recorded an issue)' "$OUT/raw.log" \
  | sed -E 's/ (failed|recorded an issue).*//' | sort -u >"$OUT/failed.txt"
grep -oE "error: .*" "$OUT/raw.log" | sort -u >"$OUT/errors.txt"
grep -oE 'Test run with [0-9]+ tests? (in [0-9]+ suites? )?(passed|failed) after [0-9.]+ seconds( with [0-9]+ issues?)?' \
  "$OUT/raw.log" | sort -u >"$OUT/totals.txt"

echo "--- totals"; cat "$OUT/totals.txt"
echo "--- distinct failing identifiers: $(wc -l <"$OUT/failed.txt" | tr -d ' ')"
cat "$OUT/failed.txt"
echo "--- compile errors: $(wc -l <"$OUT/errors.txt" | tr -d ' ')"
head -20 "$OUT/errors.txt"
exit "$RC"
