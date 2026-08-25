#!/usr/bin/env bash
# Research-only (qwen38-r1-e215): the matched-base Swift suite check for the
# E215 pin test.
#
#   usage: research/e215_suite_floor.sh
#
# The suite has a documented floor of 41 issues on this campaign base. A raw
# "41 issues" line proves nothing on its own, so this runs the SAME suite twice
# on this host, minutes apart, with and without the new test file, and compares
# the failing NAMES. The branch run must add the pin tests, keep the same issue
# count, and produce an empty failing-name diff.
#
# The base run is made by deleting the one file E215 adds under Tests/. Tests/
# is never packaged, so this patch cannot touch the submitted surface. The tree
# is restored before the script exits.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pin_test="Tests/MLXFastTests/E215DepthPriceTablePinTests.swift"
out="research/out/e215-suite"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e215_suite_floor.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }
[[ -s "${pin_test}" ]] || {
  echo "e215_suite_floor.sh: ${pin_test} is missing" >&2; exit 2; }

restore() { git checkout HEAD -- "${pin_test}" 2>/dev/null || true; }
trap restore EXIT

rm -rf "${out}"
mkdir -p "${out}"

echo "=== branch run (pin test present)"
swift test --force-resolved-versions > "${out}/branch.log" 2>&1
echo "branch exit=$?" | tee "${out}/branch.exit"

echo "=== base run (pin test removed)"
rm -f "${pin_test}"
swift test --force-resolved-versions > "${out}/base.log" 2>&1
echo "base exit=$?" | tee "${out}/base.exit"

restore
[[ -z "$(git status --porcelain)" ]] || {
  echo "e215_suite_floor.sh: tree not restored" >&2; exit 3; }

python3 research/e215_suite_report.py "${out}"
