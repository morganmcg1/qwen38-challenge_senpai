#!/usr/bin/env bash
# E162 correctness evidence: the full Swift suite, then the opt-in MLX runtime
# suite that actually executes kernels. Arm A is the correctness harness for the
# double-buffer transformation, so this is the gate that has to be green before
# arm B ships blind to the ranked M5.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/out/e162/tests"
rm -rf "${out}"
mkdir -p "${out}"

echo "head=$(git rev-parse HEAD)" > "${out}/meta.txt"
echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')" >> "${out}/meta.txt"

swift test --force-resolved-versions > "${out}/plain.log" 2>&1
plain=$?
echo "plain_exit=${plain}" >> "${out}/meta.txt"
grep -E "^Test Suite '(All tests|selected tests)'|Executed [0-9]+ tests" \
  "${out}/plain.log" | tail -4 >> "${out}/meta.txt"

MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
  > "${out}/runtime.log" 2>&1
runtime=$?
echo "runtime_exit=${runtime}" >> "${out}/meta.txt"
grep -E "^Test Suite '(All tests|selected tests)'|Executed [0-9]+ tests" \
  "${out}/runtime.log" | tail -4 >> "${out}/meta.txt"

echo "=== meta ==="
cat "${out}/meta.txt"
echo
echo "=== failures, if any ==="
grep -E "error:|failed|XCTAssert" "${out}/plain.log" "${out}/runtime.log" \
  | grep -v "warning:" | head -30
