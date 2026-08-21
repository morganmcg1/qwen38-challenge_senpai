#!/usr/bin/env bash
# E87 r2 rung 2: rebuild, run one short derived-index leg, and print every
# record the derived index left behind.
#
#   usage: research/e87_derived_debug.sh [PREFIX]     (default: e87dbg)
#
# The leg is a 64-token --local-submit, which is the cheapest run that still
# drives real draft proposals through draftTokenIDWithDeclaredRerank. It
# decides mechanism, not speed, so no thermal claim is made from it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:-e87dbg}"

research/e87_rebuild.sh || exit $?

rm -f /tmp/mlxfast-e87-derived.log
E87_TOKENS=64 research/e87_submit_gate.sh "${prefix}" derived
status=$?

echo "=== per-leg derived.log ==="
cat "research/out/${prefix}-derived/derived.log" 2>&1

echo "=== default-path derived log ==="
cat /tmp/mlxfast-e87-derived.log 2>&1

echo "=== order dump ==="
ls -l "research/out/${prefix}-derived/derived-order.bin" 2>&1

echo "=== score ==="
cat "research/out/${prefix}-derived/score.json" 2>&1

exit "${status}"
