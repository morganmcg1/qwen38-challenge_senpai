#!/usr/bin/env bash
# E137 item 2: the isolated QMV cost curve across verify widths.
#
# `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` has carried
# `sweepGatedDelta(widths: 1...12)` since E92 and the ledger names it twice as
# the first thing to run when a slot frees. It has never been run.
#
# The test loads no model and holds no checkpoint. It times the vendored MLX
# kernels the scored worker links, at the exact scored shapes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

OUT="${1:-research/e137-artifacts/item2-cost-curve.json}"
mkdir -p "$(dirname "$OUT")"

export MLXFAST_RUN_QMV_COST_CURVE=1
export MLXFAST_QMV_COST_CURVE_OUT="$ROOT/$OUT"

swift test --force-resolved-versions --filter QwenQMVCostCurve
