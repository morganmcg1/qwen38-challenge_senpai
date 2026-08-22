#!/usr/bin/env bash
# E137 item 2, corrected: the isolated QMV width curve on the SCORED path.
#
# `QwenQMVCostCurveTests` calls `quantizedMM` directly and therefore measures
# the fallback path. The scored call site is `qwen35RoutedQuantizedMM`
# (`Qwen35.swift:2252-2270`), which tries `Qwen35CustomQMV.matmul` first. This
# sweeps both paths over the same shapes, widths, weights and session.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

OUT="${1:-research/e137-artifacts/item2-routeb-curve.json}"
mkdir -p "$(dirname "$OUT")"

export MLXFAST_RUN_E137_ROUTEB_CURVE=1
export MLXFAST_E137_ROUTEB_CURVE_OUT="$ROOT/$OUT"

swift test --force-resolved-versions --filter E137RouteBCostCurve
