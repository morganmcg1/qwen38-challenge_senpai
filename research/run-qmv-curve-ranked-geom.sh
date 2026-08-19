#!/usr/bin/env bash
# Run a cost curve under the ranked box's MLX command-buffer geometry.
#
# QwenQMVCostCurveTests never constructs QwenRuntimeMTPWorker, so
# RuntimeStartupMemoryPolicy never runs and the curve normally inherits MLX's
# own architecture defaults.  On this host (applegpu_g16s, class 's') those are
# 50 ops / 50 MB; the ranked box (>=96 GiB) would get 512 MB with the same 50
# ops.  Only the byte limit differs, and MLX reads both through `static` locals
# in mlx/utils.h, so the environment must be set before the process starts.
#
# DARKBLOOM_STARTUP_MEMORY_PROFILE is deliberately NOT set: it feeds
# RuntimeStartupMemoryPolicy, which this test never reaches, so setting it would
# imply a coverage this arm does not have.  Weight residency is hard-gated on
# >=96 GiB with no env override and stays untestable on this host.
#
#   research/run-qmv-curve-ranked-geom.sh TAG BASE_SHA [run-qmv-curve args...]
set -euo pipefail

export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "run-qmv-curve-ranked-geom: MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER}" \
     "MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER}"
exec research/run-qmv-curve.sh "$@"
