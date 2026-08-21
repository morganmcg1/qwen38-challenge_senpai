#!/usr/bin/env bash
# Rebuild every artifact E87 measures, in dependency order, and certify the
# kernel arm inside the built worker.
#
#   usage: research/e87_rebuild.sh
#
# 1. mlx.metallib, published to BOTH build roots. The `quantized` family is
#    JIT-compiled from Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp,
#    so the metallib does not carry the E87 kernel edit, but its fingerprint
#    sidecar covers the whole vendored tree and goes stale on any edit there.
# 2. The trusted CLI, which research/e87_capture.sh runs as `mtp-verify`.
# 3. The participant worker, which ./benchmark-qwen-mtp.sh times. Only this
#    binary carries the quantized JIT string, so it is the only artifact whose
#    content can witness the kernel arm.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 1/3 mlx.metallib ==="
tools/build-mlx-metallib.sh --all-build-roots || exit $?

echo "=== 2/3 trusted CLI ==="
swift build -c release --force-resolved-versions || exit $?

echo "=== 3/3 participant worker + arm certificate ==="
# The E87 witnesses. The generalised single-row affine-2 kernel must be in the
# JIT string and its g64-only predecessor must be gone; the two standing
# campaign witnesses guard the base the arm is measured against.
senpai/rebuild-and-assert-worker.sh \
  --require 'qmv_fast_singlerow_affine2<T, group_size>' \
  --forbid 'qmv_fast_singlerow_affine2_g64' \
  --require 'qwen35_dual_rms_norm_concat_bf16_v1' \
  --forbid 'qwen35_dual_rms_norm_bf16_v1' \
  --require-symbol snapshotScheduleSignal || exit $?

echo "=== fingerprints ==="
echo "vendored_metal_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
for root in .build .build-worker; do
  echo "${root}_sidecar=$(cat "${root}/release/mlx.metallib.fingerprint" 2>/dev/null)"
done
echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
echo "worker_sha256=$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
