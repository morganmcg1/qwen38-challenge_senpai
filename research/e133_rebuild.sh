#!/usr/bin/env bash
# E133 rung 1: rebuild the CLI and the worker WITH the E87 hidden-state
# instrument compiled in, and certify that the instrument is actually there.
#
# research/e87_rebuild.sh cannot be reused at this base. Its arm witnesses
# describe E87's tree: `qmv_fast_singlerow_affine2<T, group_size>` has zero
# occurrences in Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp at
# 197e0550, so that guard fails a correct build.
#
# The witnesses here describe THIS experiment's only source change:
#   --require        the gate string literal, which reaches the string table.
#   --require-symbol the instrument type, which reaches the symbol table only.
# Both must hold, or the capture would run an uninstrumented binary and write
# an empty dump.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 1/3 mlx.metallib (both build roots) ==="
tools/build-mlx-metallib.sh --all-build-roots || exit $?

echo "=== 2/3 trusted CLI ==="
swift build -c release --force-resolved-versions || exit $?

echo "=== 3/3 participant worker + instrument certificate ==="
senpai/rebuild-and-assert-worker.sh \
  --require 'MLX_E87_HIDDEN_DUMP' \
  --require-symbol E87HiddenDump || exit $?

echo "=== fingerprints ==="
echo "vendored_metal_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
for root in .build .build-worker; do
  echo "${root}_sidecar=$(cat "${root}/release/mlx.metallib.fingerprint" 2>/dev/null)"
done
echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
echo "worker_sha256=$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "cli_instrument_strings=$(
  strings -a .build/release/mlxfast-swift | grep -c -F MLX_E87_HIDDEN_DUMP)"
echo "worker_instrument_strings=$(
  strings -a .build-worker/release/mlxfast-runtime-worker \
    | grep -c -F MLX_E87_HIDDEN_DUMP)"
