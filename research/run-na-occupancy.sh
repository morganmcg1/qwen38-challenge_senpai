#!/bin/bash
# Report register-pressure-limited occupancy of the crossrow inner loop for each
# experiment arm, by rebuilding the single-NA probe against that arm's quantized.h.
#
#   research/run-na-occupancy.sh CONTROL_SHA NAIVE_SHA ORDERED_SHA
set -uo pipefail

cd "$(dirname "$0")/.."
HEADER=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
PROBE=research/crossrow_na_probe.metal

swiftc -O research/crossrow_na_occupancy.swift -o /tmp/na_occupancy || exit 1

restore() { git checkout -- "$HEADER"; }
trap restore EXIT

for spec in "$@"; do
  arm="${spec%%=*}"
  sha="${spec##*=}"
  echo "=== arm=$arm sha=$sha ==="
  git checkout "$sha" -- "$HEADER" || exit 1
  xcrun -sdk macosx metal -std=metal3.1 -O2 -c "$PROBE" \
    -I Vendor/mlx-swift/Source/Cmlx/mlx -o "/tmp/probe_${arm}.air" || exit 1
  xcrun -sdk macosx metallib "/tmp/probe_${arm}.air" -o "/tmp/probe_${arm}.metallib" || exit 1
  /tmp/na_occupancy "/tmp/probe_${arm}.metallib" || exit 1
  echo
done
