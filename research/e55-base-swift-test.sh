#!/usr/bin/env bash
# Research-only (qwen38-r1-e55): run `swift test` on the BASE twins so the
# candidate's failing set can be compared against a matched measurement instead
# of a static argument.
#
# The script always restores the candidate twins, including on failure, because
# leaving base twins in the worktree would silently mismeasure everything after.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_TWINS=f2ec48a
CAND_TWINS=2267a84
HEADER=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
TWIN=Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
OUT=.mlxfast-private/e55/base-swift-test.log

mkdir -p "$(dirname "$OUT")"

restore() {
  git checkout "$CAND_TWINS" -- "$HEADER" "$TWIN"
  echo "restored candidate twins:"
  grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>' "$HEADER" "$TWIN"
}
trap restore EXIT

git checkout "$BASE_TWINS" -- "$HEADER" "$TWIN"
echo "base twins in place (expect NA <= 4 and <T, 9, 3, true>):"
grep -n 'static_assert(NA >= 2' "$HEADER"
grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>' "$HEADER" "$TWIN"

swift test --force-resolved-versions 2>&1 | tee "$OUT"
echo "swift_test_exit=${PIPESTATUS[0]}"
