#!/bin/bash
# e102: blob fingerprint of the scored QMV kernel surface for every upstream
# submission branch.
#
# A tree whose five QMV blobs all match the reference tree 9b241879 (byte
# identical kernel table to the current campaign base) carries NO wide-row
# change, so it is a valid clean control inside a same-schedule cohort.
# Also records the Sources/MLXFastModel subtree hash, because a matching
# schedule vector does not by itself prove a matching session/head policy.
#
# Writes /tmp/e102_kernel_fp.tsv (branch, 5 blob ids, mtp subtree id).
set -u
cd "$(dirname "$0")/.." || exit 1
PATHS=(
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/metal/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal"
  "Sources/MLXFastModel"
)
out=/tmp/e102_kernel_fp.tsv
: > "$out"
while read -r ref _commit _tree; do
  id="${ref#upstream/submissions/}"
  line="$id"
  while read -r _mode _type obj path; do
    line="$line	$path=$obj"
  done < <(git ls-tree "$ref" -- "${PATHS[@]}")
  printf '%s\n' "$line" >> "$out"
done < /tmp/e102_trees.txt
wc -l "$out"
