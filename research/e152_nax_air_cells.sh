#!/usr/bin/env bash
# E152 F5: does our `quantized_nax.h`, which carries E147's disarmed retile
# scaffolding, generate the same code as the frontier's unmodified header?
#
# `quantized_nax.metal` is byte-identical between the two trees and it holds
# every instantiation the metallib carries, so compiling that one file against
# each header instantiates exactly the cells the runtime can reach. The JIT
# path reads the generated twin instead of the header, and `twin_audit.py`
# already pins the twin to the header, so one header comparison settles both.
#
# Compile-only: `metal -S` plus `metal-opt`. No metallib, no MTLDevice, no
# pipeline state, no dispatch, zero GPU.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

FRONTIER="${MLXFAST_E152_FRONTIER:-0863b06ac16e26e48fc06e97444095b00feb66d4}"
HEADER="mlx/backend/metal/kernels/quantized_nax.h"
INC="Vendor/mlx-swift/Source/Cmlx/mlx"
SRC="${INC}/${HEADER%.h}.metal"
OUT="${MLXFAST_E152_AIR_OUT:-/tmp/e152/air}"

mkdir -p "${OUT}/ours/$(dirname "${HEADER}")" "${OUT}/frontier/$(dirname "${HEADER}")"
cp "${INC}/${HEADER}" "${OUT}/ours/${HEADER}"
git show "${FRONTIER}:Vendor/mlx-swift/Source/Cmlx/${HEADER#mlx/}" \
  > "${OUT}/frontier/${HEADER}" 2>/dev/null \
  || git show "${FRONTIER}:Vendor/mlx-swift/Source/Cmlx/mlx/${HEADER}" \
  > "${OUT}/frontier/${HEADER}" || { echo "FAIL: no frontier header"; exit 1; }

echo "E152 NAX per-cell AIR digest (compile-only, zero GPU)"
echo "  toolchain: $(xcrun --sdk macosx metal --version 2>&1 | head -1)"
echo "  source:    ${SRC}"
echo "  ours:      $(wc -c < "${OUT}/ours/${HEADER}") bytes"
echo "  frontier:  $(wc -c < "${OUT}/frontier/${HEADER}") bytes  (${FRONTIER:0:8})"
echo

# The flags are the ones `kernels/CMakeLists.txt:12-35` gives every kernel.
# It passes no `-std`, so the `_nax` cells compile against the toolchain
# default; `-std=metal3.1` is older than `mpp::tensor_ops::matmul2d` and fails.
for arm in ours frontier; do
  xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
    -Wno-c++17-extensions -Wno-c++20-extensions -S \
    -I "${OUT}/${arm}" -I "${INC}" "${SRC}" -o "${OUT}/${arm}.ll" \
    || { echo "FAIL: ${arm} header did not compile"; exit 1; }
  xcrun -sdk macosx metal-opt -passes='default<O3>' -S \
    "${OUT}/${arm}.ll" -o "${OUT}/${arm}.o3.ll" || exit 1
  echo "  ${arm}: AIR $(wc -l < "${OUT}/${arm}.ll") lines, O3 $(wc -l < "${OUT}/${arm}.o3.ll") lines"
done
echo

python3 research/e152_nax_air_digest.py \
  "${OUT}/ours.o3.ll" "${OUT}/frontier.o3.ll" \
  --left-label ours --right-label frontier "$@"
