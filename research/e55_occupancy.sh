#!/usr/bin/env bash
# Research-only (qwen38-r1-e55): does the NA=5 body cost the SHARED kernel any
# occupancy?
#
# The M switch lives INSIDE one kernel, so every dispatched width shares that
# kernel's register allocation. E49 Arm 2 bounded the harm from ADDING registers
# to the shipped table at |dScore| <= 0.0876 %, but it never measured the
# allocation of the composed NA=5 body itself. `vec<float,5>` pads to 32 bytes /
# 8 lanes (research/e55_vec5_check.swift), so `acc[4]` doubles from 64 to 128
# vector bytes and a spill would slow EVERY width, not just M=9.
#
# maxTotalThreadsPerThreadgroup is the cheapest direct readout of the
# register-pressure ceiling the AIR text cannot show.
#
#   research/e55_occupancy.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HEADER=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
METAL=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal
OUT="${TMPDIR:-/tmp}/e55-occupancy"
mkdir -p "${OUT}"

swiftc -O research/crossrow_na_occupancy.swift -o "${OUT}/na_occupancy" || exit 1

original="$(mktemp)"
cp "${HEADER}" "${original}"
restore() { cp "${original}" "${HEADER}"; rm -f "${original}"; }
trap restore EXIT

compile_arm() {
  local arm="$1"
  echo "=== arm=${arm} ==="
  grep -n 'crossrow_affine4_g64_m<T, 9,' "${HEADER}" | sed 's/^/    /'
  grep -n 'static_assert(NA >= 2' "${HEADER}" | sed 's/^/    /'
  xcrun -sdk macosx metal -std=metal3.1 -O2 -c "${METAL}" \
    -I Vendor/mlx-swift/Source/Cmlx/mlx -o "${OUT}/${arm}.air" || return 1
  xcrun -sdk macosx metallib "${OUT}/${arm}.air" -o "${OUT}/${arm}.metallib" || return 1
  "${OUT}/na_occupancy" "${OUT}/${arm}.metallib" \
    | grep -E 'device=|affine_qmv_fast.*gs_64_b_4|^name' \
    | tee "${OUT}/${arm}.txt"
}

# The header currently holds whichever arm the worktree is on; normalise to base
# first, then to the candidate, so the comparison never depends on entry state.
perl -0pi -e 's/qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>\(/qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(/' "${HEADER}"
perl -0pi -e 's/static_assert\(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in \[2, 5\]"\);/static_assert(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in [2, 4]");/' "${HEADER}"
compile_arm base || exit 1

perl -0pi -e 's/static_assert\(NA >= 2 && NA <= 4, "wide multi-row QMV supports NA in \[2, 4\]"\);/static_assert(NA >= 2 && NA <= 5, "wide multi-row QMV supports NA in [2, 5]");/' "${HEADER}"
perl -0pi -e 's/qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>\(/qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>(/' "${HEADER}"
compile_arm m9two || exit 1

echo "=== diff base vs m9two (empty means the shared ceiling did NOT move) ==="
diff "${OUT}/base.txt" "${OUT}/m9two.txt" && echo "    (identical)"
