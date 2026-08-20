// E72 rung 2: isolated `qmv_fast_crossrow_affine4_g64_wide` arms at one NA, so
// the AIR of each arm is attributable to that arm and the GPU can time them
// against each other inside one thermal session.
//
// Every wrapper reproduces the address arithmetic `..._m` performs before it
// calls the helper, so the helper sees the same `first_m` / `out_row`
// provenance it sees on the scored path.
//
//   python3 research/e72_wide_gen.py
//   python3 research/e72_air_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#include "../research/generated/e72_wide_arms.h"

#ifndef E72_NA
#define E72_NA 6
#endif

#define E72_PROBE_ARGS                                                     \
    const device uint32_t* w [[buffer(0)]],                                \
    const device bfloat16_t* scales [[buffer(1)]],                         \
    const device bfloat16_t* biases [[buffer(2)]],                         \
    const device bfloat16_t* x [[buffer(3)]],                              \
    device bfloat16_t* y [[buffer(4)]],                                    \
    const constant int& in_vec_size [[buffer(5)]],                         \
    const constant int& out_vec_size [[buffer(6)]],                        \
    uint3 tid [[threadgroup_position_in_grid]],                            \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lid [[thread_index_in_simdgroup]]

#define E72_CELL(tag, arm)                                                 \
  [[kernel]] void e72_cell_##tag(E72_PROBE_ARGS) {                         \
    const int first_m = int(tid.x) * E72_NA;                               \
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;                \
    arm<bfloat16_t, E72_NA, true>(                                         \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid);                                       \
  }

E72_FOR_EACH_ARM(E72_CELL)
