// E64 rung 0b: three isolated `qmv_fast_crossrow_affine4_g64_wide` arms at one
// NA, so the AIR of each arm is attributable to that arm and the GPU can time
// them against each other inside one thermal session.
//
// The wrapper reproduces the address arithmetic `..._m` performs before it calls
// the helper, so the helper sees the same `first_m` / `out_row` provenance it
// sees on the scored path and no address computation is folded away. This is the
// same wrapper geometry E63 rung 0 used.
//
//   python3 research/e64_wide_gen.py
//   python3 research/e64_air_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#include "../research/generated/e64_wide_arms.h"

#ifndef E64_NA
#define E64_NA 5
#endif

#ifndef E64_DIRECT_NIBBLES
#define E64_DIRECT_NIBBLES 1
#endif

#define E64_PROBE_ARGS                                                     \
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

#define E64_CELL(name, arm)                                                \
  [[kernel]] void name(E64_PROBE_ARGS) {                                   \
    const int first_m = int(tid.x) * E64_NA;                               \
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;                \
    arm<bfloat16_t, E64_NA, E64_DIRECT_NIBBLES>(                           \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid);                                       \
  }

E64_CELL(e64_cell_plain, qmv_fast_crossrow_affine4_g64_wide_e64plain)
E64_CELL(e64_cell_forced, qmv_fast_crossrow_affine4_g64_wide_e64forced)
E64_CELL(e64_cell_ballast, qmv_fast_crossrow_affine4_g64_wide_e64ballast)

#ifdef E64_ROWS2_CELL
// Candidate 3. Each simdgroup owns 2 output rows instead of 4, so a threadgroup
// covers 4 rows and the host must dispatch twice the threadgroups. `tid.x` keeps
// its x-group meaning; an earlier revision hijacked it for the row half, which
// left half the output unwritten and made the arm compile-only. It is kept
// behind a macro so the source the timed rung-0b arms were compiled from stays
// byte-reproducible.
[[kernel]] void e64_cell_rows2(E64_PROBE_ARGS) {
  const int out_row = int(tid.y) * 4 + int(simd_gid) * 2;
  qmv_fast_crossrow_affine4_g64_wide_e64rows2<
      bfloat16_t, E64_NA, E64_DIRECT_NIBBLES>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * E64_NA, out_row, simd_lid);
}
#endif
