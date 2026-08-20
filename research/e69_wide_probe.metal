// E69: isolated `qmv_fast_crossrow_affine4_g64_wide` arms at one NA, so the AIR
// of each arm is attributable to that arm and the GPU can time them against
// each other inside one thermal session.
//
// Every wrapper reproduces the address arithmetic `..._m` performs before it
// calls the helper, so the helper sees the same `first_m` / `out_row`
// provenance it sees on the scored path. The host grid is FROZEN at
// `grid_dims(M, ceil(N/8), B)` with `group_dims(32, 2, 1)`
// (backend/metal/quantized.cpp:253-254, not an editable path), so the two
// rows_per_simd = 8 wrappers below both live inside that grid: `rows8` folds
// two adjacent 8-row tiles into one threadgroup and returns on odd tid.y,
// `rows8idle` keeps one tile per threadgroup and returns on simd_gid == 1.
//
//   python3 research/e69_wide_gen.py
//   python3 research/e69_air_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#include "../research/generated/e69_wide_arms.h"

#ifndef E69_NA
#define E69_NA 6
#endif

#define E69_PROBE_ARGS                                                     \
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

// rows_per_simd = 4: the shipped geometry.
#define E69_CELL(name, arm)                                                \
  [[kernel]] void name(E69_PROBE_ARGS) {                                   \
    const int first_m = int(tid.x) * E69_NA;                               \
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;                \
    arm<bfloat16_t, E69_NA, true>(                                         \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid);                                       \
  }

#define E69_CELL_TG(name, arm)                                             \
  [[kernel]] void name(E69_PROBE_ARGS) {                                   \
    threadgroup bfloat16_t xs[E69_NA * 512];                               \
    const int first_m = int(tid.x) * E69_NA;                               \
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;                \
    arm<bfloat16_t, E69_NA, true>(                                         \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid, simd_gid, xs);                         \
  }

// rows_per_simd = 8, tile folding: threadgroup 2t covers rows 16t..16t+15 and
// threadgroup 2t+1 returns. Needs out_vec_size % 16 == 0, which every scored
// affine-4 projection satisfies.
#define E69_CELL_R8(name, arm)                                             \
  [[kernel]] void name(E69_PROBE_ARGS) {                                   \
    if ((tid.y & 1u) != 0u) {                                              \
      return;                                                              \
    }                                                                      \
    const int first_m = int(tid.x) * E69_NA;                               \
    const int out_row = int(tid.y) * 8 + int(simd_gid) * 8;                \
    arm<bfloat16_t, E69_NA, true>(                                         \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid);                                       \
  }

// rows_per_simd = 8, idle half: every threadgroup keeps its own 8-row tile and
// the second simdgroup returns. Same total simdgroup work as `rows8` on twice
// the threadgroups, so the contrast isolates occupancy from dispatch count.
#define E69_CELL_R8_IDLE(name, arm)                                        \
  [[kernel]] void name(E69_PROBE_ARGS) {                                   \
    if (simd_gid != 0u) {                                                  \
      return;                                                              \
    }                                                                      \
    const int first_m = int(tid.x) * E69_NA;                               \
    const int out_row = int(tid.y) * 8;                                    \
    arm<bfloat16_t, E69_NA, true>(                                         \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        first_m, out_row, simd_lid);                                       \
  }

E69_CELL(e69_cell_plain, qmv_fast_crossrow_affine4_g64_wide_e69plain)
E69_CELL(e69_cell_wvec, qmv_fast_crossrow_affine4_g64_wide_e69wvec)
E69_CELL(e69_cell_xvec, qmv_fast_crossrow_affine4_g64_wide_e69xvec)
E69_CELL(e69_cell_wxvec, qmv_fast_crossrow_affine4_g64_wide_e69wxvec)
E69_CELL_TG(e69_cell_tgx, qmv_fast_crossrow_affine4_g64_wide_e69tgx)
E69_CELL_R8(e69_cell_rows8, qmv_fast_crossrow_affine4_g64_wide_e69rows8)
E69_CELL_R8(e69_cell_rows8wxvec,
            qmv_fast_crossrow_affine4_g64_wide_e69rows8wxvec)
E69_CELL_R8_IDLE(e69_cell_rows8idle,
                 qmv_fast_crossrow_affine4_g64_wide_e69rows8)
