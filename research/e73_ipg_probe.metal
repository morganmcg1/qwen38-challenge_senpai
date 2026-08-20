// E73: one isolated entry point per legal `(M, IPG)` group partition, each
// calling the SHIPPED wrapper `qmv_fast_crossrow_affine4_g64_m` unchanged, so
// the AIR of a partition is attributable to that partition alone.
//
// The wrapper, not the helper, is the unit under study: it computes
// `first_m = tid.x * IPG`, returns on the idle x-slots, and picks the tail
// body. Calling the helper directly (as E69 did) would hide both the idle
// slots and the second inlined body that a non-zero tail creates.
//
// Legality mirrors the two static asserts in the shipped code:
// `3 <= M <= 9`, `2 <= IPG <= 6`, `M % IPG != 1`.
//
// The host grid is FROZEN at `grid_dims(M, ceil(N/8), B)` with
// `group_dims(32, 2, 1)` (backend/metal/quantized.cpp, not an editable path),
// so every arm here runs the same geometry the scored path gives it.
//
// Research only. Nothing here is on the scored path.
//
//   python3 research/e73_air_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define E73_PROBE_ARGS                                                     \
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

#define E73_CELL(M, IPG)                                                   \
  [[kernel]] void e73_cell_m##M##_ipg##IPG(E73_PROBE_ARGS) {               \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>(             \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        tid, simd_gid, simd_lid);                                          \
  }

E73_CELL(3, 3)
E73_CELL(4, 2)
E73_CELL(4, 4)
E73_CELL(5, 3)
E73_CELL(5, 5)
E73_CELL(6, 2)
E73_CELL(6, 3)
E73_CELL(6, 4)
E73_CELL(6, 6)
E73_CELL(7, 4)
E73_CELL(7, 5)
E73_CELL(8, 2)
E73_CELL(8, 3)
E73_CELL(8, 4)
E73_CELL(8, 5)
E73_CELL(8, 6)
E73_CELL(9, 3)
E73_CELL(9, 5)
E73_CELL(9, 6)
