// E63 rung 0: isolate ONE `qmv_fast_crossrow_affine4_g64_wide<T, NA, true>` cell
// so the AIR for a single NA can be read without the seven width cases inlined
// on top of it.
//
// NA is the ladder variable of the bandwidth cliff (2 -> 7). The shipped header
// asserts NA <= 5, so the census patches that assert in a SHADOW copy of
// quantized.h; this file is never edited per arm.
//
// The wrapper reproduces the address arithmetic `..._m` performs before it calls
// the helper, so the helper sees the same `first_m` / `out_row` provenance it
// sees on the scored path and no address computation is folded away.
//
//   python3 research/e63_mlp_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#ifndef E63_NA
#define E63_NA 3
#endif

#ifndef E63_DIRECT_NIBBLES
#define E63_DIRECT_NIBBLES 1
#endif

[[kernel]] void e63_wide_cell(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E63_NA;
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, E63_NA, E63_DIRECT_NIBBLES>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, out_row, simd_lid);
}
