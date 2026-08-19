// E59: AIR census of one width case, for a dispatch table that may route a
// width through a row-block wrapper instead of the shipped one.
//
//   E59_CELL_M / E59_CELL_IPG   the width case and its accumulator width
//   E59_CELL_WRAPPER            the wrapper that width's `case` actually calls
//
// The wrapper is a parameter because E59's whole question is what a width costs
// when it is reached through `..._m_rb2` or `..._m_rbx` at rows_per_simd = 2
// rather than through `..._m` at 4. Hard-coding the shipped wrapper here would
// census a cell the patched table never dispatches.
//
// The include path decides which dispatch table is measured, so an arm is a
// patched copy of quantized.h in a shadow include dir, never an edit here.
//
//   python3 research/e59_reg_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#ifndef E59_CELL_WRAPPER
#define E59_CELL_WRAPPER qmv_fast_crossrow_affine4_g64_m
#endif

[[kernel]] void e59_cell(
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
  E59_CELL_WRAPPER<bfloat16_t, E59_CELL_M, E59_CELL_IPG, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}
