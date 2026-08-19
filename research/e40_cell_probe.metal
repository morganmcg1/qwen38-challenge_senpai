// E40: every width cell that `affine_qmv_fast` inlines, isolated.
//
// `affine_qmv_fast` switches on the RUNTIME value `ntg.x` (quantized.h:1925 and
// :1971) inside ONE [[kernel]], so the compiled pipeline carries a single
// register footprint that must satisfy the worst inlined branch. E27 changed
// only `case 5` and `case 9` of the >=4096 branch, but if it raised the
// kernel-wide maximum then the code-identical cells M=3,4,6,7,8 pay too. That
// is the only route by which E27 can cost anything at an unchanged width, and
// it is the testable form of H1.
//
// This probe isolates each cell so the per-cell footprint is comparable and the
// kernel-wide maximum can be taken explicitly:
//
//   * the >=4096 dispatch cells, as shipped and as baselined;
//   * the <4096 dispatch cells `qmv_fast_crossrow_affine4_g64<T, M>`, which E27
//     never touched -- if one of those already exceeds the NA=5 footprint then
//     the kernel-wide maximum did not move and H1 is dead.
//
// Build with research/e40_cell_air.sh. Compile-only; no metallib, no MTLDevice.
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define E40_CELL_ARGS                                                      \
  const device uint32_t* w [[buffer(0)]],                                  \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]

#define E40_CELL_CALL                                                      \
  w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid

// One dispatch cell of the >=4096 branch, exactly as the switch instantiates it.
#define E40_M_CELL(name, M, IPG)                                           \
  [[kernel]] void name(E40_CELL_ARGS) {                                    \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>(             \
        E40_CELL_CALL);                                                    \
  }

// One dispatch cell of the <4096 branch, which E27 does not touch.
#define E40_N_CELL(name, M)                                                \
  [[kernel]] void name(E40_CELL_ARGS) {                                    \
    qmv_fast_crossrow_affine4_g64<bfloat16_t, M>(E40_CELL_CALL);           \
  }

// >=4096 cells. M=3,4,6,7,8 are identical in both arms; M=5 and M=9 are the
// only cells E27 rewrote, so both arms of those two are instantiated here and
// the arm that does not exist in a given commit is simply the other's control.
E40_M_CELL(e40_m3_ipg3, 3, 3)
E40_M_CELL(e40_m4_ipg4, 4, 4)
E40_M_CELL(e40_m5_ipg3_base, 5, 3)
E40_M_CELL(e40_m5_ipg5_cand, 5, 5)
E40_M_CELL(e40_m6_ipg3, 6, 3)
E40_M_CELL(e40_m7_ipg4, 7, 4)
E40_M_CELL(e40_m8_ipg4, 8, 4)
E40_M_CELL(e40_m9_ipg3_base, 9, 3)
E40_M_CELL(e40_m9_ipg5_cand, 9, 5)

// <4096 cells, untouched by E27 and inlined into the same kernel.
E40_N_CELL(e40_narrow_m2, 2)
E40_N_CELL(e40_narrow_m3, 3)
E40_N_CELL(e40_narrow_m4, 4)
E40_N_CELL(e40_narrow_m5, 5)
E40_N_CELL(e40_narrow_m6, 6)
E40_N_CELL(e40_narrow_m7, 7)
E40_N_CELL(e40_narrow_m8, 8)
E40_N_CELL(e40_narrow_m9, 9)

// The inner packing factor on its own, to reproduce the E13/E27/E32 anchor
// (62/83/104/125 for NA=2/3/4/5) on this toolchain rather than citing it.
#define E40_WIDE_CELL(name, NA)                                            \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      const constant int& first_m [[buffer(7)]],                           \
      const constant int& out_row [[buffer(8)]],                           \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true>(              \
        w, scales, biases, x, y, in_vec_size, out_vec_size, first_m,        \
        out_row, simd_lid);                                                \
  }

E40_WIDE_CELL(e40_wide_na2, 2)
E40_WIDE_CELL(e40_wide_na3, 3)
E40_WIDE_CELL(e40_wide_na4, 4)
E40_WIDE_CELL(e40_wide_na5, 5)
