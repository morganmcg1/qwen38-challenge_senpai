// E44 Gate 0: every width cell `affine_qmv_fast` inlines, plus the
// simdgroup-matrix candidate cell, isolated so the KERNEL-WIDE maximum of each
// dispatch table can be taken explicitly.
//
// `affine_qmv_fast` (quantized.h) switches on the RUNTIME value `ntg.x` inside
// ONE [[kernel]] and every helper is METAL_FUNC, so the compiled pipeline
// carries a single register allocation equal to the worst inlined branch. E27's
// revert is the campaign's proof that this is load-bearing: a cell that raises
// the kernel-wide maximum taxes the widths it never touched. So the candidate's
// register claim has to be settled before any timing claim.
//
// Compile-only. Build with research/e44_sgmm_air.sh: `metal -S` plus
// `metal-opt`, no metallib, no MTLDevice, no pipeline state, no dispatch.
//
// The candidate cells are behind -DE44_SGMM=1 so the SAME probe file measures
// the pre-implementation baseline arm and the candidate arm.
#include <metal_stdlib>

// Same include order as the AOT kernels/quantized.metal and as the JIT
// concatenation in jit_kernels.cpp get_quantized_kernel (utils, gemm,
// quantized_utils, quantized): quantized.h already REQUIRES mlx::steel from
// steel/gemm/gemm.h for qmm_t_impl, in both compile paths.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define E44_CELL_ARGS                                                      \
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

#define E44_CELL_CALL                                                      \
  w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid

// One dispatch cell of the >=4096 branch, exactly as the switch instantiates it.
#define E44_M_CELL(name, M, IPG)                                           \
  [[kernel]] void name(E44_CELL_ARGS) {                                    \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>(             \
        E44_CELL_CALL);                                                    \
  }

// One dispatch cell of the <4096 branch, shared by both arms.
#define E44_N_CELL(name, M)                                                \
  [[kernel]] void name(E44_CELL_ARGS) {                                    \
    qmv_fast_crossrow_affine4_g64<bfloat16_t, M>(E44_CELL_CALL);           \
  }

// >=4096 cells as the post-E27-revert tree dispatches them:
// <T,2> <T,3,3> <T,4,4> <T,5,3> <T,6,3> <T,7,4> <T,8,4> <T,9,3>.
E44_M_CELL(e44_m3_ipg3, 3, 3)
E44_M_CELL(e44_m4_ipg4, 4, 4)
E44_M_CELL(e44_m5_ipg3, 5, 3)
E44_M_CELL(e44_m6_ipg3, 6, 3)
E44_M_CELL(e44_m7_ipg4, 7, 4)
E44_M_CELL(e44_m8_ipg4, 8, 4)
E44_M_CELL(e44_m9_ipg3, 9, 3)

// <4096 cells, inlined into the same kernel and untouched by either arm.
E44_N_CELL(e44_narrow_m2, 2)
E44_N_CELL(e44_narrow_m3, 3)
E44_N_CELL(e44_narrow_m4, 4)
E44_N_CELL(e44_narrow_m5, 5)
E44_N_CELL(e44_narrow_m6, 6)
E44_N_CELL(e44_narrow_m7, 7)
E44_N_CELL(e44_narrow_m8, 8)
E44_N_CELL(e44_narrow_m9, 9)

// The inner packing factor on its own, to re-derive the E13/E27/E32/E40
// NA=2/3/4 anchor (62/83/104) on this toolchain and tree rather than citing it.
#define E44_WIDE_CELL(name, NA)                                            \
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

E44_WIDE_CELL(e44_wide_na2, 2)
E44_WIDE_CELL(e44_wide_na3, 3)
E44_WIDE_CELL(e44_wide_na4, 4)

#if defined(E44_SGMM)
// The candidate cell. `m_rows` is a runtime value in the shipped dispatch
// (int(ntg.x)), which is why ONE instantiation replaces six per-width ones.
#define E44_SGMM_CELL(name, MROWS)                                         \
  [[kernel]] void name(E44_CELL_ARGS) {                                    \
    qmv_fast_crossrow_affine4_g64_sgmm<bfloat16_t>(                        \
        w, scales, biases, x, y, in_vec_size, out_vec_size, MROWS, tid,     \
        simd_gid, simd_lid);                                               \
  }

// `runtime` is the shipped form. The fixed-M forms are the same code with a
// constant-folded bound and exist only to show whether constant folding, and
// with it the m_tiles == 2 path at M = 9, changes the footprint.
E44_SGMM_CELL(e44_sgmm_runtime, int(in_vec_size & 0xf))
E44_SGMM_CELL(e44_sgmm_m8, 8)
E44_SGMM_CELL(e44_sgmm_m9, 9)
#endif
