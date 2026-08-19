// Isolated single-NA instantiations of the crossrow inner kernel.
//
// The production entry point inlines every NA at once, so its AIR cannot be
// attributed to one packing factor. Compiling one NA per kernel makes the
// register/private-memory cost of each packing factor directly comparable:
//
//   xcrun -sdk macosx metal -std=metal3.1 -S -O2 research/crossrow_na_probe.metal \
//     -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/probe.ll
//   python3 research/air_kernel_stats.py /tmp/probe.ll

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define CROSSROW_NA_PROBE(name, NA)                                        \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA>(                    \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

CROSSROW_NA_PROBE(crossrow_na2, 2)
CROSSROW_NA_PROBE(crossrow_na3, 3)
CROSSROW_NA_PROBE(crossrow_na4, 4)

// Opt-in arm. NA > 5 at 4 rows per simdgroup spills the accumulator array
// (E27); it stays behind the ifdef as the evidence for why the dispatch table
// row-blocks instead of widening NA in place.
#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_NA_PROBE(crossrow_na5, 5)
CROSSROW_NA_PROBE(crossrow_na6, 6)
#endif

// Every production dispatch site passes DIRECT_NIBBLES=true, which replaces the
// shifted masks with pre-shifted activations and so changes the live-value
// profile. The NA arms above are the historical DIRECT_NIBBLES=false form; these
// are what the scored kernel actually compiles.
#define CROSSROW_DN_PROBE(name, NA)                                        \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true>(              \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

CROSSROW_DN_PROBE(crossrow_dn_na2, 2)
CROSSROW_DN_PROBE(crossrow_dn_na3, 3)
CROSSROW_DN_PROBE(crossrow_dn_na4, 4)

#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_DN_PROBE(crossrow_dn_na5, 5)
CROSSROW_DN_PROBE(crossrow_dn_na6, 6)
#endif

// E33 row-blocked arms. `_rowblocked` covers the frozen 4 rows per simdgroup as
// 4/R sequential blocks, so only R accumulators are live at a time and NA may
// exceed 5. out_row is the SHIPPED expression; the wrapper walks the blocks.
#define CROSSROW_RB_PROBE(name, NA, R)                                     \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_rowblocked<bfloat16_t, NA, true, R>(     \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

CROSSROW_RB_PROBE(crossrow_rb_na6_r2, 6, 2)

#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_RB_PROBE(crossrow_rb_na7_r2, 7, 2)
CROSSROW_RB_PROBE(crossrow_rb_na8_r2, 8, 2)
CROSSROW_RB_PROBE(crossrow_rb_na9_r2, 9, 2)
#endif

// Whole-width arms. `_m` picks one `_wide` body per input group plus a tail
// body, so its AIR shows what a candidate dispatch-table entry really costs:
// `allocas == 2` means two inlined bodies, not a spill. A spill shows up as an
// extra `[4 x <NA x float>]` accumulator alloca.
#define CROSSROW_M_PROBE(name, M, IPG)                                     \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true>(             \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        tid, simd_gid, simd_lid);                                          \
  }

CROSSROW_M_PROBE(crossrow_m4_ipg2, 4, 2)
CROSSROW_M_PROBE(crossrow_m4_ipg4, 4, 4)
CROSSROW_M_PROBE(crossrow_m5_ipg3, 5, 3)
CROSSROW_M_PROBE(crossrow_m6_ipg2, 6, 2)
CROSSROW_M_PROBE(crossrow_m6_ipg3, 6, 3)
CROSSROW_M_PROBE(crossrow_m6_ipg4, 6, 4)
CROSSROW_M_PROBE(crossrow_m7_ipg4, 7, 4)
CROSSROW_M_PROBE(crossrow_m8_ipg2, 8, 2)
CROSSROW_M_PROBE(crossrow_m8_ipg3, 8, 3)
CROSSROW_M_PROBE(crossrow_m8_ipg4, 8, 4)
CROSSROW_M_PROBE(crossrow_m9_ipg3, 9, 3)

#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_M_PROBE(crossrow_m5_ipg5, 5, 5)
CROSSROW_M_PROBE(crossrow_m7_ipg5, 7, 5)
CROSSROW_M_PROBE(crossrow_m9_ipg5, 9, 5)
#endif

// E33 whole-width row-blocked arms. These are the PRODUCTION cells, compiled
// exactly as the dispatch switch instantiates them, so the register number is
// the shipped kernel's and not a probe's approximation of it.
#define CROSSROW_M_RB_PROBE(name, M, IPG, R)                               \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true, R>(          \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        tid, simd_gid, simd_lid);                                          \
  }

CROSSROW_M_RB_PROBE(crossrow_m6_ipg6_r2, 6, 6, 2)
// E38 arm (a): the shipped 3+3 weight schedule, row-blocked. Same grid and same
// weight traffic as the shipped cell, twice the activation reads. It exists to
// price the activation tile on its own.
CROSSROW_M_RB_PROBE(crossrow_m6_ipg3_r2, 6, 3, 2)

#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_M_RB_PROBE(crossrow_m7_ipg7_r2, 7, 7, 2)
CROSSROW_M_RB_PROBE(crossrow_m8_ipg8_r2, 8, 8, 2)
CROSSROW_M_RB_PROBE(crossrow_m9_ipg9_r2, 9, 9, 2)
#endif

// E38 arm (b): the same single weight stream as crossrow_m6_ipg6_r2, but with
// the two row blocks handed to two x-blocks instead of looped in one. Register
// pressure must match the sequential form exactly -- each threadgroup still
// holds only R accumulators -- so a difference here would mean the placement
// changed more than tid.x.
#define CROSSROW_M_RBX_PROBE(name, M, IPG, R)                              \
  [[kernel]] void name(                                                    \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]) {                       \
    qmv_fast_crossrow_affine4_g64_m<bfloat16_t, M, IPG, true, R, true>(    \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        tid, simd_gid, simd_lid);                                          \
  }

CROSSROW_M_RBX_PROBE(crossrow_m6_ipg6_r2_xb, 6, 6, 2)
