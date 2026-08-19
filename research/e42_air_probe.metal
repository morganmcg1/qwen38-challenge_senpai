// E42: does the injected redundant pass survive the optimiser?
//
// The perturbation's earlier passes are dead by construction (each pass
// re-initialises the accumulators), so the compiler is entitled to delete them.
// If it does, the kernel does not slow down and the whole denominator is
// fiction. This probe instantiates the treated bodies at pass level 0, 1 and 2
// so `research/air_kernel_stats.py` can show device_loads and float_ops scaling
// with the pass count, and peak_live_regs NOT scaling (the accumulators are
// reused, so the register footprint must be flat).
//
//   xcrun -sdk macosx metal -std=metal3.1 -S -O2 research/e42_air_probe.metal \
//     -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/e42_probe.ll
//   python3 research/air_kernel_stats.py /tmp/e42_probe.ll --match e42_
//
// Requires the E42 structural patch (research/e42_perturb.py) to be applied.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define E42_WIDE_PROBE(name, NA, PASSES)                                   \
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
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true, PASSES>(       \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                 \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);     \
  }

#define E42_PAIR_PROBE(name, M, PASSES)                                    \
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
    qmv_fast_crossrow_affine4_g64<bfloat16_t, M, 2, PASSES>(               \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                 \
        tid, simd_gid, simd_lid);                                          \
  }

// Compiled against the UNPATCHED header with -DE42_BASE, the same macros drop
// the pass argument, so base and level-0 stats come from identical probe text.
#define E42_WIDE_BASE(name, NA)                                            \
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
        w, scales, biases, x, y, in_vec_size, out_vec_size,                 \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);     \
  }

#define E42_PAIR_BASE(name, M)                                             \
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
    qmv_fast_crossrow_affine4_g64<bfloat16_t, M>(                          \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                 \
        tid, simd_gid, simd_lid);                                          \
  }

// NA = 3 is the shipped M=6 cell (IPG 3); NA = 5 is the shipped M=5/M=9 cell and
// the register high-water mark of the family (125 of 128, E38).
#ifdef E42_BASE
E42_WIDE_BASE(e42_na3_p0, 3)
E42_WIDE_BASE(e42_na5_p0, 5)
E42_PAIR_BASE(e42_pair_p0, 2)
#else
E42_WIDE_PROBE(e42_na3_p0, 3, 0)
E42_WIDE_PROBE(e42_na3_p1, 3, 1)
E42_WIDE_PROBE(e42_na3_p2, 3, 2)
E42_WIDE_PROBE(e42_na5_p0, 5, 0)
E42_WIDE_PROBE(e42_na5_p1, 5, 1)
E42_WIDE_PROBE(e42_na5_p2, 5, 2)
E42_PAIR_PROBE(e42_pair_p0, 2, 0)
E42_PAIR_PROBE(e42_pair_p1, 2, 1)
E42_PAIR_PROBE(e42_pair_p2, 2, 2)
#endif
