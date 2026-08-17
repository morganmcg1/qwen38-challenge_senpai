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

// Opt-in arm. NA > 4 trips the production `static_assert(NA <= 4)`, so building
// this requires temporarily widening that bound in quantized.h; it is scaffolding
// to measure what the cap is made of, not a live-code change.
#ifdef CROSSROW_NA_PROBE_WIDE
CROSSROW_NA_PROBE(crossrow_na5, 5)
CROSSROW_NA_PROBE(crossrow_na6, 6)
CROSSROW_NA_PROBE(crossrow_na8, 8)
#endif
