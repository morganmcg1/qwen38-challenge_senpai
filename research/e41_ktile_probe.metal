// E41: AIR census for the K-tiled activation-staging ladder.
//
// Unlike E32's probe, every cell here instantiates the SHIPPED templates out of
// kernels/quantized.h directly. There is no generated copy, because the tile
// shape (ROWS_PER_SIMD, BLOCKS_PER_CALL, K_TILE_BLOCKS) is now a template
// parameter of the shipped kernel, so the census and the timed build compile the
// same source.
//
//   python3 research/e41_ktile_census.py
//
// Arms:
//   xship_na{NA}          the incumbent call, byte-for-byte as quantized.h
//                         dispatches it. These are the NON-PERTURBATION gate:
//                         E32/E36 measured 62/83/104/125 regs for NA=2..5 before
//                         the K-tile refactor, so any drift here means the
//                         refactor changed the shipped cell and the experiment
//                         is void.
//   xrb_na{NA}_r{R}       R rows per accumulator tile, 4/R sequential full-K
//                         calls. R=2 is E38's arm(a): the +10.54 % tax cell.
//   xkt_na{NA}_r{R}_kt{n} R rows per tile, 4/R tiles held live in ONE call,
//                         k-tiled every n k-blocks. n=0 means all of K, which is
//                         xrb's k-order with both tiles live. THE LADDER.
//   xctl_*                spill-gate controls, same contract as E32.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define PROBE_ARGS                                                         \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
      uint simd_lid [[thread_index_in_simdgroup]]

// The incumbent, called exactly as qmv_fast_crossrow_affine4_g64_m calls it.
#define E41_SHIPPED_PROBE(name, NA)                                        \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true>(              \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

// Sequential row blocks: 4/R calls, each its own full pass over K, one
// accumulator tile live at a time. R=2 is the E38 arm(a) tax cell.
#define E41_RB_PROBE(name, NA, R)                                          \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_rowblocked<bfloat16_t, NA, true, R, 1>(  \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

// The ladder: all 4/R tiles live in one call, interleaved every KT k-blocks.
// Only the k_tile constant separates the rungs.
#define E41_KT_PROBE(name, NA, R, KT)                                      \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_rowblocked<                              \
        bfloat16_t, NA, true, R, 4 / R, KT>(                               \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

#ifdef PROBE_CELL_SHIPPED
E41_SHIPPED_PROBE(PROBE_NAME, PROBE_NA)
#endif

#ifdef PROBE_CELL_RB
E41_RB_PROBE(PROBE_NAME, PROBE_NA, PROBE_R)
#endif

#ifdef PROBE_CELL_KT
E41_KT_PROBE(PROBE_NAME, PROBE_NA, PROBE_R, PROBE_KT)
#endif

// Spill-gate control, carried over from E32 unchanged: the accumulator array has
// the accumulator TYPE the gate matches, `vec<float, NA> acc[R]`, but is indexed
// by runtime data so it cannot be promoted to registers. This is the input that
// must make the gate fire.
#ifdef PROBE_CELL_FORCED_SPILL
[[kernel]] void PROBE_NAME(PROBE_ARGS) {
  typedef vec<float, PROBE_NA> VF;
  VF acc[PROBE_R];
  for (int r = 0; r < PROBE_R; r++) {
    acc[r] = VF(0.0f);
  }
  for (int k = 0; k < in_vec_size; k += 512) {
    const int pick = (k / 512 + int(simd_lid)) % PROBE_R;
    acc[pick] += VF(static_cast<float>(x[k + simd_lid]));
  }
  for (int r = 0; r < PROBE_R; r++) {
    for (int m = 0; m < PROBE_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[m * out_vec_size + int(tid.y) * PROBE_R + r] =
            static_cast<bfloat16_t>(reduced);
      }
    }
  }
}
#endif
