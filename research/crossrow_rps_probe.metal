// E32: NA x rows_per_simd register-budget grid for the wide crossrow QMV.
//
// E27 swept NA with rows_per_simd pinned at 4, because NA is a template
// parameter and rows_per_simd is a `constexpr int` in the body. This probe
// sweeps both, using a body generated from the shipped kernel by
// research/crossrow_rps_gen.py so it cannot drift away from what ships.
//
//   python3 research/crossrow_rps_gen.py
//   python3 research/crossrow_rps_sweep.py
//
// Arms:
//   xrps_na{NA}_r{R}      grid-relaxed:      R rows/simdgroup, one pass.
//                         Needs a host grid of out_vec_size / (2*R) groups.
//   xrb_na{NA}_r{R}       coverage-preserving: R rows/simdgroup x 4/R blocks,
//                         so the FROZEN 8-rows-per-threadgroup grid still works.
//   xship_na{NA}          the shipped template itself, unmodified, as the
//                         equivalence anchor for xrps_na{NA}_r4.
//   xctl_*                negative controls; see research/crossrow_rps_sweep.py.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#include "../research/generated/crossrow_rps_wide.h"

#define PROBE_ARGS                                                         \
      const device uint32_t* w [[buffer(0)]],                              \
      const device bfloat16_t* scales [[buffer(1)]],                       \
      const device bfloat16_t* biases [[buffer(2)]],                       \
      const device bfloat16_t* x [[buffer(3)]],                            \
      device bfloat16_t* y [[buffer(4)]],                                  \
      const constant int& in_vec_size [[buffer(5)]],                       \
      const constant int& out_vec_size [[buffer(6)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]

// Grid-relaxed. out_row follows from the geometry this cell would need:
// 2 simdgroups x R rows each, so tiles are 2*R rows tall.
#define CROSSROW_RPS_PROBE(name, NA, R)                                    \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_wide_rps<bfloat16_t, NA, true, R>(       \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * (2 * R) + int(simd_gid) * R,         \
        simd_lid);                                                         \
  }

// Coverage-preserving. out_row is the SHIPPED expression, unchanged.
#define CROSSROW_RB_PROBE(name, NA, R)                                     \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_rowblocked_rps<bfloat16_t, NA, true, R>(     \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

// The shipped template, called exactly as quantized.h:1177 calls it.
#define CROSSROW_SHIPPED_PROBE(name, NA)                                   \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true>(              \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

#ifdef PROBE_CELL_RPS
CROSSROW_RPS_PROBE(PROBE_NAME, PROBE_NA, PROBE_R)
#endif

#ifdef PROBE_CELL_RB
CROSSROW_RB_PROBE(PROBE_NAME, PROBE_NA, PROBE_R)
#endif

#ifdef PROBE_CELL_SHIPPED
CROSSROW_SHIPPED_PROBE(PROBE_NAME, PROBE_NA)
#endif

// Negative control. The accumulator array has the SAME declared type as the
// kernel's, `vec<float, NA> acc[rows_per_simd]`, but is indexed by runtime data
// so the compiler cannot promote it to registers. The gate matches allocas by
// accumulator TYPE, so this is the input that must make it fire; a control that
// spills some other shape would prove nothing about this gate.
//
// The first version of this control allocated `float ballast[NA * R]` and did
// NOT fire, correctly: `[16 x float]` is not the accumulator type. It was a
// broken control, not a broken gate, and it is recorded in the report.
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
