// E36: NA x rows_per_simd x values_per_thread register/footprint grid for the
// wide crossrow QMV.
//
// E32 swept NA x rows_per_simd at values_per_thread = 16. This adds the third
// per-lane axis, using a body generated from the shipped kernel by
// research/crossrow_vpt_gen.py so it cannot drift away from what ships.
//
//   python3 research/crossrow_vpt_gen.py
//   python3 research/crossrow_vpt_sweep.py
//
// Arms:
//   xvpt_na{NA}_r{R}_v{V}  grid-relaxed:      R rows/simdgroup, one pass.
//   xvb_na{NA}_r{R}_v{V}   coverage-preserving: R rows/simdgroup x 4/R blocks,
//                          so the FROZEN 8-rows-per-threadgroup grid still works.
//   xship_na{NA}           the shipped template itself, unmodified: the E27
//                          62/83/104/125 ladder anchor.
//   xctl_*                 controls; see research/crossrow_vpt_sweep.py.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#include "../research/generated/crossrow_vpt_wide.h"

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
#define CROSSROW_VPT_PROBE(name, NA, R, V)                                 \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_wide_vpt<bfloat16_t, NA, R, V, true>(    \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * (2 * R) + int(simd_gid) * R,         \
        simd_lid);                                                         \
  }

// Coverage-preserving. out_row is the SHIPPED expression, unchanged.
#define CROSSROW_VB_PROBE(name, NA, R, V)                                  \
  [[kernel]] void name(PROBE_ARGS) {                                       \
    qmv_fast_crossrow_affine4_g64_rowblocked_vpt<                          \
        bfloat16_t, NA, R, V, true>(                                       \
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

#ifdef PROBE_CELL_VPT
CROSSROW_VPT_PROBE(PROBE_NAME, PROBE_NA, PROBE_R, PROBE_V)
#endif

#ifdef PROBE_CELL_VB
CROSSROW_VB_PROBE(PROBE_NAME, PROBE_NA, PROBE_R, PROBE_V)
#endif

#ifdef PROBE_CELL_SHIPPED
CROSSROW_SHIPPED_PROBE(PROBE_NAME, PROBE_NA)
#endif

// Negative control A: the accumulator array in private memory. Same declared
// type as the kernel's, `vec<float, NA> acc[rows_per_simd]`, but indexed by
// runtime data so the compiler cannot promote it. This is the input the
// accumulator gate must fire on. (E32's first attempt allocated
// `float ballast[NA * R]` and correctly did NOT fire: `[16 x float]` is not the
// accumulator type. That was a broken control, not a broken gate.)
#ifdef PROBE_CELL_FORCED_ACC_SPILL
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

// Negative control B, new in E36 and the one that matters here. E36 varies
// values_per_thread, and the ONLY structure that grows with it is the uint16
// staging array `packed[rows_per_simd][values_per_thread / 4]`. The E32
// accumulator gate is blind to that type by construction, so a second gate is
// needed and it needs its own known-BAD input: the same `[R x [W x i16]]`
// array, runtime-indexed so it cannot be promoted.
#ifdef PROBE_CELL_FORCED_STAGE_SPILL
[[kernel]] void PROBE_NAME(PROBE_ARGS) {
  constexpr int W = PROBE_V / 4;
  thread uint16_t packed[PROBE_R][W];
  float acc = 0.0f;
  for (int k = 0; k < in_vec_size; k += PROBE_V * 32) {
    const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(w);
    for (int r = 0; r < PROBE_R; r++) {
      for (int i = 0; i < W; i++) {
        packed[r][i] = ws[k / 4 + r * W + i + simd_lid];
      }
    }
    const int pr = (k / 64 + int(simd_lid)) % PROBE_R;
    const int pi = (k / 64 + int(simd_lid)) % W;
    acc += float(packed[pr][pi]) * static_cast<float>(x[k + simd_lid]);
  }
  const float reduced = simd_sum(acc);
  if (simd_lid == 0) {
    y[int(tid.y)] = static_cast<bfloat16_t>(reduced);
  }
}
#endif
