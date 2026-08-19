// E54 step 0: what does `vec<float, NA>` at NA = 5 actually compile to?
//
// The wide multi-row QMV helper accumulates in `typedef vec<float, NA> VF` and
// asserts `NA <= 4`. The Metal Shading Language specification names
// `vec<T, N>` only for N in {2, 3, 4}, so before any NA = 5 timing is trusted
// the type itself has to be proved: does it exist, how large is it, how is it
// aligned, and does lane m still address input row m?
//
// COMPILE ONLY. `E54_SIZEOF` / `E54_ALIGNOF` are bisected by
// `research/e54_vec5_proof.py`: exactly one candidate pair compiles, and that
// pair is the answer straight from the Metal front end rather than from a
// buffer this file could get wrong.
//
// `E54_LANE_PERTURB` is the positive control. It shifts one lane's write by a
// single index inside the SAME type, so a lane check that cannot see it has no
// power over any real lane bug either.

#include <metal_stdlib>

using namespace metal;

#ifndef E54_NA
#define E54_NA 5
#endif

typedef vec<float, E54_NA> VF;

#if defined(E54_SIZEOF)
static_assert(sizeof(VF) == E54_SIZEOF, "E54: sizeof(VF) is not E54_SIZEOF");
#endif
#if defined(E54_ALIGNOF)
static_assert(alignof(VF) == E54_ALIGNOF, "E54: alignof(VF) is not E54_ALIGNOF");
#endif

// One accumulator array of the same shape the wide helper holds, so the AIR
// alloca type this emits is the alloca type the real cell emits.
[[kernel]] void e54_vec5_lanes(
    const device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    const constant int& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr int rows_per_simd = 4;
  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }
  for (int k = 0; k < n; k++) {
    VF a = VF(0.0f);
    for (int m = 0; m < E54_NA; m++) {
#if defined(E54_LANE_PERTURB)
      // The one-index lane shift the check must catch.
      a[m] = x[(m + (m == E54_LANE_PERTURB ? 1 : 0)) * n + k];
#else
      a[m] = x[m * n + k];
#endif
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += a * float(r + 1);
    }
  }
  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < E54_NA; m++) {
      y[(r * E54_NA + m) + int(tid) * rows_per_simd * E54_NA] = acc[r][m];
    }
  }
}
