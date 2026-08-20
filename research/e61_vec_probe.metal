// Research-only (qwen38-r1-e61): rung 0.1. Extend E55's NA=5 lane probe to
// NA = 6, 7 and 8 on the device that runs the scored kernel.
//
// E54 already proved at compile time that `vec<float,N>` has sizeof 32 and
// alignof 32 for every N in [5, 8], so NA = 5, 6, 7 and 8 share one padded
// eight-lane storage class in MEMORY. This probe asks the separate question
// that matters for the scored kernel: does every one of the N lanes still
// carry its own input row through the exact expression tree
// `qmv_fast_crossrow_affine4_g64_wide` uses, with bitwise-identical results to
// the shipped narrow types?
#include <metal_stdlib>
using namespace metal;

// out[2*(N-2)] = sizeof, out[2*(N-2)+1] = alignof, for N = 2..8.
// out[14 + (N-2)] = indexed-write lane-bleed mask: bit i is set when writing
//                   lane i disturbed any other lane.
// out[21 + (N-2)] = lane-local arithmetic fault mask: bit i is set when
//                   lane i of v*2+1 did not match its own scalar result.
#define LAYOUT_PROBE(N)                                             \
  {                                                                 \
    typedef vec<float, N> VF;                                       \
    out[2 * (N - 2)] = (uint)sizeof(VF);                            \
    out[2 * (N - 2) + 1] = (uint)alignof(VF);                       \
    VF v = VF(0.0f);                                                \
    uint bleed = 0;                                                 \
    for (int i = 0; i < N; i++) {                                   \
      VF t = VF(0.0f);                                              \
      t[i] = 1.0f;                                                  \
      for (int j = 0; j < N; j++) {                                 \
        float want = (i == j) ? 1.0f : 0.0f;                        \
        if (t[j] != want) {                                         \
          bleed |= (1u << (uint)i);                                 \
        }                                                           \
      }                                                             \
      v[i] = float(i + 1);                                          \
    }                                                               \
    out[14 + (N - 2)] = bleed;                                      \
    VF w = v * VF(2.0f) + VF(1.0f);                                 \
    uint arith = 0;                                                 \
    for (int i = 0; i < N; i++) {                                   \
      if (w[i] != float(i + 1) * 2.0f + 1.0f) {                     \
        arith |= (1u << (uint)i);                                   \
      }                                                             \
    }                                                               \
    out[21 + (N - 2)] = arith;                                      \
  }

kernel void vec_layout(device uint* out [[buffer(0)]]) {
  LAYOUT_PROBE(2)
  LAYOUT_PROBE(3)
  LAYOUT_PROBE(4)
  LAYOUT_PROBE(5)
  LAYOUT_PROBE(6)
  LAYOUT_PROBE(7)
  LAYOUT_PROBE(8)
}

// The wide helper's inner expression at compile-time NA, in the DIRECT_NIBBLES
// form the scored table uses. Each of the NA lanes must carry input row
// (first_m + m) and nothing else.
//
// PERTURB selects a deliberate lane fault so the probe has positive controls:
//   0 = faithful
//   1 = swap lanes 0 and NA-1 on the activation load  (row/lane mismatch)
//   2 = drop lane NA-1's activation to zero
//   3 = leak lane NA-2 into lane NA-1
template <int NA, int PERTURB>
void wide_lane_core(
    const device uint16_t* wpk,
    const device float* scales,
    const device float* biases,
    const device float* x,
    device float* y,
    int K,
    int rows,
    int first_m,
    int out_row) {
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < K; k += 64) {
    thread uint16_t packed[rows_per_simd][16];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      for (int i = 0; i < 16; i++) {
        packed[r][i] = wpk[row * (K / 4) + k / 4 + i];
      }
      scale_local[r] = scales[row * (K / 64) + k / 64];
      bias_local[r] = biases[row * (K / 64) + k / 64];
    }

    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 16; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {
        int src = first_m + m;
        if (PERTURB == 1) {
          if (m == 0) {
            src = first_m + NA - 1;
          } else if (m == NA - 1) {
            src = first_m;
          }
        }
        const device float* xm = x + src * K + k + 4 * i;
        thread float xc[4];
        xc[0] = xm[0];
        xc[1] = xm[1];
        xc[2] = xm[2];
        xc[3] = xm[3];
        if (PERTURB == 2 && m == NA - 1) {
          xc[0] = 0.0f;
          xc[1] = 0.0f;
          xc[2] = 0.0f;
          xc[3] = 0.0f;
        }
        sums[m] += xc[0] + xc[1] + xc[2] + xc[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      if (PERTURB == 3) {
        a0[NA - 1] = a0[NA - 2];
      }
      for (int r = 0; r < rows_per_simd; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      y[(first_m + m) * rows + out_row + r] = acc[r][m];
    }
  }
}

struct ProbeParams {
  int K;
  int rows;
  int first_m;
  int out_row;
};

#define PROBE_ENTRY(NAME, NA, PERTURB)                                    \
  kernel void NAME(                                                       \
      const device uint16_t* wpk [[buffer(0)]],                           \
      const device float* scales [[buffer(1)]],                           \
      const device float* biases [[buffer(2)]],                           \
      const device float* x [[buffer(3)]],                                \
      device float* y [[buffer(4)]],                                      \
      const constant ProbeParams& p [[buffer(5)]]) {                      \
    wide_lane_core<NA, PERTURB>(                                          \
        wpk, scales, biases, x, y, p.K, p.rows, p.first_m, p.out_row);     \
  }

PROBE_ENTRY(wide_na2_faithful, 2, 0)
PROBE_ENTRY(wide_na3_faithful, 3, 0)
PROBE_ENTRY(wide_na4_faithful, 4, 0)
PROBE_ENTRY(wide_na5_faithful, 5, 0)
PROBE_ENTRY(wide_na6_faithful, 6, 0)
PROBE_ENTRY(wide_na7_faithful, 7, 0)
PROBE_ENTRY(wide_na8_faithful, 8, 0)

PROBE_ENTRY(wide_na5_swap, 5, 1)
PROBE_ENTRY(wide_na5_zerolast, 5, 2)
PROBE_ENTRY(wide_na5_leak, 5, 3)
PROBE_ENTRY(wide_na6_swap, 6, 1)
PROBE_ENTRY(wide_na6_zerolast, 6, 2)
PROBE_ENTRY(wide_na6_leak, 6, 3)
PROBE_ENTRY(wide_na7_swap, 7, 1)
PROBE_ENTRY(wide_na7_zerolast, 7, 2)
PROBE_ENTRY(wide_na7_leak, 7, 3)
PROBE_ENTRY(wide_na8_swap, 8, 1)
PROBE_ENTRY(wide_na8_zerolast, 8, 2)
PROBE_ENTRY(wide_na8_leak, 8, 3)
