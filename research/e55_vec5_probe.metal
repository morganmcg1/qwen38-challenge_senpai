// Research-only (qwen38-r1-e55): Risk 1 gate. Prove what metal::vec<float,5>
// compiles to and prove every one of its 5 lanes carries its own row through
// the exact expression tree used by qmv_fast_crossrow_affine4_g64_wide.
#include <metal_stdlib>
using namespace metal;

typedef vec<float, 5> VF5;
typedef vec<float, 4> VF4;
typedef vec<float, 3> VF3;
typedef vec<float, 2> VF2;

// Layout facts the advisor asked for, read from the device that will run the
// scored kernel rather than assumed from the MSL spec.
kernel void vec5_layout(device uint* out [[buffer(0)]]) {
  out[0] = (uint)sizeof(VF2);
  out[1] = (uint)alignof(VF2);
  out[2] = (uint)sizeof(VF3);
  out[3] = (uint)alignof(VF3);
  out[4] = (uint)sizeof(VF4);
  out[5] = (uint)alignof(VF4);
  out[6] = (uint)sizeof(VF5);
  out[7] = (uint)alignof(VF5);
  // Does an indexed write to lane i land only in lane i?
  VF5 v = VF5(0.0f);
  uint bleed = 0;
  for (int i = 0; i < 5; i++) {
    VF5 t = VF5(0.0f);
    t[i] = 1.0f;
    for (int j = 0; j < 5; j++) {
      float want = (i == j) ? 1.0f : 0.0f;
      if (t[j] != want) {
        bleed |= (1u << (uint)(i * 5 + j));
      }
    }
    v[i] = float(i + 1);
  }
  out[8] = bleed;
  // Does arithmetic stay lane-local?
  VF5 w = v * VF5(2.0f) + VF5(1.0f);
  uint arith = 0;
  for (int i = 0; i < 5; i++) {
    if (w[i] != float(i + 1) * 2.0f + 1.0f) {
      arith |= (1u << (uint)i);
    }
  }
  out[9] = arith;
}

// The wide helper's inner expression, reproduced verbatim at compile-time NA
// with the same DIRECT_NIBBLES=true form the scored table uses. Each of the NA
// lanes must carry input row (first_m + m) and nothing else.
//
// PERTURB selects a deliberate lane fault so the probe has a positive control:
//   0 = faithful
//   1 = swap lanes 0 and 4 on the activation load  (row/lane mismatch)
//   2 = drop lane 4's activation to zero
//   3 = leak lane 3 into lane 4 (cross-lane contamination)
template <int NA, int PERTURB>
void wide_lane_core(
    const device uint16_t* wpk,   // [rows][K/4] packed nibbles, one uint16 per 4 k
    const device float* scales,   // [rows][K/64]
    const device float* biases,   // [rows][K/64]
    const device float* x,        // [M][K]
    device float* y,              // [M][rows]
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

  // One "lane" of the real kernel walks 16 values per k-block; this probe runs
  // the whole K on a single thread so the arithmetic is checkable on the CPU.
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
      if (PERTURB == 3 && NA >= 5) {
        a0[4] = a0[3];
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

PROBE_ENTRY(wide_na5_faithful, 5, 0)
PROBE_ENTRY(wide_na5_swap04, 5, 1)
PROBE_ENTRY(wide_na5_zero4, 5, 2)
PROBE_ENTRY(wide_na5_leak34, 5, 3)
PROBE_ENTRY(wide_na4_faithful, 4, 0)
PROBE_ENTRY(wide_na3_faithful, 3, 0)
PROBE_ENTRY(wide_na2_faithful, 2, 0)
