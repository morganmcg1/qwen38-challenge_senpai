// E97 rung 2 -- what does one more input row cost inside the WIDE cross-row
// kernel, and where does that cost go?
//
// Three arms, all at the scored affine-4 group-64 shape:
//
//   shipped_m{M}_ipg{IPG}
//       The exact scored template from `kernels/quantized.h`. Launched twice:
//       once with grid.x = M as the frozen host does, and once with grid.x = G,
//       the number of groups that do work. The difference is the cost of the
//       idle x-groups alone, with the work held fixed.
//
//   copy_na{NA} / bcastx_na{NA} / nofma_na{NA}
//       A research copy of the WIDE inner body and two ablations of it, each
//       launched with one working group so NA moves and the grid does not.
//       `copy` is the positive control: it must time within a few percent of
//       the shipped kernel at the same NA, or the ablations mean nothing.
//       `bcastx` loads the activation block ONCE and broadcasts it to the NA
//       lanes, so it keeps every multiply-accumulate and removes NA-1 of the
//       activation loads. `nofma` keeps every load and removes three quarters
//       of the multiply-accumulate work.
//
// The ablations do not compute the affine-4 product. They exist to price load
// issue against arithmetic issue, and nothing in this file is compiled into a
// submission: `research/` is outside `editablePaths`.
//
//   xcrun -sdk macosx metal -std=metal3.1 -O2 -c research/e97_rowcost_probe.metal \
//     -I Vendor/mlx-swift/Source/Cmlx/mlx -o /tmp/e97.air
//   xcrun -sdk macosx metallib /tmp/e97.air -o /tmp/e97.metallib

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

using namespace metal;

#define E97_COPY 0
#define E97_BCAST_X 1
#define E97_NO_FMA 2

// A copy of `qmv_fast_crossrow_affine4_g64_wide<T, NA, true>` with one
// deliberate change selected by VARIANT. Kept literal so the diff against the
// shipped body is one block, and so the copy control can prove the copy itself
// is not the effect.
template <typename T, int NA, int VARIANT>
METAL_FUNC void e97_wide(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int in_vec_size,
    const int out_vec_size,
    int first_m,
    int out_row,
    uint simd_lid) {
  typedef vec<float, NA> VF;
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 16;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint16_t packed[rows_per_simd][4];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      if (VARIANT == E97_BCAST_X) {
        const device T* xm =
            x + first_m * in_vec_size + k + simd_lid * values_per_thread + 4 * i;
        const float c0 = static_cast<float>(xm[0]);
        const float c1 = static_cast<float>(xm[1]);
        const float c2 = static_cast<float>(xm[2]);
        const float c3 = static_cast<float>(xm[3]);
        const float cs = c0 + c1 + c2 + c3;
        for (int m = 0; m < NA; m++) {
          a0[m] = c0;
          a1[m] = c1;
          a2[m] = c2;
          a3[m] = c3;
          sums[m] += cs;
        }
      } else {
        for (int m = 0; m < NA; m++) {
          const device T* xm = x + (first_m + m) * in_vec_size + k +
              simd_lid * values_per_thread + 4 * i;
          a0[m] = static_cast<float>(xm[0]);
          a1[m] = static_cast<float>(xm[1]);
          a2[m] = static_cast<float>(xm[2]);
          a3[m] = static_cast<float>(xm[3]);
          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        }
      }
      if (VARIANT == E97_NO_FMA) {
        // One of the four nibble terms survives, so every weight load stays
        // live; the other three activation vectors stay live through one add
        // each instead of four multiply-accumulates per output row.
        sums += a1 + a2 + a3;
        for (int r = 0; r < rows_per_simd; r++) {
          partial[r] += a0 * (packed[r][i] & 0x000f);
        }
      } else {
        for (int r = 0; r < rows_per_simd; r++) {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * ((packed[r][i] >> 4) & 0x000f) +
                         a2 * ((packed[r][i] >> 8) & 0x000f) +
                         a3 * ((packed[r][i] >> 12) & 0x000f));
        }
      }
    }
    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

#define E97_ABLATION(name, NA, VARIANT)                                    \
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
    e97_wide<bfloat16_t, NA, VARIANT>(                                     \
        w, scales, biases, x, y, in_vec_size, out_vec_size,                \
        int(tid.x) * NA, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);    \
  }

E97_ABLATION(copy_na2, 2, E97_COPY)
E97_ABLATION(copy_na3, 3, E97_COPY)
E97_ABLATION(copy_na4, 4, E97_COPY)
E97_ABLATION(bcastx_na2, 2, E97_BCAST_X)
E97_ABLATION(bcastx_na3, 3, E97_BCAST_X)
E97_ABLATION(bcastx_na4, 4, E97_BCAST_X)
E97_ABLATION(nofma_na2, 2, E97_NO_FMA)
E97_ABLATION(nofma_na3, 3, E97_NO_FMA)
E97_ABLATION(nofma_na4, 4, E97_NO_FMA)

#define E97_SHIPPED(name, M, IPG)                                          \
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

E97_SHIPPED(shipped_m3_ipg3, 3, 3)
E97_SHIPPED(shipped_m4_ipg4, 4, 4)
E97_SHIPPED(shipped_m5_ipg3, 5, 3)
E97_SHIPPED(shipped_m6_ipg3, 6, 3)
E97_SHIPPED(shipped_m7_ipg4, 7, 4)
E97_SHIPPED(shipped_m8_ipg4, 8, 4)
E97_SHIPPED(shipped_m9_ipg3, 9, 3)
