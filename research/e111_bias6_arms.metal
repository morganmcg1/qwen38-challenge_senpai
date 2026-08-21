// E111 rung 1: arms of the wide affine-4 QMV for the one-byte bias recoding.
//
// Research only. Nothing here is on the scored path, and no candidate file is
// modified by this experiment.
//
// Arm `a_shipped` is a line-aligned transcription of
// `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:969`
// `qmv_fast_crossrow_affine4_g64_wide<T, NA, DIRECT_NIBBLES>` reached through
// `qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>` at quantized.h:1157, which
// is what the entry switch at quantized.h:1952 selects for M == 5. The
// one-group collapse is preserved: the host launches 5 x-groups and only
// tid.x == 0 does work.
//
// The arms hold the grid, the threadgroup shape, the k-block structure and the
// summation order fixed and change exactly one thing each:
//
//   a_shipped   reference, 36 B per 64-element group
//   n_nobias    no bias load and no sums term, 34 B per group. The ceiling of
//               the WHOLE bias axis: traffic and arithmetic together.
//   n_nosums    bias still loaded as BF16, the x-sum chain removed. Isolates
//               the arithmetic half of that ceiling. Cross-checks E108's
//               independent -4.47 % measurement of the same quantity.
//   d_bias1     bias read from a uint8 array, one byte per group, value
//               deliberately wrong. 35 B per group with no reconstruction
//               cost, so this is the exact ceiling of Bias6 itself.
//   e_bias6     bias reconstructed from (scale, code) the real way. Must be
//               bit-identical to a_shipped.
//   b_constw    packed weight load replaced by a runtime-seeded value, all
//               arithmetic kept. Roofline pair, removes the stream.
//   c_loadonly  every device load kept and live, the nibble extraction and the
//               vec<float,5> FMA block removed. Roofline pair, removes the
//               arithmetic.
//   e111_stream idealised coalesced read, for the achievable-bandwidth line.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define SIMD_SIZE 32
#ifndef E111_NA
#define E111_NA 5
#endif
#define E111_ROWS_PER_SIMD 4
#define E111_VALUES_PER_THREAD 16
#define E111_BLOCK_SIZE (E111_VALUES_PER_THREAD * SIMD_SIZE)
#define E111_BYTES_PER_LANE 8

typedef vec<float, E111_NA> VF;

// The reconstruction the load-time packer validates group by group. Integer
// only, so the CPU packer and the GPU agree bit for bit with no dependence on
// a floating-point rounding mode.
static inline float e111_bias_from_code(float scale, uint code) {
  const float prod = -float(code & 0xFu) * scale;
  uint u = as_type<uint>(prod);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u += ((code & 0x30u) << 12) - 0x10000u;
  return as_type<float>(u & 0xFFFF0000u);
}

// ---------------------------------------------------------------------------
// a_shipped
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_a_shipped(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF sums = VF(0.0f);
    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        thread float xc[4];
        xc[0] = static_cast<float>(xm[0]);
        xc[1] = static_cast<float>(xm[1]);
        xc[2] = static_cast<float>(xm[2]);
        xc[3] = static_cast<float>(xm[3]);
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// n_nobias: the whole bias axis removed, 34 B per group.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_n_nobias(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
    }

    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        a0[m] = static_cast<float>(xm[0]);
        a1[m] = static_cast<float>(xm[1]);
        a2[m] = static_cast<float>(xm[2]);
        a3[m] = static_cast<float>(xm[3]);
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// n_nosums: BF16 bias still loaded, the x-sum chain removed. 36 B per group.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_n_nosums(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        a0[m] = static_cast<float>(xm[0]);
        a1[m] = static_cast<float>(xm[1]);
        a2[m] = static_cast<float>(xm[2]);
        a3[m] = static_cast<float>(xm[3]);
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// d_bias1: one byte per group, no reconstruction. Traffic-only ceiling.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_d_bias1(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = float(codes[group_index]);
    }

    VF sums = VF(0.0f);
    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        thread float xc[4];
        xc[0] = static_cast<float>(xm[0]);
        xc[1] = static_cast<float>(xm[1]);
        xc[2] = static_cast<float>(xm[2]);
        xc[3] = static_cast<float>(xm[3]);
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// e_bias6: one byte per group, exact reconstruction. Must be bit-identical to
// a_shipped.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_e_bias6(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      const float scale = scales[group_index];
      scale_local[r] = scale;
      bias_local[r] = e111_bias_from_code(scale, uint(codes[group_index]));
    }

    VF sums = VF(0.0f);
    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        thread float xc[4];
        xc[0] = static_cast<float>(xm[0]);
        xc[1] = static_cast<float>(xm[1]);
        xc[2] = static_cast<float>(xm[2]);
        xc[3] = static_cast<float>(xm[3]);
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// b_constw: the packed weight load removed, every arithmetic operation kept.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_b_constw(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  // The seed is a runtime argument and the mix depends on k, on the lane and
  // on the row, so no extraction can be constant folded or hoisted.
  const ulong lane_mix = wseed ^ (ulong(simd_lid) * 0x2545F4914F6CDD1Dul) ^
      (ulong(out_row) * 0xD1342543DE82EF95ul);

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const ulong mix = lane_mix + ulong(k + r) * 0x9E3779B97F4A7C15ul;
      for (int i = 0; i < 4; i++) {
        packed[r][i] = uint16_t((mix >> (16 * i)) & 0xFFFFul);
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF sums = VF(0.0f);
    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        thread float xc[4];
        xc[0] = static_cast<float>(xm[0]);
        xc[1] = static_cast<float>(xm[1]);
        xc[2] = static_cast<float>(xm[2]);
        xc[3] = static_cast<float>(xm[3]);
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// c_loadonly: every device load kept and live, the extraction and the
// vec<float,5> FMA block removed.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_c_loadonly(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    VF sums = VF(0.0f);
    for (int i = 0; i < 4; i++) {
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      // All four packed halves stay live, so no weight load is dead, and the
      // result is still written out below.
      const uint merged =
          uint(packed[r][0] | packed[r][1] | packed[r][2] | packed[r][3]);
      acc[r] += scale_local[r] * float(merged & 0x000fu) + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Achievable-bandwidth reference.
// ---------------------------------------------------------------------------
[[kernel]] void e111_stream(
    const device uint4* src [[buffer(0)]],
    device uint4* sink [[buffer(1)]],
    const constant uint& n_vec4 [[buffer(2)]],
    const constant uint& total_threads [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint4 acc = uint4(0);
  for (uint i = gid; i < n_vec4; i += total_threads) {
    acc ^= src[i];
  }
  if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) {
    sink[gid & 0xFFu] = acc;
  }
}

// ---------------------------------------------------------------------------
// g_pack32: the shipped values, read as one 32-bit interleaved record.
//
// a_shipped reads scales[g] and biases[g] from two buffers at the same index,
// so it issues two 16-bit loads for every group of every row. This arm reads
// the identical pair from one interleaved uint32 array. Bytes streamed,
// values and summation order do not change, so the output must stay
// bit-identical and the arm prices the second load instruction on its own.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void e111_g_pack32(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device uint8_t* codes [[buffer(3)]],
    const device T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    const constant ulong& wseed [[buffer(8)]],
    const device uint32_t* packed_sb [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int first_m = int(tid.x) * E111_NA;
  if (first_m >= E111_NA) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * E111_ROWS_PER_SIMD;
  const int in_vec_size_w = in_vec_size / 2;
  const int in_vec_size_g = in_vec_size / 64;

  VF acc[E111_ROWS_PER_SIMD];
  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    acc[r] = VF(0.0f);
  }

  for (int k = 0; k < in_vec_size; k += E111_BLOCK_SIZE) {
    thread uint16_t packed[E111_ROWS_PER_SIMD][4];
    thread float scale_local[E111_ROWS_PER_SIMD];
    thread float bias_local[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      const int row = out_row + r;
      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * E111_BYTES_PER_LANE);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
      const int group_index = row * in_vec_size_g + k / 64 + simd_lid / 4;
      const uint32_t sb = packed_sb[group_index];
      scale_local[r] = float(as_type<T>(uint16_t(sb & 0xFFFFu)));
      bias_local[r] = float(as_type<T>(uint16_t(sb >> 16)));
    }

    VF sums = VF(0.0f);
    VF partial[E111_ROWS_PER_SIMD];
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      partial[r] = VF(0.0f);
    }
    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < E111_NA; m++) {
        const device T* xm = x + (first_m + m) * in_vec_size + k +
            simd_lid * E111_VALUES_PER_THREAD + 4 * i;
        thread float xc[4];
        xc[0] = static_cast<float>(xm[0]);
        xc[1] = static_cast<float>(xm[1]);
        xc[2] = static_cast<float>(xm[2]);
        xc[3] = static_cast<float>(xm[3]);
        sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];
      }
      for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
        partial[r] += (a0 * (packed[r][i] & 0x000f) +
                       a1 * ((packed[r][i] >> 4) & 0x000f) +
                       a2 * ((packed[r][i] >> 8) & 0x000f) +
                       a3 * ((packed[r][i] >> 12) & 0x000f));
      }
    }
    for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }
  }

  for (int r = 0; r < E111_ROWS_PER_SIMD; r++) {
    for (int m = 0; m < E111_NA; m++) {
      const float reduced = simd_sum(acc[r][m]);
      if (simd_lid == 0) {
        y[(first_m + m) * out_vec_size + out_row + r] = static_cast<T>(reduced);
      }
    }
  }
}


#define instantiate_kernel(name, func, tname)  \
  template [[host_name(name)]] [[kernel]] decltype(func<tname>) func<tname>;

instantiate_kernel("a_shipped", e111_a_shipped, bfloat)
instantiate_kernel("n_nobias", e111_n_nobias, bfloat)
instantiate_kernel("n_nosums", e111_n_nosums, bfloat)
instantiate_kernel("d_bias1", e111_d_bias1, bfloat)
instantiate_kernel("e_bias6", e111_e_bias6, bfloat)
instantiate_kernel("b_constw", e111_b_constw, bfloat)
instantiate_kernel("c_loadonly", e111_c_loadonly, bfloat)
instantiate_kernel("g_pack32", e111_g_pack32, bfloat)
