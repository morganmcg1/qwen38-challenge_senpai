// E107 rung 0: arms of the coarse affine-2 draft readout, for the
// ALU-versus-bandwidth discriminator.
//
// Research only. Nothing here is on the scored path, and no candidate file is
// modified by this experiment.
//
// Arm `a_shipped` is a verbatim transcription of
// `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h:1084`
// `qmv_fast_singlerow_affine2_g64` at its single scored specialisation:
// T = bfloat, group_size 64, bits 2, in_vec_size 5120, out_vec_size 98336,
// M = 1, grid (1, 12292, 1) threadgroups of (32, 2, 1).
//
// The discriminator arms hold the grid, the threadgroup shape and the k-block
// structure fixed and remove exactly one side of the kernel:
//
//   b_constw    every arithmetic operation and every scale/bias/x load is
//               kept; the 8-byte packed-weight device load is replaced by a
//               value synthesised from a runtime seed and the k index. The
//               seed is opaque to the compiler and the value changes every
//               k-block, so no extraction can be hoisted or constant-folded.
//               Removes 125,870,080 B of the 157,337,600 B stream.
//   c_loadonly  every load is kept, including the full 8-byte packed word,
//               and the 640-operation extract-and-fma block is replaced by one
//               dependency-preserving multiply that still consumes the loaded
//               word and the full x sum, and is still written out.
//   d_stream    an idealised coalesced streaming read over an arbitrary byte
//               range, for the achievable-bandwidth reference of this device.
//
//   xcrun -sdk macosx metal -O2 -std=metal3.1 -c e107_affine2_arms.metal \
//       -o x.air && xcrun -sdk macosx metallib x.air -o e107_affine2_arms.metallib

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define SIMD_SIZE 32

// ---------------------------------------------------------------------------
// a: shipped transcription.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void e107_a_shipped(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;  // 32 values x 2 bits = 8 bytes
  const int in_vec_size_w = in_vec_size / 4;   // weight bytes per output row
  const int in_vec_size_g = in_vec_size / 64;  // scale groups per output row

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint8_t* ws = reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 4 + simd_lid * bytes_per_lane;
      packed[r] = *reinterpret_cast<const device ulong*>(ws);
      // 32 values per lane = half of one 64-value group.
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 32; j++) {
        accum += x0[j] * float((packed[r] >> (2 * j)) & 0x03ul);
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// b: weight load removed, every arithmetic operation kept.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void e107_b_constw(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    // The seed is a runtime kernel argument and the mix depends on k, on the
    // lane and on the row, so the extraction below cannot be constant-folded
    // or hoisted out of the k loop. Four multiply-adds per k-block replace
    // four device loads; the 2,560-operation extract block is untouched.
    const ulong lane_mix =
        wseed ^ (ulong(simd_lid) * 0x2545F4914F6CDD1Dul) ^
        (ulong(out_row) * 0xD1342543DE82EF95ul);
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      packed[r] = lane_mix + ulong(k + r) * 0x9E3779B97F4A7C15ul;
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 32; j++) {
        accum += x0[j] * float((packed[r] >> (2 * j)) & 0x03ul);
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// c: every load kept, extract-and-fma block removed.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void e107_c_loadonly(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 4;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const device uint8_t* ws = reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 4 + simd_lid * bytes_per_lane;
      packed[r] = *reinterpret_cast<const device ulong*>(ws);
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      // One multiply keeps the loaded word and the whole x block live, so no
      // load is dead, and the result is still written out below.
      const float accum = sum * float(packed[r] & 0x03ul);
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// e: neither the weight load nor the extract-and-fma block. Completes the 2x2
// so the load side and the arithmetic side are separately identifiable.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void e107_e_floor(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread ulong packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    const ulong lane_mix =
        wseed ^ (ulong(simd_lid) * 0x2545F4914F6CDD1Dul) ^
        (ulong(out_row) * 0xD1342543DE82EF95ul);
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      packed[r] = lane_mix + ulong(k + r) * 0x9E3779B97F4A7C15ul;
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      const float accum = sum * float(packed[r] & 0x03ul);
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// Rung 1 candidate extractions. All three keep the shipped load pattern, the
// shipped k-block structure and the shipped summation order, so each
// elementary product and every partial sum is unchanged and the output is
// expected to be BIT-IDENTICAL to a_shipped.
//
//   h_split   one 8-byte uint2 load; the value loop shifts a 32-bit word, so
//             no 64-bit shift is ever issued (advisor arm B).
//   f_mask    h_split plus mask-in-place with x pre-scaled by 4^-j, which is
//             exact in FP32 because the scaling is a power of two, so the
//             shift chain disappears entirely (advisor arm A composed with B).
//   g_bfe     h_split with `extract_bits`, the hardware bitfield extract, in
//             place of the shift-and-mask pair.
// ---------------------------------------------------------------------------

// 4^-j built directly from the FP32 exponent field, so it is exact and costs
// no divide even when the surrounding loop is not unrolled.
inline float e107_inv_pow4(int j) {
  return as_type<float>(uint((127 - 2 * j) << 23));
}

#define E107_LOAD_BLOCK()                                                     \
  thread uint2 packed[rows_per_simd];                                         \
  thread float scale_local[rows_per_simd];                                    \
  thread float bias_local[rows_per_simd];                                     \
  for (int r = 0; r < rows_per_simd; r++) {                                   \
    const int row = out_row + r;                                              \
    const device uint8_t* ws = reinterpret_cast<const device uint8_t*>(w) +   \
        row * in_vec_size_w + k / 4 + simd_lid * bytes_per_lane;              \
    packed[r] = *reinterpret_cast<const device uint2*>(ws);                   \
    const int group_index =                                                   \
        row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;   \
    scale_local[r] = scales[group_index];                                     \
    bias_local[r] = biases[group_index];                                      \
  }

template <typename T>
[[kernel]] void e107_h_split(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 4;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    E107_LOAD_BLOCK()

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[j] * float((packed[r].x >> (2 * j)) & 0x03u);
      }
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[16 + j] * float((packed[r].y >> (2 * j)) & 0x03u);
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

template <typename T>
[[kernel]] void e107_f_mask(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 4;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    E107_LOAD_BLOCK()

    // x0 holds the activation pre-divided by 4^j inside its 16-value word.
    // 4^-j is a power of two, so the division is exact in FP32 and
    // (x / 4^j) * (v * 4^j) rounds exactly like x * v.
    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < values_per_thread; i += 4) {
      const int j = i & 15;
      x0[i] = static_cast<float>(xm[i]) * e107_inv_pow4(j);
      x0[i + 1] = static_cast<float>(xm[i + 1]) * e107_inv_pow4(j + 1);
      x0[i + 2] = static_cast<float>(xm[i + 2]) * e107_inv_pow4(j + 2);
      x0[i + 3] = static_cast<float>(xm[i + 3]) * e107_inv_pow4(j + 3);
      // The incumbent BF16 expression tree for the affine bias correction is
      // preserved verbatim: these four adds round in bfloat, not in float.
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[j] * float(packed[r].x & (0x03u << (2 * j)));
      }
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[16 + j] * float(packed[r].y & (0x03u << (2 * j)));
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

template <typename T>
[[kernel]] void e107_g_bfe(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_lane = 8;
  const int in_vec_size_w = in_vec_size / 4;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    E107_LOAD_BLOCK()

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    for (int i = 0; i < values_per_thread; i += 4) {
      x0[i] = static_cast<float>(xm[i]);
      x0[i + 1] = static_cast<float>(xm[i + 1]);
      x0[i + 2] = static_cast<float>(xm[i + 2]);
      x0[i + 3] = static_cast<float>(xm[i + 3]);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[j] * float(extract_bits(packed[r].x, 2 * j, 2));
      }
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[16 + j] * float(extract_bits(packed[r].y, 2 * j, 2));
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// Unroll controls. `f_mask` is the only full arm whose activation-fill loop
// carries `#pragma unroll`, so the measured f_mask-versus-h_split difference
// confounds two changes: the extraction scheme and the fill unroll. These two
// arms cross the factors. `i_h_unroll` is h_split with the fill unrolled;
// `j_f_nounroll` is f_mask with the fill left rolled. Both must stay bit-exact
// against a_shipped, because neither touches an elementary product or the
// summation order.
// ---------------------------------------------------------------------------

#define E107_HEAD_PROLOGUE()                                                  \
  constexpr int rows_per_simd = 4;                                            \
  constexpr int values_per_thread = 32;                                       \
  constexpr int block_size = values_per_thread * SIMD_SIZE;                   \
  constexpr int bytes_per_lane = 8;                                           \
  const int in_vec_size_w = in_vec_size / 4;                                  \
  const int in_vec_size_g = in_vec_size / 64;                                 \
  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;         \
  thread float result[rows_per_simd];                                         \
  for (int r = 0; r < rows_per_simd; r++) {                                   \
    result[r] = 0.0f;                                                         \
  }

#define E107_HEAD_EPILOGUE()                                                  \
  for (int r = 0; r < rows_per_simd; r++) {                                   \
    const float reduced = simd_sum(result[r]);                                \
    if (simd_lid == 0) {                                                      \
      y[out_row + r] = static_cast<T>(reduced);                               \
    }                                                                         \
  }

#define E107_XFILL(PRAGMA, SCALE)                                             \
  thread float x0[values_per_thread];                                         \
  const device T* xm = x + k + simd_lid * values_per_thread;                  \
  float sum = 0.0f;                                                           \
  PRAGMA                                                                      \
  for (int i = 0; i < values_per_thread; i += 4) {                            \
    const int jj = i & 15;                                                    \
    (void)jj;                                                                 \
    x0[i] = static_cast<float>(xm[i]) * SCALE(jj);                            \
    x0[i + 1] = static_cast<float>(xm[i + 1]) * SCALE(jj + 1);                \
    x0[i + 2] = static_cast<float>(xm[i + 2]) * SCALE(jj + 2);                \
    x0[i + 3] = static_cast<float>(xm[i + 3]) * SCALE(jj + 3);                \
    sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];                         \
  }

#define E107_SCALE_ONE(j) 1.0f
#define E107_SCALE_INV_POW4(j) e107_inv_pow4(j)

#define E107_VALUES_SHIFT()                                                   \
  for (int r = 0; r < rows_per_simd; r++) {                                   \
    float accum = 0.0f;                                                       \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < 16; j++) {                                            \
      accum += x0[j] * float((packed[r].x >> (2 * j)) & 0x03u);               \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < 16; j++) {                                            \
      accum += x0[16 + j] * float((packed[r].y >> (2 * j)) & 0x03u);          \
    }                                                                         \
    result[r] += scale_local[r] * accum + sum * bias_local[r];                \
  }

#define E107_VALUES_MASK()                                                    \
  for (int r = 0; r < rows_per_simd; r++) {                                   \
    float accum = 0.0f;                                                       \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < 16; j++) {                                            \
      accum += x0[j] * float(packed[r].x & (0x03u << (2 * j)));               \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < 16; j++) {                                            \
      accum += x0[16 + j] * float(packed[r].y & (0x03u << (2 * j)));          \
    }                                                                         \
    result[r] += scale_local[r] * accum + sum * bias_local[r];                \
  }

#define E107_CROSSED_ARM(NAME, PRAGMA, SCALE, VALUES)                         \
  template <typename T>                                                       \
  [[kernel]] void NAME(                                                       \
      const device uint32_t* w [[buffer(0)]],                                 \
      const device T* scales [[buffer(1)]],                                   \
      const device T* biases [[buffer(2)]],                                   \
      const device T* x [[buffer(3)]],                                        \
      device T* y [[buffer(4)]],                                              \
      const constant int& in_vec_size [[buffer(5)]],                          \
      const constant int& out_vec_size [[buffer(6)]],                         \
      const constant ulong& wseed [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]) {                          \
    E107_HEAD_PROLOGUE()                                                      \
    for (int k = 0; k < in_vec_size; k += block_size) {                       \
      E107_LOAD_BLOCK()                                                       \
      E107_XFILL(PRAGMA, SCALE)                                               \
      VALUES()                                                                \
    }                                                                         \
    E107_HEAD_EPILOGUE()                                                      \
  }

E107_CROSSED_ARM(e107_i_h_unroll, _Pragma("unroll"), E107_SCALE_ONE,
                 E107_VALUES_SHIFT)
E107_CROSSED_ARM(e107_j_f_nounroll, , E107_SCALE_INV_POW4, E107_VALUES_MASK)

// ---------------------------------------------------------------------------
// b2: the mask-and-pre-scale extraction with the weight load removed, so the
// pure issue-rate difference against b_constw is visible with no memory time
// to hide behind.
// ---------------------------------------------------------------------------

template <typename T>
[[kernel]] void e107_b2_maskalu(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const constant ulong& wseed [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simd = 4;
  constexpr int values_per_thread = 32;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  const int in_vec_size_g = in_vec_size / 64;

  const int out_row = int(tid.y) * 8 + int(simd_gid) * rows_per_simd;

  thread float result[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    result[r] = 0.0f;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    thread uint2 packed[rows_per_simd];
    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    const ulong lane_mix =
        wseed ^ (ulong(simd_lid) * 0x2545F4914F6CDD1Dul) ^
        (ulong(out_row) * 0xD1342543DE82EF95ul);
    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;
      const ulong synth = lane_mix + ulong(k + r) * 0x9E3779B97F4A7C15ul;
      packed[r] = uint2(uint(synth), uint(synth >> 32));
      const int group_index =
          row * in_vec_size_g + k / 64 + (simd_lid * values_per_thread) / 64;
      scale_local[r] = scales[group_index];
      bias_local[r] = biases[group_index];
    }

    thread float x0[values_per_thread];
    const device T* xm = x + k + simd_lid * values_per_thread;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < values_per_thread; i += 4) {
      const int j = i & 15;
      x0[i] = static_cast<float>(xm[i]) * e107_inv_pow4(j);
      x0[i + 1] = static_cast<float>(xm[i + 1]) * e107_inv_pow4(j + 1);
      x0[i + 2] = static_cast<float>(xm[i + 2]) * e107_inv_pow4(j + 2);
      x0[i + 3] = static_cast<float>(xm[i + 3]) * e107_inv_pow4(j + 3);
      sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
    }

    for (int r = 0; r < rows_per_simd; r++) {
      float accum = 0.0f;
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[j] * float(packed[r].x & (0x03u << (2 * j)));
      }
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        accum += x0[16 + j] * float(packed[r].y & (0x03u << (2 * j)));
      }
      result[r] += scale_local[r] * accum + sum * bias_local[r];
    }
  }

  for (int r = 0; r < rows_per_simd; r++) {
    const float reduced = simd_sum(result[r]);
    if (simd_lid == 0) {
      y[out_row + r] = static_cast<T>(reduced);
    }
  }
}

// ---------------------------------------------------------------------------
// d: idealised coalesced stream, for the achievable-bandwidth reference.
// ---------------------------------------------------------------------------

[[kernel]] void e107_d_stream(
    const device uint4* src [[buffer(0)]],
    device uint* sink [[buffer(1)]],
    const constant uint& n_vec4 [[buffer(2)]],
    const constant uint& total_threads [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint4 acc = uint4(0);
  for (uint i = gid; i < n_vec4; i += total_threads) {
    acc ^= src[i];
  }
  const uint folded = acc.x ^ acc.y ^ acc.z ^ acc.w;
  // Never true for the values used, but opaque to the compiler, so the whole
  // read chain stays live without a store on the timed path.
  if (folded == 0xFFFFFFFFu) {
    sink[gid % 1024] = folded;
  }
}

#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

instantiate_kernel("a_shipped", e107_a_shipped, bfloat)
instantiate_kernel("b_constw", e107_b_constw, bfloat)
instantiate_kernel("c_loadonly", e107_c_loadonly, bfloat)
instantiate_kernel("e_floor", e107_e_floor, bfloat)
instantiate_kernel("h_split", e107_h_split, bfloat)
instantiate_kernel("f_mask", e107_f_mask, bfloat)
instantiate_kernel("g_bfe", e107_g_bfe, bfloat)
instantiate_kernel("b2_maskalu", e107_b2_maskalu, bfloat)
instantiate_kernel("i_h_unroll", e107_i_h_unroll, bfloat)
instantiate_kernel("j_f_nounroll", e107_j_f_nounroll, bfloat)
