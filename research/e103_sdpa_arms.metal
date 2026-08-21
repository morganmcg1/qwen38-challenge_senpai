// E103 rungs 1 and 2: arms of the scored full-attention decode SDPA kernel.
//
// Research only. Nothing here is on the scored path. One file serves both
// rungs: `research/agx_crossarch.py census` reads registers and spill from the
// compiled metallib (rung 1), and `research/e103_sdpa_ab.m` times the same
// metallib in one process with ABBA ordering (rung 2).
//
// Arm `a` is a transcription of the shipped
// `Vendor/mlx-swift/.../kernels/sdpa_vector.h:16-177` `sdpa_vector` template at
// the scored specialisation: bfloat16, D = V = 256, no mask array, query not
// transposed, no sinks. The mask and sink branches are dropped because the
// trusted dispatcher pins those function constants to false for this model
// (`scaled_dot_product_attention.cpp:361-378`, no array mask is ever built).
// `CAUSAL` stays a template parameter because the dispatcher sets it from
// `q.shape(2) > 1`.
//
//   xcrun -sdk macosx metal -O2 -std=metal3.1 -c e103_sdpa_arms.metal -o x.air
//   xcrun -sdk macosx metallib x.air -o e103_sdpa_arms.metallib

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// ---------------------------------------------------------------------------
// a: shipped transcription.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_a_shipped(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset = o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// b: 4-wide vector loads for K and V. Arithmetic and reduction order are
// unchanged, so the output must be bit-identical to arm a.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_b_vecload(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int kvec = qk_per_thread / 4;
  constexpr int vvec = v_per_thread / 4;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;
  typedef vec<T, 4> T4;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      const device T4* k4 = reinterpret_cast<const device T4*>(keys);
      U score = 0;
      for (int c = 0; c < kvec; c++) {
        T4 kv4 = k4[c];
        for (int j = 0; j < 4; j++) {
          score += q[4 * c + j] * static_cast<U>(kv4[j]);
        }
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      const device T4* v4 = reinterpret_cast<const device T4*>(values);
      for (int c = 0; c < vvec; c++) {
        T4 vv4 = v4[c];
        for (int j = 0; j < 4; j++) {
          o[4 * c + j] = o[4 * c + j] * factor +
              exp_score * static_cast<U>(vv4[j]);
        }
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// c: arm b plus the unit-rescale fast path. `factor` is exactly 1.0 whenever
// the running maximum does not move, and the branch is simdgroup uniform
// because `score` comes out of `simd_sum`. Whether the fast path is
// bit-identical to arm a depends on how the compiler contracts
// `o*1 + exp*v`, so `research/e103_sdpa_ab.m` compares the two outputs.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_c_fastpath(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int kvec = qk_per_thread / 4;
  constexpr int vvec = v_per_thread / 4;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;
  typedef vec<T, 4> T4;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      const device T4* k4 = reinterpret_cast<const device T4*>(keys);
      U score = 0;
      for (int c = 0; c < kvec; c++) {
        T4 kv4 = k4[c];
        for (int j = 0; j < 4; j++) {
          score += q[4 * c + j] * static_cast<U>(kv4[j]);
        }
      }
      score = simd_sum(score);

      const device T4* v4 = reinterpret_cast<const device T4*>(values);
      if (score <= max_score) {
        U exp_score = fast::exp(score - max_score);
        sum_exp_score = sum_exp_score * 1.0f + exp_score;
        for (int c = 0; c < vvec; c++) {
          T4 vv4 = v4[c];
          for (int j = 0; j < 4; j++) {
            o[4 * c + j] = o[4 * c + j] * 1.0f +
                exp_score * static_cast<U>(vv4[j]);
          }
        }
      } else {
        U factor = fast::exp(max_score - score);
        U exp_score = fast::exp(0.0f);
        max_score = score;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int c = 0; c < vvec; c++) {
          T4 vv4 = v4[c];
          for (int j = 0; j < 4; j++) {
            o[4 * c + j] = o[4 * c + j] * factor +
                exp_score * static_cast<U>(vv4[j]);
          }
        }
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// d: P query heads packed into one threadgroup. One K element and one V element
// are loaded once and used by all P heads, so the device-memory traffic falls
// by P and the load instruction count per query falls by the same factor.
//
// The grid is (num_q_heads / P, q_seq_len, 1). The shipped grid is fixed by
// trusted code, so this arm can only ever be dispatched by an editable
// Swift-side custom kernel, never by `sdpa_vector.h`.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, int P, bool CAUSAL>
[[kernel]] void sdpa_d_pack(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int kvec = qk_per_thread / 4;
  constexpr int vvec = v_per_thread / 4;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;
  typedef vec<T, 4> T4;

  thread U q[P][qk_per_thread];
  thread U o[P][v_per_thread];
  thread U max_score[P];
  thread U sum_exp_score[P];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int head_group = tid.x;
  const int q_seq_idx = tid.y;
  const int first_q_head = head_group * P;
  const int kv_head_idx = first_q_head / gqa_factor;
  const int q_seq_len = int(tpg.y);

  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;

  for (int p = 0; p < P; p++) {
    const device T* qp =
        queries + ((first_q_head + p) * q_seq_len + q_seq_idx) * D +
        simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[p][i] = static_cast<U>(scale) * qp[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      o[p][i] = 0;
    }
    max_score[p] = -INFINITY;
    sum_exp_score[p] = 0;
  }

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - q_seq_len + q_seq_idx);
    }
    if (use_key) {
      const device T4* k4 = reinterpret_cast<const device T4*>(keys);
      U score[P];
      for (int p = 0; p < P; p++) {
        score[p] = 0;
      }
      for (int c = 0; c < kvec; c++) {
        T4 kv4 = k4[c];
        for (int j = 0; j < 4; j++) {
          U kj = static_cast<U>(kv4[j]);
          for (int p = 0; p < P; p++) {
            score[p] += q[p][4 * c + j] * kj;
          }
        }
      }
      for (int p = 0; p < P; p++) {
        score[p] = simd_sum(score[p]);
      }

      const device T4* v4 = reinterpret_cast<const device T4*>(values);
      U factor[P], exp_score[P];
      for (int p = 0; p < P; p++) {
        U new_max = max(max_score[p], score[p]);
        factor[p] = fast::exp(max_score[p] - new_max);
        exp_score[p] = fast::exp(score[p] - new_max);
        max_score[p] = new_max;
        sum_exp_score[p] = sum_exp_score[p] * factor[p] + exp_score[p];
      }
      for (int c = 0; c < vvec; c++) {
        T4 vv4 = v4[c];
        for (int j = 0; j < 4; j++) {
          U vj = static_cast<U>(vv4[j]);
          for (int p = 0; p < P; p++) {
            o[p][4 * c + j] = o[p][4 * c + j] * factor[p] + exp_score[p] * vj;
          }
        }
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  for (int p = 0; p < P; p++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score[p];
      sum_exp_scores[simd_gid] = sum_exp_score[p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U mine = max_scores[simd_lid];
    U new_max = simd_max(mine);
    U factor = fast::exp(mine - new_max);
    U total = simd_sum(sum_exp_scores[simd_lid] * factor);

    device T* outp =
        out + ((first_q_head + p) * q_seq_len + q_seq_idx) * V +
        simd_gid * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) {
      U value = o[p][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      outputs[simd_lid * BD + simd_gid] = value;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      U reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
      reduced = total == 0 ? reduced : (reduced / total);
      if (simd_lid == 0) {
        outp[i] = static_cast<T>(reduced);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// e: traffic-free control. Same instruction stream as arm a, but the key and
// value pointers never advance, so every iteration reads one resident tile.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_e_resident(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// f: softmax-free control. Same pointer walk and the same loads as arm a, but
// no cross-lane reduction, no exponential and no rescaling.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_f_nosoftmax(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      for (int j = 0; j < v_per_thread; j++) {
        o[j] += score * values[j];
      }
    }
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// g: positive control. Arm a with the key loop run twice. It must be close to
// twice arm a; if it is not, the harness cannot resolve this kernel at all.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_g_double(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  const device T* keys0 = keys + kv_head_idx * k_head_stride +
      simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  const device T* values0 = values + kv_head_idx * v_head_stride +
      simd_gid * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int pass = 0; pass < 2; pass++) {
    const device T* kp = keys0;
    const device T* vp = values0;
    for (int i = simd_gid; i < N; i += BN) {
      bool use_key = true;
      if (CAUSAL) {
        use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
      }
      if (use_key) {
        for (int j = 0; j < qk_per_thread; j++) {
          k[j] = kp[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * vp[j];
        }
      }
      kp += inner_k_stride;
      vp += inner_v_stride;
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// h: tail-free control. The key loop is byte for byte the one in arm a, but the
// cross-simdgroup output reduction and its sixteen threadgroup barriers are
// removed and every simdgroup writes its own partial accumulator. The answer is
// wrong on purpose; the arm exists only to price the tail.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_h_tailfree(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (CAUSAL) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    }
    if (use_key) {
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i] + sum_exp_score);
    }
  }
}

// ---------------------------------------------------------------------------
// j: launch-only control. The query load, the pointer arithmetic and the output
// write survive; the key loop and the tail do not. This is the floor that no
// change inside the kernel can go below while the trusted dispatcher keeps one
// threadgroup per query vector.
// ---------------------------------------------------------------------------

template <typename T, int D, int V, bool CAUSAL>
[[kernel]] void sdpa_j_launchonly(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;
  thread U q[qk_per_thread];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += o_offset * D + simd_lid * qk_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(q[i % qk_per_thread] +
                              static_cast<U>(N) * 0.0f +
                              static_cast<U>(gqa_factor) * 0.0f);
    }
  }
}

instantiate_kernel("a_shipped_c", sdpa_a_shipped, bfloat, 256, 256, true)
instantiate_kernel("a_shipped_nc", sdpa_a_shipped, bfloat, 256, 256, false)
instantiate_kernel("b_vecload_c", sdpa_b_vecload, bfloat, 256, 256, true)
instantiate_kernel("c_fastpath_c", sdpa_c_fastpath, bfloat, 256, 256, true)
instantiate_kernel("d_pack1_c", sdpa_d_pack, bfloat, 256, 256, 1, true)
instantiate_kernel("d_pack2_c", sdpa_d_pack, bfloat, 256, 256, 2, true)
instantiate_kernel("d_pack3_c", sdpa_d_pack, bfloat, 256, 256, 3, true)
instantiate_kernel("d_pack6_c", sdpa_d_pack, bfloat, 256, 256, 6, true)
instantiate_kernel("e_resident_c", sdpa_e_resident, bfloat, 256, 256, true)
instantiate_kernel("f_nosoftmax_c", sdpa_f_nosoftmax, bfloat, 256, 256, true)
instantiate_kernel("g_double_c", sdpa_g_double, bfloat, 256, 256, true)
instantiate_kernel("h_tailfree_c", sdpa_h_tailfree, bfloat, 256, 256, true)
instantiate_kernel("j_launchonly_c", sdpa_j_launchonly, bfloat, 256, 256, true)
