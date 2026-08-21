#!/usr/bin/env python3
"""E109 rung 1a/1b: generate the threadgroup-shape arms for the two kernels
that carry the intra-kernel latency residual, plus the probe's buffer spec.

    usage: research/e109_shape_arms.py --outdir DIR [--family prework|qkrope]

THE QUESTION. E105 rung 1 split each live dispatch family's interval into
launch, memory and a residual. Two families are mostly residual:

    GDN prework      11.36 us/dispatch, 6.92 us residual (60.9 %), 400 tg of
                     ONE simd group each
    q/k norm + RoPE   9.17 us/dispatch, 5.80 us residual (63.3 %), 140 tg of
                     TWO simd groups each

The residual is neither bytes nor launch. Three hypotheses explain it and they
predict different curves when the same total work is folded into fewer, wider
threadgroups: occupancy (falls then saturates), a dependent chain (flat), or
per-threadgroup granularity (falls as 1/threadgroups). This module builds that
sweep.

FOLDING IS BIT-EXACT BY CONSTRUCTION. Each arm moves grid dimensions into the
threadgroup and nothing else. Every reduction stays inside the simd group (or
the pair of simd groups) that already performed it, so no thread's arithmetic
or accumulation order changes; only which thread is resident beside which does.
The per-unit threadgroup arrays are the whole of the difference. The probe
checks that claim against arm 0's output bytes rather than asserting it.

WHY THE FOLD AXIS MATTERS FOR PREWORK. `logical_head` selects one of three
classes: 0...15 query, 16...31 key, 32...79 value. Query and key take a
barrier path that value does not, so a threadgroup that mixed classes would
execute `threadgroup_barrier` non-uniformly. Folding along the head axis with
a group size that divides 16 and 48 keeps every threadgroup inside one class,
which is why the sweep folds z and not y.
"""

from __future__ import annotations

import argparse
import json
import pathlib

PREAMBLE = """#include "mlx_preamble.h"

typedef bfloat16_t InT;
"""

# `custom_kernel.cpp:71` builds every MLXFast.metalKernel library as
# `metal::utils() + source_`, and `metal::utils()` is the raw-string body of
# this generated file. Reproducing an arm against a hand-written preamble
# instead would change which `abs`, `exp` and `log1p` overloads bind, so the
# arm would no longer be the shipped kernel.
MLX_UTILS_CPP = (pathlib.Path(__file__).resolve().parent.parent
                 / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/utils.cpp")
MLX_UTILS_OPEN = 'R"preamble('
MLX_UTILS_CLOSE = ')preamble"'


def mlx_preamble() -> str:
    text = MLX_UTILS_CPP.read_text()
    start = text.index(MLX_UTILS_OPEN) + len(MLX_UTILS_OPEN)
    end = text.index(MLX_UTILS_CLOSE, start)
    return text[start:end]

PREWORK_HEADER = """
inline InT qwen35_prework_sigmoid(InT x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

inline InT qwen35_prework_beta(InT x) {
  const uint16_t bits = as_type<uint16_t>(x);
  if (bits == uint16_t(0xC0DB)) {
    return as_type<InT>(uint16_t(0x3A8B));
  }
  return qwen35_prework_sigmoid(x);
}

inline InT qwen35_prework_logaddexp(InT x, InT y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<InT>::quiet_NaN();
  }
  constexpr InT inf = metal::numeric_limits<InT>::infinity();
  InT maxval = metal::max(x, y);
  InT minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + log1p(metal::exp(minval - maxval)));
}
"""

# Body is the shipped `qwen35_packed_gdn_prework` source with three edits:
# `unit` selects the logical head inside the threadgroup, and the two
# threadgroup arrays gain a leading UNITS dimension. Nothing else moves.
PREWORK_BODY = """
[[kernel]] void __NAME__(
    const device InT* qkv [[buffer(0)]],
    const constant int64_t* qkv_strides [[buffer(1)]],
    const device InT* a [[buffer(2)]],
    const constant int64_t* a_strides [[buffer(3)]],
    const device InT* b [[buffer(4)]],
    const constant int64_t* b_strides [[buffer(5)]],
    const device InT* conv_state [[buffer(6)]],
    const constant int64_t* conv_state_strides [[buffer(7)]],
    const device InT* conv_weight [[buffer(8)]],
    const constant int64_t* conv_weight_strides [[buffer(9)]],
    const device InT* a_log [[buffer(10)]],
    const device InT* dt_bias [[buffer(11)]],
    const constant InT& q_scale [[buffer(12)]],
    const constant InT& k_scale [[buffer(13)]],
    device InT* q_out [[buffer(14)]],
    device InT* k_out [[buffer(15)]],
    device InT* v_out [[buffer(16)]],
    device InT* conv_out [[buffer(17)]],
    device float* g_out [[buffer(18)]],
    device float* beta_out [[buffer(19)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int Hk = 16;
  constexpr int Dk = 128;
  constexpr int Hv = 48;
  constexpr int Dv = 128;
  constexpr int NKeep = 3;
  constexpr int C = 10240;
  constexpr int T = 5;
  constexpr uint UNITS = __UNITS__;

  const uint unit = thread_position_in_threadgroup.z;
  const uint lane = thread_position_in_threadgroup.x;
  const uint row = threadgroup_position_in_grid.y;
  const uint logical_head = threadgroup_position_in_grid.z * UNITS + unit;

  constexpr uint q_heads = Hk;
  constexpr uint k_head_base = Hk;
  constexpr uint v_head_base = 2 * Hk;
  const bool is_q = logical_head < q_heads;
  const bool is_k = logical_head >= k_head_base
                 && logical_head < v_head_base;
  const uint head = is_q ? logical_head
                   : (is_k ? logical_head - k_head_base
                           : logical_head - v_head_base);
  const uint channel_base = is_q ? head * Dk
                            : (is_k ? Hk * Dk + head * Dk
                                    : 2 * Hk * Dk + head * Dv);

  InT activated[4];
  float sumsq = 0.0f;
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 4; ++i) {
    const uint channel = channel_base + lane * 4 + i;
    float acc = 0.0f;
    #pragma clang loop unroll(full)
    for (uint tap = 0; tap < 4; ++tap) {
      const uint input_row = row + tap;
      const ulong input_offset = input_row < NKeep
          ? ulong(input_row) * ulong(conv_state_strides[1])
              + ulong(channel) * ulong(conv_state_strides[2])
          : ulong(input_row - NKeep) * ulong(qkv_strides[1])
              + ulong(channel) * ulong(qkv_strides[2]);
      const InT xv = input_row < NKeep
          ? conv_state[input_offset]
          : qkv[input_offset];
      const ulong weight_offset =
          ulong(channel) * ulong(conv_weight_strides[0])
          + ulong(tap) * ulong(conv_weight_strides[1]);
      acc += static_cast<float>(xv) * conv_weight[weight_offset];
    }
    const InT conv = static_cast<InT>(acc);
    const InT act = conv * qwen35_prework_sigmoid(conv);
    activated[i] = act;
    const float value = static_cast<float>(act);
    sumsq += value * value;
  }

  if (is_q || is_k) {
__REDUCE__

    const InT scale = is_q ? q_scale : k_scale;
    const uint output_base = (row * Hk + head) * Dk + lane * 4;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
      const InT rms = InT(1) * static_cast<InT>(
          static_cast<float>(activated[i]) * __INVMEAN__);
      const InT value = scale * rms;
      if (is_q) {
        q_out[output_base + i] = value;
      } else {
        k_out[output_base + i] = value;
      }
    }
  } else {
    const uint output_base = (row * Hv + head) * Dv + lane * 4;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
      v_out[output_base + i] = activated[i];
    }

    if (lane == 0) {
      const ulong a_offset = ulong(row) * ulong(a_strides[1])
          + ulong(head) * ulong(a_strides[2]);
      const ulong b_offset = ulong(row) * ulong(b_strides[1])
          + ulong(head) * ulong(b_strides[2]);
      const InT shifted = a[a_offset] + dt_bias[head];
      const InT softplus = qwen35_prework_logaddexp(shifted, InT(0));
      const float exp_a = metal::precise::exp(
          static_cast<float>(a_log[head]));
      const float neg_exp_a = -exp_a;
      const float product = neg_exp_a * static_cast<float>(softplus);
      const uint scalar_output = row * Hv + head;
      g_out[scalar_output] = metal::precise::exp(product);
      beta_out[scalar_output] = static_cast<float>(
          qwen35_prework_beta(b[b_offset]));
    }
  }

  if (row + NKeep >= uint(T)) {
    const uint state_row = row + NKeep - T;
    const ulong raw_base = ulong(row) * ulong(qkv_strides[1])
        + ulong(channel_base + lane * 4) * ulong(qkv_strides[2]);
    const uint state_base = state_row * C + channel_base + lane * 4;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
      conv_out[state_base + i] =
          qkv[raw_base + ulong(i) * ulong(qkv_strides[2])];
    }
  }
}
"""

# Body is the shipped `qwen35_attention_qk_rms_rope_bf16_v1` source with one
# edit: ROWS rows share a threadgroup, so `sub` selects the row and the three
# threadgroup arrays gain a leading ROWS dimension.
QKROPE_BODY = """
[[kernel]] void __NAME__(
    const device bfloat16_t* q [[buffer(0)]],
    const constant int* q_shape [[buffer(1)]],
    const constant int64_t* q_strides [[buffer(2)]],
    const device bfloat16_t* k [[buffer(3)]],
    const constant int* k_shape [[buffer(4)]],
    const constant int64_t* k_strides [[buffer(5)]],
    const device bfloat16_t* q_weight [[buffer(6)]],
    const constant int64_t* q_weight_strides [[buffer(7)]],
    const device bfloat16_t* k_weight [[buffer(8)]],
    const constant int64_t* k_weight_strides [[buffer(9)]],
    const constant float& eps [[buffer(10)]],
    const constant int& offset [[buffer(11)]],
    const constant float& log2_base [[buffer(12)]],
    device bfloat16_t* q_out [[buffer(13)]],
    device bfloat16_t* k_out [[buffer(14)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint n_reads = 4;
  constexpr uint simd_size = 32;
  constexpr uint rotary_dimensions = 64;
  constexpr uint rotary_pairs = rotary_dimensions / 2;
  constexpr uint ROWS = __ROWS__;

  uint sub = thread_position_in_threadgroup.x / 64;
  uint row = threadgroup_position_in_grid.x * ROWS + sub;
  uint thread_id = thread_position_in_threadgroup.x % 64;
  uint simd_thread = thread_index_in_simdgroup;
  uint simd_group = simdgroup_index_in_threadgroup % 2;

  uint batch_size = uint(q_shape[0]);
  uint sequence_length = uint(q_shape[1]);
  uint query_heads = uint(q_shape[2]);
  uint key_heads = uint(k_shape[2]);
  uint axis_size = uint(q_shape[3]);
  uint query_rows = batch_size * query_heads * sequence_length;
  bool is_query = row < query_rows;
  uint local_row = is_query ? row : row - query_rows;
  uint head_count = is_query ? query_heads : key_heads;
  uint batch = local_row / (head_count * sequence_length);
  uint head_sequence = local_row % (head_count * sequence_length);
  uint head = head_sequence / sequence_length;
  uint sequence = head_sequence % sequence_length;

  ulong input_base;
  ulong input_axis_stride;
  ulong weight_stride;
  ulong output_base = ulong(local_row) * ulong(axis_size);
  if (is_query) {
      input_base = ulong(batch) * ulong(q_strides[0])
          + ulong(sequence) * ulong(q_strides[1])
          + ulong(head) * ulong(q_strides[2]);
      input_axis_stride = ulong(q_strides[3]);
      weight_stride = ulong(q_weight_strides[0]);
  } else {
      input_base = ulong(batch) * ulong(k_strides[0])
          + ulong(sequence) * ulong(k_strides[1])
          + ulong(head) * ulong(k_strides[2]);
      input_axis_stride = ulong(k_strides[3]);
      weight_stride = ulong(k_weight_strides[0]);
  }

  threadgroup float local_inv_mean[ROWS];
  threadgroup float local_sums[ROWS][simd_size];
  threadgroup bfloat normalized[ROWS][256];

  float acc = 0.0f;
  uint first = thread_id * n_reads;
  for (uint i = 0; i < n_reads; ++i) {
      uint element = first + i;
      if (element < axis_size) {
          ulong index = input_base + ulong(element) * input_axis_stride;
          float value = is_query ? float(q[index]) : float(k[index]);
          acc += value * value;
      }
  }

  acc = simd_sum(acc);
  if (simd_group == 0) {
      local_sums[sub][simd_thread] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_thread == 0) {
      local_sums[sub][simd_group] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group == 0) {
      acc = simd_sum(local_sums[sub][simd_thread]);
      if (simd_thread == 0) {
          local_inv_mean[sub] = metal::precise::rsqrt(
              acc / axis_size + eps);
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_mean = local_inv_mean[sub];
  for (uint i = 0; i < n_reads; ++i) {
      uint element = first + i;
      if (element < axis_size) {
          ulong index = input_base + ulong(element) * input_axis_stride;
          bfloat input_value = is_query ? q[index] : k[index];
          bfloat rms_value = bfloat(float(input_value) * inv_mean);
          bfloat weight = is_query
              ? q_weight[ulong(element) * weight_stride]
              : k_weight[ulong(element) * weight_stride];
          normalized[sub][element] = weight * rms_value;
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = 0; i < n_reads; ++i) {
      uint element = first + i;
      if (element >= rotary_dimensions && element < axis_size) {
          if (is_query) {
              q_out[output_base + ulong(element)] = normalized[sub][element];
          } else {
              k_out[output_base + ulong(element)] = normalized[sub][element];
          }
      }
  }

  if (thread_id < rotary_pairs / n_reads) {
      for (uint i = 0; i < n_reads; ++i) {
          uint pair = first + i;
          float d = float(pair) / float(rotary_pairs);
          float inv_freq = metal::exp2(-d * float(log2_base));
          float position = float(int(sequence) + int(offset));
          float theta = position * inv_freq;
          float costheta = metal::fast::cos(theta);
          float sintheta = metal::fast::sin(theta);
          float x1 = float(normalized[sub][pair]);
          float x2 = float(normalized[sub][pair + rotary_pairs]);
          bfloat rx1 = bfloat(x1 * costheta - x2 * sintheta);
          bfloat rx2 = bfloat(x1 * sintheta + x2 * costheta);
          if (is_query) {
              q_out[output_base + ulong(pair)] = rx1;
              q_out[output_base + ulong(pair + rotary_pairs)] = rx2;
          } else {
              k_out[output_base + ulong(pair)] = rx1;
              k_out[output_base + ulong(pair + rotary_pairs)] = rx2;
          }
      }
  }
}
"""

# Live verify geometry, S = 5: the width the scored MTP round actually runs.
S = 5
HK, DK, HV, DV, NKEEP = 16, 128, 48, 128, 3
C = 2 * HK * DK + HV * DV          # 10240 logical channels
QKV_ROW_STRIDE = 16480             # the fused in-proj carrier's live stride
QUERY_HEADS, KEY_HEADS, HEAD_DIM = 24, 4, 256


# The shipped RMS reduction, verbatim.
PREWORK_REDUCE_SHIPPED = """    threadgroup float local_inv_mean[UNITS];
    threadgroup float local_sums[UNITS][32];
    sumsq = simd_sum(sumsq);
    local_sums[unit][lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      local_sums[unit][0] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = simd_sum(local_sums[unit][lane]);
    if (lane == 0) {
      local_inv_mean[unit] = metal::precise::rsqrt(sumsq / Dk + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);"""

# The same reduction with a redundant threadgroup round trip removed.
#
# `simd_sum` is a BROADCAST reduction: after the first call every lane already
# holds the complete 32-lane sum. The shipped code then zeroes a threadgroup
# array, has lane 0 store that sum into slot 0, and runs a second `simd_sum`
# over `[sumsq, 0, 0, ... 0]`. That reproduces the number every lane already
# had, at the cost of two threadgroup barriers and a threadgroup round trip.
# Once `sumsq` is known to be live in every lane, `local_inv_mean` is also
# unnecessary: each lane evaluates the same `rsqrt` of the same input, which
# removes a third barrier and the second threadgroup array.
#
# WHY THIS IS BIT-EXACT.
#   * The removed second reduction adds 31 zeros to `sumsq`. `sumsq` is a sum
#     of squares, so it is never -0.0, and x + 0.0 == x exactly for every other
#     float in every summation order. So the second `simd_sum` is the identity.
#   * `rsqrt` is deterministic, so evaluating it per lane gives every lane the
#     same bits lane 0 would have broadcast.
# The probe still byte-compares this arm against arm 0 rather than trusting the
# argument.
PREWORK_REDUCE_DIRECT = """    sumsq = simd_sum(sumsq);
    const float inv_mean = metal::precise::rsqrt(sumsq / Dk + 1e-6f);"""


def prework_body(name: str, units: int, reduce_src: str, inv_mean: str) -> str:
    return (PREWORK_BODY
            .replace("__NAME__", name)
            .replace("__REDUCE__", reduce_src)
            .replace("__INVMEAN__", inv_mean)
            .replace("__UNITS__", str(units)))


def prework_spec(units: list[int]) -> dict:
    buffers = [
        {"name": "qkv", "kind": "in", "dtype": "bf16", "count": S * QKV_ROW_STRIDE},
        {"name": "qkv_strides", "kind": "i64", "values":
            [S * QKV_ROW_STRIDE, QKV_ROW_STRIDE, 1]},
        {"name": "a", "kind": "in", "dtype": "bf16", "count": S * HV},
        {"name": "a_strides", "kind": "i64", "values": [S * HV, HV, 1]},
        {"name": "b", "kind": "in", "dtype": "bf16", "count": S * HV},
        {"name": "b_strides", "kind": "i64", "values": [S * HV, HV, 1]},
        {"name": "conv_state", "kind": "in", "dtype": "bf16", "count": NKEEP * C},
        {"name": "conv_state_strides", "kind": "i64", "values": [NKEEP * C, C, 1]},
        {"name": "conv_weight", "kind": "in", "dtype": "bf16", "count": C * 4},
        {"name": "conv_weight_strides", "kind": "i64", "values": [4, 1, 1]},
        {"name": "a_log", "kind": "in", "dtype": "bf16", "count": HV},
        {"name": "dt_bias", "kind": "in", "dtype": "bf16", "count": HV},
        {"name": "q_scale", "kind": "in", "dtype": "bf16", "count": 1},
        {"name": "k_scale", "kind": "in", "dtype": "bf16", "count": 1},
        {"name": "q_out", "kind": "out", "dtype": "bf16", "count": S * HK * DK},
        {"name": "k_out", "kind": "out", "dtype": "bf16", "count": S * HK * DK},
        {"name": "v_out", "kind": "out", "dtype": "bf16", "count": S * HV * DV},
        {"name": "conv_out", "kind": "out", "dtype": "bf16", "count": NKEEP * C},
        {"name": "g_out", "kind": "out", "dtype": "f32", "count": S * HV},
        {"name": "beta_out", "kind": "out", "dtype": "f32", "count": S * HV},
    ]
    arms = []
    for unit in units:
        arms.append(
            {
                "name": f"g{unit}",
                "function": f"e109_prework_g{unit}",
                "source": f"arm_prework_g{unit}.metal",
                "grid": [32, S, 2 * HK + HV],
                "threadgroup": [32, 1, unit],
                "threadgroups": S * (2 * HK + HV) // unit,
                "simdgroups_per_threadgroup": unit,
                "shipped": unit == 1,
                "exact_vs_arm0": True,
                "reduction": "shipped",
            }
        )
    # The shape sweep names the mechanism; this arm prices one concrete fix for
    # H2 at the shipped shape, so a "dependent chain" verdict arrives with a
    # candidate lever instead of only a diagnosis.
    arms.append(
        {
            "name": "g1nored",
            "function": "e109_prework_g1nored",
            "source": "arm_prework_g1nored.metal",
            "grid": [32, S, 2 * HK + HV],
            "threadgroup": [32, 1, 1],
            "threadgroups": S * (2 * HK + HV),
            "simdgroups_per_threadgroup": 1,
            "shipped": False,
            "exact_vs_arm0": True,
            "reduction": "direct",
        }
    )
    return {
        "family": "prework",
        "live_kernel": "qwen35_packed_gdn_prework",
        "live_grid": [32, S, 2 * HK + HV],
        "live_threadgroup": [32, 1, 1],
        "dispatch": "dispatchThreads",
        "buffers": buffers,
        "arms": arms,
    }


def qkrope_spec(rows_per_tg: list[int]) -> dict:
    total_rows = S * (QUERY_HEADS + KEY_HEADS)
    buffers = [
        {"name": "q", "kind": "in", "dtype": "bf16",
         "count": S * QUERY_HEADS * HEAD_DIM},
        {"name": "q_shape", "kind": "i32",
         "values": [1, S, QUERY_HEADS, HEAD_DIM]},
        {"name": "q_strides", "kind": "i64",
         "values": [S * QUERY_HEADS * HEAD_DIM, QUERY_HEADS * HEAD_DIM,
                    HEAD_DIM, 1]},
        {"name": "k", "kind": "in", "dtype": "bf16",
         "count": S * KEY_HEADS * HEAD_DIM},
        {"name": "k_shape", "kind": "i32",
         "values": [1, S, KEY_HEADS, HEAD_DIM]},
        {"name": "k_strides", "kind": "i64",
         "values": [S * KEY_HEADS * HEAD_DIM, KEY_HEADS * HEAD_DIM,
                    HEAD_DIM, 1]},
        {"name": "q_weight", "kind": "in", "dtype": "bf16", "count": HEAD_DIM},
        {"name": "q_weight_strides", "kind": "i64", "values": [1]},
        {"name": "k_weight", "kind": "in", "dtype": "bf16", "count": HEAD_DIM},
        {"name": "k_weight_strides", "kind": "i64", "values": [1]},
        {"name": "eps", "kind": "f32", "values": [1e-6]},
        {"name": "offset", "kind": "i32", "values": [512]},
        {"name": "log2_base", "kind": "f32", "values": [21.0]},
        {"name": "q_out", "kind": "out", "dtype": "bf16",
         "count": S * QUERY_HEADS * HEAD_DIM},
        {"name": "k_out", "kind": "out", "dtype": "bf16",
         "count": S * KEY_HEADS * HEAD_DIM},
    ]
    arms = []
    for rows in rows_per_tg:
        arms.append(
            {
                "name": f"r{rows}",
                "function": f"e109_qkrope_r{rows}",
                "source": f"arm_qkrope_r{rows}.metal",
                "grid": [total_rows * 64, 1, 1],
                "threadgroup": [64 * rows, 1, 1],
                "threadgroups": total_rows // rows,
                "simdgroups_per_threadgroup": 2 * rows,
                "shipped": rows == 1,
                "exact_vs_arm0": True,
            }
        )
    return {
        "family": "qkrope",
        "live_kernel": "qwen35_attention_qk_rms_rope_bf16_v1",
        "live_grid": [total_rows * 64, 1, 1],
        "live_threadgroup": [64, 1, 1],
        "dispatch": "dispatchThreads",
        "buffers": buffers,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--family", default="both",
                        choices=["prework", "qkrope", "both"])
    parser.add_argument("--prework-units", default="1,2,4,8,16")
    parser.add_argument("--qkrope-rows", default="1,2,4")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mlx_preamble.h").write_text(
        "#pragma once\n" + mlx_preamble())
    written = []

    if args.family in ("prework", "both"):
        units = [int(x) for x in args.prework_units.split(",")]
        for unit in units:
            if (2 * HK) % unit or HV % unit:
                raise SystemExit(
                    f"prework unit {unit} would mix query/key/value classes"
                    " inside one threadgroup")
            path = outdir / f"arm_prework_g{unit}.metal"
            path.write_text(PREAMBLE + PREWORK_HEADER + prework_body(
                f"e109_prework_g{unit}", unit, PREWORK_REDUCE_SHIPPED,
                "local_inv_mean[unit]"))
            written.append(path)
        path = outdir / "arm_prework_g1nored.metal"
        path.write_text(PREAMBLE + PREWORK_HEADER + prework_body(
            "e109_prework_g1nored", 1, PREWORK_REDUCE_DIRECT, "inv_mean"))
        written.append(path)
        spec = prework_spec(units)
        (outdir / "spec_prework.json").write_text(json.dumps(spec, indent=2))

    if args.family in ("qkrope", "both"):
        rows_list = [int(x) for x in args.qkrope_rows.split(",")]
        total_rows = S * (QUERY_HEADS + KEY_HEADS)
        for rows in rows_list:
            if total_rows % rows:
                raise SystemExit(
                    f"qkrope rows {rows} does not divide {total_rows} rows")
            body = QKROPE_BODY.replace("__NAME__", f"e109_qkrope_r{rows}")
            body = body.replace("__ROWS__", str(rows))
            path = outdir / f"arm_qkrope_r{rows}.metal"
            path.write_text(PREAMBLE + body)
            written.append(path)
        spec = qkrope_spec(rows_list)
        (outdir / "spec_qkrope.json").write_text(json.dumps(spec, indent=2))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
