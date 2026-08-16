#!/usr/bin/env python3
"""Component-level arithmetic floor for the charged 512-token seed prefill.

Times every op the Qwen 3.8 27B target executes during `begin` at its exact
scored shape, dtype and quantization, then sums the components into a modelled
prefill wall. Comparing that model with the parent-clock measurement of P from
`research/prefill_amdahl.py` says how much of P is irreducible kernel work and
how much is schedulable overhead.

MLX python 0.32.0 drives the same C++/Metal kernels as the vendored
Vendor/mlx-swift 0.32.0 the scored worker links, so component walls transfer.

Research-only: never packaged into a submission.
"""

import argparse
import json
import time

import mlx.core as mx

H = 5120
INTERMEDIATE = 17408
SEED = 512

N_LINEAR_LAYERS = 48
N_FULL_LAYERS = 16

LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 48
LINEAR_KEY_HEAD_DIM = 128
LINEAR_VALUE_HEAD_DIM = 128
CONV_KERNEL = 4

ATTN_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ATTN_OUTPUT_GATE = True

VOCAB = 248320
QBITS = 4
QGROUP = 64

KEY_DIM = LINEAR_KEY_HEAD_DIM * LINEAR_KEY_HEADS
VALUE_DIM = LINEAR_VALUE_HEAD_DIM * LINEAR_VALUE_HEADS
CONV_DIM = KEY_DIM * 2 + VALUE_DIM

GATED_DELTA_SOURCE = """
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""


def timeit(fn, reps, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def make_qlinear(out_features, in_features):
    w = mx.random.normal([out_features, in_features]).astype(mx.bfloat16) * 0.02
    return mx.quantize(w, group_size=QGROUP, bits=QBITS)


def qmm_fn(x, packed):
    w, scales, biases = packed

    def run():
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=QGROUP, bits=QBITS
        )

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--measured-prefill-seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mx.random.seed(0)
    x = mx.random.normal([1, SEED, H]).astype(mx.bfloat16)
    comps = []

    def record(name, count, fn, macs_per_call, reps=None):
        sec = timeit(fn, reps or args.reps)
        comps.append(
            {
                "component": name,
                "per_call_seconds": sec,
                "calls_per_prefill": count,
                "total_seconds": sec * count,
                "macs_per_call": macs_per_call,
                "tflops_achieved": (2.0 * macs_per_call) / sec / 1e12,
            }
        )

    # Dense bf16 GEMM ceiling at the dominant prefill shape (no dequantization).
    wd = mx.random.normal([H, INTERMEDIATE]).astype(mx.bfloat16)
    xd = x.reshape(SEED, H)
    record("ceiling:dense_bf16_gemm_512x5120x17408", 0, lambda: xd @ wd, SEED * H * INTERMEDIATE)

    # Linear-attention (Gated DeltaNet) layer, 48x.
    fused_out = CONV_DIM + VALUE_DIM + LINEAR_VALUE_HEADS * 2
    qkvzba = make_qlinear(fused_out, H)
    record("linear_attn:in_proj_fused_qkvzba", N_LINEAR_LAYERS, qmm_fn(x, qkvzba), SEED * H * fused_out)

    qkv = make_qlinear(CONV_DIM, H)
    z = make_qlinear(VALUE_DIM, H)
    ba = make_qlinear(LINEAR_VALUE_HEADS * 2, H)

    def unfused():
        a = mx.quantized_matmul(x, *qkv, transpose=True, group_size=QGROUP, bits=QBITS)
        b = mx.quantized_matmul(x, *z, transpose=True, group_size=QGROUP, bits=QBITS)
        c = mx.quantized_matmul(x, *ba, transpose=True, group_size=QGROUP, bits=QBITS)
        return (a, b, c)

    record("linear_attn:in_proj_unfused_alt", 0, unfused, SEED * H * fused_out)

    conv_w = mx.random.normal([CONV_DIM, CONV_KERNEL, 1]).astype(mx.bfloat16)
    conv_x = mx.random.normal([1, SEED + CONV_KERNEL - 1, CONV_DIM]).astype(mx.bfloat16)
    record(
        "linear_attn:conv1d_depthwise_k4",
        N_LINEAR_LAYERS,
        lambda: mx.conv1d(conv_x, conv_w, groups=CONV_DIM),
        SEED * CONV_DIM * CONV_KERNEL,
    )

    kernel = mx.fast.metal_kernel(
        name="gated_delta_step",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=GATED_DELTA_SOURCE,
    )
    qq = mx.random.normal([1, SEED, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM]).astype(mx.bfloat16)
    kk = mx.random.normal([1, SEED, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM]).astype(mx.bfloat16)
    vv = mx.random.normal([1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM]).astype(mx.bfloat16)
    gg = mx.random.uniform(shape=[1, SEED, LINEAR_VALUE_HEADS]).astype(mx.float32)
    bb = mx.random.uniform(shape=[1, SEED, LINEAR_VALUE_HEADS]).astype(mx.float32)
    st = mx.zeros([1, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM, LINEAR_KEY_HEAD_DIM], dtype=mx.float32)

    def delta():
        return kernel(
            inputs=[qq, kk, vv, gg, bb, st, mx.array(SEED)],
            template=[
                ("InT", mx.bfloat16),
                ("StT", mx.float32),
                ("Dk", LINEAR_KEY_HEAD_DIM),
                ("Dv", LINEAR_VALUE_HEAD_DIM),
                ("Hk", LINEAR_KEY_HEADS),
                ("Hv", LINEAR_VALUE_HEADS),
            ],
            grid=(32, LINEAR_VALUE_HEAD_DIM, LINEAR_VALUE_HEADS),
            threadgroup=(32, 4, 1),
            output_shapes=[[1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM], st.shape],
            output_dtypes=[mx.bfloat16, mx.float32],
        )

    record(
        "linear_attn:gated_delta_kernel_T512",
        N_LINEAR_LAYERS,
        delta,
        SEED * LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM * LINEAR_KEY_HEAD_DIM * 4,
    )

    lin_out = make_qlinear(H, VALUE_DIM)
    xv = mx.random.normal([1, SEED, VALUE_DIM]).astype(mx.bfloat16)
    record("linear_attn:out_proj", N_LINEAR_LAYERS, qmm_fn(xv, lin_out), SEED * VALUE_DIM * H)

    # Full-attention layer, 16x.
    q_out = ATTN_HEADS * HEAD_DIM * (2 if ATTN_OUTPUT_GATE else 1)
    qp = make_qlinear(q_out, H)
    kp = make_qlinear(KV_HEADS * HEAD_DIM, H)
    vp = make_qlinear(KV_HEADS * HEAD_DIM, H)
    op = make_qlinear(H, ATTN_HEADS * HEAD_DIM)

    def attn_proj():
        a = mx.quantized_matmul(x, *qp, transpose=True, group_size=QGROUP, bits=QBITS)
        b = mx.quantized_matmul(x, *kp, transpose=True, group_size=QGROUP, bits=QBITS)
        c = mx.quantized_matmul(x, *vp, transpose=True, group_size=QGROUP, bits=QBITS)
        return (a, b, c)

    record(
        "full_attn:qkv_proj",
        N_FULL_LAYERS,
        attn_proj,
        SEED * H * (q_out + 2 * KV_HEADS * HEAD_DIM),
    )

    xo = mx.random.normal([1, SEED, ATTN_HEADS * HEAD_DIM]).astype(mx.bfloat16)
    record("full_attn:o_proj", N_FULL_LAYERS, qmm_fn(xo, op), SEED * ATTN_HEADS * HEAD_DIM * H)

    qa = mx.random.normal([1, ATTN_HEADS, SEED, HEAD_DIM]).astype(mx.bfloat16)
    ka = mx.random.normal([1, KV_HEADS, SEED, HEAD_DIM]).astype(mx.bfloat16)
    va = mx.random.normal([1, KV_HEADS, SEED, HEAD_DIM]).astype(mx.bfloat16)
    record(
        "full_attn:sdpa_causal_512",
        N_FULL_LAYERS,
        lambda: mx.fast.scaled_dot_product_attention(qa, ka, va, scale=1.0 / 16.0, mask="causal"),
        ATTN_HEADS * SEED * SEED * HEAD_DIM,
    )

    # Dense MLP, all 64 layers. Gate/up stay unfused above S == 9.
    gate = make_qlinear(INTERMEDIATE, H)
    up = make_qlinear(INTERMEDIATE, H)
    down = make_qlinear(H, INTERMEDIATE)
    xi = mx.random.normal([1, SEED, INTERMEDIATE]).astype(mx.bfloat16)

    def mlp():
        a = mx.quantized_matmul(x, *gate, transpose=True, group_size=QGROUP, bits=QBITS)
        b = mx.quantized_matmul(x, *up, transpose=True, group_size=QGROUP, bits=QBITS)
        h = mx.fast.silu(a) * b if hasattr(mx.fast, "silu") else (a * mx.sigmoid(a)) * b
        return mx.quantized_matmul(h, *down, transpose=True, group_size=QGROUP, bits=QBITS)

    record("mlp:gate_up_down", N_LINEAR_LAYERS + N_FULL_LAYERS, mlp, 3 * SEED * H * INTERMEDIATE)

    # Head work `begin` actually performs: final norm plus a single-row readout.
    lm = make_qlinear(VOCAB, H)
    x1 = mx.random.normal([1, 1, H]).astype(mx.bfloat16)
    record("head:lm_head_single_row", 1, qmm_fn(x1, lm), H * VOCAB)

    norm_w = mx.random.normal([H]).astype(mx.bfloat16)
    record(
        "head:final_norm_full_512",
        1,
        lambda: mx.fast.rms_norm(x, norm_w, 1e-6),
        SEED * H,
    )

    modelled = sum(c["total_seconds"] for c in comps if c["calls_per_prefill"] > 0)
    measured = args.measured_prefill_seconds
    out = {
        "host": {"chip": "Apple M4 Pro", "gpu_cores": 20, "mlx_python": "0.32.0", "mlx_swift": "0.32.0"},
        "seed_tokens": SEED,
        "components": sorted(comps, key=lambda c: -c["total_seconds"]),
        "modelled_prefill_seconds": modelled,
        "measured_prefill_seconds": measured,
        "unattributed_seconds": measured - modelled,
        "unattributed_fraction_of_measured": (measured - modelled) / measured,
        "modelled_fraction_of_measured": modelled / measured,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'component':44s} {'per_call_ms':>12s} {'x':>4s} {'total_s':>9s} {'TFLOP/s':>9s}")
    for c in out["components"]:
        print(
            f"{c['component']:44s} {c['per_call_seconds']*1e3:12.4f} "
            f"{c['calls_per_prefill']:4d} {c['total_seconds']:9.4f} {c['tflops_achieved']:9.3f}"
        )
    print(f"\nmodelled prefill   = {modelled:.4f} s")
    print(f"measured prefill P = {measured:.4f} s")
    print(f"unattributed       = {measured - modelled:.4f} s ({100*(measured-modelled)/measured:.1f}%)")


if __name__ == "__main__":
    main()
