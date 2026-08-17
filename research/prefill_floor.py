#!/usr/bin/env python3
"""Component-level arithmetic floor for the charged 512-token seed prefill.

Times every op the Qwen 3.8 27B target executes during `begin` at its exact
scored shape, dtype and quantization, then sums the components into a modelled
prefill wall. Comparing that model with the parent-clock measurement of P from
`research/prefill_amdahl.py` says how much of P is irreducible kernel work and
how much is schedulable overhead.

The available MLX python wheel is 0.29.3 while the scored worker links vendored
mlx-swift 0.32.0, so absolute component rates are directional and only the
internal comparisons made inside one invocation of this script are load-bearing.
The recorded `mlx_version` in the output names the wheel actually used.

Research-only: never packaged into a submission.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
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
FULL_ATTENTION_INTERVAL = 4
ROPE_DIMS = 64
ROPE_THETA = 10_000_000.0

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


def host_info():
    def sysctl(key):
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    try:
        mlx_version = importlib.metadata.version("mlx")
    except Exception:
        mlx_version = None
    gpu_cores = None
    try:
        for line in subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, check=True
        ).stdout.splitlines():
            if "Total Number of Cores" in line:
                gpu_cores = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "gpu_cores": gpu_cores,
        "cpu_perf_cores": sysctl("hw.perflevel0.physicalcpu"),
        "cpu_cores": sysctl("hw.ncpu"),
        "memsize_bytes": sysctl("hw.memsize"),
        "macos_build": sysctl("kern.osversion"),
        # The Swift build vendors mlx-swift's C++ core (Vendor/mlx-swift
        # version.h); the Python wheel is a separate release line, so these two
        # numbers are not expected to match and the floor is only directional
        # for absolute kernel rates.
        "mlx_python": mlx_version,
        "python": platform.python_version(),
    }


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


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


def timeit_lazy(make_outputs, reps, warmup=2):
    """Build a whole lazy graph, then time one terminal eval.

    Returns (build_seconds, eval_seconds) medians. The graph is rebuilt every
    rep because evaluated arrays are already materialized. Splitting the two
    walls is the point: `eval_seconds` is the GPU wall with MLX free to
    pipeline every launch, which a per-op `eval`+`synchronize` loop forbids.
    """
    for _ in range(warmup):
        mx.eval(make_outputs())
        mx.synchronize()
    builds, evals = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = make_outputs()
        t1 = time.perf_counter()
        mx.eval(outs)
        mx.synchronize()
        t2 = time.perf_counter()
        builds.append(t1 - t0)
        evals.append(t2 - t1)
    return median(builds), median(evals)


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


def qmm(x, packed):
    w, scales, biases = packed
    return mx.quantized_matmul(
        x, w, scales, biases, transpose=True, group_size=QGROUP, bits=QBITS
    )


def rms(x, w):
    return mx.fast.rms_norm(x, w, 1e-6)


def silu(x):
    return mx.fast.silu(x) if hasattr(mx.fast, "silu") else x * mx.sigmoid(x)


def build_pipelined_prefill(w, x0, delta_kernel):
    """The 512-row prefill as one lazy graph: no intermediate eval, real data
    dependencies between all 64 layers, weights shared across same-kind layers.

    Every op the target runs is present, including the per-layer norms,
    residual adds, rope, gating and activations that the per-component floor
    leaves out. Shared weights change cache residency, not kernel work.
    """
    h = x0
    for i in range(N_LINEAR_LAYERS + N_FULL_LAYERS):
        n = rms(h, w["norm"])
        if (i + 1) % FULL_ATTENTION_INTERVAL != 0:
            y = qmm(n, w["qkvzba"])
            conv_in = y[..., :CONV_DIM]
            z = y[..., CONV_DIM : CONV_DIM + VALUE_DIM]
            ba = y[..., CONV_DIM + VALUE_DIM :]
            conv_in = mx.pad(conv_in, [(0, 0), (CONV_KERNEL - 1, 0), (0, 0)])
            c = silu(mx.conv1d(conv_in, w["conv"], groups=CONV_DIM))
            q = c[..., :KEY_DIM].reshape(1, SEED, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM)
            k = c[..., KEY_DIM : 2 * KEY_DIM].reshape(
                1, SEED, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM
            )
            v = c[..., 2 * KEY_DIM :].reshape(1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
            g = mx.sigmoid(ba[..., :LINEAR_VALUE_HEADS]).astype(mx.float32)
            beta = mx.sigmoid(ba[..., LINEAR_VALUE_HEADS:]).astype(mx.float32)
            core, _ = delta_kernel(
                inputs=[q, k, v, g, beta, w["state"], mx.array(SEED)],
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
                output_shapes=[
                    [1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM],
                    w["state"].shape,
                ],
                output_dtypes=[mx.bfloat16, mx.float32],
            )
            core = rms(core, w["dnorm"]) * mx.sigmoid(
                z.reshape(1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
            )
            r = qmm(core.reshape(1, SEED, VALUE_DIM), w["lin_out"])
        else:
            qg = qmm(n, w["qp"])
            kk = qmm(n, w["kp"])
            vv = qmm(n, w["vp"])
            gate = qg[..., ATTN_HEADS * HEAD_DIM :]
            q = qg[..., : ATTN_HEADS * HEAD_DIM].reshape(1, SEED, ATTN_HEADS, HEAD_DIM)
            q = rms(q, w["hnorm"]).transpose(0, 2, 1, 3)
            k = rms(kk.reshape(1, SEED, KV_HEADS, HEAD_DIM), w["hnorm"]).transpose(0, 2, 1, 3)
            v = vv.reshape(1, SEED, KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, ROPE_DIMS, traditional=False, base=ROPE_THETA, scale=1.0, offset=0)
            k = mx.fast.rope(k, ROPE_DIMS, traditional=False, base=ROPE_THETA, scale=1.0, offset=0)
            o = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / 16.0, mask="causal"
            )
            o = o.transpose(0, 2, 1, 3).reshape(1, SEED, ATTN_HEADS * HEAD_DIM)
            r = qmm(o * mx.sigmoid(gate), w["op"])
        h = h + r
        n2 = rms(h, w["norm"])
        gu = silu(qmm(n2, w["gate"])) * qmm(n2, w["up"])
        h = h + qmm(gu, w["down"])
    h = rms(h, w["norm"])
    return [qmm(h[:, -1:, :], w["lm"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--chain", type=int, default=0,
                    help="per-component copies built into one graph and evaluated once")
    ap.add_argument("--pipeline-reps", type=int, default=0,
                    help="reps of the whole-prefill single-eval graph (0 disables)")
    ap.add_argument("--measured-prefill-seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mx.random.seed(0)
    x = mx.random.normal([1, SEED, H]).astype(mx.bfloat16)
    comps = []

    def record(name, count, fn, macs_per_call, reps=None, is_gemm=False):
        sec = timeit(fn, reps or args.reps)
        entry = {
            "component": name,
            "per_call_seconds": sec,
            "calls_per_prefill": count,
            "total_seconds": sec * count,
            "macs_per_call": macs_per_call,
            "tflops_achieved": (2.0 * macs_per_call) / sec / 1e12,
            "is_gemm": is_gemm,
        }
        if args.chain > 0:
            n = args.chain
            _, one = timeit_lazy(lambda: [fn()], max(3, (reps or args.reps) // 3))
            _, many = timeit_lazy(lambda: [fn() for _ in range(n)], max(3, (reps or args.reps) // 3))
            per_call = many / n
            entry.update(
                {
                    "chain": n,
                    "chained_per_call_seconds": per_call,
                    "chained_total_seconds": per_call * count,
                    "chained_tflops_achieved": (2.0 * macs_per_call) / per_call / 1e12,
                    # ~n proves MLX evaluated n distinct copies instead of
                    # collapsing identical subgraphs, which would silently
                    # divide the measured rate by the chain length.
                    "chain_scaling": many / one,
                    "sync_overhead_per_call_seconds": sec - per_call,
                }
            )
        comps.append(entry)

    # Dense bf16 GEMM ceiling at the dominant prefill shape (no dequantization).
    wd = mx.random.normal([H, INTERMEDIATE]).astype(mx.bfloat16)
    xd = x.reshape(SEED, H)
    record("ceiling:dense_bf16_gemm_512x5120x17408", 0, lambda: xd @ wd, SEED * H * INTERMEDIATE)

    # Linear-attention (Gated DeltaNet) layer, 48x.
    fused_out = CONV_DIM + VALUE_DIM + LINEAR_VALUE_HEADS * 2
    qkvzba = make_qlinear(fused_out, H)
    record(
        "linear_attn:in_proj_fused_qkvzba",
        N_LINEAR_LAYERS,
        qmm_fn(x, qkvzba),
        SEED * H * fused_out,
        is_gemm=True,
    )

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
    record(
        "linear_attn:out_proj",
        N_LINEAR_LAYERS,
        qmm_fn(xv, lin_out),
        SEED * VALUE_DIM * H,
        is_gemm=True,
    )

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
        is_gemm=True,
    )

    xo = mx.random.normal([1, SEED, ATTN_HEADS * HEAD_DIM]).astype(mx.bfloat16)
    record(
        "full_attn:o_proj",
        N_FULL_LAYERS,
        qmm_fn(xo, op),
        SEED * ATTN_HEADS * HEAD_DIM * H,
        is_gemm=True,
    )

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

    record(
        "mlp:gate_up_down",
        N_LINEAR_LAYERS + N_FULL_LAYERS,
        mlp,
        3 * SEED * H * INTERMEDIATE,
        is_gemm=True,
    )

    # Head work `begin` actually performs: final norm plus a single-row readout.
    lm = make_qlinear(VOCAB, H)
    x1 = mx.random.normal([1, 1, H]).astype(mx.bfloat16)
    record("head:lm_head_single_row", 1, qmm_fn(x1, lm), H * VOCAB, is_gemm=True)

    norm_w = mx.random.normal([H]).astype(mx.bfloat16)
    record(
        "head:final_norm_full_512",
        1,
        lambda: mx.fast.rms_norm(x, norm_w, 1e-6),
        SEED * H,
    )

    # Elementwise work the original component list omitted: two RMSNorms and
    # two residual adds per layer, plus the per-kind gating/rope glue.
    record("layer:rms_norm_5120_512", 2 * (N_LINEAR_LAYERS + N_FULL_LAYERS),
           lambda: mx.fast.rms_norm(x, norm_w, 1e-6), 0)
    record("layer:residual_add_512", 2 * (N_LINEAR_LAYERS + N_FULL_LAYERS),
           lambda: x + x, 0)

    hnorm_w = mx.random.normal([HEAD_DIM]).astype(mx.bfloat16)
    gate_x = mx.random.normal([1, SEED, ATTN_HEADS * HEAD_DIM]).astype(mx.bfloat16)

    def attn_glue():
        q = rms(qa.transpose(0, 2, 1, 3), hnorm_w).transpose(0, 2, 1, 3)
        k = rms(ka.transpose(0, 2, 1, 3), hnorm_w).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, ROPE_DIMS, traditional=False, base=ROPE_THETA, scale=1.0, offset=0)
        k = mx.fast.rope(k, ROPE_DIMS, traditional=False, base=ROPE_THETA, scale=1.0, offset=0)
        return (q, k, xo * mx.sigmoid(gate_x))

    record("full_attn:qk_norm_rope_gate", N_FULL_LAYERS, attn_glue, 0)

    dnorm_w = mx.random.normal([LINEAR_VALUE_HEAD_DIM]).astype(mx.bfloat16)
    zx = mx.random.normal([1, SEED, VALUE_DIM]).astype(mx.bfloat16)
    conv_out = mx.random.normal([1, SEED, CONV_DIM]).astype(mx.bfloat16)
    ba_x = mx.random.normal([1, SEED, LINEAR_VALUE_HEADS * 2]).astype(mx.bfloat16)

    def delta_glue():
        c = silu(conv_out)
        g = mx.sigmoid(ba_x[..., :LINEAR_VALUE_HEADS]).astype(mx.float32)
        b = mx.sigmoid(ba_x[..., LINEAR_VALUE_HEADS:]).astype(mx.float32)
        core = rms(vv, dnorm_w) * mx.sigmoid(
            zx.reshape(1, SEED, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        )
        return (c, g, b, core)

    record("linear_attn:conv_silu_gate_norm", N_LINEAR_LAYERS, delta_glue, 0)

    measured = args.measured_prefill_seconds
    modelled = sum(c["total_seconds"] for c in comps if c["calls_per_prefill"] > 0)
    ceiling = next(c for c in comps if c["component"].startswith("ceiling:"))["tflops_achieved"]

    gemms = [c for c in comps if c["is_gemm"] and c["calls_per_prefill"] > 0]
    gemm_seconds = sum(c["total_seconds"] for c in gemms)
    gemm_macs = sum(c["macs_per_call"] * c["calls_per_prefill"] for c in gemms)
    gemm_at_ceiling = (2.0 * gemm_macs) / (ceiling * 1e12)
    nongemm_seconds = modelled - gemm_seconds
    dequant = gemm_seconds - gemm_at_ceiling
    # Signed closing budget for the measured prefill. The floor proper is the
    # GEMM work at the dense ceiling plus the non-GEMM components; everything
    # between that subtotal and the measurement is the residual, and the
    # residual is *named* by two signed terms rather than absorbed.
    floor_subtotal = gemm_at_ceiling + nongemm_seconds
    residual = measured - floor_subtotal
    overlap_credit = modelled - measured
    budget = {
        "ceiling_tflops": ceiling,
        "gemm_tflop_total": 2.0 * gemm_macs / 1e12,
        "gemm_seconds_measured": gemm_seconds,
        "gemm_tflops_achieved": (2.0 * gemm_macs) / gemm_seconds / 1e12,
        "gemm_at_ceiling_seconds": gemm_at_ceiling,
        "gemm_at_ceiling_fraction_of_measured": gemm_at_ceiling / measured,
        "gemm_fraction_of_ceiling": gemm_at_ceiling / gemm_seconds,
        "nongemm_seconds": nongemm_seconds,
        "floor_subtotal_seconds": floor_subtotal,
        "floor_subtotal_fraction_of_measured": floor_subtotal / measured,
        "residual_seconds": residual,
        "residual_fraction_of_measured": residual / measured,
        "dequant_overhead_seconds": dequant,
        "dequant_fraction_of_measured": dequant / measured,
        # Positive means MLX overlapped work the per-op floor charges serially;
        # negative means the floor is missing work the model actually performs.
        "overlap_credit_seconds": overlap_credit,
        "overlap_credit_fraction_of_measured": overlap_credit / measured,
        # floor + dequant - overlap == measured, so this must be ~0.
        "closure_error_seconds": floor_subtotal + dequant - overlap_credit - measured,
        "modelled_prefill_seconds": modelled,
        "measured_prefill_seconds": measured,
    }
    if args.chain > 0:
        chained = sum(c.get("chained_total_seconds", 0.0) for c in comps if c["calls_per_prefill"] > 0)
        budget["chained_modelled_seconds"] = chained
        budget["chained_vs_synced_ratio"] = chained / modelled
        budget["chained_vs_measured_ratio"] = chained / measured
        budget["min_chain_scaling_normalized"] = min(
            c["chain_scaling"] for c in comps if "chain_scaling" in c
        ) / args.chain

    pipelined = None
    if args.pipeline_reps > 0:
        w = {
            "norm": norm_w, "dnorm": dnorm_w, "hnorm": hnorm_w,
            "conv": conv_w, "state": st,
            "qkvzba": qkvzba, "lin_out": lin_out,
            "qp": qp, "kp": kp, "vp": vp, "op": op,
            "gate": gate, "up": up, "down": down, "lm": lm,
        }
        b_sec, e_sec = timeit_lazy(
            lambda: build_pipelined_prefill(w, x, kernel), args.pipeline_reps
        )
        pipelined = {
            "reps": args.pipeline_reps,
            "build_seconds": b_sec,
            "eval_seconds": e_sec,
            "total_seconds": b_sec + e_sec,
            "eval_vs_measured_ratio": e_sec / measured,
            "total_vs_measured_ratio": (b_sec + e_sec) / measured,
            "eval_vs_modelled_ratio": e_sec / modelled,
            "implied_tflops": (2.0 * gemm_macs) / e_sec / 1e12,
        }
        budget["pipelined_build_seconds"] = b_sec
        budget["pipelined_eval_seconds"] = e_sec

    out = {
        "host": host_info(),
        "seed_tokens": SEED,
        "components": sorted(comps, key=lambda c: -c["total_seconds"]),
        "budget": budget,
        "pipelined_prefill": pipelined,
        "modelled_prefill_seconds": modelled,
        "measured_prefill_seconds": measured,
        "unattributed_seconds": measured - modelled,
        "unattributed_fraction_of_measured": (measured - modelled) / measured,
        "modelled_fraction_of_measured": modelled / measured,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    chained_cols = args.chain > 0
    hdr = f"{'component':44s} {'per_call_ms':>12s} {'x':>4s} {'total_s':>9s} {'TFLOP/s':>9s}"
    if chained_cols:
        hdr += f" {'chain_ms':>10s} {'chainTF/s':>10s} {'scale/n':>8s}"
    print(hdr)
    for c in out["components"]:
        line = (
            f"{c['component']:44s} {c['per_call_seconds']*1e3:12.4f} "
            f"{c['calls_per_prefill']:4d} {c['total_seconds']:9.4f} {c['tflops_achieved']:9.3f}"
        )
        if chained_cols and "chained_per_call_seconds" in c:
            line += (
                f" {c['chained_per_call_seconds']*1e3:10.4f} "
                f"{c['chained_tflops_achieved']:10.3f} {c['chain_scaling']/args.chain:8.3f}"
            )
        print(line)
    print(f"\nmodelled prefill   = {modelled:.4f} s")
    print(f"measured prefill P = {measured:.4f} s")
    print(f"unattributed       = {measured - modelled:.4f} s ({100*(measured-modelled)/measured:.1f}%)")
    print(
        f"\nGEMM {gemm_seconds:.4f} s at {budget['gemm_tflops_achieved']:.3f} TFLOP/s "
        f"({100*budget['gemm_fraction_of_ceiling']:.1f}% of the {ceiling:.3f} TFLOP/s ceiling)"
    )
    print(
        f"  = {gemm_at_ceiling:.4f} s at ceiling + {budget['dequant_overhead_seconds']:.4f} s dequant"
    )
    print(f"non-GEMM           = {nongemm_seconds:.4f} s")
    print(
        f"floor subtotal     = {floor_subtotal:.4f} s "
        f"({100*budget['floor_subtotal_fraction_of_measured']:.2f}% of P), residual "
        f"{residual:+.4f} s ({100*budget['residual_fraction_of_measured']:+.2f}%)"
    )
    print(
        f"  residual named as dequant {dequant:+.4f} s "
        f"({100*budget['dequant_fraction_of_measured']:+.2f}%) minus overlap credit "
        f"{overlap_credit:+.4f} s ({100*budget['overlap_credit_fraction_of_measured']:+.2f}%), "
        f"closure error {budget['closure_error_seconds']:+.2e} s"
    )
    if args.chain > 0:
        print(
            f"chained modelled   = {budget['chained_modelled_seconds']:.4f} s "
            f"({budget['chained_vs_synced_ratio']:.3f}x per-op-synced), "
            f"min chain scaling {budget['min_chain_scaling_normalized']:.3f}"
        )
    if pipelined:
        print(
            f"pipelined prefill  = build {pipelined['build_seconds']:.4f} s + eval "
            f"{pipelined['eval_seconds']:.4f} s = {pipelined['total_seconds']:.4f} s "
            f"({pipelined['total_vs_measured_ratio']:.3f}x measured P), "
            f"implied {pipelined['implied_tflops']:.3f} TFLOP/s on GEMM MACs"
        )


if __name__ == "__main__":
    main()
