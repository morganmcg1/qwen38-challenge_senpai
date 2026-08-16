#!/usr/bin/env python3
"""Component-level floor for one scored MTP decode round at depth d.

Decode analogue of `research/prefill_floor.py`. For every op a round executes
at its exact scored shape, width and quantization it reports two floors:

  roofline_floor        max(bytes / BW_eff, flops / FLOPS_eff) summed over the round
  achieved_kernel_floor sum of the isolated *measured* kernel walls at the same shapes

`roofline_floor -> achieved_kernel_floor` is the kernel-efficiency gap (what a
better-dispatched quantized matmul could recover). `achieved_kernel_floor ->
measured_block_seconds` is the scheduling/host gap that remains once every
kernel is charged at the speed this machine actually runs it.

BW_eff and FLOPS_eff are measured here rather than assumed, so the roofline is
anchored to the same host and thermal state as the achieved numbers.

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
VOCAB = 248320
QBITS = 4
QGROUP = 64
BYTES_PER_WEIGHT = (QGROUP * QBITS + 32) / QGROUP / 8.0  # 4-bit + bf16 scale/bias per group

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

KEY_DIM = LINEAR_KEY_HEAD_DIM * LINEAR_KEY_HEADS
VALUE_DIM = LINEAR_VALUE_HEAD_DIM * LINEAR_VALUE_HEADS
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
FUSED_IN_PROJ_OUT = CONV_DIM + VALUE_DIM + LINEAR_VALUE_HEADS * 2
ATTN_Q_OUT = ATTN_HEADS * HEAD_DIM * 2  # output-gated
ATTN_KV_OUT = KV_HEADS * HEAD_DIM

# Compact draft vocabulary the head reads instead of the full lm_head.
COMPACT_DRAFT_ROWS = 98_336

# Gated DeltaNet recurrent state, float32, snapshotted once per round.
RECURRENT_STATE_BYTES = LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM * LINEAR_KEY_HEAD_DIM * 4

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


INNER = 16
# Distinct weight copies per component. Consecutive calls rotate through them so
# a small weight cannot sit in cache across the back-to-back batch; the scored
# round reads a different layer's weights on every one of its calls.
WEIGHT_COPIES = 2


def _median(samples):
    samples.sort()
    return samples[len(samples) // 2]


def _release():
    for name in ("clear_cache", "reset_peak_memory"):
        fn = getattr(mx, name, None)
        if fn is not None:
            fn()


def timeit_isolated(fn, reps, warmup=3):
    """One submit + one sync per call.

    At decode widths most kernels are far smaller than a command-buffer round
    trip, so this over-charges them. It is reported only as a per-dispatch
    overhead diagnostic, never summed into a floor.
    """
    for _ in range(warmup):
        mx.eval(fn(0))
    mx.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(0))
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return _median(samples)


def timeit_amortized(fn, reps, inner=INNER, warmup=2):
    """`inner` independent calls inside one submit, divided by `inner`.

    `fn(i)` must build a graph that differs by `i` so MLX cannot common-
    subexpression the copies away. This is the floor an ideal scheduler
    reaches: every kernel is charged, no host stall or sync is charged, and
    independent work may overlap exactly as it would under a good submission.
    """
    for _ in range(warmup):
        mx.eval([fn(i) for i in range(inner)])
    mx.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval([fn(i) for i in range(inner)])
        mx.synchronize()
        samples.append((time.perf_counter() - t0) / inner)
    return _median(samples)


def make_qlinear(out_features, in_features):
    w = mx.random.normal([out_features, in_features]).astype(mx.bfloat16) * 0.02
    return mx.quantize(w, group_size=QGROUP, bits=QBITS)


def qbank(out_features, in_features, copies=WEIGHT_COPIES):
    return [make_qlinear(out_features, in_features) for _ in range(copies)]


def inputs(shape, dtype=mx.bfloat16, n=INNER, uniform=False):
    gen = mx.random.uniform if uniform else mx.random.normal
    return [gen(shape=shape).astype(dtype) for _ in range(n)]


def qmm_fn(xs, bank):
    def run(i):
        w, scales, biases = bank[i % len(bank)]
        return mx.quantized_matmul(
            xs[i % len(xs)], w, scales, biases, transpose=True, group_size=QGROUP, bits=QBITS
        )

    return run


def qweight_bytes(out_features, in_features):
    return out_features * in_features * BYTES_PER_WEIGHT


def measure_machine_balance(reps):
    """Measure BW_eff from a pure weight-streaming qmv and FLOPS_eff from a wide GEMM.

    Both probes use the amortized path so the anchors describe the machine, not
    the command-buffer round trip.
    """
    lm = qbank(VOCAB, H, copies=1)
    t_bw = timeit_amortized(qmm_fn(inputs([1, 1, H]), lm), reps)
    bw_eff = qweight_bytes(VOCAB, H) / t_bw
    del lm
    _release()

    wide = qbank(INTERMEDIATE, H, copies=1)
    t_fl = timeit_amortized(qmm_fn(inputs([1, 512, H]), wide), reps)
    flops_eff = (2.0 * 512 * H * INTERMEDIATE) / t_fl
    del wide
    _release()

    return {
        "bw_eff_bytes_per_s": bw_eff,
        "bw_probe": "quantized_matmul lm_head M=1 K=5120 N=248320",
        "bw_probe_seconds": t_bw,
        "flops_eff": flops_eff,
        "flops_probe": "quantized_matmul M=512 K=5120 N=17408",
        "flops_probe_seconds": t_fl,
        "machine_balance_flop_per_byte": flops_eff / bw_eff,
    }


def build_components(width, kv_len, depth, reps):
    """Every op one round at `depth` executes, at its true scored width."""
    comps = []
    M = width

    def record(name, count, fn, byts, flops, inner=INNER, note=None):
        sec = timeit_amortized(fn, reps, inner=inner)
        iso = timeit_isolated(fn, reps)
        entry = {
            "component": name,
            "calls_per_round": count,
            "per_call_seconds": sec,
            "per_call_seconds_isolated": iso,
            "dispatch_overhead_seconds": iso - sec,
            "measured_total_seconds": sec * count,
            "bytes_per_call": byts,
            "flops_per_call": flops,
        }
        if note:
            entry["note"] = note
        comps.append(entry)
        _release()

    xM = inputs([1, M, H])
    x1 = inputs([1, 1, H])

    # ---- target verify pass, once per round at width M -----------------
    in_proj = qbank(FUSED_IN_PROJ_OUT, H)
    record(
        "verify:lin_attn:in_proj_fused_qkvzba",
        N_LINEAR_LAYERS,
        qmm_fn(xM, in_proj),
        qweight_bytes(FUSED_IN_PROJ_OUT, H),
        2 * M * H * FUSED_IN_PROJ_OUT,
    )

    conv_w = [
        mx.random.normal(shape=[CONV_DIM, CONV_KERNEL, 1]).astype(mx.bfloat16)
        for _ in range(WEIGHT_COPIES)
    ]
    conv_x = inputs([1, M + CONV_KERNEL - 1, CONV_DIM])
    record(
        "verify:lin_attn:conv1d_depthwise_k4",
        N_LINEAR_LAYERS,
        lambda i: mx.conv1d(conv_x[i % INNER], conv_w[i % WEIGHT_COPIES], groups=CONV_DIM),
        CONV_DIM * CONV_KERNEL * 2 + (M + CONV_KERNEL - 1) * CONV_DIM * 2,
        2 * M * CONV_DIM * CONV_KERNEL,
    )

    kernel = mx.fast.metal_kernel(
        name="gated_delta_step",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=GATED_DELTA_SOURCE,
    )
    qq = inputs([1, M, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM])
    kk = inputs([1, M, LINEAR_KEY_HEADS, LINEAR_KEY_HEAD_DIM])
    vv = inputs([1, M, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM])
    gg = inputs([1, M, LINEAR_VALUE_HEADS], dtype=mx.float32, uniform=True)
    bb = inputs([1, M, LINEAR_VALUE_HEADS], dtype=mx.float32, uniform=True)
    state_shape = [1, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM, LINEAR_KEY_HEAD_DIM]
    st = [mx.zeros(state_shape, dtype=mx.float32) for _ in range(WEIGHT_COPIES)]
    tarr = mx.array(M)

    def delta(i):
        j = i % INNER
        return kernel(
            inputs=[qq[j], kk[j], vv[j], gg[j], bb[j], st[i % WEIGHT_COPIES], tarr],
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
            output_shapes=[[1, M, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM], state_shape],
            output_dtypes=[mx.bfloat16, mx.float32],
        )

    record(
        "verify:lin_attn:gated_delta_kernel",
        N_LINEAR_LAYERS,
        delta,
        2 * RECURRENT_STATE_BYTES,
        M * LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM * LINEAR_KEY_HEAD_DIM * 4 * 2,
    )

    lin_out = qbank(H, VALUE_DIM)
    xv = inputs([1, M, VALUE_DIM])
    record(
        "verify:lin_attn:out_proj",
        N_LINEAR_LAYERS,
        qmm_fn(xv, lin_out),
        qweight_bytes(H, VALUE_DIM),
        2 * M * VALUE_DIM * H,
    )

    qp = qbank(ATTN_Q_OUT, H)
    kp = qbank(ATTN_KV_OUT, H)
    vp = qbank(ATTN_KV_OUT, H)
    op = qbank(H, ATTN_HEADS * HEAD_DIM)

    def proj3(xs, banks):
        def run(i):
            x = xs[i % INNER]
            j = i % WEIGHT_COPIES
            return tuple(
                mx.quantized_matmul(x, *b[j], transpose=True, group_size=QGROUP, bits=QBITS)
                for b in banks
            )

        return run

    qkv_out = ATTN_Q_OUT + 2 * ATTN_KV_OUT
    record(
        "verify:full_attn:qkv_proj",
        N_FULL_LAYERS,
        proj3(xM, (qp, kp, vp)),
        qweight_bytes(qkv_out, H),
        2 * M * H * qkv_out,
    )

    qa = inputs([1, ATTN_HEADS, M, HEAD_DIM])
    ka = inputs([1, KV_HEADS, kv_len, HEAD_DIM], n=WEIGHT_COPIES)
    va = inputs([1, KV_HEADS, kv_len, HEAD_DIM], n=WEIGHT_COPIES)
    record(
        "verify:full_attn:sdpa",
        N_FULL_LAYERS,
        lambda i: mx.fast.scaled_dot_product_attention(
            qa[i % INNER],
            ka[i % WEIGHT_COPIES],
            va[i % WEIGHT_COPIES],
            scale=1.0 / 16.0,
            mask=None,
        ),
        2 * KV_HEADS * kv_len * HEAD_DIM * 2,
        2 * 2 * ATTN_HEADS * M * kv_len * HEAD_DIM,
    )

    xo = inputs([1, M, ATTN_HEADS * HEAD_DIM])
    record(
        "verify:full_attn:o_proj",
        N_FULL_LAYERS,
        qmm_fn(xo, op),
        qweight_bytes(H, ATTN_HEADS * HEAD_DIM),
        2 * M * ATTN_HEADS * HEAD_DIM * H,
    )

    gate = qbank(INTERMEDIATE, H)
    up = qbank(INTERMEDIATE, H)
    down = qbank(H, INTERMEDIATE)

    def mlp(xs):
        def run(i):
            x = xs[i % INNER]
            j = i % WEIGHT_COPIES
            a = mx.quantized_matmul(x, *gate[j], transpose=True, group_size=QGROUP, bits=QBITS)
            b = mx.quantized_matmul(x, *up[j], transpose=True, group_size=QGROUP, bits=QBITS)
            h = a * mx.sigmoid(a) * b
            return mx.quantized_matmul(h, *down[j], transpose=True, group_size=QGROUP, bits=QBITS)

        return run

    mlp_bytes = 2 * qweight_bytes(INTERMEDIATE, H) + qweight_bytes(H, INTERMEDIATE)
    record(
        "verify:mlp:gate_up_down",
        N_LINEAR_LAYERS + N_FULL_LAYERS,
        mlp(xM),
        mlp_bytes,
        3 * 2 * M * H * INTERMEDIATE,
    )

    lm = qbank(VOCAB, H, copies=1)
    record(
        "verify:lm_head_full_vocab",
        1,
        qmm_fn(xM, lm),
        qweight_bytes(VOCAB, H),
        2 * M * H * VOCAB,
        note="untied lm_head; embed_tokens is gathered, not streamed",
    )

    # ---- recurrent state snapshot, once per round ----------------------
    snap_shape = [N_LINEAR_LAYERS, LINEAR_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM, LINEAR_KEY_HEAD_DIM]
    snap_src = [mx.zeros(snap_shape, dtype=mx.float32) for _ in range(WEIGHT_COPIES)]
    record(
        "state:recurrent_snapshot_48_layers",
        1,
        lambda i: mx.contiguous(snap_src[i % WEIGHT_COPIES] + float(i)),
        2 * N_LINEAR_LAYERS * RECURRENT_STATE_BYTES,
        N_LINEAR_LAYERS * RECURRENT_STATE_BYTES // 4,
        inner=4,
    )

    # ---- proposal head, `depth` times per round at width 1 -------------
    fc = qbank(H, 2 * H)
    x2h = inputs([1, 1, 2 * H])
    record("head:fc_concat_proj", depth, qmm_fn(x2h, fc), qweight_bytes(H, 2 * H), 2 * 2 * H * H)

    record(
        "head:attn_qkv_proj",
        depth,
        proj3(x1, (qp, kp, vp)),
        qweight_bytes(qkv_out, H),
        2 * 1 * H * qkv_out,
    )

    qh = inputs([1, ATTN_HEADS, 1, HEAD_DIM])
    record(
        "head:attn_sdpa",
        depth,
        lambda i: mx.fast.scaled_dot_product_attention(
            qh[i % INNER],
            ka[i % WEIGHT_COPIES],
            va[i % WEIGHT_COPIES],
            scale=1.0 / 16.0,
            mask=None,
        ),
        2 * KV_HEADS * kv_len * HEAD_DIM * 2,
        2 * 2 * ATTN_HEADS * 1 * kv_len * HEAD_DIM,
    )

    xo1 = inputs([1, 1, ATTN_HEADS * HEAD_DIM])
    record(
        "head:attn_o_proj",
        depth,
        qmm_fn(xo1, op),
        qweight_bytes(H, ATTN_HEADS * HEAD_DIM),
        2 * 1 * ATTN_HEADS * HEAD_DIM * H,
    )

    record("head:mlp_gate_up_down", depth, mlp(x1), mlp_bytes, 3 * 2 * 1 * H * INTERMEDIATE)

    draft_lm = qbank(COMPACT_DRAFT_ROWS, H, copies=1)
    record(
        "head:draft_lm_head_compact",
        depth,
        qmm_fn(x1, draft_lm),
        qweight_bytes(COMPACT_DRAFT_ROWS, H),
        2 * 1 * H * COMPACT_DRAFT_ROWS,
        note="single weight bank: 283 MB exceeds any plausible reuse window",
    )

    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--depth", type=int, default=8, help="draft count d; verify width is d+1")
    ap.add_argument("--kv-len", type=int, default=768)
    ap.add_argument(
        "--measured-block-seconds",
        type=float,
        required=True,
        help="parent-clock seconds for one round at this depth",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mx.random.seed(0)
    width = args.depth + 1

    balance = measure_machine_balance(args.reps)
    bw = balance["bw_eff_bytes_per_s"]
    fl = balance["flops_eff"]

    comps = build_components(width, args.kv_len, args.depth, args.reps)
    for c in comps:
        bw_sec = c["bytes_per_call"] / bw
        fl_sec = c["flops_per_call"] / fl
        c["roofline_per_call_seconds"] = max(bw_sec, fl_sec)
        c["roofline_bound"] = "memory" if bw_sec >= fl_sec else "compute"
        c["roofline_total_seconds"] = c["roofline_per_call_seconds"] * c["calls_per_round"]
        c["kernel_efficiency"] = (
            c["roofline_per_call_seconds"] / c["per_call_seconds"]
            if c["per_call_seconds"] > 0
            else 0.0
        )
        c["achieved_bytes_per_s"] = c["bytes_per_call"] / c["per_call_seconds"]
        c["isolated_total_seconds"] = c["per_call_seconds_isolated"] * c["calls_per_round"]

    roofline = sum(c["roofline_total_seconds"] for c in comps)
    achieved = sum(c["measured_total_seconds"] for c in comps)
    isolated = sum(c["isolated_total_seconds"] for c in comps)
    measured = args.measured_block_seconds

    out = {
        "host": {"chip": "Apple M4 Pro", "mlx_python": mx.__version__, "mlx_swift": "0.32.0"},
        "depth": args.depth,
        "verify_width": width,
        "kv_len": args.kv_len,
        "bytes_per_weight": BYTES_PER_WEIGHT,
        "machine": balance,
        "components": sorted(comps, key=lambda c: -c["measured_total_seconds"]),
        "roofline_floor_seconds": roofline,
        "achieved_kernel_floor_seconds": achieved,
        "measured_block_seconds": measured,
        "kernel_efficiency_gap_seconds": achieved - roofline,
        "scheduling_gap_seconds": measured - achieved,
        "achieved_over_roofline": achieved / roofline if roofline else 0.0,
        "measured_over_achieved": measured / achieved if achieved else 0.0,
        "measured_over_roofline": measured / roofline if roofline else 0.0,
        "isolated_kernel_sum_seconds": isolated,
        "dispatch_overhead_seconds": isolated - achieved,
        "isolated_note": (
            "one-submit-per-call timing; not a floor. Difference from "
            "achieved_kernel_floor_seconds is per-dispatch command-buffer overhead that an "
            "ideal scheduler overlaps away."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"BW_eff  = {bw/1e9:.1f} GB/s   FLOPS_eff = {fl/1e12:.3f} TFLOP/s   "
        f"balance = {balance['machine_balance_flop_per_byte']:.1f} FLOP/byte\n"
    )
    print(
        f"{'component':42s} {'x':>3s} {'roof_ms':>9s} {'meas_ms':>9s} "
        f"{'eff':>6s} {'bound':>7s} {'iso_ms':>9s}"
    )
    for c in out["components"]:
        print(
            f"{c['component']:42s} {c['calls_per_round']:3d} "
            f"{c['roofline_total_seconds']*1e3:9.3f} {c['measured_total_seconds']*1e3:9.3f} "
            f"{c['kernel_efficiency']*100:5.1f}% {c['roofline_bound']:>7s} "
            f"{c['isolated_total_seconds']*1e3:9.3f}"
        )
    print(f"\nroofline_floor        = {roofline*1e3:9.2f} ms")
    print(f"achieved_kernel_floor = {achieved*1e3:9.2f} ms  ({out['achieved_over_roofline']:.2f}x roofline)")
    print(f"measured_block        = {measured*1e3:9.2f} ms  ({out['measured_over_achieved']:.2f}x achieved)")
    print(f"\nkernel-efficiency gap = {(achieved-roofline)*1e3:9.2f} ms")
    print(f"scheduling gap        = {(measured-achieved)*1e3:9.2f} ms")
    print(
        f"\n[diag] isolated one-submit-per-call sum = {isolated*1e3:9.2f} ms; "
        f"per-dispatch overhead = {(isolated-achieved)*1e3:.2f} ms (not part of the floor)"
    )


if __name__ == "__main__":
    main()
