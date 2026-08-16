#!/usr/bin/env python3
"""Stock-MLX control for the verify-width cost curve of `quantized_matmul`.

Runs the same sweep as `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` at the
same scored Qwen 3.8 27B shapes, but against whatever MLX the interpreter
imports. Comparing the two curves says whether the vendored checkout the scored
worker links carries a small-M quantized kernel the public release does not --
which decides whether a python-only measurement can stand in for the scored one.

Research-only: never packaged into a submission.
"""

import argparse
import json
import platform
import time

import mlx.core as mx

QBITS = 4
QGROUP = 64

SCORED_SHAPES = [
    ("linear_attn.in_proj_fused_qkvzba", 5120, 16480, 48),
    ("linear_attn.out_proj", 6144, 5120, 48),
    ("full_attn.qkv_proj_fused", 5120, 14336, 16),
    ("full_attn.o_proj", 6144, 5120, 16),
    ("mlp.gate_up_fused", 5120, 34816, 64),
    ("mlp.down", 17408, 5120, 64),
    ("head.lm_head", 5120, 248320, 1),
]

BOUNDARY_PROBES = [
    ("probe.k2048_n2048", 2048, 2048, list(range(1, 23)), 18),
    ("probe.k4096_n4096", 4096, 4096, list(range(1, 17)), 12),
    ("probe.k5120_n5120", 5120, 5120, list(range(1, 15)), 10),
]

FAST_PATH_PROBES = [
    ("fastprobe.k5120_n16480", 5120, 16480),
    ("fastprobe.k5184_n16480", 5184, 16480),
    ("fastprobe.k5632_n16480", 5632, 16480),
]


def weight_bytes(k, n):
    return n * (k // 2) + 2 * (n * (k // QGROUP) * 2)


def synthetic_weight(k, n):
    words = k // 8
    tile = mx.array(
        [(i * 2654435761) % (1 << 32) ^ 0x9E3779B9 for i in range(words)],
        dtype=mx.uint32,
    ).reshape(1, words)
    w = tile + mx.arange(n, dtype=mx.uint32).reshape(n, 1)

    groups = k // QGROUP
    jitter = mx.arange(n, dtype=mx.float32).reshape(n, 1) * 1e-6
    scale_tile = mx.array(
        [0.006 + 0.004 * ((i * 37) % 61) / 61 for i in range(groups)], dtype=mx.float32
    ).reshape(1, groups)
    bias_tile = mx.array(
        [-0.05 - 0.02 * ((i * 23) % 53) / 53 for i in range(groups)], dtype=mx.float32
    ).reshape(1, groups)
    scales = (scale_tile + jitter).astype(mx.bfloat16)
    biases = (bias_tile + jitter).astype(mx.bfloat16)
    mx.eval(w, scales, biases)
    return w, scales, biases


def synthetic_activations(m, k, salt):
    tile = mx.array(
        [((i * 131 + salt * 7919) % 251) / 251 - 0.5 for i in range(k)],
        dtype=mx.float32,
    ).reshape(1, k)
    jitter = mx.arange(m, dtype=mx.float32).reshape(m, 1) * 0.01
    return (tile + jitter).astype(mx.bfloat16)


def median(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples


def sweep(name, k, n, widths, calls_per_verify, reps, inner, extra=None):
    w, scales, biases = synthetic_weight(k, n)
    rows = []
    reference_row0 = None
    for m in widths:
        xs = [synthetic_activations(m, k, salt) for salt in range(inner)]
        mx.eval(xs)

        def run():
            mx.eval(
                [
                    mx.quantized_matmul(
                        x, w, scales, biases, transpose=True,
                        group_size=QGROUP, bits=QBITS,
                    )
                    for x in xs
                ]
            )

        samples = median(run, reps)
        row0 = mx.quantized_matmul(
            xs[0], w, scales, biases, transpose=True, group_size=QGROUP, bits=QBITS
        )[0].astype(mx.float32)
        mx.eval(row0)
        row0 = row0.tolist()
        if reference_row0 is None:
            reference_row0 = row0
        rows.append(
            {
                "m": m,
                "seconds_per_call": samples[len(samples) // 2] / inner,
                "seconds_per_call_min": samples[0] / inner,
                "seconds_per_call_max": samples[-1] / inner,
                "row0_bitwise_matches_m1": row0 == reference_row0,
                "row0_max_abs_delta_vs_m1": max(
                    abs(a - b) for a, b in zip(row0, reference_row0)
                ),
            }
        )
    del w, scales, biases
    out = {
        "name": name,
        "k": k,
        "n": n,
        "calls_per_verify": calls_per_verify,
        "weight_bytes": weight_bytes(k, n),
        "flops_per_row": 2 * k * n,
        "rows": rows,
    }
    out.update(extra or {})
    return out


def roofline(reps):
    stream_elements = 256 * 1024 * 1024
    stream = mx.zeros([stream_elements], dtype=mx.bfloat16)
    mx.eval(stream)
    stream_seconds = median(lambda: mx.eval(stream + 1), reps)[reps // 2]

    dim = 4096
    a = mx.zeros([dim, dim], dtype=mx.bfloat16)
    b = mx.zeros([dim, dim], dtype=mx.bfloat16)
    mx.eval(a, b)
    gemm_seconds = median(lambda: mx.eval(a @ b), reps)[reps // 2]
    return {
        "stream_bytes": 2 * 2 * stream_elements,
        "stream_seconds": stream_seconds,
        "peak_bandwidth_bytes_per_second": 2 * 2 * stream_elements / stream_seconds,
        "gemm_flops": 2 * dim**3,
        "gemm_seconds": gemm_seconds,
        "peak_flops_per_second": 2 * dim**3 / gemm_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--max-width", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    widths = list(range(1, args.max_width + 1))
    payload = {
        "source": "pip-mlx",
        "mlx_version": mx.__version__,
        "python": platform.python_version(),
        "reps": args.reps,
        "inner_calls_per_rep": args.inner,
        "widths": widths,
        "roofline": roofline(args.reps),
        "shapes": [
            sweep(name, k, n, widths, calls, args.reps, args.inner)
            for name, k, n, calls in SCORED_SHAPES
        ],
        "dispatch_boundary_probes": [
            sweep(
                name, k, n, probe_widths, 0, args.reps, args.inner,
                {"predicted_vector_limit": limit},
            )
            for name, k, n, probe_widths, limit in BOUNDARY_PROBES
        ],
        "fast_path_probes": [
            sweep(name, k, n, [1, 4, 8, 9], 0, args.reps, args.inner)
            for name, k, n in FAST_PATH_PROBES
        ],
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"QMV_COST_CURVE_OUT {args.out}")


if __name__ == "__main__":
    main()
