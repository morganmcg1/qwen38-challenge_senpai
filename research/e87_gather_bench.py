#!/usr/bin/env python3
"""E87: does arm C's byte saving convert into time on this GPU?

The rung-1 screen prices arm C in BYTES. The price list turns bytes into
score on the assumption that the coarse stage is bandwidth bound, which is
true of the shipped dense scorer. Arm C does not merely read fewer bytes: it
reads them through a DIFFERENT kernel. The shipped stage is one dense
`quantized_matmul` over 98,336 rows; arm C is a small dense centroid pass
plus a `gather_qmm` over `C` gathered `[rowsPerCluster, 5120]` blocks.

`gather_qmm` at batch 1 with thousands of tiny matrices is exactly the shape
where a gather kernel can fail to reach bandwidth. If it does, arm C's
predicted gain is illusory and arm G is the winner by default. This costs a
few minutes and can save a large Swift implementation.

Read the caveat before you quote a number:

  This runs against the HOMEBREW mlx wheel, not the repository's vendored
  MLX. The vendored build carries the campaign's promoted hand-written
  `qmv_fast_singlerow_affine2` kernel for the 98,336-row single-row case,
  which the wheel does not have. So the DENSE column here is stock MLX and
  is, if anything, SLOWER than the shipped path. That biases this screen in
  arm C's favour, which is the safe direction: a gather path that loses here
  loses harder in the real build.

  The decisive quantity is therefore not the speedup. It is arm C's achieved
  bandwidth against the dense scorer's achieved bandwidth. A gather path that
  reads 6x fewer bytes at 6x worse bandwidth buys nothing.

Usage:
  research/e87_gather_bench.py [--rule spherical] [--rpc 8,16,32]
                               [--p 0.05,0.064,0.10,0.25] [--iters 200]
                               [--out research/e87-gather-bench.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402

CLUSTER_DIR = Path.home() / ".cache/mlxfast/qwen3.8-27b-mtp-v1/e87/clusters"


def sync() -> None:
    mx.eval(mx.array(0.0))


def timed(fn, iters: int, warmup: int = 20) -> float:
    """Median-of-3 mean seconds per call, so one stray interrupt cannot win."""
    for _ in range(warmup):
        mx.eval(fn())
    runs = []
    for _ in range(3):
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        runs.append((time.perf_counter() - t0) / iters)
    return float(np.median(runs))


def coarse_bytes(rows: int) -> int:
    """2-bit g64 packed weight plus bf16 scales and biases, per row."""
    return rows * (320 * 4 + 80 * 2 * 2)


def fit_fixed_cost(mb: np.ndarray, us: np.ndarray) -> tuple[float, float]:
    """Least-squares `(fixed us, marginal GB/s)` for a pass.

    In this regime a dispatch is not free, so a pass costs a fixed launch
    term plus a bandwidth term. The fixed term is what decides whether
    splitting one dense read into two smaller passes can pay at all.
    """
    a = np.vstack([mb, np.ones_like(mb)]).T
    slope, intercept = np.linalg.lstsq(a, us, rcond=None)[0]
    return float(intercept), float(1000.0 / slope)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="spherical")
    ap.add_argument("--rpc", default="8,16,32")
    ap.add_argument("--p", default="0.05,0.064,0.10,0.25")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out")
    args = ap.parse_args()

    coarse = H.load_coarse()
    w, s, b = coarse["weight"], coarse["scales"], coarse["biases"]
    assert w.shape == (H.PADDED_COUNT, 320), w.shape
    mx.eval(w, s, b)

    x = mx.random.normal((1, H.HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)

    dense_t = timed(
        lambda: mx.quantized_matmul(x, w, s, b, transpose=True,
                                    group_size=64, bits=2),
        args.iters)
    dense_bytes = coarse_bytes(H.PADDED_COUNT)
    print(f"dense 98336 rows   {dense_t * 1e6:8.1f} us   "
          f"{dense_bytes / dense_t / 1e9:7.1f} GB/s")

    # The scored path does not stop at the matmul: it also selects the top 32.
    # The shipped build fuses that into `qwen35DraftTop32`, which this wheel
    # does not have, so `argpartition` is an UPPER bound on the dense stage.
    def dense_pipe():
        sc = mx.quantized_matmul(x, w, s, b, transpose=True,
                                 group_size=64, bits=2)
        return mx.argpartition(-sc, 32, axis=-1)[:, :32]

    dense_pipe_t = timed(dense_pipe, args.iters)
    print(f"dense + top32      {dense_pipe_t * 1e6:8.1f} us   "
          f"(select costs {(dense_pipe_t - dense_t) * 1e6:.1f} us)")

    # Arm G is the same kernel over the same rows with half the scale and bias
    # traffic, so measure it rather than assuming the byte model holds.
    g = H.requantize(H.dequantized(coarse), 128, 2)
    gw, gs, gb = g["weight"], g["scales"], g["biases"]
    mx.eval(gw, gs, gb)

    def g128_pipe():
        sc = mx.quantized_matmul(x, gw, gs, gb, transpose=True,
                                 group_size=128, bits=2)
        return mx.argpartition(-sc, 32, axis=-1)[:, :32]

    g128_t = timed(
        lambda: mx.quantized_matmul(x, gw, gs, gb, transpose=True,
                                    group_size=128, bits=2),
        args.iters)
    g128_pipe_t = timed(g128_pipe, args.iters)
    g128_bytes = H.PADDED_COUNT * (320 * 4 + 40 * 2 * 2)
    stage_pct = 36.78 * 0.0815
    print(f"armG g128          {g128_t * 1e6:8.1f} us   "
          f"{g128_bytes / g128_t / 1e9:7.1f} GB/s   "
          f"bytes {g128_bytes / dense_bytes:6.1%}   "
          f"time {g128_pipe_t / dense_pipe_t:6.1%}   "
          f"byte-model +{stage_pct * (1 - g128_bytes / dense_bytes):.2f}%   "
          f"measured +{stage_pct * (1 - g128_pipe_t / dense_pipe_t):.2f}%")
    del g, gw, gs, gb

    results = {
        "dense": {"rows": H.PADDED_COUNT, "seconds": dense_t,
                  "bytes": dense_bytes,
                  "gbps": dense_bytes / dense_t / 1e9},
        "dense_pipeline_seconds": dense_pipe_t,
        "armG_g128": {
            "seconds": g128_t, "pipeline_seconds": g128_pipe_t,
            "bytes": g128_bytes,
            "byte_fraction_of_dense": g128_bytes / dense_bytes,
            "time_fraction_of_dense": g128_pipe_t / dense_pipe_t,
            "predicted_pct_byte_model": stage_pct * (1 - g128_bytes / dense_bytes),
            "predicted_pct_measured_time": stage_pct * (1 - g128_pipe_t / dense_pipe_t),
        },
        "cells": [],
    }

    ps = [float(v) for v in args.p.split(",")]
    for rpc in [int(v) for v in args.rpc.split(",")]:
        k = H.PADDED_COUNT // rpc
        path = CLUSTER_DIR / f"{args.rule}-k{k}.npz"
        if not path.exists():
            print(f"skip {path.name}: not built")
            continue
        assign = np.load(path)["assign"].astype(np.int32)
        # Permute rows so a cluster is one contiguous block, exactly as the
        # Swift implementation would ship it.
        order = mx.array(np.argsort(assign, kind="stable").astype(np.int32))
        wp = mx.take(w, order, axis=0).reshape(k, rpc, 320)
        sp = mx.take(s, order, axis=0).reshape(k, rpc, 80)
        bp = mx.take(b, order, axis=0).reshape(k, rpc, 80)
        mx.eval(wp, sp, bp)

        # The centroid table is 2-bit g64 over K rows, priced the same way.
        cent = H.requantize(mx.random.normal((k, H.HIDDEN)).astype(mx.bfloat16), 64, 2)
        cw, cs, cb = cent["weight"], cent["scales"], cent["biases"]
        mx.eval(cw, cs, cb)
        cent_t = timed(
            lambda: mx.quantized_matmul(x, cw, cs, cb, transpose=True,
                                        group_size=64, bits=2),
            args.iters)
        cent_bytes = coarse_bytes(k)
        print(f"  centroids K={k:<6} {cent_t * 1e6:8.1f} us   "
              f"{cent_bytes / cent_t / 1e9:7.1f} GB/s")

        xg = x.reshape(1, 1, H.HIDDEN)
        for p in ps:
            c = max(1, math.ceil(p * k))
            # Real probes are the top-C centroids, so the indices are sorted
            # by score, not by value. `sorted_indices` needs value order, so
            # sort them: the Swift path would do the same.
            idx = mx.array(np.sort(
                np.random.default_rng(0).choice(k, size=c, replace=False)
            ).astype(np.uint32))
            lhs = mx.zeros((c,), dtype=mx.uint32)
            mx.eval(idx, lhs)

            gather_t = timed(
                lambda: mx.gather_qmm(xg, wp, scales=sp, biases=bp,
                                      lhs_indices=lhs, rhs_indices=idx,
                                      transpose=True, group_size=64, bits=2,
                                      sorted_indices=True),
                args.iters)

            # The honest arm C stage: score centroids, pick the top C, gather
            # those blocks, pick the top 32 of what came back. Steps 2 and 4
            # are real dispatches that the byte model does not price at all.
            def armc_pipe():
                cscore = mx.quantized_matmul(x, cw, cs, cb, transpose=True,
                                             group_size=64, bits=2)
                top = mx.argpartition(-cscore, c, axis=-1)[0, :c].astype(mx.uint32)
                got = mx.gather_qmm(xg, wp, scales=sp, biases=bp,
                                    lhs_indices=lhs, rhs_indices=mx.sort(top),
                                    transpose=True, group_size=64, bits=2,
                                    sorted_indices=True)
                flat = got.reshape(1, c * rpc)
                return mx.argpartition(-flat, 32, axis=-1)[:, :32]

            pipe_t = timed(armc_pipe, args.iters)
            gather_bytes = coarse_bytes(c * rpc)
            total_t = pipe_t
            total_bytes = cent_bytes + gather_bytes
            byte_frac = total_bytes / dense_bytes
            time_frac = total_t / dense_pipe_t
            print(f"    rpc={rpc:<3} p={p:<6g} C={c:<6} "
                  f"gather {gather_t * 1e6:7.1f} us "
                  f"({gather_bytes / gather_t / 1e9:6.1f} GB/s)  "
                  f"pipe {total_t * 1e6:7.1f} us  "
                  f"bytes {byte_frac:6.1%}  time {time_frac:6.1%}  "
                  f"byte-model +{stage_pct * (1 - byte_frac):.2f}%  "
                  f"measured +{stage_pct * (1 - time_frac):.2f}%")
            results["cells"].append({
                "rule": args.rule, "rows_per_cluster": rpc, "k": k, "p": p,
                "clusters_probed": c, "rows_probed": c * rpc,
                "centroid_seconds": cent_t, "gather_seconds": gather_t,
                "pipeline_seconds": total_t,
                "centroid_bytes": cent_bytes, "gather_bytes": gather_bytes,
                "total_bytes": total_bytes,
                "gather_gbps": gather_bytes / gather_t / 1e9,
                "byte_fraction_of_dense": byte_frac,
                "time_fraction_of_dense": time_frac,
                "speedup": dense_pipe_t / total_t,
                "predicted_pct_byte_model": stage_pct * (1 - byte_frac),
                "predicted_pct_measured_time": stage_pct * (1 - time_frac),
            })
        del wp, sp, bp, cw, cs, cb

    cells = results["cells"]
    if cells:
        g_fixed, g_bw = fit_fixed_cost(
            np.array([c["gather_bytes"] / 1e6 for c in cells]),
            np.array([c["gather_seconds"] * 1e6 for c in cells]))
        seen = {c["k"]: c for c in cells}.values()
        c_fixed, c_bw = fit_fixed_cost(
            np.array([c["centroid_bytes"] / 1e6 for c in seen]),
            np.array([c["centroid_seconds"] * 1e6 for c in seen]))
        print(f"\ngather   fixed {g_fixed:6.1f} us   marginal {g_bw:6.0f} GB/s")
        print(f"centroid fixed {c_fixed:6.1f} us   marginal {c_bw:6.0f} GB/s")
        print(f"two-pass launch floor {g_fixed + c_fixed:.0f} us = "
              f"{(g_fixed + c_fixed) / (dense_pipe_t * 1e6):.1%} of the dense stage")
        results["fit"] = {
            "gather_fixed_us": g_fixed, "gather_marginal_gbps": g_bw,
            "centroid_fixed_us": c_fixed, "centroid_marginal_gbps": c_bw,
        }
        best = max(cells, key=lambda c: c["predicted_pct_measured_time"])
        print(f"best cell: rpc={best['rows_per_cluster']} p={best['p']:g} "
              f"-> byte model +{best['predicted_pct_byte_model']:.2f}%, "
              f"measured +{best['predicted_pct_measured_time']:.2f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
