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

    results = {
        "dense": {"rows": H.PADDED_COUNT, "seconds": dense_t,
                  "bytes": dense_bytes,
                  "gbps": dense_bytes / dense_t / 1e9},
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
            gather_bytes = coarse_bytes(c * rpc)
            total_t = cent_t + gather_t
            total_bytes = cent_bytes + gather_bytes
            print(f"    rpc={rpc:<3} p={p:<6g} C={c:<6} "
                  f"gather {gather_t * 1e6:8.1f} us "
                  f"({gather_bytes / gather_t / 1e9:6.1f} GB/s)   "
                  f"stage1 total {total_t * 1e6:8.1f} us   "
                  f"bytes {total_bytes / dense_bytes:6.1%}   "
                  f"time {total_t / dense_t:6.1%}   "
                  f"speedup {dense_t / total_t:5.2f}x")
            results["cells"].append({
                "rule": args.rule, "rows_per_cluster": rpc, "k": k, "p": p,
                "clusters_probed": c, "rows_probed": c * rpc,
                "centroid_seconds": cent_t, "gather_seconds": gather_t,
                "total_seconds": total_t,
                "centroid_bytes": cent_bytes, "gather_bytes": gather_bytes,
                "total_bytes": total_bytes,
                "gather_gbps": gather_bytes / gather_t / 1e9,
                "byte_fraction_of_dense": total_bytes / dense_bytes,
                "time_fraction_of_dense": total_t / dense_t,
                "speedup": dense_t / total_t,
            })
        del wp, sp, bp, cw, cs, cb

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
