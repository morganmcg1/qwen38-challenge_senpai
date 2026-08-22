#!/usr/bin/env python3
"""E133 rung 2: rebuild the LIVE derived cluster index offline.

WHY THIS FILE EXISTS AND research/e87_screen.py's `cluster` COMMAND DOES NOT
SERVE. E87 screened a shortlist arm against the DENSE coarse readout, and its
arm-C tables come from its own capacity-balanced k-means (`cmd_cluster`,
e87_screen.py:334-399). The scored path at 197e0550 is neither of those: it is
the E121 derived index built by `buildDerivedClusterIndex`
(Qwen35.swift:5523-5589) from a balanced BISECTING 2-means
(`qwen35BisectingPartition`, :4586-4645). C1 must be priced against what ships
today, so the baseline chain has to be the derived index, reproduced rule for
rule:

  rows        dequantize(compact affine-4 g64 lm_head)      [98_336, 5_120]
  permutation balanced bisecting 2-means, 8 rows per leaf, 8 iterations
  order       ascending compact ids inside each leaf
  centroids   float32 mean of the 8 exact rows, bf16, then affine-2 g64
  clusterPerm order, with the six padding rows folded back onto their source

No RNG anywhere: `qwen35ClusterFurthestPair` (:4607-4622) seeds each split from
the furthest-point rule, so this port is deterministic and comparable with the
runtime table.

Usage:
  research/e133_index.py build [--out PATH]
  research/e133_index.py check [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e133"))
DEFAULT_OUT = CACHE / "derived-index.npz"

ROWS_PER_LEAF = 8
ITERATIONS = 8
CENTROID_BITS = 2
PROBE_FRACTION = 0.25


def _squared_distance(xf: mx.array, xn: mx.array, centres: mx.array) -> mx.array:
    """`[m, s, c]`, formed as `||x||^2 - 2 x.c + ||c||^2`."""
    projection = mx.matmul(xf, mx.swapaxes(centres, 1, 2))
    centre_norm = mx.sum(centres * centres, axis=2)
    return xn[:, :, None] - 2.0 * projection + centre_norm[:, None, :]


def _furthest_pair(xf: mx.array, xn: mx.array) -> mx.array:
    nodes, span, hidden = xf.shape
    flat = xf.reshape(nodes * span, hidden)
    row_base = mx.arange(nodes, dtype=mx.int32) * span
    mean = mx.mean(xf, axis=1)[:, None, :]
    first = mx.argmax(_squared_distance(xf, xn, mean)[:, :, 0], axis=1).astype(mx.int32)
    centre_a = mx.take(flat, row_base + first, axis=0)[:, None, :]
    second = mx.argmax(_squared_distance(xf, xn, centre_a)[:, :, 0], axis=1).astype(mx.int32)
    centre_b = mx.take(flat, row_base + second, axis=0)[:, None, :]
    return mx.concatenate([centre_a, centre_b], axis=1)


def _balanced_split(xf: mx.array, xn: mx.array, split: mx.array, iterations: int) -> mx.array:
    nodes, span = xf.shape[0], xf.shape[1]
    centres = _furthest_pair(xf, xn)
    order = mx.broadcast_to(mx.arange(span, dtype=mx.int32)[None, :], (nodes, span))
    for _ in range(iterations):
        distance = _squared_distance(xf, xn, centres)
        order = mx.argsort(distance[:, :, 0] - distance[:, :, 1], axis=1).astype(mx.int32)
        rank = mx.argsort(order, axis=1).astype(mx.int32)
        left = (rank < split[:, None]).astype(mx.float32)
        membership = mx.stack([left, 1.0 - left], axis=1)
        counts = mx.maximum(mx.sum(membership, axis=2), 1.0)
        centres = mx.matmul(membership, xf) / counts[:, :, None]
        mx.eval(centres, order)
    return order


def bisecting_partition(rows: mx.array, rows_per_leaf: int, iterations: int) -> np.ndarray:
    count, hidden = rows.shape
    work = rows
    permutation = mx.arange(count, dtype=mx.int32)
    nodes = [(0, count, count // rows_per_leaf)]  # (start, span, leaves)
    level = 0

    while any(node[2] > 1 for node in nodes):
        t0 = time.time()
        by_span: dict[int, list[int]] = {}
        for index, node in enumerate(nodes):
            if node[2] > 1:
                by_span.setdefault(node[1], []).append(index)
        next_order = np.arange(count, dtype=np.int32)
        cuts: dict[int, int] = {}
        for span in sorted(by_span):
            members = by_span[span]
            starts = np.array([nodes[i][0] for i in members], dtype=np.int32)
            gather = (starts[:, None] + np.arange(span, dtype=np.int32)[None, :]).reshape(-1)
            block = mx.take(work, mx.array(gather), axis=0)
            block = block.reshape(len(members), span, hidden).astype(mx.float32)
            block_norm = mx.sum(block * block, axis=2)
            targets = np.array(
                [rows_per_leaf * ((nodes[i][2] + 1) // 2) for i in members], dtype=np.int32)
            order = _balanced_split(block, block_norm, mx.array(targets), iterations)
            mx.eval(order)
            order_host = np.asarray(order)
            del block, block_norm
            for member, index in enumerate(members):
                start = nodes[index][0]
                next_order[start: start + span] = start + order_host[member]
                cuts[index] = int(targets[member])
        reorder = mx.array(next_order)
        work = mx.take(work, reorder, axis=0)
        permutation = mx.take(permutation, reorder, axis=0)
        mx.eval(work, permutation)

        nxt = []
        for index, node in enumerate(nodes):
            start, span, leaves = node
            if leaves <= 1:
                nxt.append(node)
                continue
            cut = cuts[index]
            nxt.append((start, cut, cut // rows_per_leaf))
            nxt.append((start + cut, span - cut, leaves - cut // rows_per_leaf))
        nodes = nxt
        print(f"  level {level:2d}: {len(nodes)} nodes, "
              f"max leaves {max(n[2] for n in nodes)}, {time.time() - t0:.1f}s", flush=True)
        level += 1

    return np.asarray(permutation)


def cmd_build(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exact = H.load_exact()
    coarse = H.load_coarse()
    rows = H.dequantized(exact)
    mx.eval(rows)
    print(f"rows {rows.shape} {rows.dtype}", flush=True)

    t0 = time.time()
    permutation = bisecting_partition(rows, ROWS_PER_LEAF, ITERATIONS)
    leaves = H.PADDED_COUNT // ROWS_PER_LEAF
    order = np.sort(permutation.reshape(leaves, ROWS_PER_LEAF), axis=1).reshape(-1)
    print(f"partition in {time.time() - t0:.1f}s; leaves {leaves}", flush=True)

    order_mx = mx.array(order.astype(np.int32))
    centroids = (
        mx.take(rows, order_mx, axis=0)
        .reshape(leaves, ROWS_PER_LEAF, H.HIDDEN)
        .astype(mx.float32)
        .mean(axis=1)
        .astype(mx.bfloat16)
    )
    cw, cs, cb = mx.quantize(centroids, group_size=64, bits=CENTROID_BITS)
    mx.eval(cw, cs, cb)
    cluster_perm = np.where(order >= H.REAL_COUNT, order - H.REAL_COUNT, order)

    # The runtime gathers the coarse affine-2 rows into leaf blocks. Offline the
    # gather is implicit: `order` indexes the unpermuted coarse table, so only
    # the order and the centroids need storing.
    np.savez(
        out,
        order=order.astype(np.int32),
        cluster_perm=cluster_perm.astype(np.int32),
        centroid_weight=np.asarray(cw),
        centroid_scales=np.asarray(cs.view(mx.uint16)),
        centroid_biases=np.asarray(cb.view(mx.uint16)),
        rows_per_leaf=np.int32(ROWS_PER_LEAF),
        iterations=np.int32(ITERATIONS),
        centroid_bits=np.int32(CENTROID_BITS),
    )
    print(f"wrote {out}")
    _report(out, rows, coarse)


def _report(out: Path, rows: mx.array | None = None, coarse=None) -> None:
    blob = np.load(out)
    order = blob["order"]
    leaves = order.size // ROWS_PER_LEAF
    assert order.size == H.PADDED_COUNT, order.size
    assert np.array_equal(np.sort(order), np.arange(H.PADDED_COUNT)), "order is not a bijection"
    grouped = order.reshape(leaves, ROWS_PER_LEAF)
    assert np.all(np.diff(grouped, axis=1) > 0), "leaf rows are not ascending"
    probes = max(1, int(np.ceil(PROBE_FRACTION * leaves)))
    print(f"order bijection ok, leaves {leaves}, probes {probes}, "
          f"rows scored {probes * ROWS_PER_LEAF}")
    if rows is None:
        return
    # How tight are the leaves? A useful index puts near rows together, so the
    # mean within-leaf cosine should sit far above the global mean cosine.
    idx = mx.array(grouped[:512].reshape(-1).astype(np.int32))
    block = mx.take(rows, idx, axis=0).reshape(512, ROWS_PER_LEAF, H.HIDDEN).astype(mx.float32)
    unit = block / mx.maximum(mx.linalg.norm(block, axis=2, keepdims=True), 1e-6)
    gram = mx.matmul(unit, mx.swapaxes(unit, 1, 2))
    off = (mx.sum(gram, axis=(1, 2)) - ROWS_PER_LEAF) / (ROWS_PER_LEAF * (ROWS_PER_LEAF - 1))
    rand = mx.take(rows, mx.array(np.random.default_rng(0).choice(
        H.REAL_COUNT, 4096, replace=False).astype(np.int32)), axis=0).astype(mx.float32)
    rand = rand / mx.maximum(mx.linalg.norm(rand, axis=1, keepdims=True), 1e-6)
    rand_gram = mx.matmul(rand, rand.T)
    rand_mean = (mx.sum(rand_gram) - 4096) / (4096 * 4095)
    print(f"within-leaf mean cosine {float(mx.mean(off)):.4f} "
          f"vs random-pair mean cosine {float(rand_mean):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("check")
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.set_defaults(func=lambda a: _report(Path(a.out)))
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
