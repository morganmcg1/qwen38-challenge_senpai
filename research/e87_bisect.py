#!/usr/bin/env python3
"""E87 r2 rung 1: derive the arm-C partition with balanced bisecting 2-means.

Arm C as measured ships a capacity-balanced k-means partition inside the
proposal head. Option B keeps the declared head untouched and DERIVES the
partition at load, so the partition function must be cheap, deterministic and
implementable with the same MLX ops the Swift runtime already uses.

This writes a cluster table in exactly the format `research/e87_screen.py`
reads, so the screen runs unchanged:

    /opt/homebrew/bin/python3 research/e87_bisect.py --rule bisect
    /opt/homebrew/bin/python3 research/e87_screen.py screen \\
        --rules bisect,plain --rpc 8 --p 0.25 --out research/e87-screen-b.json

Algorithm, stated so a second implementation can reproduce it bit for bit:

1. Rows are the dequantized affine-4 g64 compact `lm_head` rows, `[98336, 5120]`
   bfloat16, in compact row order. This is the same input the shipped k-means
   used, so the screen compares partition functions and nothing else.
2. A node holds a contiguous span of the current row order and a leaf target
   `L`. The root is the whole table with `L = 12292`.
3. A node with `L > 1` splits into leaf targets `ceil(L/2)` and `floor(L/2)`,
   so the left child takes `8*ceil(L/2)` rows. Every leaf therefore holds
   exactly 8 rows and no node needs padding.
4. Initial centres are the furthest-point pair: `a = argmax ||x - mean||^2`,
   then `b = argmax ||x - x_a||^2`. There is no RNG anywhere.
5. Each iteration scores `d_i = (||x_i - c_1||^2 - ||x_i - c_0||^2)`, keeps the
   `n_0` smallest for cluster 0 and recomputes both means. Balancing inside the
   loop is what makes an empty cluster impossible and removes the separate
   rebalance pass.
6. Nodes of one level that share a row count are processed as one batched
   `[m, s, 5120]` tensor, so the cost is a fixed number of full-table passes
   per level rather than one dispatch per node.

`--balance natural` screens the second partition function: the split point is
the natural 2-means boundary rounded to the nearest multiple of 8 and clamped
to `[8, s-8]`, so cluster sizes follow the data instead of halving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402
from e87_screen import CLUSTER_DIR, cluster_path  # noqa: E402

ROWS_PER_LEAF = 8


def _sq_to(xf: mx.array, xn: mx.array, centres: mx.array) -> mx.array:
    """`[m, s, c]` squared distance from every row to every centre.

    Formed as `||x||^2 - 2 x.c + ||c||^2` so no `[m, s, D]` difference tensor
    is ever materialised.
    """
    proj = mx.matmul(xf, centres.transpose(0, 2, 1))
    cn = mx.sum(centres ** 2, axis=2)
    return xn[:, :, None] - 2.0 * proj + cn[:, None, :]


def _furthest_pair(xf: mx.array, xn: mx.array) -> mx.array:
    """`[m, 2, D]` initial centres from the furthest-point rule."""
    m = xf.shape[0]
    rowsel = mx.arange(m)
    mu = mx.mean(xf, axis=1)[:, None, :]
    a = mx.argmax(_sq_to(xf, xn, mu)[:, :, 0], axis=1)
    ca = xf[rowsel, a][:, None, :]
    b = mx.argmax(_sq_to(xf, xn, ca)[:, :, 0], axis=1)
    cb = xf[rowsel, b][:, None, :]
    return mx.concatenate([ca, cb], axis=1)


def _split_batch(xf: mx.array, xn: mx.array, n0, iters: int,
                 balance: str) -> tuple[mx.array, mx.array]:
    """Balanced 2-means over `[m, s, D]`. Returns `(order, split)`.

    `order` is the within-node permutation that puts cluster 0 first. `split`
    is the per-node cluster-0 size, so the caller can slice each child span.
    """
    m, s = xf.shape[0], xf.shape[1]
    centres = _furthest_pair(xf, xn)
    split = mx.full((m,), n0, dtype=mx.int32) if isinstance(n0, int) else n0
    order = mx.broadcast_to(mx.arange(s)[None, :], (m, s))

    for _ in range(iters):
        d2 = _sq_to(xf, xn, centres)
        d = d2[:, :, 0] - d2[:, :, 1]
        order = mx.argsort(d, axis=1)
        rank = mx.argsort(order, axis=1)
        if balance == "natural":
            natural = mx.sum(d < 0, axis=1).astype(mx.int32)
            rounded = ((natural + 4) // ROWS_PER_LEAF) * ROWS_PER_LEAF
            split = mx.clip(rounded, ROWS_PER_LEAF, s - ROWS_PER_LEAF)
        left = (rank < split[:, None]).astype(mx.float32)
        weights = mx.stack([left, 1.0 - left], axis=1)
        counts = mx.maximum(mx.sum(weights, axis=2), 1.0)
        centres = mx.matmul(weights, xf) / counts[:, :, None]
        mx.eval(centres, order, split)

    return order, split


def bisect(rows: mx.array, leaves: int, iters: int, balance: str,
           verbose: bool = True) -> np.ndarray:
    """`assign[compact_row] = leaf id`, leaves in left-to-right tree order."""
    n = rows.shape[0]
    if n != leaves * ROWS_PER_LEAF:
        raise SystemExit(f"{n} rows do not divide into {leaves} leaves of {ROWS_PER_LEAF}")

    perm = mx.arange(n, dtype=mx.int32)
    work = rows
    # (start, size, leaf target). Kept sorted by start so the row order and the
    # node list stay consistent after every level.
    nodes = [(0, n, leaves)]
    level = 0

    while any(leaf > 1 for _, _, leaf in nodes):
        t0 = time.time()
        by_size: dict[int, list[int]] = {}
        for i, (_, size, leaf) in enumerate(nodes):
            if leaf > 1:
                by_size.setdefault(size, []).append(i)

        new_order = np.arange(n, dtype=np.int32)
        splits: dict[int, int] = {}
        for size, members in sorted(by_size.items()):
            starts = np.array([nodes[i][0] for i in members], dtype=np.int32)
            idx = mx.array(starts)[:, None] + mx.arange(size, dtype=mx.int32)[None, :]
            block = mx.take(work, idx.reshape(-1), axis=0).reshape(
                len(members), size, work.shape[1]).astype(mx.float32)
            block_norm = mx.sum(block * block, axis=2)
            if balance == "natural":
                order, split = _split_batch(
                    block, block_norm, size // 2, iters, balance)
            else:
                targets = mx.array(np.array(
                    [ROWS_PER_LEAF * ((nodes[i][2] + 1) // 2) for i in members],
                    dtype=np.int32))
                order, split = _split_batch(
                    block, block_norm, targets, iters, balance)
            mx.eval(order, split)
            order_np = np.asarray(order)
            split_np = np.asarray(split)
            for j, i in enumerate(members):
                start = nodes[i][0]
                new_order[start:start + size] = start + order_np[j]
                splits[i] = int(split_np[j])
            del block, block_norm, order, split, idx

        gather = mx.array(new_order)
        work = mx.take(work, gather, axis=0)
        perm = mx.take(perm, gather, axis=0)
        mx.eval(work, perm)

        nxt: list[tuple[int, int, int]] = []
        for i, (start, size, leaf) in enumerate(nodes):
            if leaf <= 1:
                nxt.append((start, size, leaf))
                continue
            cut = splits[i]
            left_leaf = cut // ROWS_PER_LEAF
            nxt.append((start, cut, left_leaf))
            nxt.append((start + cut, size - cut, leaf - left_leaf))
        nodes = nxt
        level += 1
        if verbose:
            sizes = [size for _, size, leaf in nodes if leaf > 1]
            print(f"  level {level:2d} nodes={len(nodes)} "
                  f"open={len(sizes)} size[{min(sizes) if sizes else 0},"
                  f"{max(sizes) if sizes else 0}] {time.time() - t0:.1f}s", flush=True)

    assign = np.empty(n, dtype=np.int32)
    perm_np = np.asarray(perm)
    for leaf_id, (start, size, leaf) in enumerate(nodes):
        if size != ROWS_PER_LEAF or leaf != 1:
            raise SystemExit(f"leaf {leaf_id} has {size} rows and target {leaf}")
        assign[perm_np[start:start + size]] = leaf_id
    return assign


def order_fnv1a64(order: np.ndarray) -> str:
    """The runtime's `qwen35ClusterOrderDigest`, over little-endian int32."""
    digest = 0xCBF29CE484222325
    for byte in order.astype("<i4").tobytes():
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="bisect")
    ap.add_argument("--balance", default="half", choices=["half", "natural"])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--from-dump",
                    help="read the row order the Swift runtime wrote instead of "
                         "deriving one, so the screen prices the exact partition "
                         "the candidate uses")
    ap.add_argument("--out")
    ap.add_argument("--report", default="research/e87-bisect.json")
    args = ap.parse_args()

    leaves = H.PADDED_COUNT // ROWS_PER_LEAF
    rows = H.dequantized(H.load_exact())
    mx.eval(rows)

    if args.from_dump:
        order = np.fromfile(args.from_dump, dtype="<i4")
        if order.shape != (H.PADDED_COUNT,):
            raise SystemExit(f"{args.from_dump}: {order.shape[0]} rows")
        if not np.array_equal(np.sort(order), np.arange(H.PADDED_COUNT)):
            raise SystemExit(f"{args.from_dump} is not a permutation")
        assign = np.empty(H.PADDED_COUNT, dtype=np.int32)
        assign[order] = np.arange(H.PADDED_COUNT, dtype=np.int32) // ROWS_PER_LEAF
        build_seconds = 0.0
    else:
        t0 = time.time()
        assign = bisect(rows, leaves, args.iters, args.balance)
        build_seconds = time.time() - t0

    counts = np.bincount(assign, minlength=leaves)
    if counts.min() != ROWS_PER_LEAF or counts.max() != ROWS_PER_LEAF:
        raise SystemExit(f"ragged leaves: {counts.min()}..{counts.max()}")

    canonical = np.argsort(assign, kind="stable").astype(np.int32)
    if args.from_dump and not np.array_equal(canonical, order):
        raise SystemExit(f"{args.from_dump} is not in canonical within-leaf order")
    members = mx.array(canonical)
    probe = mx.mean(
        mx.take(rows, members, axis=0).reshape(leaves, ROWS_PER_LEAF, H.HIDDEN).astype(
            mx.float32),
        axis=1)
    mx.eval(probe)

    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else cluster_path(args.rule, leaves)
    np.savez(path, assign=assign, probe=np.asarray(probe),
             rows_per_cluster=ROWS_PER_LEAF)

    digest = hashlib.sha256(assign.tobytes()).hexdigest()
    report = {
        "rule": args.rule,
        "balance": args.balance,
        "iters": args.iters,
        "from_dump": args.from_dump,
        "leaves": leaves,
        "rows_per_leaf": ROWS_PER_LEAF,
        "build_seconds": build_seconds,
        "assign_sha256": digest,
        "order_fnv1a64": order_fnv1a64(canonical),
        "probe_sha256": hashlib.sha256(np.asarray(probe).tobytes()).hexdigest(),
        "path": str(path),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
