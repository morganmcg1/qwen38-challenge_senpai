#!/usr/bin/env python3
"""E87 rungs 0 and 1: screen candidate coarse shortlist scorers offline.

One question, asked of every candidate scorer `S`:

  over a corpus of real head hidden states `x`, what fraction `m` of samples
  has `argmax_exact(x)` OUTSIDE `top32_S(x)`?

`argmax_exact` is the argmax of the affine-4 g64 compact `lm_head` over all
98,330 reachable rows. `research/e87_head.py` proves the shipped affine-2 g64
coarse head is exactly `mx.quantize(dequantize(that), 64, 2)`, so both stages
come from one source and the arms below are variations of one rule.

Why a rank and not a sort. `top32_S(x)` contains row `r` exactly when fewer
than 32 rows outscore `r` under `S`. Counting `sum(S > S[r])` answers that in
one pass and never sorts 98,330 columns.

The reported quantity is NOT the absolute `m`. The shipped shortlist already
misses at some rate and the baseline is already priced with it, so every arm
carries the paired discordance `(worse - better) / n` against the shipped
scorer, per work and per domain. The decision runs on the WORST domain.

Arm C's primary grid is `rowsPerCluster in {8, 16, 32}`, the three exact
zero-padding factorisations of 98,336 (`K in {12292, 6146, 3073}`), which
bracket FlashHead's published 16 rows per cluster. Centroids are priced AND
scored at 2-bit g64 so the price column and the `m` column describe one table.

Sub-commands:
  validate   positive control. Offline `argmax_exact` must equal the proposal
             the runtime returned, and the shipped g64 scorer must report
             m == 0. A deliberately damaged scorer must report a large m.
  cluster    build the balanced cluster tables arm C needs.
  screen     the `m(K, p)` tables for arm G and arm C against the price list.

  /opt/homebrew/bin/python3 research/e87_screen.py validate --limit 4000
  /opt/homebrew/bin/python3 research/e87_screen.py cluster --rpc 8,16,32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402

DUMP_DIR = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e87/screen/hidden"))
CLUSTER_DIR = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e87/clusters"))
MANIFEST = Path(__file__).resolve().parent / "e87-corpus-manifest.json"
CANDIDATES = 32
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
# corpus


def domains() -> dict[str, str]:
    return {s["name"]: s["domain"] for s in json.load(MANIFEST.open())["seeds"]}


def seed_files(limit_seeds: int = 0) -> list[tuple[str, Path, Path]]:
    """One entry per dump shard, named by the seed that produced it.

    The instrument shards by process, so a seed can appear more than once.
    Shards are grouped by seed so a `--seeds` limit still means whole seeds.
    """
    by_seed: dict[str, list[tuple[str, Path, Path]]] = {}
    for x_path in sorted(DUMP_DIR.glob("*.pid*.x.f32")):
        stem = x_path.name[: -len(".x.f32")]
        tok_path = DUMP_DIR / f"{stem}.tok.i32"
        if not tok_path.exists():
            continue
        seed = stem.rsplit(".pid", 1)[0]
        by_seed.setdefault(seed, []).append((seed, x_path, tok_path))
    seeds = sorted(by_seed)
    if limit_seeds:
        seeds = seeds[:limit_seeds]
    return [shard for seed in seeds for shard in by_seed[seed]]


def load_seed(x_path: Path, tok_path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, H.HIDDEN)
    tok = np.memmap(tok_path, dtype=np.int32, mode="r")
    n = min(x.shape[0], tok.shape[0])
    return x[:n], tok[:n]


def chunks(limit_samples: int, limit_seeds: int, batch: int):
    """Yield `(seed, domain, x_batch_bf16, proposal_batch)`."""
    dom = domains()
    total = 0
    for name, x_path, tok_path in seed_files(limit_seeds):
        x, tok = load_seed(x_path, tok_path)
        for start in range(0, x.shape[0], batch):
            if limit_samples and total >= limit_samples:
                return
            stop = start + batch
            if limit_samples:
                stop = min(stop, start + (limit_samples - total))
            xb = mx.array(np.ascontiguousarray(x[start:stop])).astype(mx.bfloat16)
            yield name, dom.get(name, "?"), xb, np.asarray(tok[start:stop])
            total += xb.shape[0]


# --------------------------------------------------------------------------
# scorers


def rank_of(score: mx.array, row: mx.array) -> mx.array:
    """How many rows strictly outscore `row`. `< 32` means the shortlist keeps it."""
    at = mx.take_along_axis(score, row[:, None], axis=1)
    return mx.sum(score > at, axis=1)


def masked_rank_reference(coarse: mx.array, row: mx.array, row_probed: mx.array) -> mx.array:
    """Arm C's rank, written the slow, obvious way: mask then rank.

    Kept as the reference the fast cluster-aggregate path is checked against.
    """
    masked = mx.where(row_probed, coarse, mx.array(-3.0e38, mx.float32))
    return rank_of(masked, row)


# --------------------------------------------------------------------------
# clustering


def _cluster_scores(work_chunk: mx.array, centres: mx.array,
                    centre_norm: mx.array | None) -> mx.array:
    """Assignment score. Euclidean argmin reduces to argmax of
    `r . c - 0.5 ||c||^2`; the spherical rule uses the plain dot."""
    score = mx.matmul(work_chunk, centres.T).astype(mx.float32)
    return score if centre_norm is None else score - centre_norm


def _preference_lists(work: mx.array, centres: mx.array, centre_norm, width: int,
                      chunk: int) -> tuple[np.ndarray, np.ndarray]:
    n = work.shape[0]
    pref = np.empty((n, width), dtype=np.int32)
    margin = np.empty(n, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        score = _cluster_scores(work[a:b], centres, centre_norm)
        idx = mx.argpartition(-score, kth=width - 1, axis=1)[:, :width]
        top = mx.take_along_axis(score, idx, axis=1)
        order = mx.argsort(-top, axis=1)
        idx = mx.take_along_axis(idx, order, axis=1)
        top = mx.take_along_axis(top, order, axis=1)
        mx.eval(idx, top)
        pref[a:b] = np.asarray(idx)
        margin[a:b] = np.asarray(top[:, 0] - top[:, 1])
        del score, idx, top, order
    return pref, margin


def _greedy(pref: np.ndarray, order: np.ndarray, counts: np.ndarray,
            assign: np.ndarray, rpc: int, rows: np.ndarray | None = None) -> int:
    """Place each listed row in the best cluster of `pref` that still has room.

    `pref` and `order` share one index space. `rows` maps that space onto
    global row numbers when the caller is repairing a subset.
    """
    placed = 0
    for i in order:
        row = i if rows is None else rows[i]
        if assign[row] >= 0:
            continue
        for c in pref[i]:
            if counts[c] < rpc:
                assign[row] = c
                counts[c] += 1
                placed += 1
                break
    return placed


def balanced_assign(work: mx.array, centres: mx.array, centre_norm, k: int, rpc: int,
                    width: int, chunk: int, rng) -> tuple[np.ndarray, np.ndarray, int]:
    """Exactly `rpc` rows per cluster, with no reliance on a truncated list.

    Pass 1 is a confidence-ordered greedy over each row's best `width`
    clusters. With zero padding the total capacity equals the row count, so
    the last rows can find every listed cluster full; pass 2 repairs those
    rows against the clusters that are ACTUALLY free, which keeps a spilled
    row near its own centroid instead of scattering it at random.
    """
    n = work.shape[0]
    pref, margin = _preference_lists(work, centres, centre_norm, min(width, k), chunk)
    counts = np.zeros(k, dtype=np.int32)
    assign = np.full(n, -1, dtype=np.int32)
    _greedy(pref, np.argsort(-margin), counts, assign, rpc)
    del pref, margin

    spill = int(np.sum(assign < 0))
    left = np.flatnonzero(assign < 0)
    while left.size:
        free_mask = mx.array(counts < rpc)
        placed = 0
        repair_width = min(k, 256)
        for a in range(0, left.size, chunk):
            rows = left[a: a + chunk]
            score = _cluster_scores(work[mx.array(rows)], centres, centre_norm)
            score = mx.where(free_mask[None, :], score, mx.array(-3.0e38, mx.float32))
            idx = mx.argpartition(-score, kth=repair_width - 1, axis=1)[:, :repair_width]
            top = mx.take_along_axis(score, idx, axis=1)
            idx = np.asarray(mx.take_along_axis(idx, mx.argsort(-top, axis=1), axis=1))
            placed += _greedy(idx, np.arange(rows.size), counts, assign, rpc, rows)
            del score, idx, top
        left = np.flatnonzero(assign < 0)
        if placed == 0 and left.size:
            slots = np.repeat(np.arange(k), rpc - counts)
            rng.shuffle(slots)
            assign[left] = slots[: left.size]
            counts = np.bincount(assign, minlength=k).astype(np.int32)
            break
    return assign, counts, spill


def kmeans(rows: mx.array, k: int, rows_per_cluster: int, spherical: bool,
           iters: int, width: int, chunk: int, seed: int = 0) -> tuple[np.ndarray, mx.array]:
    """Capacity-balanced k-means. Returns `(assignment, probe_centroids)`.

    Every cluster holds exactly `rows_per_cluster` slots so the run-time gather
    can use one fixed row block per cluster.

    `probe_centroids` are what the run time scores `x` against. For the
    spherical rule the direction and the norm are carried separately, exactly
    as the arm describes: the centroid is the mean unit row rescaled by the
    cluster's mean row norm, so `x . centroid` still estimates `x . row`.
    """
    n = rows.shape[0]
    norms = mx.linalg.norm(rows.astype(mx.float32), axis=1)
    unit = (rows.astype(mx.float32) / mx.maximum(norms[:, None], 1e-6)).astype(mx.bfloat16)
    work = unit if spherical else rows

    def group_mean(values: mx.array, assign_mx: mx.array) -> mx.array:
        shape = (k,) + values.shape[1:]
        total = mx.zeros(shape, mx.float32).at[assign_mx].add(values.astype(mx.float32))
        count = mx.zeros((k,), mx.float32).at[assign_mx].add(mx.ones((n,), mx.float32))
        count = mx.maximum(count, 1.0)
        return total / (count[:, None] if values.ndim > 1 else count)

    rng = np.random.default_rng(seed)
    centres = work[mx.array(rng.choice(n, size=k, replace=False))]
    assign = np.zeros(n, dtype=np.int32)

    for it in range(iters):
        t0 = time.time()
        centre_norm = None
        if not spherical:
            centre_norm = 0.5 * mx.sum(centres.astype(mx.float32) ** 2, axis=1)[None, :]
        assign, counts, spill = balanced_assign(
            work, centres, centre_norm, k, rows_per_cluster, width, chunk, rng)
        centres = group_mean(work, mx.array(assign)).astype(mx.bfloat16)
        mx.eval(centres)
        print(f"    iter {it:2d} spill={spill} ({spill / n:.2%}) "
              f"counts[{counts.min()},{counts.max()}] {time.time() - t0:.1f}s", flush=True)

    if spherical:
        mean_norm = group_mean(norms, mx.array(assign))
        probe = (centres.astype(mx.float32) * mean_norm[:, None]).astype(mx.bfloat16)
    else:
        probe = centres
    return assign, probe


def cluster_path(rule: str, k: int) -> Path:
    return CLUSTER_DIR / f"{rule}-k{k}.npz"


def rows_per_cluster_for(k: int) -> int:
    """Rows in every cluster block, as a multiple of 8.

    `98,336 = 2^5 x 7 x 439` divides exactly by 8, 16 and 32, so the primary
    grid `K in {12292, 6146, 3073}` has NO padded rows at all. The coarse
    trend cells still need a pad, and the multiple of 8 keeps `gather_qmm` on
    the fast path (`quantized.cpp:992`, `N % 8 == 0`).
    """
    if H.PADDED_COUNT % k == 0 and (H.PADDED_COUNT // k) % 8 == 0:
        return H.PADDED_COUNT // k
    need = math.ceil(H.PADDED_COUNT / k)
    return need + (-need % 8)


def cluster_counts(args) -> list[int]:
    ks = [int(v) for v in args.k.split(",") if v]
    for rpc in [int(v) for v in args.rpc.split(",") if v]:
        if H.PADDED_COUNT % rpc:
            raise SystemExit(f"rows-per-cluster {rpc} does not divide {H.PADDED_COUNT}")
        ks.append(H.PADDED_COUNT // rpc)
    return sorted(set(ks))


def cmd_cluster(args) -> None:
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    exact = H.load_exact()
    rows = H.dequantized(exact)
    mx.eval(rows)
    for rule in args.rules.split(","):
        for k in cluster_counts(args):
            path = cluster_path(rule, k)
            if path.exists() and not args.force:
                print(f"skip {path}")
                continue
            rpc = rows_per_cluster_for(k)
            print(f"=== {rule} K={k} rows_per_cluster={rpc} "
                  f"padded={k * rpc} ({k * rpc / H.PADDED_COUNT - 1:.2%} pad) ===", flush=True)
            t0 = time.time()
            assign, probe = kmeans(rows, k, rpc, rule == "spherical",
                                   args.iters, args.width, args.chunk)
            np.savez(
                path,
                assign=assign,
                probe=np.asarray(probe.astype(mx.float32)),
                rows_per_cluster=rpc,
            )
            print(f"    wrote {path} in {time.time() - t0:.1f}s", flush=True)


# --------------------------------------------------------------------------
# byte and score model


def coarse_row_bytes(group_size: int) -> int:
    groups = H.HIDDEN // group_size
    return 320 * 4 + groups * 2 * 2


def centroid_bytes(k: int, bits: int) -> int:
    if bits == 16:
        return k * H.HIDDEN * 2
    return k * (H.HIDDEN * bits // 8 + (H.HIDDEN // 64) * 2 * 2)


def price(stage2_bytes: int, centroid_read: int = 0) -> dict:
    read_bytes = stage2_bytes + centroid_read
    removed = H.COARSE_STAGE_BYTES - read_bytes
    head_pct = 100.0 * removed / H.DECLARED_HEAD_TENSOR_BYTES
    gain = head_pct * H.BYTES_TO_SCORE_PCT
    return {
        "centroid_bytes": centroid_read,
        "stage2_bytes": stage2_bytes,
        "read_bytes": read_bytes,
        "removed_bytes": removed,
        "head_pct": head_pct,
        "score_gain_pct": gain,
        "breakeven_m": gain / H.MISS_TO_SCORE_PCT,
    }


# --------------------------------------------------------------------------
# commands


def cmd_validate(args) -> None:
    exact = H.load_exact()
    coarse = H.load_coarse()
    rows = H.dequantized(exact)
    # A scorer that cannot fail is not an instrument: zero all but the first
    # 1/16 of every row so the damaged index keeps only a fraction of the
    # signal and must lose the exact argmax often.
    broken_rows = mx.concatenate(
        [rows[:, : H.HIDDEN // 16], mx.zeros_like(rows[:, H.HIDDEN // 16:])], axis=1)
    damaged = H.requantize(broken_rows, 64, 2)
    mx.eval(damaged["weight"], damaged["scales"], damaged["biases"])
    del broken_rows, rows

    n = tok_ok = miss64 = miss_damaged = 0
    for _, _, x, proposal in chunks(args.limit, args.seeds, args.batch):
        ex = H.scores(exact, x)
        r = mx.argmax(ex, axis=1)
        vocab = np.asarray(H.compact_to_vocab(r))
        tok_ok += int(np.sum(vocab == proposal))
        miss64 += int(mx.sum(rank_of(H.scores(coarse, x), r) >= CANDIDATES).item())
        miss_damaged += int(mx.sum(rank_of(H.scores(damaged, x), r) >= CANDIDATES).item())
        n += x.shape[0]
        if n % (args.batch * 20) == 0:
            print(f"  {n} samples", flush=True)

    p, lo, hi = wilson(miss64, n)
    pd, _, _ = wilson(miss_damaged, n)
    report = {
        "samples": n,
        "proposal_match": tok_ok / n,
        "m_shipped_g64": {"misses": miss64, "p": p, "lo": lo, "hi": hi},
        "m_damaged_control": {"misses": miss_damaged, "p": pd},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


def cmd_screen(args) -> None:
    exact = H.load_exact()
    coarse = H.load_coarse()
    rows = H.dequantized(exact)
    g128 = H.requantize(rows, 128, 2)
    mx.eval(g128["weight"], g128["scales"], g128["biases"])
    del rows

    ks = cluster_counts(args)
    ps = [float(v) for v in args.p.split(",")]
    rules = args.rules.split(",")

    tables = {}
    for rule in rules:
        for k in ks:
            path = cluster_path(rule, k)
            if not path.exists():
                continue
            blob = np.load(path)
            probe = mx.array(blob["probe"]).astype(mx.bfloat16)
            assign = blob["assign"].astype(np.int32)
            rpc = int(blob["rows_per_cluster"])
            counts = np.bincount(assign, minlength=k)
            if counts.min() != rpc or counts.max() != rpc:
                raise SystemExit(f"{path.name}: clusters are not exactly {rpc} rows")
            entry = {
                "assign": mx.array(assign),
                "row_cluster": mx.array(assign[: H.REAL_COUNT]),
                # Members of each cluster, grouped. `assign` is exactly
                # balanced, so a stable argsort reshapes to (K, rpc) and turns
                # the per-cluster reduction into one gather plus one sum.
                "members": mx.array(
                    np.argsort(assign, kind="stable").astype(np.int32).reshape(k, rpc)),
                # Padded rows are unreachable, so a cluster's real width can be
                # below rpc and the probed-row count must use the real width.
                "n_real": mx.array(
                    np.bincount(assign[: H.REAL_COUNT], minlength=k).astype(np.int32)),
                "rpc": rpc,
                "probe": probe,
                "quant": None,
            }
            # Price and quality must describe the same table: the centroid
            # column in the price list is 2-bit g64, so the probe that decides
            # `m` is scored from a 2-bit g64 table too.
            if args.centroid_bits != 16:
                entry["quant"] = H.requantize(probe, 64, args.centroid_bits)
                mx.eval(*(entry["quant"][t] for t in ("weight", "scales", "biases")))
            tables[(rule, k)] = entry
    if not tables:
        raise SystemExit("no cluster tables; run `cluster` first")

    counters: dict[str, dict] = {}
    groups: dict[str, dict] = {}

    def bump(key: str, labels: list[str], misses: mx.array, base: mx.array,
             total: int) -> None:
        """Accumulate absolute misses AND the paired discordance vs shipped.

        The score cost of an arm is not its absolute miss rate. The shipped
        g64 shortlist already misses at some rate, and that rate is priced
        into the baseline. Only the NET change matters, so count the two
        discordant cells directly (`worse` = arm misses where shipped hits,
        `better` = the reverse) and price `(worse - better) / n`.
        """
        hit = int(mx.sum(misses).item())
        worse = int(mx.sum(misses & ~base).item())
        better = int(mx.sum(base & ~misses).item())
        for target in [counters.setdefault(key, {})] + [
            groups.setdefault(key, {}).setdefault(label, {}) for label in labels
        ]:
            for field, value in (("miss", hit), ("worse", worse),
                                 ("better", better), ("n", total)):
                target[field] = target.get(field, 0) + value

    pad = H.PADDED_COUNT - H.REAL_COUNT
    n = 0
    checked_fast_path = False
    t0 = time.time()
    for name, domain, x, _ in chunks(args.limit, args.seeds, args.batch):
        b = x.shape[0]
        labels = [f"domain:{domain}", f"work:{name.rsplit('-', 1)[0]}"]
        ex = H.scores(exact, x)
        r = mx.argmax(ex, axis=1)
        coarse_scores = H.scores(coarse, x)
        # `gt` is the only full-width comparison arm C needs: masking a score
        # cannot change which OTHER rows outscore the exact argmax, and a
        # masked row is never strictly greater than another masked row.
        at = mx.take_along_axis(coarse_scores, r[:, None], axis=1)
        gt = coarse_scores > at
        base_miss = mx.sum(gt, axis=1) >= CANDIDATES
        mx.eval(base_miss, gt)
        bump("shipped-g64", labels, base_miss, base_miss, b)
        bump("armG-g128", labels, rank_of(H.scores(g128, x), r) >= CANDIDATES, base_miss, b)
        gt_pad = mx.concatenate([gt, mx.zeros((b, pad), gt.dtype)], axis=1)

        for (rule, k), table in tables.items():
            if table["quant"] is None:
                probe_scores = mx.matmul(x, table["probe"].T).astype(mx.float32)
            else:
                probe_scores = H.scores_all(table["quant"], x)
            cluster_of_r = mx.take(table["row_cluster"], r)
            # Per-cluster counts of outscoring rows and of real rows.
            s_gt = mx.sum(mx.take(gt_pad, table["members"], axis=1), axis=2)
            order_c = mx.argsort(-probe_scores, axis=1)
            cum_gt = mx.cumsum(mx.take_along_axis(s_gt, order_c, axis=1), axis=1)
            cum_n = mx.cumsum(
                mx.take_along_axis(
                    mx.broadcast_to(table["n_real"][None, :], (b, k)), order_c, axis=1),
                axis=1)
            # Probe rank of the exact argmax's own cluster.
            crank_r = mx.argmax(
                (order_c == cluster_of_r[:, None]).astype(mx.int8), axis=1)
            mx.eval(cum_gt, cum_n, crank_r)
            for p in ps:
                c = max(1, int(round(p * k)))
                # A row survives only if its cluster is probed AND it still
                # ranks in the top 32 of the probed rows under the SAME coarse
                # g64 score the run time uses. Restricting the pool can only
                # improve a row's rank, so arm C can also beat the shipped
                # shortlist on samples the global top-32 loses. When the exact
                # argmax's own cluster is not probed, every probed real row
                # outscores its masked sentinel, so the rank is that count.
                probed_r = crank_r < c
                rank = mx.where(probed_r, cum_gt[:, c - 1], cum_n[:, c - 1])
                miss = rank >= CANDIDATES
                mx.eval(miss)
                if not checked_fast_path:
                    row_probed = mx.take(
                        mx.argsort(order_c, axis=1) < c, table["row_cluster"], axis=1)
                    want = masked_rank_reference(coarse_scores, r, row_probed) >= CANDIDATES
                    if not bool(mx.all(want == miss).item()):
                        raise SystemExit(
                            f"fast cluster path disagrees with the masked reference "
                            f"on {rule} K={k} p={p:g}")
                    del row_probed, want
                bump(f"armC-{rule}-K{k}-p{p:g}", labels, miss, base_miss, b)
                del probed_r, rank, miss
            del probe_scores, order_c, cum_gt, cum_n, crank_r, s_gt
        checked_fast_path = True
        n += b
        if n % (args.batch * 20) == 0:
            print(f"  {n} samples  {time.time() - t0:.0f}s", flush=True)

    out = {"samples": n, "cells": []}
    for key, c in counters.items():
        p_hat, lo, hi = wilson(c["miss"], c["n"])
        net = (c["worse"] - c["better"]) / c["n"] if c["n"] else float("nan")
        # Discordant-pair interval: the paired count that moves the score is
        # `worse - better` out of `worse + better` trials.
        disc = c["worse"] + c["better"]
        _, dlo, dhi = wilson(c["worse"], disc)
        net_lo = (2 * dlo - 1) * disc / c["n"] if disc and c["n"] else 0.0
        net_hi = (2 * dhi - 1) * disc / c["n"] if disc and c["n"] else 0.0
        if key == "shipped-g64":
            model = price(H.COARSE_STAGE_BYTES)
        elif key == "armG-g128":
            model = price(H.PADDED_COUNT * coarse_row_bytes(128))
        else:
            _, rule, ktag, ptag = key.split("-")
            k = int(ktag[1:])
            pf = float(ptag[1:])
            rpc = tables[(rule, k)]["rpc"]
            cc = max(1, int(round(pf * k)))
            model = price(cc * rpc * coarse_row_bytes(64),
                          centroid_bytes(k, args.centroid_bits))
        by_group = {
            label: {"n": v["n"], "m": v["miss"] / v["n"],
                    "net": (v["worse"] - v["better"]) / v["n"]}
            for label, v in groups[key].items()
        }
        by_domain = {label[7:]: v for label, v in by_group.items()
                     if label.startswith("domain:")}
        by_work = {label[5:]: v for label, v in by_group.items()
                   if label.startswith("work:")}
        # FlashHead's own containment collapses on ONE of three corpora, so
        # the decision runs on the worst domain, not the pooled mean.
        worst_domain = max(by_domain.values(), key=lambda v: v["net"])["net"] \
            if by_domain else net
        worst_work = max(by_work.values(), key=lambda v: v["net"])["net"] \
            if by_work else net
        cell = {
            "arm": key,
            "misses": c["miss"],
            "n": c["n"],
            "m": p_hat,
            "m_lo": lo,
            "m_hi": hi,
            "worse_than_shipped": c["worse"],
            "better_than_shipped": c["better"],
            "net_miss_vs_shipped": net,
            "net_miss_lo": net_lo,
            "net_miss_hi": net_hi,
            "net_miss_worst_domain": worst_domain,
            "net_miss_worst_work": worst_work,
            **model,
            "predicted_score_pct": model["score_gain_pct"] - H.MISS_TO_SCORE_PCT * net,
            "predicted_worst_pct": model["score_gain_pct"] - H.MISS_TO_SCORE_PCT * net_hi,
            "predicted_worst_domain_pct":
                model["score_gain_pct"] - H.MISS_TO_SCORE_PCT * worst_domain,
            "by_domain": by_domain,
            "by_work": by_work,
        }
        out["cells"].append(cell)
    out["cells"].sort(key=lambda c: -c["predicted_worst_domain_pct"])

    print(f"{'arm':30s} {'cent MB':>8s} {'st2 MB':>8s} {'m':>9s} {'net':>10s} "
          f"{'wrstDom':>10s} {'gain%':>7s} {'break m':>9s} {'pred%':>7s} {'wdom%':>7s}")
    for cell in out["cells"]:
        print(f"{cell['arm']:30s} {cell['centroid_bytes'] / 1e6:8.2f} "
              f"{cell['stage2_bytes'] / 1e6:8.2f} {cell['m']:9.5f} "
              f"{cell['net_miss_vs_shipped']:10.2e} {cell['net_miss_worst_domain']:10.2e} "
              f"{cell['score_gain_pct']:7.3f} {cell['breakeven_m']:9.2e} "
              f"{cell['predicted_score_pct']:7.3f} "
              f"{cell['predicted_worst_domain_pct']:7.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--seeds", type=int, default=0)
    v.add_argument("--batch", type=int, default=256)
    v.add_argument("--out")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("cluster")
    c.add_argument("--rules", default="spherical,plain")
    # Primary grid: the three exact, zero-padding factorisations of 98,336
    # that bracket FlashHead's 16 rows per cluster.
    c.add_argument("--rpc", default="8,16,32")
    c.add_argument("--k", default="")
    c.add_argument("--iters", type=int, default=8)
    c.add_argument("--width", type=int, default=128)
    c.add_argument("--chunk", type=int, default=8192)
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_cluster)

    s = sub.add_parser("screen")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--seeds", type=int, default=0)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--rules", default="spherical,plain")
    s.add_argument("--rpc", default="8,16,32")
    s.add_argument("--k", default="")
    s.add_argument("--p", default="0.05,0.064,0.10,0.15,0.25,0.50")
    s.add_argument("--centroid-bits", type=int, default=2)
    s.add_argument("--out")
    s.set_defaults(func=cmd_screen)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
