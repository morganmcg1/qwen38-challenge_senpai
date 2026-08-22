#!/usr/bin/env python3
"""The exact probe-fraction recall ladder, from one corpus pass.

    usage: research/e139_probe_margin.py [--batch N] [--limit N] [--out PATH]

WHY THIS REPLACES A SWEEP. E139 F3 asks for the recall-margin distribution
below `p = 0.10` and suggests sweeping `p` and watching a count approach 32.
Reading the shipped chain in `e133_screen.Screen.shipped()` shows the ladder
does not need a sweep, because the two failure modes are separable and only
one of them is active below the shipped point.

The chain probes `c = ceil(p * 12292)` derived-cluster leaves in affine-2
centroid order, gathers their `8 * c` rows, keeps the affine-2 top 32, and
returns the affine-4 argmax of those 32. The true answer is lost in exactly
two ways:

    leaf miss   crank_a2 >= c
                the leaf holding the true argmax row is not probed.
    cut miss    cum_gt[c - 1] >= 32
                at least 32 probed rows score strictly above the true argmax
                under affine-2, so it does not survive the top-32 cut.

`crank_a2` is the rank of that leaf under affine-2 centroid order and `tau`
is the affine-2 score of the true argmax row itself, so `cum_gt[c - 1]`
counts rows strictly above `tau` inside the probed prefix.

Both quantities are computed once per query and neither depends on `p`.
`crank_a2` does not move at all, and `cum_gt` is a cumulative sum whose
value at every rung is read from the same vector. So:

    recall(p) = P( crank_a2 < c(p)  AND  cum_gt[c(p) - 1] < 32 )

is an EXACT function of two per-query integers, evaluated at any `p` for
free. A sweep would re-measure one distribution once per rung and add
sampling noise for nothing.

WHICH TERM BINDS, AND WHY THE DIRECTION IN F3 IS INVERTED. Lowering `p`
lowers `c`, which makes the leaf-miss term WORSE and the cut-miss term
BETTER: fewer probed leaves means fewer competitors above `tau`. So the
count of rows above `tau` moves AWAY from 32 as `p` falls, and it is not
the quantity that breaks recall. The quantity that breaks recall is
`crank_a2`, and the knee is a QUANTILE OF ITS DISTRIBUTION: recall reaches
`1 - q` at exactly `c = 1 + Quantile_{1-q}(crank_a2)`. This report gives
both terms so the claim is checkable rather than asserted.

WORST DOMAIN. Every rate is reported pooled and per stratum, and the
worst-gating column takes the lowest recall over strata, matching the
`*_worst_gating` convention in `e133_screen.summarize()`.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e133_screen as S  # noqa: E402

# The rungs E139 F3 asks about, plus the two already measured live.
LADDER = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
          0.075, 0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25)

# Gross candidate-leg per-cent against the shipped p = 0.25 anchor, from the
# E136 rung-5a byte model, UNSCALED. FINDING 192 measured this model against
# two independent ranked receipts at p = 0.15 and got 0.973, so it is
# calibrated for byte removal at a constant retrieved set and for nothing
# else. A rung whose recall is below 1 does NOT hold the retrieved set
# constant, and its gross figure must not be read through that calibration.
BYTES_PER_ROW = 1600
MODEL_RATIO_AT_P015 = 0.9730


def gross_pct(clusters: int) -> float:
    ship = math.ceil(S.SHIPPED_PROBE_FRACTION * S.LEAVES)
    removed = (ship - clusters) * S.IX.ROWS_PER_LEAF * BYTES_PER_ROW
    return S.price(removed)["pct_head_share_7"]


def quantiles(v: np.ndarray, qs) -> dict:
    if v.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.quantile(v, q)) for q in qs}


def collect(batch: int, limit: int) -> dict:
    index = S.Index(pathlib.Path(S.IX.DEFAULT_OUT))
    screen = S.Screen(index)
    crank: dict[str, list[np.ndarray]] = {}
    cum: dict[str, list[np.ndarray]] = {}
    clusters = sorted({max(1, math.ceil(p * S.LEAVES)) for p in LADDER})
    idx = mx.array(np.array([c - 1 for c in clusters], dtype=np.int32))

    n = 0
    for _, stratum, x, _, _, _, _ in S.chunks(batch, limit):
        f = screen.front(x)
        # `cum_gt[i, j]` counts rows strictly above tau in the first j + 1
        # probed leaves, so one gather gives every rung at once.
        cum_gt = mx.cumsum(
            mx.take_along_axis(f["gt_leaf"], f["order_a2"], axis=1), axis=1)
        at_rungs = mx.take(cum_gt, idx, axis=1)
        mx.eval(at_rungs, f["crank_a2"])
        crank.setdefault(stratum, []).append(
            np.asarray(f["crank_a2"]).astype(np.int64))
        cum.setdefault(stratum, []).append(
            np.asarray(at_rungs).astype(np.int64))
        n += int(x.shape[0])
        if n % 4096 < batch:
            print(f"  {n} samples", flush=True)

    strata = {k: (np.concatenate(v), np.concatenate(cum[k], axis=0))
              for k, v in crank.items()}
    return {"clusters": clusters, "strata": strata, "samples": n}


def ladder(data: dict) -> list[dict]:
    clusters = data["clusters"]
    strata = data["strata"]
    real = {k: v for k, v in strata.items() if k in S.ALL_STRATA}
    pool_crank = np.concatenate([v[0] for v in strata.values()])
    pool_cum = np.concatenate([v[1] for v in strata.values()], axis=0)

    rows = []
    for p in LADDER:
        c = max(1, math.ceil(p * S.LEAVES))
        j = clusters.index(c)

        def rates(crank_v, cum_v):
            leaf_miss = crank_v >= c
            cut_miss = (~leaf_miss) & (cum_v[:, j] >= S.SHORTLIST)
            return {
                "n": int(crank_v.size),
                "recall": float(np.mean(~(leaf_miss | cut_miss))),
                "leaf_miss_rate": float(np.mean(leaf_miss)),
                "cut_miss_rate": float(np.mean(cut_miss)),
                # How many more leaves the worst query would have needed.
                # Negative means it was already covered with room to spare.
                "leaf_margin_min": int(c - 1 - crank_v.max()),
                # How many more rows could outrank the true argmax before the
                # top-32 cut would drop it. This is the count F3 asks to watch
                # approach 32, expressed as its distance from 32.
                "cut_margin_min": int(S.SHORTLIST - 1 - cum_v[:, j].max()),
            }

        pooled = rates(pool_crank, pool_cum)
        per = {k: rates(v[0], v[1]) for k, v in sorted(real.items())}
        worst = min(per.values(), key=lambda r: r["recall"]) if per else pooled
        rows.append({
            "p": p,
            "probes": c,
            "coarse_rows": c * S.IX.ROWS_PER_LEAF,
            "gross_pct_unscaled": gross_pct(c),
            "pooled": pooled,
            "recall_worst_gating": worst["recall"],
            "leaf_miss_rate_worst_gating": max(
                r["leaf_miss_rate"] for r in per.values()) if per else None,
            "cut_margin_min_worst_gating": min(
                r["cut_margin_min"] for r in per.values()) if per else None,
            "per_stratum": per,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="research/e139-probe-margin.json")
    args = ap.parse_args()

    data = collect(args.batch, args.limit)
    rows = ladder(data)
    strata = data["strata"]
    pool_crank = np.concatenate([v[0] for v in strata.values()])
    qs = (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)

    print(f"\n=== crank_a2, the leaf rank of the true affine-4 argmax "
          f"({pool_crank.size} samples) ===")
    print("  this distribution IS the ladder: recall reaches 1 - q at "
          "probes = 1 + quantile_{1-q}(crank_a2)")
    print("  pooled quantiles: " + "  ".join(
        f"q{q}={v:.0f}" for q, v in quantiles(pool_crank, qs).items()))
    for k in sorted(strata):
        v = strata[k][0]
        print(f"  {k:<14} n={v.size:<7} max={v.max():<7} "
              + "  ".join(f"q{q}={x:.0f}"
                          for q, x in quantiles(v, (0.99, 0.9999, 1.0)).items()))

    print("\n=== exact recall ladder, unscaled ===")
    head = ("      p  probes  coarseRows   gross%    recall  recallWG  "
            "leafMiss   cutMiss  leafMargin  cutMargin")
    print(head)
    for r in rows:
        pl = r["pooled"]
        print(f"  {r['p']:>5.4f}  {r['probes']:>6}  {r['coarse_rows']:>10}  "
              f"{r['gross_pct_unscaled']:>7.4f}  {pl['recall']:>8.6f}  "
              f"{r['recall_worst_gating']:>8.6f}  "
              f"{pl['leaf_miss_rate']:>8.6f}  {pl['cut_miss_rate']:>8.6f}  "
              f"{pl['leaf_margin_min']:>10}  {pl['cut_margin_min']:>9}")

    intact = [r for r in rows if r["recall_worst_gating"] >= 1.0]
    knee = min(intact, key=lambda r: r["probes"]) if intact else None
    print("\n=== knee ===")
    if knee is None:
        print("  no rung on this ladder holds worst-gating recall 1.0")
    else:
        print(f"  lowest rung with worst-gating recall 1.000000: "
              f"p={knee['p']} probes={knee['probes']} "
              f"gross={knee['gross_pct_unscaled']:+.4f} % unscaled")
        print(f"  exact break point: probes = {int(pool_crank.max()) + 1} "
              f"= p {(int(pool_crank.max()) + 1) / S.LEAVES:.6f}, the first "
              f"probe count that still covers every observed argmax leaf")
    print(f"  byte model calibration at p=0.15 against two ranked receipts: "
          f"{MODEL_RATIO_AT_P015}, valid only for byte removal at a "
          f"CONSTANT retrieved set")

    out = {
        "samples": data["samples"],
        "leaves": S.LEAVES,
        "shortlist": S.SHORTLIST,
        "bytes_per_row": BYTES_PER_ROW,
        "model_ratio_at_p015": MODEL_RATIO_AT_P015,
        "crank_quantiles_pooled": quantiles(pool_crank, qs),
        "crank_max_pooled": int(pool_crank.max()),
        "crank_quantiles_by_stratum": {
            k: quantiles(v[0], qs) | {"max": int(v[0].max()),
                                      "n": int(v[0].size)}
            for k, v in sorted(strata.items())},
        "ladder": rows,
        "knee": knee,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
