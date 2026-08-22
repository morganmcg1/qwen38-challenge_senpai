#!/usr/bin/env python3
"""E125 stage 0 - test H6 (roofline distance) on already-measured E123 cells.

Zero GPU cost. For every (shape, width) cell of the E123 session this reads the
per-cell achieved read bandwidth and the per-cell effect of a real deletion, and
reports the correlation between them.

H6 predicts a negative correlation: a cell close to its bandwidth roofline
cannot pay out an instruction deletion, so the gain shrinks as achieved
bandwidth rises.

  python3 research/e125_bandwidth_scan.py --rate research/e123-artifacts/rate.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

DRAM_PEAK_GBS = 273.0


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fisher_ci(r, n, z=1.96):
    if not math.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    zr = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = zr - z * se, zr + z * se
    return (math.tanh(lo), math.tanh(hi))


def load_cells(path, arms, warmup_blocks=1):
    """Return {(shape, m): {"gbs": float, "pct": {arm: percent of a_base}}}."""
    doc = json.load(open(path))
    by_cell = defaultdict(list)
    for rec in doc["measurements"]:
        if rec.get("kind") != "timing":
            continue
        if rec["block"] < warmup_blocks:
            continue
        by_cell[(rec["shape"], rec["m"])].append(rec)

    cells = {}
    for key, recs in sorted(by_cell.items()):
        base = statistics.median(r["seconds"]["a_base"] for r in recs)
        read_bytes = recs[0]["read_bytes"]
        entry = [r["gpu_temp_entry_c"] for r in recs if r.get("gpu_temp_entry_c")]
        pct = {}
        for arm in arms:
            if arm not in recs[0]["seconds"]:
                continue
            t = statistics.median(r["seconds"][arm] for r in recs)
            pct[arm] = 100.0 * (base - t) / base
        cells[key] = {
            "gbs": read_bytes / base / 1e9,
            "roofline_frac": (read_bytes / base / 1e9) / DRAM_PEAK_GBS,
            "seconds_base": base,
            "read_bytes": read_bytes,
            "blocks": len(recs),
            "entry_c_min": min(entry) if entry else None,
            "entry_c_max": max(entry) if entry else None,
            "pct": pct,
        }
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="research/e123-artifacts/rate.json")
    ap.add_argument(
        "--arms",
        default="n_nosums,n_halfsums_free,k_alu16,k_ld16,k_shuf16,k_tgld16,q_scaffold",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    cells = load_cells(args.rate, arms)

    print(f"{'shape':34s} {'m':>2s} {'GB/s':>7s} {'/peak':>6s} " + " ".join(f"{a[:12]:>12s}" for a in arms))
    for (shape, m), c in cells.items():
        row = f"{shape:34s} {m:2d} {c['gbs']:7.1f} {c['roofline_frac']:6.3f} "
        row += " ".join(f"{c['pct'].get(a, float('nan')):12.4f}" for a in arms)
        print(row)

    print()
    print("H6 test: correlation of deletion gain with achieved bandwidth")
    print(f"{'arm':18s} {'n':>3s} {'r':>7s} {'ci95_lo':>8s} {'ci95_hi':>8s} {'slope %/(GB/s)':>16s}")
    stats = {}
    for arm in arms:
        pts = [(c["gbs"], c["pct"][arm]) for c in cells.values() if arm in c["pct"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        r = pearson(xs, ys)
        lo, hi = fisher_ci(r, len(pts))
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")
        stats[arm] = {"n": len(pts), "r": r, "ci95": [lo, hi], "slope_pct_per_gbs": slope}
        print(f"{arm:18s} {len(pts):3d} {r:7.3f} {lo:8.3f} {hi:8.3f} {slope:16.5f}")

    print()
    print("within-width correlation (removes the width effect)")
    print(f"{'arm':18s} {'m':>2s} {'n':>3s} {'r':>7s}")
    within = defaultdict(dict)
    for arm in arms:
        for m in sorted({k[1] for k in cells}):
            pts = [(c["gbs"], c["pct"][arm]) for k, c in cells.items() if k[1] == m and arm in c["pct"]]
            if len(pts) < 3:
                continue
            r = pearson([p[0] for p in pts], [p[1] for p in pts])
            within[arm][m] = {"n": len(pts), "r": r}
            print(f"{arm:18s} {m:2d} {len(pts):3d} {r:7.3f}")

    if args.out:
        payload = {
            "source": args.rate,
            "dram_peak_gbs": DRAM_PEAK_GBS,
            "cells": {f"{s}|m{m}": c for (s, m), c in cells.items()},
            "pooled": stats,
            "within_width": {a: {str(m): v for m, v in d.items()} for a, d in within.items()},
        }
        json.dump(payload, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
