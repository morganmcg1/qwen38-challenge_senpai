#!/usr/bin/env python3
"""Aggregate every measured E120 fill cell and test the additivity model.

Model under test (advisor, PR #121 comment 5377942628):

    gain(shape, M) = sum over accumulator groups of gain_NA(shape, NA_of_group)
    net(shape, M)  = gain(shape, M) - fill(shape, M)

The per-NA basis comes from the single-group widths M=3 -> [3], M=4 -> [4] and
M=5 -> [5].  Every multi-group width is then a pure prediction.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

GROUPS = {3: [3], 4: [4], 5: [5], 6: [3, 3], 7: [4, 3], 8: [4, 4], 9: [3, 3, 3]}
DEFAULT_RUNS = [
    "research/out/e120-rung5d-fill",
    "research/out/e120-rung5d-na",
    "research/out/e120-rung5d-na2",
]


def bytes_moved(k, n, m):
    """Affine-4/group-64 weight traffic plus bf16 activation traffic."""
    return k * n * 36 // 64 + 2 * m * k + 2 * m * n


def load(paths):
    raw = {}
    for p in paths:
        f = Path(p)
        if f.is_dir():
            f = f / "fill.json"
        doc = json.loads(f.read_text())
        for c in doc["cells"]:
            raw.setdefault((c["shape"], c["width"]), []).append((f.parent.name, c))
    return raw


def aggregate(raw):
    """Median the raw arm times first, then difference.

    The three arm times are the primitive measurements and the differences
    between them are small, so differencing per block first and taking a median
    of the differences amplifies noise. research/e120_fill_report.py uses the
    same estimator.
    """
    out = {}
    for (shape, m), rows in sorted(raw.items()):
        cs = [c for _, c in rows]
        layers = cs[0]["layers"]

        def arm(name):
            s = [
                v
                for c in cs
                for a in c["arms"]
                if a["arm"] == name
                for v in (a["forward_us"], a["reverse_us"])
            ]
            return statistics.median(s)

        replica, noconsume, table = arm("a_replica"), arm("b_fill_noconsume"), arm("c_sumtable")
        base = replica / layers
        k, n = cs[0]["hidden"], cs[0]["outputs"]
        gain = (noconsume - table) / layers
        fill = (noconsume - replica) / layers
        out[(shape, m)] = {
            "shape": shape,
            "m": m,
            "k": k,
            "n": n,
            "k_blocks": cs[0]["k_blocks"],
            "blocks": len(cs),
            "runs": sorted({r for r, _ in rows}),
            "base_us": base,
            "gain_us": gain,
            "fill_us": fill,
            "net_us": (replica - table) / layers,
            # cross-check: median of the per-block difference
            "net_alt_us": statistics.median(c["net_us_per_matvec"] for c in cs),
            "gb_per_s": bytes_moved(k, n, m) / base / 1000.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="research/out/e120-additivity.json")
    args = ap.parse_args()

    cells = aggregate(load(args.runs))
    shapes = sorted({s for s, _ in cells})

    print("== measured cells ==")
    hdr = f"{'shape':<14}{'M':>2} {'groups':<10}{'base_us':>10}{'GB/s':>7}"
    hdr += f"{'gain_us':>9}{'gain%':>7}{'fill_us':>9}{'net_us':>9}{'net%':>7} blocks"
    print(hdr)
    for s in shapes:
        for m in range(3, 10):
            c = cells.get((s, m))
            if not c:
                continue
            g = 100 * c["gain_us"] / c["base_us"]
            nt = 100 * c["net_us"] / c["base_us"]
            print(
                f"{s:<14}{m:>2} {str(GROUPS[m]):<10}{c['base_us']:>10.2f}"
                f"{c['gb_per_s']:>7.1f}{c['gain_us']:>9.3f}{g:>7.2f}"
                f"{c['fill_us']:>9.3f}{c['net_us']:>9.3f}{nt:>7.2f}{c['blocks']:>7}"
            )
        print()

    print("== per-NA gain basis (single-group widths) ==")
    basis = {}
    print(f"{'shape':<14}{'NA=3':>9}{'NA=4':>9}{'NA=5':>9}   (us gain per matvec)")
    for s in shapes:
        row = {}
        for na, m in ((3, 3), (4, 4), (5, 5)):
            c = cells.get((s, m))
            if c:
                row[na] = c["gain_us"]
        basis[s] = row
        f = lambda na: f"{row[na]:>9.3f}" if na in row else f"{'-':>9}"
        print(f"{s:<14}{f(3)}{f(4)}{f(5)}")
    print()

    print("== additivity test: multi-group widths ==")
    print(
        f"{'shape':<14}{'M':>2} {'groups':<10}{'pred_gain':>10}{'meas_gain':>10}"
        f"{'ratio':>7}{'err_us':>9}{'err%base':>9}"
    )
    ratios = []
    checks = []
    for s in shapes:
        for m in (6, 7, 8, 9):
            c = cells.get((s, m))
            if not c:
                continue
            gs = GROUPS[m]
            if any(na not in basis[s] for na in gs):
                continue
            pred = sum(basis[s][na] for na in gs)
            meas = c["gain_us"]
            ratio = meas / pred if pred else float("nan")
            err = meas - pred
            checks.append(
                {
                    "shape": s,
                    "m": m,
                    "groups": gs,
                    "pred_gain_us": pred,
                    "meas_gain_us": meas,
                    "ratio": ratio,
                    "err_us": err,
                    "err_pct_of_base": 100 * err / c["base_us"],
                }
            )
            ratios.append(ratio)
            print(
                f"{s:<14}{m:>2} {str(gs):<10}{pred:>10.3f}{meas:>10.3f}"
                f"{ratio:>7.3f}{err:>9.3f}{100 * err / c['base_us']:>9.2f}"
            )
    print()
    if ratios:
        print(
            f"measured/predicted gain ratio: n={len(ratios)} "
            f"mean={statistics.mean(ratios):.3f} median={statistics.median(ratios):.3f} "
            f"min={min(ratios):.3f} max={max(ratios):.3f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "cells": [dict(v) for v in cells.values()],
                "basis": {s: {str(k): v for k, v in b.items()} for s, b in basis.items()},
                "additivity_checks": checks,
            },
            indent=1,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
