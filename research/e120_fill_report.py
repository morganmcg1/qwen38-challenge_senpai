#!/usr/bin/env python3
"""Aggregate E120 fillCost blocks into per-cell medians and a break-even verdict."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BREAK_EVEN_US = 22.7

# Round census: how many activation tensors of each K feed exactly one wide QMV.
CENSUS = {5120: 129, 17408: 64, 6144: 16}


def load_blocks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return raw["cells"] if isinstance(raw, dict) else raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fill_json", type=Path)
    args = ap.parse_args()

    blocks = load_blocks(args.fill_json)
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for b in blocks:
        cells[(b["shape"], b["width"])].append(b)

    rows = []
    for (shape, width), bs in sorted(cells.items()):
        layers = bs[0]["layers"]

        def arm_samples(name: str) -> list[float]:
            out = []
            for b in bs:
                for a in b["arms"]:
                    if a["arm"] == name:
                        out.append(a["forward_us"])
                        out.append(a["reverse_us"])
            return out

        a = statistics.median(arm_samples("a_replica"))
        bb = statistics.median(arm_samples("b_fill_noconsume"))
        c = statistics.median(arm_samples("c_sumtable"))
        rows.append(
            {
                "shape": shape,
                "width": width,
                "k": bs[0]["hidden"],
                "n": bs[0]["outputs"],
                "k_blocks": bs[0]["k_blocks"],
                "table_bytes": bs[0]["table_bytes"],
                "layers": layers,
                "blocks": len(bs),
                "samples_per_arm": 2 * len(bs),
                "a_replica_us": a,
                "b_fill_noconsume_us": bb,
                "c_sumtable_us": c,
                "fill_us_per_dispatch": (bb - a) / layers,
                "consumer_gain_us_per_matvec": (bb - c) / layers,
                "net_us_per_matvec": (a - c) / layers,
                "net_pct_of_matvec": 100.0 * (a - c) / a,
                "temp_entry_c": min(x["gpu_temp_entry_c"] for x in bs),
                "temp_exit_c": max(x["gpu_temp_exit_c"] for x in bs),
            }
        )

    hdr = (
        f"{'shape':<14}{'M':>3}{'K':>7}{'N':>8}{'kb':>4}{'tblB':>7}"
        f"{'fill_us':>10}{'gain_us':>10}{'net_us':>9}{'net_%':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['shape']:<14}{r['width']:>3}{r['k']:>7}{r['n']:>8}{r['k_blocks']:>4}"
            f"{r['table_bytes']:>7}{r['fill_us_per_dispatch']:>10.3f}"
            f"{r['consumer_gain_us_per_matvec']:>10.3f}{r['net_us_per_matvec']:>9.3f}"
            f"{r['net_pct_of_matvec']:>8.3f}"
        )

    fills = [r["fill_us_per_dispatch"] for r in rows]
    print()
    print(f"fill cost per in-stream dispatch: min={min(fills):.3f} "
          f"median={statistics.median(fills):.3f} max={max(fills):.3f} us")
    print(f"pre-registered break-even        : {BREAK_EVEN_US:.3f} us")
    print("verdict: " + ("BELOW break-even (prediction FALSIFIED, Route B viable)"
                         if max(fills) < BREAK_EVEN_US
                         else "AT/ABOVE break-even for some cells"))

    by_k = defaultdict(list)
    for r in rows:
        by_k[r["k"]].append(r)
    print()
    print("per-K summary (median over widths):")
    for k in sorted(by_k):
        rs = by_k[k]
        print(
            f"  K={k:<6} fill={statistics.median(r['fill_us_per_dispatch'] for r in rs):.3f} "
            f"gain={statistics.median(r['consumer_gain_us_per_matvec'] for r in rs):.3f} "
            f"net={statistics.median(r['net_us_per_matvec'] for r in rs):.3f} us "
            f"count/round={CENSUS.get(k, '?')}"
        )

    out = args.fill_json.with_name("fill_report.json")
    out.write_text(json.dumps({"break_even_us": BREAK_EVEN_US, "cells": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
