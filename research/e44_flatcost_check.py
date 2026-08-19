#!/usr/bin/env python3
"""Test the post-hoc cost model for the E44 simdgroup-matrix QMV cell.

The pre-registered mechanism was weight-stream halving, which predicts the win
concentrates at M=5..8 and is larger on the DRAM-bound mlp_down shape. The
measured pattern contradicts both halves, so this script tests the alternative
that the data suggests: the MMA cell always evaluates a full 8-row tile, so its
cost is independent of M for M<=8 and doubles at M=9 when a second tile appears,
while the base cost rises with M. That predicts a crossover, a win only above it,
and a large loss below it.

    research/e44_flatcost_check.py RUN_DIR
"""

from __future__ import annotations

import json
import statistics as st
import sys


def main() -> int:
    run_dir = sys.argv[1]
    with open(f"{run_dir}/ab.json") as handle:
        data = json.load(handle)
    rows = [r for r in data["measurements"] if r["kind"] == "timing"]

    for shape in sorted({r["shape"] for r in rows}):
        print(f"\n=== {shape} ===")
        base, cand = {}, {}
        for m in range(1, 10):
            sel = [r for r in rows if r["shape"] == shape and r["m"] == m]
            if not sel:
                continue
            base[m] = st.mean(r["base_s"] for r in sel) * 1e6
            cand[m] = st.mean(r["cand_s"] for r in sel) * 1e6

        print(f"{'M':>2} {'base_us':>9} {'cand_us':>9} {'cand/base':>10} {'cand step':>10}")
        for m in sorted(base):
            step = f"{cand[m] - cand[m - 1]:+10.2f}" if m - 1 in cand else " " * 10
            print(f"{m:>2} {base[m]:9.2f} {cand[m]:9.2f} {cand[m] / base[m]:10.4f} {step}")

        plateau = [cand[m] for m in range(4, 9) if m in cand]
        mean_p = st.mean(plateau)
        sd_p = st.stdev(plateau)
        print(f"  cand plateau M=4..8 : mean {mean_p:8.2f} us  sd {sd_p:5.2f} us "
              f"({100 * sd_p / mean_p:.2f} % of mean)")
        base_span = [base[m] for m in range(4, 9) if m in base]
        print(f"  base  span   M=4..8 : {base_span[0]:8.2f} -> {base_span[-1]:8.2f} us "
              f"(+{100 * (base_span[-1] / base_span[0] - 1):.1f} %)")
        if 9 in cand:
            print(f"  cand M=9 / plateau  : {cand[9] / mean_p:.3f}x "
                  f"(a second 8-row tile predicts ~2x)")

        # Crossover: the bracket where the rising base curve overtakes the flat
        # candidate cost. Only the first sign change is meaningful, because the
        # candidate stops being flat once a second tile appears at M=9.
        wins = [m for m in sorted(base) if m >= 4 and base[m] > cand[m]]
        print(f"  widths where cand wins: {wins if wins else 'none'}")
        bracket = None
        for m in sorted(base):
            if m >= 5 and base[m - 1] <= cand[m - 1] and base[m] > cand[m]:
                bracket = (m - 1, m)
                break
        if bracket:
            lo, hi = bracket
            span = base[hi] - base[lo]
            frac = (mean_p - base[lo]) / span if span else float("nan")
            print(f"  crossover bracketed in M=[{lo}, {hi}], "
                  f"flat-cost interpolation M ~ {lo + frac:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
