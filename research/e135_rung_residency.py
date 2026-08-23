#!/usr/bin/env python3
"""Which verify-width rungs does the ranked run actually sit on?

`onePass6` and `onePass67` differ at exactly one rung. Both send width 6 to a
single one-pass column; `onePass67` also sends width 7 to one column while
`onePass6` leaves it at `[4+3]`. So the whole distance between the two arms is
the ranked residency of width 7, and the distance from `shipped` to either arm
is the residency of the rungs each one changes.

The receipt publishes a per-prompt mean verify width, `1 + edl`, and no
histogram. The maximum-entropy distribution on the routed width support with
that mean is the least committed reconstruction: a truncated geometric. It is
an assumption, not a measurement, and it is labelled as one everywhere below.

The local benchmark fixture publishes its real histogram in the pipeline trace,
so the local column is measured and only the ranked column is reconstructed.

  research/e135_rung_residency.py [--board PATH] [--receipt ID]
                                  [--trace research/out/TAG/trace.txt]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_receipt_read as rr  # noqa: E402

SUPPORT = tuple(range(2, 10))

# The rung each arm serves width by width, from the compiled witnesses.
ARMS = {
    "shipped":    {6: 2, 7: 2, 8: 2},
    "onePass6":   {6: 1, 7: 2, 8: 2},
    "onePass67":  {6: 1, 7: 1, 8: 2},
    "onePass678": {6: 1, 7: 1, 8: 1},
}


def maxent(mean: float, support=SUPPORT) -> dict:
    """Truncated geometric on `support` with the requested mean."""
    lo, hi = -20.0, 20.0
    for _ in range(200):
        lam = (lo + hi) / 2.0
        w = [math.exp(lam * m) for m in support]
        got = sum(m * x for m, x in zip(support, w)) / sum(w)
        if got < mean:
            lo = lam
        else:
            hi = lam
    w = [math.exp(((lo + hi) / 2.0) * m) for m in support]
    total = sum(w)
    return {m: x / total for m, x in zip(support, w)}


def local_histogram(path: str) -> dict:
    counts: dict[int, int] = {}
    with open(path) as fh:
        for line in fh:
            if "mtp-trace:" not in line:
                continue
            hit = re.search(r"\bd=(\d+)", line.split("mtp-trace:", 1)[1])
            if hit:
                counts[int(hit.group(1)) + 1] = counts.get(
                    int(hit.group(1)) + 1, 0) + 1
    total = sum(counts.values())
    return {m: c / total for m, c in sorted(counts.items())}, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    ap.add_argument("--receipt", default="0cf1637e")
    ap.add_argument("--trace", default="research/out/e135shipexact/trace.txt")
    args = ap.parse_args()

    row = rr.find(rr.load_board(args.board), args.receipt)
    entries = rr.per_prompt(row)

    ranked = {m: 0.0 for m in SUPPORT}
    print(f"receipt {row['id'][:8]}  maxent reconstruction, Rule 148 weights")
    print("  prompt      weight   mean width   P(6)     P(7)     P(8)")
    for name, weight in sorted(rr.RULE148_WEIGHTS.items(),
                               key=lambda kv: -kv[1]):
        mean = 1.0 + entries[name]["effective_mean_draft_len"]
        dist = maxent(mean)
        for m in SUPPORT:
            ranked[m] += weight * dist[m]
        print(f"  {name:<9} {weight:8.4f} {mean:11.4f}"
              f" {dist[6]:8.4f} {dist[7]:8.4f} {dist[8]:8.4f}")

    scale = sum(rr.RULE148_WEIGHTS.values())
    ranked = {m: p / scale for m, p in ranked.items()}

    local, rounds = local_histogram(args.trace)
    print(f"\n  local fixture histogram, MEASURED over {rounds} rounds: "
          f"{ {m: round(p, 4) for m, p in local.items()} }")

    print("\n## Rung residency")
    print("  width   ranked (maxent)   local (measured)   ranked/local")
    for m in SUPPORT:
        lp = local.get(m, 0.0)
        ratio = f"{ranked[m] / lp:8.2f}" if lp else "       -"
        print(f"  {m:5d} {ranked[m]:16.4f} {lp:18.4f} {ratio}")
    six_seven = ranked[6] + ranked[7]
    lo_six_seven = local.get(6, 0.0) + local.get(7, 0.0)
    print(f"  widths 6+7      {six_seven:.4f}          {lo_six_seven:.4f}"
          f"     {six_seven / lo_six_seven:.2f}")

    print("\n## What separates the arms")
    print("  arm          columns at 6/7/8   ranked column mass over 6..8")
    for name, plan in ARMS.items():
        mass = sum(ranked[m] * plan[m] for m in (6, 7, 8))
        print(f"  {name:<11}  {plan[6]}/{plan[7]}/{plan[8]}"
              f"              {mass:.4f}")
    print(f"\n  onePass6 -> onePass67 removes one column from width 7 only:"
          f" {ranked[7]:.4f} of ranked rounds.")
    print(f"  shipped  -> onePass6  removes one column from width 6 only:"
          f" {ranked[6]:.4f} of ranked rounds.")
    print("  The maxent reconstruction is an assumption. Rule 83 still applies:")
    print("  transfer the column count, never the local microseconds.")


if __name__ == "__main__":
    main()
