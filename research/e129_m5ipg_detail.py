#!/usr/bin/env python3
"""E129 rung 2d -- per-block detail behind the isolated M=5 IPG probe.

    usage: research/e129_m5ipg_detail.py OUT_DIR

Prints every palindromic block mean so a reader can see the dispersion the
medians hide, then reports the implied residency-to-time coefficient on the
non-M=5 cells and the pass-count ratio on the M=5 cells.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# `(5,5)` -> `(5,3)` moves this host's shared entry point from 32 to 33
# derived resident simdgroups. Every non-M=5 cell executes identical code, so
# residency is the only channel that can move them.
G16S_RESIDENCY_STEP = 1.0 / 32.0


def block_means(blocks: list[dict]) -> tuple[list[float], list[float]]:
    a = [0.5 * (b["arms"][0]["forward_us"] + b["arms"][0]["reverse_us"]) for b in blocks]
    b = [0.5 * (b["arms"][1]["forward_us"] + b["arms"][1]["reverse_us"]) for b in blocks]
    return a, b


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    payload = json.loads((Path(sys.argv[1]) / "m5ipg.json").read_text())

    cells: dict[tuple[str, int], list[dict]] = {}
    for record in payload["cells"]:
        cells.setdefault((record["shape"], record["width"]), []).append(record)

    print("per-block arm means, microseconds. a = IPG 5 at M=5, b = IPG 3 at M=5")
    for (shape, width), blocks in sorted(cells.items()):
        a, b = block_means(blocks)
        print(
            f"{shape:<13}M={width} a " + " ".join(f"{v:7.1f}" for v in a)
            + " | b " + " ".join(f"{v:7.1f}" for v in b)
        )

    print("\nnon-M=5 cells: identical executed code, residency is the only channel")
    # M=4 is the first cell measured for each shape and carries the weight
    # allocation and the second pipeline's first compile; four of its six
    # shapes show single-block outliers above 1.5x the cell median. fa.o_proj
    # is bimodal at every width: its `a` arm alternates between the `b` arm's
    # value and a value 12 % above it. Both are reported and then excluded.
    selections = [
        ("all cells", lambda s, m: m != 5),
        ("M>=6 only", lambda s, m: m >= 6),
        ("M>=6, no fa.o_proj", lambda s, m: m >= 6 and s != "fa.o_proj"),
    ]
    for label, keep in selections:
        fractions = []
        for (shape, width), blocks in cells.items():
            if not keep(shape, width):
                continue
            a, b = block_means(blocks)
            fractions.append(statistics.median((x - y) / x for x, y in zip(a, b)))
        mean = statistics.mean(fractions)
        sd = statistics.stdev(fractions)
        print(
            f"  {label:<22} n={len(fractions):>2} mean={100 * mean:+.4f}% "
            f"sd={100 * sd:.4f}% sem={100 * sd / len(fractions) ** 0.5:.4f}% "
            f"c_implied={mean / G16S_RESIDENCY_STEP:+.4f}"
        )

    print("\nM=5 cells: IPG 3 needs two passes over the weight matrix, IPG 5 needs one")
    for (shape, width), blocks in sorted(cells.items()):
        if width != 5:
            continue
        a, b = block_means(blocks)
        ipg5 = statistics.median(a)
        ipg3 = statistics.median(b)
        neighbour = cells.get((shape, 6))
        note = ""
        if neighbour:
            m6 = statistics.median(block_means(neighbour)[0])
            note = f"  M=6 two-pass reference {m6:8.1f} us, ipg3/M6 = {ipg3 / m6:.4f}"
        print(
            f"  {shape:<13} ipg5 {ipg5:8.1f} us  ipg3 {ipg3:8.1f} us  "
            f"ratio {ipg3 / ipg5:.4f}{note}"
        )

    # The shipped table sends M=3,4,5 through one pass and M=6,7,8 through two.
    # Fit the one-pass cost from the M=4 and M=5 arm-a cells, extrapolate it to
    # M=6,7,8, and compare with the measured two-pass cost. This prices the
    # opposite lever: raising IPG so the wide widths make one pass.
    print("\none-pass line fitted on M=4,5 (arm a), against the measured two-pass cells")
    for shape in sorted({s for s, _ in cells}):
        a4 = cells.get((shape, 4))
        a5 = cells.get((shape, 5))
        if not a4 or not a5:
            continue
        # Arm a at M=4 shares the shape's first-cell outliers, so take the
        # minimum block rather than the median as the one-pass floor.
        c4 = min(block_means(a4)[0])
        c5 = statistics.median(block_means(a5)[0])
        slope = c5 - c4
        parts = []
        for width in (6, 7, 8):
            cell = cells.get((shape, width))
            if not cell:
                continue
            two = statistics.median(block_means(cell)[0])
            one = c4 + slope * (width - 4)
            parts.append(f"M={width} {two:7.1f} -> {one:7.1f} ({100 * (one / two - 1):+6.1f}%)")
        print(f"  {shape:<13} one-pass {c4:7.1f} + {slope:6.1f}/row   " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
