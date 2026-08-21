#!/usr/bin/env python3
"""E92: per-position acceptance from width-pinned legs.

A pinned leg proposes exactly `d` drafts on every round, so position `i <= d`
is observed on every round whatever happened at earlier positions. That removes
the survivorship bias an adaptive schedule introduces, where deep positions are
only reached on rounds that were already going well.

    P(i | d)      fraction of pinned-`d` rounds whose accepted prefix reaches i
    conditional   P(i | d) / P(i-1 | d)

    usage: research/e92_acceptance.py [GLOB ...] [--output PATH]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

from e90_intervals import read_meta, read_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("globs", nargs="*",
                        default=["research/out/e92[wp][1-9][abcd]/trace.txt"])
    parser.add_argument("--skip-rounds", type=int, default=8)
    parser.add_argument("--drop-last", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    accepted = collections.defaultdict(collections.Counter)
    legs = collections.defaultdict(list)
    for pattern in arguments.globs:
        for path in sorted(glob.glob(pattern)):
            tag = Path(path).parent.name
            pin = int(read_meta(tag)["e92_pinned_drafts"])
            if pin == 0:
                continue
            rows = read_rounds(tag)[arguments.skip_rounds:]
            if arguments.drop_last:
                rows = rows[:-arguments.drop_last]
            # A round clamped by the remaining decode budget proposes fewer
            # drafts than the pin, so it observes a shorter position set and
            # must not enter another depth's bucket.
            for row in rows:
                if row["d"] == pin:
                    accepted[pin][row["acc"]] += 1
            legs[pin].append(tag)

    table = {}
    for depth in sorted(accepted):
        counts = accepted[depth]
        total = sum(counts.values())
        reach = [sum(v for k, v in counts.items() if k >= i) / total
                 for i in range(1, depth + 1)]
        table[depth] = {
            "rounds": total,
            "legs": sorted(legs[depth]),
            "reach": reach,
            "conditional": [reach[0]] + [
                reach[i] / reach[i - 1] if reach[i - 1] else None
                for i in range(1, depth)],
            "mean_accepted": sum(k * v for k, v in counts.items()) / total,
            "expected_tokens": 1.0 + sum(reach),
        }

    print("pinned d  rounds  " + "".join("  P%-6d" % i for i in range(1, 9))
          + "  mean_acc   Y(d)")
    for depth, row in table.items():
        cells = "".join("  %.4f" % p for p in row["reach"])
        print("%8d %7d  %s%s  %8.4f %6.4f"
              % (depth, row["rounds"], cells, "        " * (8 - depth),
                 row["mean_accepted"], row["expected_tokens"]))

    print()
    print("conditional P(accept position i | positions 1..i-1 accepted)")
    print("pinned d  " + "".join("  i=%-5d" % i for i in range(1, 9)))
    for depth, row in table.items():
        cells = "".join(
            "  %.4f" % c if c is not None else "     -  "
            for c in row["conditional"])
        print("%8d  %s" % (depth, cells))

    # Pool each position over every pinned depth that observes it. A single
    # pinned leg carries its own sampling noise into Y(d); the pooled profile
    # separates the shape of acceptance from that noise.
    pooled = []
    for position in range(1, 9):
        at_risk = sum(v for depth, counts in accepted.items() if depth >= position
                      for k, v in counts.items() if k >= position - 1)
        reached = sum(v for depth, counts in accepted.items() if depth >= position
                      for k, v in counts.items() if k >= position)
        pooled.append(reached / at_risk if at_risk else None)

    running, pooled_yield = 1.0, [1.0]
    for q in pooled:
        running *= q
        pooled_yield.append(pooled_yield[-1] + running)

    print()
    print("pooled conditional acceptance and the Y(d) it implies")
    print("  i  q_i      d  Y(d)")
    for index, q in enumerate(pooled):
        print("%3d  %.4f  %3d  %.4f"
              % (index + 1, q, index + 1, pooled_yield[index + 1]))

    result = {"per_pinned_depth": table,
              "pooled_conditional": pooled,
              "pooled_expected_tokens": pooled_yield}
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2,
                                               sort_keys=True))


if __name__ == "__main__":
    main()
