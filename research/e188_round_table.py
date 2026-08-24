#!/usr/bin/env python3
"""E188 stage 2: the per-round host-phase table for one or more legs.

harness=local. The commit phase has two regimes inside a single 128-token leg,
and realized draft depth `d` ramps over the same early rounds, so depth and
round index are confounded. This table prints both against every host phase so
the confound is visible rather than assumed away.

Usage: python3 research/e188_round_table.py TAG [TAG ...]
"""
import re
import sys

FIELDS = ["d", "acc", "commit_us", "clear_release_us", "clear_release_count",
          "readout_us", "upkeep_us", "round_us"]


def rows(tag):
    out = []
    with open(f"research/out/{tag}/trace.txt") as fh:
        for line in fh:
            if not line.startswith("mtp-trace: round="):
                continue
            out.append(dict(re.findall(r"(\w+)=([-\w.]+)", line)))
    return out


def main():
    for tag in sys.argv[1:]:
        print(f"--- {tag}")
        print("  round " + " ".join(f"{f:>11s}" for f in FIELDS))
        for r in rows(tag):
            print(f"  {int(r['round']):5d} " +
                  " ".join(f"{float(r.get(f, 'nan')):11.1f}" for f in FIELDS))


if __name__ == "__main__":
    main()
