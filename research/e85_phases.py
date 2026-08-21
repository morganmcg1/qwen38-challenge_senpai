#!/usr/bin/env python3
"""Per-leg host-phase medians for an E85 traced session, keyed by position.

    usage: research/e85_phases.py SESSION_DIR [--json OUT]

`qwen-alphonse` found that a leg at a session extreme carries several times
the host cost of an interior leg, mostly in `commit_us`, `d_submit1_us` and
`d_head1_us`. If one arm owns the extremes, every contrast measured against it
is biased. This script makes that contamination visible, or shows it is absent,
before any arm contrast is read.

His `research/e86_phases.py` is on PR #88, which this launch's isolation rules
place outside the branches I may inspect, so the host-sum definition here is
rebuilt from his reported field list rather than copied from his code.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from e85_round_pairs import parse_rounds, timed_segment

HOST_FIELDS = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
               "d_chain_us", "readout_us", "commit_us", "upkeep_us"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.session)
    with (root / "legs.tsv").open() as handle:
        legs = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for position, row in enumerate(legs):
        leg = int(row["leg"])
        rounds = timed_segment(
            parse_rounds(root / f"leg{leg:02d}-{row['arm']}" / "rounds.txt"))
        entry = {
            "leg": leg,
            "position": position,
            "arm": row["arm"],
            "host_sum_us": statistics.median(
                sum(r[f] for f in HOST_FIELDS) for r in rounds),
            "round_us": statistics.median(r["round_us"] for r in rounds),
        }
        for field in HOST_FIELDS:
            entry[field] = statistics.median(r[field] for r in rounds)
        rows.append(entry)

    interior = [r["host_sum_us"] for r in rows if 0 < r["position"] < len(rows) - 1]
    interior_median = statistics.median(interior)

    print(f"{'pos':>3s} {'leg':>3s} {'arm':<5s} {'host_sum':>9s} {'vs int':>7s} "
          + " ".join(f"{f.replace('_us', ''):>9s}" for f in HOST_FIELDS))
    for r in rows:
        ratio = r["host_sum_us"] / interior_median
        mark = "  <-- extreme" if r["position"] in (0, len(rows) - 1) else ""
        print(f"{r['position']:3d} {r['leg']:3d} {r['arm']:<5s} "
              f"{r['host_sum_us']:9.0f} {ratio:6.2f}x "
              + " ".join(f"{r[f]:9.0f}" for f in HOST_FIELDS) + mark)

    print(f"\ninterior host_sum median = {interior_median:.0f} us/round")
    worst = max(rows, key=lambda r: r["host_sum_us"])
    print(f"worst leg = position {worst['position']} ({worst['arm']}) at "
          f"{worst['host_sum_us'] / interior_median:.2f}x interior")

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["position"])
    print("\nmean session position per arm (equal means = position balanced)")
    for arm, positions in sorted(by_arm.items()):
        print(f"  {arm:<5s} positions {positions} mean "
              f"{statistics.fmean(positions):.2f}")

    report = {
        "legs": rows,
        "interior_host_sum_median_us": interior_median,
        "mean_position_per_arm": {
            a: statistics.fmean(p) for a, p in by_arm.items()},
        "extreme_positions_by_arm": {
            a: [p for p in ps if p in (0, len(rows) - 1)]
            for a, ps in by_arm.items()},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
