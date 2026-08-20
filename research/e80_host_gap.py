#!/usr/bin/env python3
"""Close the wall-minus-GPU gap of an E80 census leg into named host costs.

The census records, per round and per phase, the host nanoseconds spent inside
the dispatch call (`dispatch_ns`), inside `commit` (`commit_ns`) and blocked in
`waitUntilCompleted` (`wait_ns`), alongside the counts of each. This script sums
them per round and compares the total with `wall - sum(GPU)`, so the closure gap
is decomposed rather than left as a residual.

    usage: research/e80_host_gap.py research/out/e80-census-w6-default ...
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e80_blocks as B
import e80_census_report as R

FIELDS = ("dispatch_ns", "commit_ns", "wait_ns", "commits", "dispatches",
          "barriers", "waits")


def host_rows(path: pathlib.Path):
    rounds = collections.Counter()
    acc = collections.defaultdict(collections.Counter)
    for line in path.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "round":
            continue
        width = rec["width"]
        rounds[width] += 1
        acc[width]["wall_ns"] += rec.get("wall_ns", 0)
        for entry in (rec.get("phases") or {}).values():
            for field in FIELDS:
                acc[width][field] += entry.get(field, 0)
    return rounds, acc


def gpu_per_round(leg, width, rules, min_rounds):
    """Measured GPU ms/round for a width, summed over its live phases."""
    total = 0.0
    rnds = 0
    for phase in ("target_forward", "target_verify", "draft_head"):
        n = leg.round_count(width, phase)
        if n < min_rounds:
            continue
        att, n = B.attribute(leg, phase, width, rules)
        total += sum(v["gpu_ns"] for v in att.values()) / n / 1e6
        rnds = max(rnds, n)
    return total, rnds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--min-rounds", type=int, default=20)
    args = ap.parse_args()

    print("| leg | width | rounds | wall ms | GPU ms | gap ms | dispatch ms | "
          "commit ms | wait ms | host sum ms | unexplained ms | dispatches | "
          "commits | waits | barriers |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
          "---:|---:|")
    for path in args.legs:
        directory = pathlib.Path(path)
        census = directory / "census.jsonl"
        if not census.exists():
            continue
        leg = R.Leg([census])
        rules = B.learn_axis_rules(leg)
        rounds, acc = host_rows(census)
        for width in sorted(rounds):
            n = rounds[width]
            if n < args.min_rounds:
                continue
            a = acc[width]
            gpu, _ = gpu_per_round(leg, width, rules, args.min_rounds)
            wall = a["wall_ns"] / n / 1e6
            disp = a["dispatch_ns"] / n / 1e6
            commit = a["commit_ns"] / n / 1e6
            wait = a["wait_ns"] / n / 1e6
            host = disp + commit + wait
            print(f"| {directory.name} | {width} | {n} | {wall:.3f} | "
                  f"{gpu:.3f} | {wall - gpu:.3f} | {disp:.3f} | {commit:.3f} | "
                  f"{wait:.3f} | {host:.3f} | {wall - gpu - host:.3f} | "
                  f"{a['dispatches']/n:.1f} | {a['commits']/n:.1f} | "
                  f"{a['waits']/n:.2f} | {a['barriers']/n:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
