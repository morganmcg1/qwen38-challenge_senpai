#!/usr/bin/env python3
"""E185 helper: where the gaps of one millisecond and more sit.

    usage: research/e185_biggap.py TAG [TAG...] [--json OUT]

Ninety percent of the ledger's idle total is in gaps of at least one
millisecond, so the coherent-slice verdict turns on where those few gaps are
and whether they recur in every round. This reader groups them by the phase
they start in and reports how many rounds contain one.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e185_gap_table import gaps_in, phase_of, read_anchors, read_intervals, union
from e185_residency import largest_round_block, read_rounds

BIG_US = 1000.0


def report(tag: str) -> list[dict]:
    out = Path("research/out") / tag
    block = largest_round_block(read_rounds(out / "trace.txt"))[8:]
    pid = block[0].pid
    anchors = {a["round"]: a for a in read_anchors(out / "trace.txt") if a.get("pid") == pid}
    merged = union(read_intervals(out / "gpu-intervals.jsonl", pid))

    big = []
    for r in block:
        a = anchors[r.index]
        for s, e in gaps_in(merged, r.t0, r.t1):
            us = (e - s) / 1000.0
            if us >= BIG_US:
                big.append((us, phase_of(a, s), r.index))

    inter = [
        (e - s) / 1000.0
        for r0, r1 in zip(block, block[1:])
        for s, e in gaps_in(merged, r0.t1, r1.t0)
    ]

    total: dict[str, float] = defaultdict(float)
    count: Counter[str] = Counter()
    for us, phase, _ in big:
        total[phase] += us
        count[phase] += 1

    rows = []
    print(f"== {tag}: rounds={len(block)} gaps>={BIG_US:g}us n={len(big)}")
    for phase in sorted(total, key=lambda p: -total[p]):
        med = st.median([u for u, p, _ in big if p == phase])
        rows.append({
            "tag": tag,
            "start_phase": phase,
            "gaps": count[phase],
            "rounds": len(block),
            "rounds_with_one": len({i for _, p, i in big if p == phase}),
            "us_per_round": total[phase] / len(block),
            "median_gap_us": med,
        })
        print(
            f"   {phase:14s} n={count[phase]:3d}"
            f"  total/round={total[phase] / len(block):8.1f}us"
            f"  median={med:8.1f}us"
        )
    print(f"   rounds containing one: {len({i for _, _, i in big})}/{len(block)}")
    print(
        f"   inter_round_gap n={len(inter)}"
        f" mean/round={sum(inter) / len(block):.1f}us"
        f" median={st.median(inter):.1f}us max={max(inter):.1f}us"
    )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json", dest="json_path")
    parsed = ap.parse_args()
    rows: list[dict] = []
    for tag in parsed.tags:
        rows += report(tag)
    if parsed.json_path:
        Path(parsed.json_path).write_text(json.dumps(rows, indent=2) + "\n")
