#!/usr/bin/env python3
"""Which prompts can actually move the published score, and by how much.

METHOD
------
The published score is the median of eight per-prompt ``raw_ratio_of_means``
values.  With eight values the median is the mean of the fourth and fifth
after sorting ascending.  Only those two slots enter the score, so a prompt
is worth something only while it occupies one of them, or while an
improvement can carry it into one of them.

This tool sorts every official board run's eight prompts ascending, reports
which prompt occupies each rank, and measures the relative gap between
adjacent ranks.  It then computes, for one reference run, the exact
derivative and the exact ceiling of a single-prompt improvement by replaying
the median rule under a multiplier applied to that prompt alone.

WHY THE GAPS MATTER
-------------------
A prompt below rank 4 is worth zero until it closes the whole gap to rank 4.
A prompt above rank 5 is worth zero, full stop.  A prompt sitting in rank 4
or rank 5 is worth ``0.5 * its own share of the median`` per unit, but only
until it crosses the next prompt above it, after which the slot is handed to
that prompt and the derivative drops to zero.

The observed structure at the frontier is extreme: rank 4 is a single prompt
in every strong run, the three ranks below it are far away, and the four
ranks above it form a tight cluster.  That makes the score

    published = 0.5 * raw_rank4  +  0.5 * min(the rank-5 cluster)

with only the first term free.

CAVEATS
-------
The ranking is a property of the current operating point.  A large enough
single-prompt gain reorders the slots, which is exactly what the ceiling
computation measures.  Re-run this after any promotion that changes the
schedule, because the schedule sets the per-prompt draft lengths and
therefore the ranking.

USAGE
-----
    python3 research/board_median_lock.py                 # default reference run
    python3 research/board_median_lock.py <submission-id-prefix> ...

Reads the board dump written by the campaign refresh helper at
``/tmp/yukon-board/full.json``.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter

BOARD = "/tmp/yukon-board/full.json"
STRONG = 3.25

PROMPT_BY_SHA8 = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def load_rows(path: str = BOARD) -> list[dict]:
    payload = json.load(open(path))
    rows = payload
    for key in ("submissions", "rows", "data", "items"):
        if isinstance(rows, dict) and key in rows:
            rows = rows[key]
            break
    return [r for r in rows if isinstance(r, dict)]


def scored_runs(rows: list[dict]) -> list[tuple[dict, list[tuple[float, str]]]]:
    """Return (row, ascending [(raw_ratio, prompt_name)]) for every fully scored run."""
    out = []
    for r in rows:
        metrics = r.get("officialMetrics") or {}
        per_prompt = metrics.get("per_prompt") or []
        if len(per_prompt) != 8 or not r.get("officialScore"):
            continue
        vals = sorted(
            (pp["raw_ratio_of_means"], PROMPT_BY_SHA8.get(pp["prompt_sha256"][:8], "?"))
            for pp in per_prompt
        )
        out.append((r, vals))
    return out


def median_of_eight(values: list[float]) -> float:
    s = sorted(values)
    return (s[3] + s[4]) / 2.0


def rank_occupancy(runs, minimum_score: float | None = None) -> None:
    sel = [(r, v) for r, v in runs if minimum_score is None or r["officialScore"] >= minimum_score]
    label = "all scored runs" if minimum_score is None else f"published >= {minimum_score}"
    print(f"\nwho occupies each sorted rank, {label}, n = {len(sel)}")
    for rank in range(1, 9):
        counts = Counter(v[rank - 1][1] for _, v in sel)
        total = sum(counts.values())
        top = ", ".join(f"{n} {c * 100 / total:.1f}%" for n, c in counts.most_common(4))
        mark = "   <-- median pair" if rank in (4, 5) else ""
        print(f"  rank {rank}: {top}{mark}")


def rank_gaps(runs, minimum_score: float = STRONG) -> None:
    sel = [(r, v) for r, v in runs if r["officialScore"] >= minimum_score]
    print(f"\nrelative gap between adjacent sorted ranks, published >= {minimum_score}, n = {len(sel)}")
    for lo in range(3, 8):
        g = [(v[lo][0] - v[lo - 1][0]) / v[lo - 1][0] * 100 for _, v in sel]
        note = ""
        if lo == 3:
            note = "   the floor below the median pair"
        elif lo == 4:
            note = "   the headroom of the rank-4 prompt"
        print(
            f"  rank {lo} to {lo + 1}: median {statistics.median(g):7.3f} %"
            f"   min {min(g):7.3f}   max {max(g):7.3f}{note}"
        )


def prompt_value(runs, prefix: str) -> None:
    match = next((x for x in runs if str(x[0].get("id", "")).startswith(prefix)), None)
    if match is None:
        print(f"\n{prefix}: not found among scored runs")
        return
    row, vals = match
    base = [v for v, _ in vals]
    names = [n for _, n in vals]
    old = median_of_eight(base)
    solver = row.get("solverUsername", "?")
    print(f"\n{prefix}  {solver}  published {row['officialScore']:.8f}  recomputed median {old:.8f}")
    print(f"  {'#':>2s} {'prompt':9s} {'raw ratio':>11s}   gap to next")
    for i, (v, n) in enumerate(vals, 1):
        gap = f"{(vals[i][0] - v) / v * 100:+7.3f} %" if i < 8 else "       -"
        mark = "   <-- median pair" if i in (4, 5) else ""
        print(f"  {i:2d} {n:9s} {v:11.6f}   {gap}{mark}")

    def median_with(name: str, mult: float) -> float:
        return median_of_eight([v * (mult if n == name else 1.0) for v, n in vals])

    print(f"\n  value of a single-prompt improvement on {prefix}")
    print(f"  {'prompt':9s} {'raw':>10s} {'1 % alone':>11s} {'ceiling':>10s} {'reached at':>11s}")
    for name in names:
        slope = (median_with(name, 1.01) / old - 1) * 100
        best, at = 0.0, 0.0
        step = 0.1
        pct = step
        while pct <= 60.0 + 1e-9:
            gain = (median_with(name, 1 + pct / 100) / old - 1) * 100
            if gain > best + 1e-12:
                best, at = gain, pct
            pct += step
        print(f"  {name:9s} {dict(zip(names, base))[name]:10.6f} {slope:+10.4f} % {best:+9.4f} % {at:10.1f} %")
    uni = (median_of_eight([v * 1.01 for v in base]) / old - 1) * 100
    print(f"  {'uniform':9s} {'':10s} {uni:+10.4f} % {'unbounded':>10s}")

    rank4, rank5 = vals[3], vals[4]
    cluster = [v for v, _ in vals[4:]]
    reachable = min(cluster[1:]) if len(cluster) > 1 else cluster[0]
    ceiling_median = (rank5[0] + reachable) / 2.0
    print(
        f"\n  closing the whole rank-4 deficit on {rank4[1]} would give median "
        f"{ceiling_median:.8f}, which is {(ceiling_median / old - 1) * 100:+.3f} %"
    )


def main() -> None:
    rows = load_rows()
    runs = scored_runs(rows)
    print(f"board rows {len(rows)}, fully scored runs {len(runs)}")
    rank_occupancy(runs)
    rank_occupancy(runs, STRONG)
    rank_gaps(runs)
    targets = sys.argv[1:] or ["8819b108", "cb8aeefb"]
    for prefix in targets:
        prompt_value(runs, prefix)


if __name__ == "__main__":
    main()
