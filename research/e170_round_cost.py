#!/usr/bin/env python3
"""Measure the per-round cost law directly, from untraced timed legs.

    usage: research/e170_round_cost.py LEG_DIR [LEG_DIR ...] [--json OUT]

`04-mtp-timed.json` already carries `block_request_seconds` and
`effective_draft_lengths` as PER-ROUND lists. Together they give round wall
time against verified width `M = 1 + drafts` at no extra GPU cost, and they are
written by the trusted parent whether or not the candidate trace is on. So this
is a timing measurement taken from a trace-free leg, which is the only kind of
leg whose wall time means anything.

The quantity under test is the group structure. `activeInputGroups` in
`Qwen35.swift` is a static candidate-side table, so `groups(M)` is fixed:

    M       2  3  4  5  6  7  8  9
    IPG     2  3  4  5  3  4  4  3
    groups  1  1  1  1  2  2  2  3

A group is one complete pass over the backbone weights. If that is what the
hardware does, the cost curve must show a large jump between `M = 5` and
`M = 6`, and must NOT show one anywhere else below `M = 9`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

# `Qwen35.swift:1709-1728`, a compile-time table, so this transfers between
# hosts exactly. Only the cost of one pass is hardware.
GROUPS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}
HEAD_STEP_MS = 0.96


def load_rounds(directory: str) -> list[tuple[int, float]]:
    """Per-round `(M, milliseconds)` from one untraced timed leg."""
    path = pathlib.Path(directory) / "reports" / "04-mtp-timed.json"
    report = json.loads(path.read_text())
    widths = report["effective_draft_lengths"]
    seconds = report["block_request_seconds"]
    if len(widths) != len(seconds):
        raise SystemExit(
            f"{path}: {len(widths)} widths against {len(seconds)} round times"
        )
    meta = pathlib.Path(directory) / "meta.txt"
    if meta.exists() and "rounds_traced=0" not in meta.read_text():
        raise SystemExit(f"{directory}: traced leg, its wall time is perturbed")
    return [(1 + w, s * 1000.0) for w, s in zip(widths, seconds)]


def cost_table(rounds: list[tuple[int, float]], min_rounds: int) -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for width, ms in rounds:
        buckets.setdefault(width, []).append(ms)
    rows = []
    previous = None
    for width in sorted(buckets):
        values = buckets[width]
        if len(values) < min_rounds:
            continue
        median = statistics.median(values)
        rows.append(
            {
                "m": width,
                "rounds": len(values),
                "median_ms": median,
                "mean_ms": statistics.fmean(values),
                "sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                "increment_ms": None if previous is None else median - previous,
                "groups": GROUPS.get(width),
            }
        )
        previous = median
    return rows


def structural_checks(table: list[dict]) -> dict:
    """Identities the group model must satisfy, and a linear model cannot.

    Writing `round(M) = F + h*(M-1) + sum_g pass(NA_g)`, the two-group widths
    give two independent readings of the same difference:

        round(7) - round(6) = h + pass(4) - pass(3)
        round(8) - round(7) = h + pass(4) - pass(3)

    and the group-doubling pairs give two independent readings of the fixed
    per-round term:

        round(6) - 2*round(3) = h - F
        round(8) - 2*round(4) = h - F

    Neither identity is available to a linear law, so agreement is evidence for
    the group structure rather than a restatement of it.
    """
    median = {row["m"]: row["median_ms"] for row in table}
    result: dict = {}
    if {6, 7, 8} <= median.keys():
        first = median[7] - median[6]
        second = median[8] - median[7]
        result["two_group_step_a_ms"] = first
        result["two_group_step_b_ms"] = second
        result["two_group_step_disagreement_ms"] = abs(first - second)
    if {3, 6} <= median.keys():
        result["fixed_term_from_3_6_ms"] = HEAD_STEP_MS - (
            median[6] - 2 * median[3]
        )
    if {4, 8} <= median.keys():
        result["fixed_term_from_4_8_ms"] = HEAD_STEP_MS - (
            median[8] - 2 * median[4]
        )
    increments = {
        row["m"]: row["increment_ms"]
        for row in table
        if row["increment_ms"] is not None
    }
    if increments:
        crossing = increments.get(6)
        others = [v for m, v in increments.items() if m != 6]
        if crossing is not None and others:
            result["crossing_increment_ms"] = crossing
            result["largest_other_increment_ms"] = max(others)
            result["crossing_ratio"] = crossing / max(others)
    return result


def histogram(rounds: list[tuple[int, float]]) -> dict:
    counts: dict[int, int] = {}
    for width, _ in rounds:
        counts[width] = counts.get(width, 0) + 1
    total = sum(counts.values())
    multi = sum(n for m, n in counts.items() if GROUPS.get(m, 1) >= 2)
    return {
        "rounds": total,
        "histogram": dict(sorted(counts.items())),
        "multi_pass_rounds": multi,
        "multi_pass_fraction": multi / total if total else 0.0,
        "mean_m": sum(m * n for m, n in counts.items()) / total if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legs", nargs="+")
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--json")
    args = parser.parse_args()

    pooled: list[tuple[int, float]] = []
    per_leg = {}
    for leg in args.legs:
        rounds = load_rounds(leg)
        per_leg[leg] = histogram(rounds)
        pooled.extend(rounds)

    table = cost_table(pooled, args.min_rounds)
    checks = structural_checks(table)

    print("=== per-round cost against verified width M ===")
    print(
        f"{'M':>3}{'groups':>8}{'rounds':>8}{'median ms':>11}"
        f"{'sd':>8}{'increment':>11}"
    )
    for row in table:
        increment = (
            "-" if row["increment_ms"] is None else f"{row['increment_ms']:+.2f}"
        )
        print(
            f"{row['m']:>3}{row['groups']:>8}{row['rounds']:>8}"
            f"{row['median_ms']:>11.2f}{row['sd_ms']:>8.2f}{increment:>11}"
        )

    print("\n=== identities the group model must satisfy ===")
    for name, value in checks.items():
        print(f"  {name:<34}{value:>10.3f}")

    print("\n=== width histogram per leg ===")
    for leg, stats in per_leg.items():
        print(
            f"  {leg:<44} n={stats['rounds']:<5} mean M {stats['mean_m']:.3f}"
            f"  multi-pass {stats['multi_pass_fraction']:.3f}"
        )
        print(f"  {'':<44} {stats['histogram']}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {"table": table, "checks": checks, "legs": per_leg},
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
