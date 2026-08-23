#!/usr/bin/env python3
"""Measure how badly published-median deltas fail to add (Rules 121/128/129).

E135 F39 turns on a question the campaign keeps hitting: when two receipts
differ, can you price a mechanism by subtracting published medians?

The published median is an order statistic over eight per-prompt ratios. It is
a sum only if the same two prompts hold the middle two ranks in both receipts.
When a mechanism reshuffles the ranking, the subtraction silently prices a pair
that never occurred.

This measures the size of that error on the real board, with no knowledge of
any solver's mechanisms:

  1. Which prompts hold the median pair, and how often (Rule 123 check).
  2. For every ordered receipt pair, the HELD-PAIR delta -- the delta you get
     if you keep the first receipt's median pair and read those same two
     prompts in the second -- against the REALISED delta, which re-sorts.
     Their difference is the order-statistic error, in percentage points.

Read-only. Uses the public board only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import statistics
import urllib.request

BOARD_URL = (
    "https://api.yukon.org/api/benchmarks/"
    "5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true"
)

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def load(path: str | None) -> list[dict]:
    if path and os.path.exists(path):
        raw = json.load(open(path))
    else:
        token = os.environ["YUKON_API_TOKEN"]
        req = urllib.request.Request(
            BOARD_URL, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=120) as fh:
            raw = json.load(fh)
        if path:
            json.dump(raw, open(path, "w"))
    return raw if isinstance(raw, list) else raw.get(
        "submissions", raw.get("data", []))


def prompt_name(entry: dict) -> str | None:
    for key in ("prompt_sha256", "prompt_sha", "prompt_id", "prompt"):
        value = entry.get(key)
        if isinstance(value, str):
            return PROMPT_NAMES.get(value[:8], value[:8])
    return None


def ratios(row: dict) -> dict[str, float] | None:
    metrics = row.get("officialMetrics") or {}
    entries = metrics.get("per_prompt") or []
    if len(entries) != 8:
        return None
    out = {}
    for entry in entries:
        name = prompt_name(entry)
        value = entry.get("raw_ratio_of_means")
        if name is None or value is None:
            return None
        out[name] = value
    return out if len(out) == 8 else None


def median_pair(values: dict[str, float]) -> tuple[str, str]:
    order = sorted(values, key=lambda k: values[k])
    return order[3], order[4]


def published(values: dict[str, float]) -> float:
    lo, hi = median_pair(values)
    return (values[lo] + values[hi]) / 2.0


UPPER_SLOT_CANDIDATES = ("medicine", "essays", "republic", "botany")


def contrast(rows: list[dict], want_a: str, want_b: str) -> None:
    def find(want: str) -> dict[str, float]:
        hit = next((r for r in rows if r["id"].startswith(want)), None)
        if hit is None:
            raise SystemExit(f"no receipt with a complete block for {want}")
        return ratios(hit)

    a, b = find(want_a), find(want_b)
    base = published(a)
    lo_a, hi_a = median_pair(a)
    lo_b, hi_b = median_pair(b)

    realised = (published(b) - base) / base * 100.0
    held_b = (b[lo_a] + b[hi_a]) / 2.0
    held = (held_b - base) / base * 100.0
    worst = min(UPPER_SLOT_CANDIDATES, key=lambda k: (b[k] - a[k]) / a[k])
    r129 = (((b[lo_a] / a[lo_a]) + (b[worst] / a[worst])) / 2.0 - 1.0) * 100.0

    print(f"## Contrast {want_a} -> {want_b}")
    print(f"  median pair {want_a}   ({lo_a}, {hi_a})")
    print(f"  median pair {want_b}   ({lo_b}, {hi_b})"
          f"{'   SAME' if (lo_a, hi_a) == (lo_b, hi_b) else '   CHANGED'}")
    print(f"  realised published delta        {realised:+.4f} %")
    print(f"  held-pair delta                 {held:+.4f} %")
    print(f"  order-statistic error           {realised - held:+.4f} pp")
    print(f"  Rule 129 worst upper slot       {worst}")
    print(f"  Rule 129 reading                {r129:+.4f} %")
    print("  per prompt, candidate ladder change:")
    for name in sorted(a, key=lambda k: (b[k] - a[k]) / a[k]):
        print(f"    {name:<9} {a[name]:.6f} -> {b[name]:.6f}   "
              f"{(b[name] - a[name]) / a[name] * 100.0:+8.4f} %")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="restrict to receipts at or above this published median, so the "
             "error is measured in the regime the campaign actually works in",
    )
    ap.add_argument(
        "--show",
        nargs="*",
        default=[],
        help="print the median pair of these receipt id prefixes, so a chain "
             "of subtractions can be checked before it is trusted",
    )
    ap.add_argument(
        "--contrast",
        nargs=2,
        metavar=("FROM", "TO"),
        help="price one named contrast three ways: realised, held-pair, and "
             "the Rule 129 worst-upper-slot reading",
    )
    args = ap.parse_args()

    rows = [r for r in load(args.board) if ratios(r)]

    if args.contrast:
        contrast(rows, *args.contrast)

    if args.show:
        print("## Median pair of the named receipts")
        for want in args.show:
            hit = next((r for r in rows if r["id"].startswith(want)), None)
            if hit is None:
                print(f"  {want:<10} NOT FOUND with a complete 8-prompt block")
                continue
            values = ratios(hit)
            lo, hi = median_pair(values)
            print(f"  {want:<10} ({lo}, {hi})  lo {values[lo]:.6f}  "
                  f"hi {values[hi]:.6f}  published {published(values):.6f}")
            order = sorted(values, key=lambda k: values[k])
            rung = "  ".join(
                f"{'[' if i in (3, 4) else ' '}{k[:4]} {values[k]:.4f}"
                f"{']' if i in (3, 4) else ' '}"
                for i, k in enumerate(order))
            print(f"             {rung}")
        print()

    if args.min_score > 0.0:
        rows = [r for r in rows
                if (r.get("officialScore") or 0.0) >= args.min_score]
    data = [(r["id"][:8], ratios(r)) for r in rows]
    print(f"receipts with a complete 8-prompt block: {len(data)}"
          f"  (min_score {args.min_score})")

    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for _, values in data:
        lo, hi = median_pair(values)
        lower[lo] = lower.get(lo, 0) + 1
        upper[hi] = upper.get(hi, 0) + 1
        pairs[(lo, hi)] = pairs.get((lo, hi), 0) + 1

    print("\n## Rule 123 check: who holds the LOWER median slot")
    for name, count in sorted(lower.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<9} {count:5d}  {count / len(data) * 100:6.2f} %")

    print("\n## Who holds the UPPER median slot")
    for name, count in sorted(upper.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<9} {count:5d}  {count / len(data) * 100:6.2f} %")

    print("\n## Realised median pairs")
    for (lo, hi), count in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  ({lo}, {hi}){'':<4} {count:5d}  "
              f"{count / len(data) * 100:6.2f} %")

    rng = random.Random(args.seed)
    universe = list(itertools.combinations(range(len(data)), 2))
    sample = (universe if len(universe) <= args.pairs
              else rng.sample(universe, args.pairs))

    errors, changed, realised_all = [], 0, []
    for i, j in sample:
        _, a = data[i]
        _, b = data[j]
        realised = (published(b) - published(a)) / published(a) * 100.0
        lo, hi = median_pair(a)
        held_b = (b[lo] + b[hi]) / 2.0
        held = (held_b - published(a)) / published(a) * 100.0
        errors.append(realised - held)
        realised_all.append(realised)
        if median_pair(b) != (lo, hi):
            changed += 1

    absolute = [abs(e) for e in errors]
    absolute.sort()

    def pct(fraction: float) -> float:
        return absolute[min(len(absolute) - 1,
                            int(fraction * len(absolute)))]

    print(f"\n## Order-statistic error over {len(sample)} receipt pairs")
    print(f"  median pair CHANGES identity in {changed}/{len(sample)} pairs "
          f"({changed / len(sample) * 100:.2f} %)")
    print(f"  realised - held-pair delta, mean  {statistics.fmean(errors):+.4f} pp")
    print(f"  absolute error, median            {pct(0.50):.4f} pp")
    print(f"  absolute error, 90th percentile   {pct(0.90):.4f} pp")
    print(f"  absolute error, 99th percentile   {pct(0.99):.4f} pp")
    print(f"  absolute error, max               {absolute[-1]:.4f} pp")
    print(f"  for scale, realised delta abs mean "
          f"{statistics.fmean(abs(v) for v in realised_all):.4f} pp")
    print("\n  A held-pair subtraction is only safe where this error is small "
          "relative\n  to the effect being priced. Compare it with the at-zero "
          "MDE of 0.1154 pp.")


if __name__ == "__main__":
    main()
