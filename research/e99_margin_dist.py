#!/usr/bin/env python3
"""Empirical pending-primary top-2 margin distribution per width cap.

The margin gate is a fixed ABSOLUTE threshold on the pending primary token's
target top-2 logit margin, so the share of rounds it fires on is a property of
the margin distribution, not of the cap. This reads that distribution straight
out of the traced legs (`m=` on every scheduled round) so the transfer of the
fired share across caps can be checked without another timed session.

    usage: research/e99_margin_dist.py TAG [TAG ...]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
ROUND_RE = re.compile(r" m=(-?[0-9.]+) ")
FIRE_RE = re.compile(r" fire=(\d) ")
DEPTH_RE = re.compile(r"round=\d+ d=(\d+) acc=(\d+) ")
THRESHOLDS = (4.0, 8.25, 9.4375, 11.5625, 16.0)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def read_leg(tag: str) -> dict:
    trace = OUT / tag / "trace.txt"
    meta = (OUT / tag / "meta.txt").read_text().splitlines()
    fields = dict(
        line.split("=", 1) for line in meta if "=" in line and " " not in line
    )
    margins: list[float] = []
    fired = 0
    widths: list[int] = []
    for line in trace.read_text().splitlines():
        found = ROUND_RE.search(line)
        if not found:
            continue
        margins.append(float(found.group(1)))
        fire = FIRE_RE.search(line)
        fired += int(fire.group(1)) if fire else 0
        depth = DEPTH_RE.search(line)
        if depth:
            widths.append(int(depth.group(1)))
    return {
        "tag": tag,
        "cap": fields.get("e99_cap", "?"),
        "gate": fields.get("e99_gate", "?"),
        "threshold": fields.get("e99_gate_threshold", "?"),
        "rounds": len(margins),
        "fired": fired,
        "margins": margins,
        "mean_width": sum(widths) / len(widths) if widths else math.nan,
    }


def main(tags: list[str]) -> int:
    legs = [read_leg(tag) for tag in tags]
    header = (
        f"{'leg':<12}{'cap':>4}{'gate':>9}{'rounds':>7}{'fired':>6}"
        f"{'share':>8}{'width':>7}"
        f"{'p05':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p95':>8}"
    )
    print(header)
    print("-" * len(header))
    for leg in legs:
        margins = leg["margins"]
        share = leg["fired"] / leg["rounds"] if leg["rounds"] else math.nan
        print(
            f"{leg['tag']:<12}{leg['cap']:>4}{leg['gate']:>9}"
            f"{leg['rounds']:>7}{leg['fired']:>6}{share:>8.3f}"
            f"{leg['mean_width']:>7.3f}"
            + "".join(
                f"{quantile(margins, q):>8.3f}"
                for q in (0.05, 0.25, 0.50, 0.75, 0.95)
            )
        )
    print()
    print("share of rounds at or below each candidate threshold")
    head = f"{'leg':<12}{'cap':>4}{'gate':>9}" + "".join(
        f"{'t=' + format(t, 'g'):>12}" for t in THRESHOLDS
    )
    print(head)
    print("-" * len(head))
    for leg in legs:
        margins = leg["margins"]
        row = f"{leg['tag']:<12}{leg['cap']:>4}{leg['gate']:>9}"
        for threshold in THRESHOLDS:
            count = sum(1 for value in margins if value <= threshold)
            row += f"{count:>5}/{leg['rounds']:<3}{count / leg['rounds']:>4.2f}"
        print(row)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
