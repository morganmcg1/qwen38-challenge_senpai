#!/usr/bin/env python3
"""Predict the fixed-depth curve from a measured cost table and acceptance.

    usage: research/e170_depth_predict.py --cost research/out/e168/round_cost.json \
             --pinned research/out/e168/p7/english research/out/e168/p7/benchfixture

Both inputs are measurements, not fits:

  * the cost table is per-round wall time by verified width `M`, taken from
    untraced legs by `research/e170_round_cost.py`;
  * the acceptance profile is the per-position conditional accept rate from the
    pinned depth-7 census, which is the only uncensored source for positions
    the shipped clamp refuses to draft.

`ms/token = round_ms(M) / (1 + sum_k S_k)` with `S_k` the chain survival, so
the only modelled step is the assumption that acceptance at a position does not
depend on how many drafts were offered. That assumption is stated, not hidden:
if offering fewer drafts raises acceptance, every shallow arm below is
pessimistic.
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e168_margin_calibration import (  # noqa: E402
    chain_survival,
    load_legs,
    position_profile,
)


def curve(cost: dict[int, float], profile: list[float], max_depth: int = 8):
    rows = []
    for depth in range(max_depth + 1):
        width = 1 + depth
        if width not in cost:
            continue
        accepted = sum(chain_survival(profile, k) for k in range(depth))
        tokens = 1.0 + accepted
        rows.append(
            {
                "depth": depth,
                "m": width,
                "round_ms": cost[width],
                "tokens_per_round": tokens,
                "ms_per_token": cost[width] / tokens,
            }
        )
    best = min(row["ms_per_token"] for row in rows)
    for row in rows:
        row["excess_pct"] = (row["ms_per_token"] / best - 1.0) * 100.0
        row["best"] = row["ms_per_token"] == best
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", required=True)
    parser.add_argument("--pinned", nargs="+", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()

    table = json.loads(pathlib.Path(args.cost).read_text())["table"]
    cost = {row["m"]: row["median_ms"] for row in table}
    print("measured round ms by M: " + "  ".join(
        f"{m}:{ms:.2f}" for m, ms in sorted(cost.items())
    ))

    report = {}
    for directory in args.pinned:
        label = pathlib.Path(directory).name
        legs = load_legs([directory], "pinned")
        profile = [row["q"] for row in position_profile(legs[0]["rounds"])]
        rows = curve(cost, profile)
        report[label] = {"profile": profile, "curve": rows}
        print(f"\n--- {label}: p_d = " + " ".join(f"{p:.4f}" for p in profile))
        print(
            f"  {'depth':>6}{'M':>4}{'groups':>8}{'round ms':>10}"
            f"{'tok/rnd':>9}{'ms/token':>10}{'vs best':>10}"
        )
        for row in rows:
            groups = 2 if row["m"] >= 6 else (3 if row["m"] >= 9 else 1)
            mark = "  <-- best" if row["best"] else ""
            print(
                f"  {row['depth']:>6}{row['m']:>4}{groups:>8}"
                f"{row['round_ms']:>10.2f}{row['tokens_per_round']:>9.3f}"
                f"{row['ms_per_token']:>10.3f}{row['excess_pct']:>+9.1f}%{mark}"
            )

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
