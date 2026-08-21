#!/usr/bin/env python3
"""How much does the arm-C round saving depend on drafts per round?

The published score is (raw_beagle + raw_essays) / 2, and those two prompts run
different mean draft counts. A per-draft mechanism can therefore be worth a
different amount on each. This script uses the same trace records and the same
absolute host gate as `e87_paired.py`, groups the paired rounds by the number
of drafts actually proposed, and fits both the base round time and the
arm-minus-base saving against that draft count. It then reports the predicted
per-round gain at named draft counts.

Both fits are local, single-fixture, and directional. They cannot replace a
per-prompt measurement on the ranked host.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

from e87_paired import OUT, arm_of, rounds


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def fit(prefix: str, base_arm: str, arm: str) -> dict:
    """Fit base round time and arm saving against drafts proposed per round."""
    legs: dict[str, list[list[dict]]] = defaultdict(list)
    for path in sorted(OUT.iterdir()):
        if not path.name.startswith(prefix + "-"):
            continue
        legs[arm_of(path.name)].append(rounds(path.name))

    for name in (base_arm, arm):
        if name not in legs:
            raise SystemExit(f"session has no {name} legs")

    base, cand = legs[base_arm], legs[arm]
    n = min(min(len(r) for r in base), min(len(r) for r in cand))

    points = []
    for i in range(n):
        b = [rs[i] for rs in base if rs[i]["clean"]]
        a = [rs[i] for rs in cand if rs[i]["clean"]]
        if not b or not a:
            continue
        drafts = st.median(r["d"] for r in b)
        if drafts <= 0:
            continue
        bm = st.median(r["round_us"] for r in b)
        am = st.median(r["round_us"] for r in a)
        points.append({"round": i, "drafts": drafts, "base_us": bm, "saving_us": bm - am})

    if len(points) < 3:
        raise SystemExit("too few comparable rounds to fit")

    xs = [p["drafts"] for p in points]
    base_int, base_slope = ols(xs, [p["base_us"] for p in points])
    sav_int, sav_slope = ols(xs, [p["saving_us"] for p in points])

    grouped: dict[float, list[dict]] = defaultdict(list)
    for p in points:
        grouped[p["drafts"]].append(p)
    by_depth = {}
    for depth in sorted(grouped):
        group = grouped[depth]
        bm = st.median(p["base_us"] for p in group)
        sm = st.median(p["saving_us"] for p in group)
        by_depth[str(depth)] = {
            "rounds": len(group),
            "median_base_us": bm,
            "median_saving_us": sm,
            "median_gain_pct": 100.0 * sm / bm,
        }

    return {
        "prefix": prefix,
        "harness": "local",
        "base_arm": base_arm,
        "arm": arm,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "rounds_used": len(points),
        "drafts_observed_min": min(xs),
        "drafts_observed_max": max(xs),
        "base_round_us_fit": {"intercept_us": base_int, "slope_us_per_draft": base_slope},
        "saving_us_fit": {"intercept_us": sav_int, "slope_us_per_draft": sav_slope},
        "by_observed_depth": by_depth,
    }


def round_gain_pct(model: dict, drafts: float) -> float:
    b = model["base_round_us_fit"]
    s = model["saving_us_fit"]
    base_us = b["intercept_us"] + b["slope_us_per_draft"] * drafts
    sav_us = s["intercept_us"] + s["slope_us_per_draft"] * drafts
    return 100.0 * sav_us / base_us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--base", default="declared")
    ap.add_argument("--arm", default="derived25")
    ap.add_argument("--draft-counts", default="beagle=4.382,essays=5.087,republic=4.989")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    doc = fit(args.prefix, args.base, args.arm)
    predicted = {}
    for item in args.draft_counts.split(","):
        name, _, value = item.partition("=")
        d = float(value)
        b = doc["base_round_us_fit"]
        s = doc["saving_us_fit"]
        predicted[name] = {
            "drafts_per_round": d,
            "predicted_base_round_us": b["intercept_us"] + b["slope_us_per_draft"] * d,
            "predicted_saving_us": s["intercept_us"] + s["slope_us_per_draft"] * d,
            "predicted_round_gain_pct": round_gain_pct(doc, d),
            "extrapolated_below_observed": d < doc["drafts_observed_min"],
        }
    doc["predicted"] = predicted

    text = json.dumps(doc, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
