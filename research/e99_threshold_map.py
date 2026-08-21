#!/usr/bin/env python3
"""Build the combined margin-gate threshold map across offered caps.

Reads every rung-5 ABBA artifact and the rung-6/7 session artifact, groups the
gated legs by (offered cap, threshold), and prints the ranked-curve gain of each
cell against the unchanged off baseline measured at the same cap.

Usage:
  python3 research/e99_threshold_map.py [--json research/e99-artifacts/threshold-map.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

ART = pathlib.Path(__file__).resolve().parent / "e99-artifacts"

RUNG5 = {
    8: "rung5-abba.json",
    6: "rung5-cap6.json",
    5: "rung5-cap5.json",
    4: "rung5-cap4.json",
}
RUNG5_OLD_BUILD = {4: "rung5-cap4-w945d67b3.json"}
RUNG67 = "rung67.json"


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def legs_of(doc: dict) -> list[dict]:
    if "legs" in doc:
        return list(doc["legs"].values())
    out: list[dict] = []
    for key in ("off", "on", "sweep"):
        out.extend(doc.get(key, []))
    return out


DEFAULT_DEPTH = 3


def collect() -> tuple[dict, dict]:
    """Return (baselines, cells) keyed by (cap, worker) and (cap, threshold, depth, worker)."""
    baselines: dict[tuple[int, str], list[float]] = {}
    cells: dict[tuple[int, float, int, str], list[dict]] = {}

    sources = [(cap, load(name)) for cap, name in RUNG5.items()]
    sources += [(cap, load(name)) for cap, name in RUNG5_OLD_BUILD.items()]
    sources += [(None, load(RUNG67))]

    for _, doc in sources:
        for leg in legs_of(doc):
            cap = int(leg["offered_cap"])
            worker = leg["worker_sha256"][:8]
            ranked = float(leg["ranked_us_per_token"])
            if leg["gate"] == "off":
                baselines.setdefault((cap, worker), []).append(ranked)
            else:
                depth = int(leg.get("gate_depth", DEFAULT_DEPTH))
                cells.setdefault((cap, float(leg["threshold"]), depth, worker), []).append(leg)
    return baselines, cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ART / "threshold-map.json"))
    args = ap.parse_args()

    baselines, cells = collect()

    base_mean = {k: statistics.fmean(v) for k, v in baselines.items()}
    print("Off baselines (ranked us/token, mean of replicates):")
    for (cap, worker), vals in sorted(baselines.items()):
        print(f"  cap {cap}  worker {worker}  n={len(vals)}  {statistics.fmean(vals):10.1f}")

    rows = []
    for (cap, thr, depth, worker), legs in sorted(cells.items()):
        base = base_mean.get((cap, worker))
        if base is None:
            continue
        ranked = statistics.fmean(float(x["ranked_us_per_token"]) for x in legs)
        rows.append(
            {
                "cap": cap,
                "threshold": thr,
                "gate_depth": depth,
                "worker": worker,
                "n": len(legs),
                "legs": [x["leg"] for x in legs],
                "baseline_ranked_us": base,
                "ranked_us": ranked,
                "gain_pct": (base - ranked) / base * 100.0,
                "fired_share": statistics.fmean(float(x["fired_share"]) for x in legs),
                "mean_width": statistics.fmean(float(x["mean_width"]) for x in legs),
            }
        )

    print("\n| cap | threshold | depth | worker | n | fired share | mean width | ranked us/tok | gain % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["cap"], r["threshold"], r["gate_depth"], r["worker"])):
        print(
            f"| {r['cap']} | {r['threshold']:g} | {r['gate_depth']} | {r['worker']} | {r['n']} |"
            f" {r['fired_share']*100:5.1f} % | {r['mean_width']:.3f} |"
            f" {r['ranked_us']:9.1f} | {r['gain_pct']:+6.3f} |"
        )

    best = {}
    for r in rows:
        if r["gate_depth"] != DEFAULT_DEPTH:
            continue
        cur = best.get(r["cap"])
        if cur is None or r["gain_pct"] > cur["gain_pct"]:
            best[r["cap"]] = r
    print(f"\nBest threshold per cap at shipped depth {DEFAULT_DEPTH}:")
    for cap in sorted(best):
        r = best[cap]
        print(
            f"  cap {cap}: t={r['threshold']:g} gain {r['gain_pct']:+.3f} %"
            f" (shipped t=9.4375 default)"
        )

    out = {"baselines": {f"{c}|{w}": v for (c, w), v in base_mean.items()}, "cells": rows}
    pathlib.Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
