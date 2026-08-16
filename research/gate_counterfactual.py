#!/usr/bin/env python3
"""Streak-gate counterfactual and EMA separation from one traced MTP run.

The deep cap opens when `fullAcceptStreak >= segmentedStreakGate`. The trace
records the streak and the per-position acceptance EMA that the policy saw on
entry to every round, so the rounds a looser gate WOULD have opened can be
counted exactly, and the EMA a threshold rule would have keyed on can be
scored against what the round actually achieved.

This is bookkeeping over an existing trace, not a prediction: a looser gate
changes the trajectory from its first divergent round onward. It bounds how
much room a gate sweep can possibly have.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
KV_RE = re.compile(r"([a-z_0-9]+)=([^\s]+)")


def parse(trace: str):
    phase = None
    rounds = []
    for line in open(trace):
        if "measuring native-MTP decode" in line:
            phase = "mtp"
        elif "measuring the TRUE serial control" in line:
            phase = "serial"
        elif "generating the MTP reference" in line:
            phase = "reference"
        m = ROUND_RE.search(line)
        if not m or phase != "mtp":
            continue
        kv = dict(KV_RE.findall(m.group(4)))
        ema = [float(x) for x in kv.get("ema_in", "").split(",") if x]
        rounds.append(
            {
                "round": int(m.group(1)),
                "depth": int(m.group(2)),
                "accepted": int(m.group(3)),
                "streak_in": int(kv.get("streak_in", -1)),
                "cap_in": int(kv.get("cap", -1)),
                "round_us": int(kv.get("round_us", 0)),
                "ema_in": ema,
            }
        )
    return rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--shallow-cap", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    rounds = parse(args.trace)
    shallow = [r for r in rounds if r["depth"] <= args.shallow_cap]
    deep = [r for r in rounds if r["depth"] > args.shallow_cap]

    report = {
        "trace": args.trace,
        "round_count": len(rounds),
        "depth_histogram": dict(sorted(Counter(r["depth"] for r in rounds).items())),
        "streak_in_histogram": dict(
            sorted(Counter(r["streak_in"] for r in rounds).items())),
        "gate_counterfactual": {},
        "ema_separation": {},
        "throughput": {},
    }

    for gate in (1, 2, 3, 4):
        would_open = [r for r in rounds if r["streak_in"] >= gate]
        newly = [r for r in would_open if r["depth"] <= args.shallow_cap]
        report["gate_counterfactual"][str(gate)] = {
            "rounds_opened": len(would_open),
            "share_opened": round(len(would_open) / len(rounds), 4),
            "newly_opened_vs_observed": len(newly),
            "newly_opened_rounds": [r["round"] for r in newly],
        }

    # Did the EMA the policy saw on entry separate the rounds that went on to
    # accept everything from the rounds that rejected?
    for index in range(0, 8):
        clean = [
            r["ema_in"][index]
            for r in rounds
            if len(r["ema_in"]) > index and r["accepted"] == r["depth"]
        ]
        dirty = [
            r["ema_in"][index]
            for r in rounds
            if len(r["ema_in"]) > index and r["accepted"] < r["depth"]
        ]
        if not clean or not dirty:
            continue
        report["ema_separation"][f"ema[{index}]"] = {
            "full_accept_rounds": len(clean),
            "full_accept_mean": round(statistics.fmean(clean), 4),
            "full_accept_min": round(min(clean), 4),
            "reject_rounds": len(dirty),
            "reject_mean": round(statistics.fmean(dirty), 4),
            "reject_max": round(max(dirty), 4),
            "separated": min(clean) > max(dirty),
        }

    def summarise(name, group):
        if not group:
            return
        emitted = sum(r["accepted"] + 1 for r in group)
        micros = sum(r["round_us"] for r in group)
        report["throughput"][name] = {
            "rounds": len(group),
            "emitted_tokens": emitted,
            "mean_round_us": round(micros / len(group), 1),
            "us_per_token": round(micros / emitted, 1),
            "accept_rate": round(
                sum(r["accepted"] for r in group)
                / max(1, sum(r["depth"] for r in group)),
                4,
            ),
        }

    summarise("shallow", shallow)
    summarise("deep", deep)
    summarise("all", rounds)

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
