#!/usr/bin/env python3
"""Realised positionAcceptEMA and fullAcceptStreak distributions from an MTP trace.

The deep-round gate is currently `fullAcceptStreak >= segmentedStreakGate`.
An EMA-based gate would instead read `positionAcceptEMA[i]`.  Whether that is a
different gate at all depends on the *measured* joint distribution of the two,
which is what this reports: any EMA threshold that never separates the rounds
the streak gate separates is a no-op dressed up as a new mechanism.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
PHASE_MARKERS = (
    ("reference", "generating the MTP reference rows"),
    ("serial", "measuring the TRUE serial control"),
    ("mtp", "measuring native-MTP decode"),
)


def parse(path: str) -> list[dict]:
    rounds: list[dict] = []
    phase = "reference"
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            for name, marker in PHASE_MARKERS:
                if marker in line:
                    phase = name
            m = ROUND_RE.search(line)
            if not m:
                continue
            rec: dict = {"phase": phase, "round": int(m.group(1))}
            for k, v in KV_RE.findall(m.group(2)):
                if k == "ema_in":
                    continue
                try:
                    rec[k] = int(v)
                except ValueError:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
            ema = re.search(r"ema_in=([0-9.,eE+-]+)", m.group(2))
            if ema:
                rec["ema"] = [float(x) for x in ema.group(1).split(",") if x]
            rounds.append(rec)
    return [r for r in rounds if r["phase"] == "mtp"]


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[min(n - 1, int(p * n))]
    return {
        "n": n,
        "min": s[0],
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--gate", type=int, default=3)
    ap.add_argument("--ema-index", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    rounds = parse(args.trace)
    out: dict = {"label": args.label, "rounds": len(rounds), "streak_gate": args.gate}

    streaks = [r["streak_in"] for r in rounds if "streak_in" in r]
    out["streak_in_histogram"] = dict(sorted(Counter(streaks).items()))
    out["streak_in"] = quantiles([float(s) for s in streaks])
    out["streak_gate_open_rate"] = (
        sum(1 for s in streaks if s >= args.gate) / len(streaks) if streaks else 0.0
    )

    idx = args.ema_index
    per_index = {}
    for i in range(8):
        vals = [r["ema"][i] for r in rounds if r.get("ema") and len(r["ema"]) > i]
        per_index[i] = quantiles(vals)
    out["ema_in_by_index"] = per_index

    # Would an EMA threshold ever disagree with the streak gate?
    joint = []
    for r in rounds:
        if "streak_in" not in r or not r.get("ema") or len(r["ema"]) <= idx:
            continue
        joint.append((r["streak_in"], r["ema"][idx], r.get("d"), r.get("acc")))
    out["ema_index_used"] = idx

    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    sep = {}
    for t in thresholds:
        streak_open = [s >= args.gate for s, _, _, _ in joint]
        ema_open = [e >= t for _, e, _, _ in joint]
        disj = [a or b for a, b in zip(streak_open, ema_open)]
        sep[f"{t:.2f}"] = {
            "ema_alone_open_rate": sum(ema_open) / len(joint) if joint else 0.0,
            "disjunction_open_rate": sum(disj) / len(joint) if joint else 0.0,
            "rounds_ema_opens_that_streak_closes": sum(
                1 for a, b in zip(streak_open, ema_open) if b and not a
            ),
            "rounds_streak_opens_that_ema_closes": sum(
                1 for a, b in zip(streak_open, ema_open) if a and not b
            ),
        }
    out["ema_threshold_separation"] = sep
    out["streak_open_rate_reference"] = (
        sum(1 for s, _, _, _ in joint if s >= args.gate) / len(joint) if joint else 0.0
    )
    out["ema_at_index_when_streak_closed"] = quantiles(
        [e for s, e, _, _ in joint if s < args.gate]
    )
    out["ema_at_index_when_streak_open"] = quantiles(
        [e for s, e, _, _ in joint if s >= args.gate]
    )

    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
