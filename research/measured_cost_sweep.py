#!/usr/bin/env python3
"""Re-derive the deep-round gate sweep using the *measured* Run C cost model.

The earlier sweep in `streak_gate_analysis.py` scored trajectories with the
shipped cost model (`1 + d * headStepCostRatio`, h = 0.20). Run C measured the
real thing on this host:

    round_us ~= FIXED_MS + SLOPE_MS * width,   width = depth + 1
    one serial target forward = SERIAL_FORWARD_MS

In serial-forward units that is `0.186 + 0.343 * d` per draft plus a 0.343
base, i.e. a marginal-to-base cost ratio of 0.648 rather than the 0.20 the
shipped schedule assumes. Deep rounds are therefore far more expensive than
the schedule believes, which flips the sign of several sweep conclusions.

Cost is charged on the *requested* width (depth + 1) because the verify batch
evaluates every drafted row before acceptance is known.
"""
from __future__ import annotations

import argparse
import json
import statistics

from streak_gate_analysis import simulate_trajectory

FIXED_MS = 12.21
SLOPE_MS = 22.49
SERIAL_FORWARD_MS = 65.58

# Run C realised per-position acceptance (256-token window, cap 8, all depths
# reachable). Positions 0-5 never rejected; the curve collapses at i=7, which is
# the only index where the shipped 0.98^i prior is optimistic.
MEASURED_CURVE = [1.0000, 0.9737, 1.0000, 0.9722, 1.0000, 1.0000, 0.9444, 0.8462]


def round_ms(depth: int, fixed: float, slope: float) -> float:
    return fixed + slope * (depth + 1)


def score(traj: dict, fixed: float, slope: float, serial_ms: float) -> dict:
    total_ms = sum(round_ms(r["depth"], fixed, slope) for r in traj["trajectory"])
    tokens = traj["trajectory"][-1]["emitted"]
    ms_per_token = total_ms / tokens
    return {
        "tokens": tokens,
        "rounds": traj["rounds"],
        "ms_per_token": ms_per_token,
        "raw_speedup": serial_ms / ms_per_token,
        "mean_depth": traj["mean_depth"],
        "depth_histogram": traj["depth_histogram"],
    }


def arm(label: str, p, tokens: int, seeds: int, fixed: float, slope: float,
        serial_ms: float, **kw) -> dict:
    runs = [
        score(simulate_trajectory(p=p, tokens=tokens, seed=s, **kw), fixed, slope, serial_ms)
        for s in range(seeds)
    ]
    return {
        "label": label,
        "p": p,
        "ms_per_token": statistics.fmean(r["ms_per_token"] for r in runs),
        "raw_speedup": statistics.fmean(r["raw_speedup"] for r in runs),
        "mean_depth": statistics.fmean(r["mean_depth"] for r in runs),
        "deep_occupancy": statistics.fmean(
            sum(v for k, v in r["depth_histogram"].items() if k >= 7) / r["rounds"]
            for r in runs
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acceptance", type=float, nargs="*", default=[0.96, 0.95, 0.93, 0.90])
    ap.add_argument(
        "--per-position", action="store_true",
        help="use the Run C measured per-position curve instead of an i.i.d. scalar")
    ap.add_argument(
        "--hardness", type=float, nargs="*", default=[1.0],
        help="exponent applied to the measured curve; >1 emulates harder prose prompts")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--fixed-ms", type=float, default=FIXED_MS)
    ap.add_argument("--slope-ms", type=float, default=SLOPE_MS)
    ap.add_argument("--serial-ms", type=float, default=SERIAL_FORWARD_MS)
    ap.add_argument("--out")
    args = ap.parse_args()

    common = dict(
        tokens=args.tokens, seeds=args.seeds, fixed=args.fixed_ms,
        slope=args.slope_ms, serial_ms=args.serial_ms,
    )
    arms = [
        ("gate3 cap8 (shipped base)", dict(gate=3, deep_cap=8)),
        ("gate2 cap8", dict(gate=2, deep_cap=8)),
        ("gate1 cap8", dict(gate=1, deep_cap=8)),
        ("gate3 cap7 (this PR)", dict(gate=3, deep_cap=7)),
        ("gate2 cap7", dict(gate=2, deep_cap=7)),
        ("gate1 cap7", dict(gate=1, deep_cap=7)),
        ("gate3 cap6", dict(gate=3, deep_cap=6)),
        ("gate1 cap6", dict(gate=1, deep_cap=6)),
        ("ema[4]>=0.90 OR gate3, cap8", dict(gate=3, deep_cap=8, ema_gate=0.90)),
        ("ema[4]>=0.90 OR gate3, cap7", dict(gate=3, deep_cap=7, ema_gate=0.90)),
        ("ema[4]>=0.98 OR gate3, cap7", dict(gate=3, deep_cap=7, ema_gate=0.98)),
    ]

    results = {}
    print("cost model: round_ms = %.2f + %.2f * width   serial forward = %.2f ms"
          % (args.fixed_ms, args.slope_ms, args.serial_ms))
    print("            = %.3f + %.3f * d serial-forward units (shipped model: 1 + 0.20*d)"
          % (args.fixed_ms / args.serial_ms, args.slope_ms / args.serial_ms))

    if args.per_position:
        cases = [("measured curve ^%.2f" % k, [v ** k for v in MEASURED_CURVE])
                 for k in args.hardness]
    else:
        cases = [("p = %.3f" % p, p) for p in args.acceptance]

    for name, p in cases:
        rows = [arm(label, p, **common, **kw) for label, kw in arms]
        base = rows[0]["ms_per_token"]
        print()
        if isinstance(p, list):
            print("## %s  ->  %s" % (name, ", ".join("%.3f" % v for v in p)))
            print("   (%d tokens x %d seeds)" % (args.tokens, args.seeds))
        else:
            print("## %s  (%d tokens x %d seeds)" % (name, args.tokens, args.seeds))
        print()
        print("| arm | ms/token | vs base | raw speedup | mean depth | depth>=7 occupancy |")
        print("|:--|---:|---:|---:|---:|---:|")
        for r in rows:
            r["pct_vs_base"] = 100.0 * (base - r["ms_per_token"]) / base
            print("| %s | %.3f | %+.2f%% | %.3f | %.2f | %.1f%% |" % (
                r["label"], r["ms_per_token"], r["pct_vs_base"],
                r["raw_speedup"], r["mean_depth"], 100.0 * r["deep_occupancy"]))
        results[name] = rows

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "cost_model": {
                    "fixed_ms": args.fixed_ms,
                    "slope_ms": args.slope_ms,
                    "serial_forward_ms": args.serial_ms,
                },
                "arms": results,
            }, fh, indent=2)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
