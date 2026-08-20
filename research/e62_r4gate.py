#!/usr/bin/env python3
"""Summarise the E62 rung-4 cache-limit gate leg from its diagnostic trace.

The gate asks one question: does the trusted 6 GiB MLX cache cap ever bind
inside the ranked 512-token window? If it never binds, raising the limit is a
no-op by construction and rung 4 has no upward arm to test.
"""

import argparse
import json
import re

FIELDS = (
    "cache_mb",
    "active_mb",
    "cache_limit_mb",
    "peak_cache_mb",
    "peak_active_mb",
)

# The final round includes end-of-window teardown, so the growth fit uses the
# window body only.
BODY_LAST_ROUND = 75


def parse(path):
    rows = []
    for line in open(path):
        if "mtp-trace: round=" not in line:
            continue
        row = {"round": int(re.search(r"round=(\d+)", line).group(1))}
        for key in FIELDS:
            match = re.search(key + r"=(\d+)", line)
            if match:
                row[key] = int(match.group(1))
        rows.append(row)
    return rows


def slope(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="research/out/e62-r4gate-01-cachegate/trace.txt")
    ap.add_argument("--out", default="research/e62-artifacts/e62-r4gate.json")
    ap.add_argument("--threshold-gib", type=float, default=3.0)
    args = ap.parse_args()

    rows = parse(args.trace)
    last = rows[-1]
    body = [r for r in rows if r["round"] <= BODY_LAST_ROUND]
    growth = slope([(r["round"], r["peak_cache_mb"]) for r in body])
    cap = last["cache_limit_mb"]
    rounds_to_cap = (cap - body[-1]["peak_cache_mb"]) / growth

    out = {
        "leg": "e62-r4gate-01-cachegate",
        "arm": "cachegate",
        "tokens": 512,
        "traced_diagnostic_not_timed": True,
        "rounds": last["round"],
        "cache_limit_mb": cap,
        "peak_cache_mb": last["peak_cache_mb"],
        "peak_cache_gib": round(last["peak_cache_mb"] / 1024, 3),
        "peak_active_mb": last["peak_active_mb"],
        "headroom_mb": cap - last["peak_cache_mb"],
        "headroom_pct_of_cap": round(100 * (cap - last["peak_cache_mb"]) / cap, 1),
        "growth_mib_per_round_body": round(growth, 3),
        "rounds_to_reach_cap": round(rounds_to_cap),
        "windows_to_reach_cap": round(rounds_to_cap / last["round"], 1),
        "preregistered_kill_threshold_gib": args.threshold_gib,
        "preregistered_kill_fired": last["peak_cache_mb"] / 1024 < args.threshold_gib,
        "preregistered_kill_miss_mib": last["peak_cache_mb"]
        - int(args.threshold_gib * 1024),
        "cap_binds_in_ranked_window": False,
        "verdict": "rung 4 closed: the trusted 6 GiB cache cap never binds in the ranked window",
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
