#!/usr/bin/env python3
"""Test whether each extra draft pays for itself, using fully-accepted rounds.

Restricting to rounds where accepted == depth removes acceptance-outcome bias,
so the remaining per-width differences are pure cost. A draft depth is worth
adding when its marginal round cost is below the running microseconds per token
of the next-shallower depth; otherwise that depth is strictly dominated even at
100% acceptance.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
KV_RE = re.compile(r"(\w+)=([0-9.]+)")


def load(path, drop_first=True):
    out = []
    for line in open(path, errors="replace"):
        m = ROUND_RE.search(line)
        if not m:
            continue
        kv = {k: float(v) for k, v in KV_RE.findall(m.group(4)) if k != "ema_in"}
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), kv))
    return [r for r in out if r[0] > 1] if drop_first else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    warm = load(args.trace)
    full = defaultdict(list)
    for _, d, acc, kv in warm:
        if acc == d:
            full[d].append(kv["round_us"])

    table = []
    prev = None
    for d in sorted(full):
        mean = st.mean(full[d])
        tokens = d + 1
        row = {
            "depth": d,
            "width": d + 1,
            "n": len(full[d]),
            "round_us": mean,
            "best_case_tokens": tokens,
            "us_per_token": mean / tokens,
        }
        if prev is not None:
            marginal = (mean - prev[1]) / (d - prev[0])
            running = prev[1] / (prev[0] + 1)
            row["marginal_us_per_draft"] = marginal
            row["running_us_per_token_at_prev_depth"] = running
            row["pays_for_itself"] = marginal < running
        table.append(row)
        prev = (d, mean)

    best = min(table, key=lambda r: r["us_per_token"])
    report = {
        "fully_accepted_only": True,
        "table": table,
        "cost_optimal_depth": best["depth"],
        "cost_optimal_width": best["width"],
        "cost_optimal_us_per_token": best["us_per_token"],
        "dominated_depths": [
            r["depth"] for r in table if r.get("pays_for_itself") is False
        ],
    }

    print("FULLY-ACCEPTED ROUNDS ONLY (pure cost, no acceptance-outcome bias)")
    print("depth width  n  round_ms  tokens  us/token   marginal_ms  pays?")
    for r in table:
        marg = r.get("marginal_us_per_draft")
        marg_s = f"{marg/1000:11.2f}" if marg is not None else " " * 11
        pays = r.get("pays_for_itself")
        pays_s = "" if pays is None else ("  YES" if pays else "   NO")
        print(
            f"{r['depth']:5d} {r['width']:5d} {r['n']:3d} {r['round_us']/1000:9.1f}"
            f" {r['best_case_tokens']:7d} {r['us_per_token']:9.0f} {marg_s} {pays_s}"
        )
    print()
    print(f"cost-optimal depth = {best['depth']} (width {best['width']})")
    print(f"dominated depths   = {report['dominated_depths']}")

    if args.out:
        with open(args.out, "w") as h:
            json.dump(report, h, indent=2)


if __name__ == "__main__":
    main()
