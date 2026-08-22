#!/usr/bin/env python3
"""E87 rung 1 decision: price every screened cell with MEASURED time.

`e87_screen.py` prices a cell in bytes, because bytes are all an offline
retention screen can see. `e87_gather_bench.py` then measures what those
bytes actually cost on this GPU. The two disagree, and they disagree in a
direction the byte model cannot represent: arm C splits one dense read into
two dispatches, and each dispatch carries about 100 us of fixed cost that no
byte count predicts.

This joins the two and applies the assignment's decision rule to the WORST
domain, as advisor feedback 1 requires:

    predicted score gain = measured stage-1 time saving
                           - 206.6 * worst-domain net miss rate

Timing depends on `K`, `rowsPerCluster` and the probe count, not on which
clustering rule produced the table, so a bench cell is matched to a screen
cell by `(K, p)` alone.

Usage:
  research/e87_decide.py [--screen research/e87-screen.json]
                         [--bench research/e87-gather-bench.json]
                         [--out research/e87-decision.json] [--top 20]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Finding 69 exchange rate, superseding the earlier 206.6 from E82 rung 0.
# Keep this equal to `e87_head.MISS_TO_SCORE_PCT` (E133 feedback F1 section 7).
MISS_TO_SCORE_PCT = 203.0
ARM_C = re.compile(r"armC-(?P<rule>\w+)-K(?P<k>\d+)-p(?P<p>[\d.]+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="research/e87-screen.json")
    ap.add_argument("--bench", default="research/e87-gather-bench.json")
    ap.add_argument("--out", default="research/e87-decision.json")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    screen = json.loads(Path(args.screen).read_text())
    bench = json.loads(Path(args.bench).read_text())
    timing = {(c["k"], round(c["p"], 6)): c for c in bench["cells"]}

    rows = []
    for cell in screen["cells"]:
        name = cell["arm"]
        if name == "shipped-g64":
            continue
        if name == "armG-g128":
            measured = bench["armG_g128"]["predicted_pct_measured_time"]
        else:
            m = ARM_C.match(name)
            if not m:
                continue
            key = (int(m["k"]), round(float(m["p"]), 6))
            if key not in timing:
                continue
            measured = timing[key]["predicted_pct_measured_time"]
        worst = cell["net_miss_worst_domain"]
        rows.append({
            "arm": name,
            "m": cell["m"],
            "net": cell["net_miss_vs_shipped"],
            "worst_domain_net": worst,
            "worst_work_net": cell["net_miss_worst_work"],
            "byte_model_gain_pct": cell["score_gain_pct"],
            "measured_gain_pct": measured,
            "byte_model_worst_pct": cell["predicted_worst_domain_pct"],
            "measured_worst_pct": measured - MISS_TO_SCORE_PCT * worst,
            "by_domain": cell["by_domain"],
        })

    rows.sort(key=lambda r: -r["measured_worst_pct"])
    print(f"{'arm':<32}{'m':>9}{'wrstDom':>10}"
          f"{'byteGain':>10}{'measGain':>10}{'bytePred':>10}{'measPred':>10}")
    for r in rows[: args.top]:
        print(f"{r['arm']:<32}{r['m']:9.5f}{r['worst_domain_net']:10.2e}"
              f"{r['byte_model_gain_pct']:10.3f}{r['measured_gain_pct']:10.3f}"
              f"{r['byte_model_worst_pct']:10.3f}{r['measured_worst_pct']:10.3f}")

    best = rows[0]
    print(f"\nwinner on the worst domain: {best['arm']}")
    print(f"  byte model said  +{best['byte_model_worst_pct']:.3f} %")
    print(f"  measured time    +{best['measured_worst_pct']:.3f} %")
    print(f"  overstatement    {best['byte_model_worst_pct'] - best['measured_worst_pct']:+.3f} pp")
    armg = next(r for r in rows if r["arm"] == "armG-g128")
    print(f"armG-g128 measured +{armg['measured_worst_pct']:.3f} % "
          f"(worst-domain net {armg['worst_domain_net']:.2e}, "
          f"break-even 1.45e-3)")

    Path(args.out).write_text(json.dumps(
        {"samples": screen["samples"], "ranked": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
