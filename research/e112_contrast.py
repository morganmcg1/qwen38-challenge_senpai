#!/usr/bin/env python3
"""Drift-corrected arm contrasts for the E112 rung-1 ABBA session.

    usage: research/e112_contrast.py [--report research/out/e112-rung1-report.json]

The session order is `b k b k b k b k b`, so every candidate leg has an `off`
leg on each side. Contrasting a candidate leg against the mean of its two
neighbours removes any monotone session drift to first order and is a paired
statistic, unlike the raw arm-mean difference.

Also reports the per-leg spread of absolute candidate MTP seconds per token,
which is the local measurement floor this session can actually resolve.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st


def neighbour_contrast(legs, key):
    """Percent difference of each `on` leg against its two `off` neighbours."""
    out = []
    for index, leg in enumerate(legs):
        if leg["arm"] != "on":
            continue
        neighbours = [
            legs[j][key]
            for j in (index - 1, index + 1)
            if 0 <= j < len(legs) and legs[j]["arm"] == "off"
        ]
        base = st.mean(neighbours)
        out.append(100.0 * (leg[key] - base) / base)
    return out


def report_line(label, values):
    mean = st.mean(values)
    sd = st.stdev(values)
    se = sd / math.sqrt(len(values))
    print(f"{label:<34}{mean:+9.4f} %  sd {sd:6.4f}  se {se:6.4f}  "
          f"t {mean / se:+5.2f}  n {len(values)}  "
          f"{[round(value, 3) for value in values]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",
                        default="research/out/e112-rung1-report.json")
    args = parser.parse_args()

    report = json.load(open(args.report, encoding="utf-8"))
    legs = sorted(report["legs"], key=lambda leg: leg["position"])
    for leg in legs:
        crossing = leg["mtp_crossings"][0]
        leg["cross_us"] = crossing["round_us"]
        leg["cross_ratio"] = crossing["round_us"] / leg["mtp_median_round_us"]

    print("-- neighbour-averaged local contrast, candidate arm minus base --")
    for key, label in (
        ("mtp_s_per_tok", "absolute candidate MTP s/token"),
        ("speedup", "local serial-to-MTP ratio"),
        ("mtp_median_round_us", "median round us"),
        ("cross_us", "crossing round r78 us"),
        ("cross_ratio", "r78 us / same-leg median"),
    ):
        report_line(label, neighbour_contrast(legs, key))

    off = [leg["mtp_s_per_tok"] for leg in legs if leg["arm"] == "off"]
    on = [leg["mtp_s_per_tok"] for leg in legs if leg["arm"] == "on"]
    pooled_sd = math.sqrt(
        (st.stdev(off) ** 2 * (len(off) - 1) + st.stdev(on) ** 2 * (len(on) - 1))
        / (len(off) + len(on) - 2))
    grand = st.mean(off + on)
    leg_pct = 100.0 * pooled_sd / grand
    leg_us = 512 * grand * 1e6
    cross_delta = st.mean([leg["cross_us"] for leg in legs if leg["arm"] == "on"]) \
        - st.mean([leg["cross_us"] for leg in legs if leg["arm"] == "off"])

    print("\n-- local measurement floor from this session --")
    print(f"per-leg sd, off arm            {100 * st.stdev(off) / st.mean(off):.4f} %")
    print(f"per-leg sd, on arm             {100 * st.stdev(on) / st.mean(on):.4f} %")
    print(f"pooled within-arm per-leg sd   {leg_pct:.4f} %")
    print(f"arm-mean-difference SE (5 v 4) "
          f"{leg_pct * math.sqrt(1 / len(off) + 1 / len(on)):.4f} %")
    print(f"leg wall time                  {leg_us:,.0f} us")
    print(f"crossing-round arm delta       {cross_delta:+,.0f} us "
          f"= {100 * cross_delta / leg_us:.5f} % of the leg")


if __name__ == "__main__":
    main()
