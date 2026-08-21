#!/usr/bin/env python3
"""E106 rung 0 -- price the per-tensor trace split over the whole census round.

    usage: research/e106_round_account.py TRACE_JSON --round-us US
                                          [--m1-f US] [--m1-s US_PER_GB]

`TRACE_JSON` is the payload written by `research/e106_trace_split.py`. The
round total comes from the same census leg's phase table.

Three quantities matter to the advisor's rung 0 question:

  * the streaming rate the round actually achieves, against the 273 GB/s DRAM
    peak, which tests the "no bandwidth headroom" premise of Finding 36;
  * the gap between the measured M=5 round and the same dispatches priced at
    the M=1 law, which is the rate(NA) axis and belongs to E104; and
  * the N=5120 excess that survives inside the M=5 refit, which is E106.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import json

DRAM_PEAK_GB_S = 273.0
GPU_CORES = 20
ORDER = ("lm_head", "mlp.gate_up", "gdn.in_proj", "fa.qkv",
         "gdn.out_proj", "fa.o_proj", "mlp.down")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_json")
    ap.add_argument("--round-us", type=float, required=True)
    ap.add_argument("--m1-f", type=float, default=10.73)
    ap.add_argument("--m1-s", type=float, default=4003.3)
    args = ap.parse_args()

    payload = json.load(open(args.trace_json))
    for tag, d in payload.items():
        t = d["tensors"]
        f_us = d["fit"]["F_us"]
        s_us = d["fit"]["S_us_per_gb"]
        rnd = args.round_us
        print(f"=== {tag}   M={d['width']}   round={rnd:.0f} us")
        print(f"    refit F = {f_us:.2f} us/dispatch   S = {s_us:.1f} us/GB "
              f"= {1e6 / s_us:.1f} GB/s = "
              f"{100 * (1e6 / s_us) / DRAM_PEAK_GB_S:.1f} % of DRAM peak")

        print(f"\n  {'tensor':<14} {'n/rnd':>6} {'us/rnd':>9} {'exc/rnd':>9} "
              f"{'% round':>8} {'TG act':>7} {'TG tot':>7} {'act/core':>9} "
              f"{'M=1 law':>10}")
        tot = dict(us=0.0, exc=0.0, gb=0.0, n=0.0, m1=0.0)
        for label in ORDER:
            if label not in t:
                continue
            e = t[label]
            n = e["per_round"]
            us = n * e["mean_us"]
            exc = n * e["excess_us"]
            tg_act = e["n"] // 8
            m1 = n * (args.m1_f + e["gb"] * args.m1_s)
            tot["us"] += us
            tot["exc"] += exc
            tot["gb"] += n * e["gb"]
            tot["n"] += n
            tot["m1"] += m1
            print(f"  {label:<14} {n:6.0f} {us:9.1f} {exc:9.1f} "
                  f"{100 * exc / rnd:8.3f} {tg_act:7d} {tg_act * d['width']:7d} "
                  f"{tg_act / GPU_CORES:9.1f} {m1:10.1f}")
        print(f"  {'TOTAL':<14} {tot['n']:6.0f} {tot['us']:9.1f} "
              f"{tot['exc']:9.1f} {100 * tot['exc'] / rnd:8.3f} "
              f"{'':7} {'':7} {'':9} {tot['m1']:10.1f}")

        gross = tot["gb"] / (tot["us"] * 1e-6)
        after_f = tot["gb"] / ((tot["us"] - tot["n"] * f_us) * 1e-6)
        print(f"\n  streaming bytes per round     {tot['gb']:.4f} GB")
        print(f"  streaming time per round      {tot['us']:.0f} us "
              f"= {100 * tot['us'] / rnd:.1f} % of the round")
        print(f"  gross streaming rate          {gross:.1f} GB/s "
              f"= {100 * gross / DRAM_PEAK_GB_S:.1f} % of DRAM peak")
        print(f"  rate after removing F         {after_f:.1f} GB/s "
              f"= {100 * after_f / DRAM_PEAK_GB_S:.1f} % of DRAM peak")
        print(f"  fixed cost priced             {tot['n'] * f_us:.0f} us "
              f"= {100 * tot['n'] * f_us / rnd:.2f} % of the round")

        print(f"\n  same dispatches at the M=1 law   {tot['m1']:.0f} us")
        print(f"  M=5 measured minus M=1 law       {tot['us'] - tot['m1']:.0f} us "
              f"= {100 * (tot['us'] - tot['m1']) / rnd:.1f} % of the round "
              f"[rate(NA) axis, E104]")
        print(f"  N=5120 excess inside the M=5 refit {tot['exc']:.0f} us "
              f"= {100 * tot['exc'] / rnd:.3f} % of the round [E106]")
        for label in ("gdn.out_proj", "fa.o_proj", "mlp.down"):
            if label not in t:
                continue
            e = t[label]
            share = 100 * e["per_round"] * e["excess_us"] / tot["exc"]
            print(f"      {label:<14} {share:5.1f} % of that excess")
        print()


if __name__ == "__main__":
    main()
