#!/usr/bin/env python3
"""E89: how much of the drafting host work is charged to the scored metric?

The answer decides whether "reduce host work per drafting round" is a real
follow-up or a dead end. Host work that overlaps GPU execution costs nothing.

Primary estimate is leg level: regress each leg's mtp_seconds_per_token on its
mean host phase sum per round, then read the slope as the charged fraction.
A round-level split is reported too, but it compares rounds drawn from
different legs, so it is only a shape diagnostic.

usage: research/e89_critical_path.py PREFIX [WARMUP]
"""
import glob
import json
import os
import re
import statistics as st
import sys

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
        "d_chain_us", "readout_us", "commit_us", "upkeep_us"]
FIELDS = HOST + ["round_us"]


def legs(prefix, warmup):
    out = []
    for d in sorted(glob.glob(f"research/out/{prefix}-*")):
        trace, score = os.path.join(d, "trace.txt"), os.path.join(d, "score.json")
        if not (os.path.exists(trace) and os.path.exists(score)):
            continue
        rows = []
        for line in open(trace):
            if "round_us=" not in line:
                continue
            rec = {}
            for key in FIELDS:
                m = re.search(rf"\b{key}=([0-9.]+)", line)
                if m:
                    rec[key] = float(m.group(1))
            if len(rec) == len(FIELDS):
                rec["host"] = sum(rec[k] for k in HOST)
                rows.append(rec)
        rows = rows[warmup:]
        if not rows:
            continue
        metrics = json.load(open(score))["metrics"]
        tag = os.path.basename(d)
        out.append({"tag": tag, "arm": tag[len(prefix) + 1:].rsplit("-", 1)[0],
                    "rows": rows, "nrounds": len(rows),
                    "spt": metrics["mtp_seconds_per_token"],
                    "host_mean": st.mean(r["host"] for r in rows)})
    return out


def main():
    prefix = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    data = [L for L in legs(prefix, warmup) if L["arm"] != "bg"]
    nr = st.mean(L["nrounds"] for L in data) + warmup
    tokens = 512.0
    tpr = tokens / nr
    print(f"{len(data)} legs, {nr:.0f} rounds each, {tokens:.0f} decode tokens, "
          f"{tpr:.2f} tokens per round")

    xs = [L["host_mean"] for L in data]
    ys = [L["spt"] * 1e6 * tpr for L in data]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    vy = st.pvariance(ys)
    r2 = 1 - st.pvariance(resid) / vy if vy else float("nan")
    se = (st.pvariance(resid) * len(xs) / (len(xs) - 2) / sxx) ** 0.5

    print("\nLEG-LEVEL REGRESSION, us of scored round time per us of host work")
    print(f"  mean host phase sum per round   {mx:8.0f} us")
    print(f"  mean scored time per round      {my:8.0f} us")
    print(f"  CHARGED FRACTION (slope)        {slope:8.3f}   se {se:.3f}   "
          f"r2 {r2:.3f}")
    print("  A slope near 1 means the host work is on the critical path.")
    print("  A slope near 0 means it hides under GPU execution and is free.")

    clean = [L for L in data if L["host_mean"] < 1200]
    stuck = [L for L in data if L["host_mean"] >= 1200]
    hc = st.mean(L["host_mean"] for L in clean)
    rc = st.mean(L["spt"] for L in clean) * 1e6 * tpr
    big = st.mean(r["d_submit1_us"] + r["d_chain_us"]
                  for L in clean for r in L["rows"])
    print("\nCEILING FOR A HOST-WORK REDUCTION, clean state")
    print(f"  clean legs {len(clean)}, host {hc:.0f} us, scored round {rc:.0f} us")
    print(f"  removing ALL clean-state host work saves at most "
          f"{100 * slope * hc / rc:.3f} %")
    print(f"  removing d_submit1 and d_chain ({big:.0f} us) saves "
          f"{100 * slope * big / rc:.3f} %")

    if stuck:
        hs = st.mean(L["host_mean"] for L in stuck)
        rs = st.mean(L["spt"] for L in stuck) * 1e6 * tpr
        print("\nPRIZE IF THE HOST WORK WERE HIDDEN INSTEAD OF SHORTENED")
        print(f"  stuck legs {len(stuck)}, host {hs:.0f} us, scored round {rs:.0f} us")
        print(f"  fully hiding host work removes {100 * slope * hs / rs:.3f} % "
              f"from a stuck leg")
        print(f"  measured clean-to-stuck penalty is "
              f"{100 * (rs - rc) / rc:.3f} %")

    allrows = [r for L in data for r in L["rows"]]
    ordered = sorted(r["host"] for r in allrows)
    cut = 2.0 * st.median(ordered[: len(ordered) // 2])
    fast = [r for r in allrows if r["host"] <= cut]
    slow = [r for r in allrows if r["host"] > cut]
    print(f"\nROUND-LEVEL PHASE SHAPE, cut {cut:.0f} us, fast {len(fast)}, "
          f"slow {len(slow)}")
    print(f"  {'phase':<16}{'fast med':>10}{'slow med':>10}{'delta':>10}{'ratio':>8}")
    for key in ["host"] + HOST:
        f, s = st.median(r[key] for r in fast), st.median(r[key] for r in slow)
        print(f"  {key:<16}{f:10.0f}{s:10.0f}{s - f:10.0f}{s / f:8.2f}")


main()
