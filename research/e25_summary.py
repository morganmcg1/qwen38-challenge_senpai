#!/usr/bin/env python3
"""Median-of-8 summary rows for the E25 result record."""
import json
import pathlib
import statistics as st

PROMPTS = [
    "english",
    "narrative",
    "technical",
    "dramatic",
    "travel",
    "philosophy",
    "natural_history",
    "medicine",
]

timed = json.loads(pathlib.Path("research/e25-phase1.json").read_text())["reduced"]["timed_runs"]


def col(arm, fn):
    return [fn(timed[arm][p]) for p in PROMPTS]


for arm in ("BASE", "PRICE"):
    serial = col(arm, lambda r: r["serial_true_decode"] / 512.0)
    mtp = col(arm, lambda r: r["mtp_true_decode"] / 512.0)
    speedup = [s / m for s, m in zip(serial, mtp)]
    depth = col(arm, lambda r: r["mtp_mean_depth"])
    rate = col(arm, lambda r: r["mtp_accepted_rate"])
    print(arm)
    print(f"  serial s/tok median   {st.median(serial):.7f}")
    print(f"  mtp    s/tok median   {st.median(mtp):.7f}")
    print(f"  serial-rel speedup    median {st.median(speedup):.6f}  min {min(speedup):.6f} max {max(speedup):.6f}")
    print(f"  mean draft length     median {st.median(depth):.4f}")
    print(f"  accepted draft rate   median {st.median(rate):.4f}")
