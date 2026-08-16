#!/usr/bin/env python3
"""Reconcile the raw = G*T/(1 + h*d) feasibility bound with a measured trace.

The advisor's bound treats a round as one target forward plus d head steps.
A traced run gives the round cost directly, so h and the round's fixed cost
can be measured instead of assumed, and the implied speedup can be compared
with the published campaign score without needing G as a free variable.

Also emits the realised per-position acceptance curve. A round of depth d
that accepted a tokens is evidence that positions 0..a-1 accepted and, when
a < d, that position a rejected; positions past a are unobserved.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
BEGIN_RE = re.compile(r"mtp-trace: begin seed=(\d+) build_us=(\d+) eval_wall_us=(\d+)")
KV_RE = re.compile(r"(\w+)=([0-9.]+)")


def load(path):
    rounds, begins = [], []
    for line in open(path, errors="replace"):
        b = BEGIN_RE.search(line)
        if b:
            begins.append(
                {
                    "seed": int(b.group(1)),
                    "build_us": int(b.group(2)),
                    "eval_wall_us": int(b.group(3)),
                }
            )
        m = ROUND_RE.search(line)
        if m:
            kv = {k: float(v) for k, v in KV_RE.findall(m.group(4)) if k != "ema_in"}
            rounds.append(
                {
                    "round": int(m.group(1)),
                    "depth": int(m.group(2)),
                    "accepted": int(m.group(3)),
                    **kv,
                }
            )
    return rounds, begins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--promoted-score", type=float, default=2.9042110287045)
    ap.add_argument("--out")
    args = ap.parse_args()

    rounds, begins = load(args.trace)
    score = json.load(open(args.score))["metrics"]
    tokens = score["decode_tokens"]

    serial_leg_s = score["serial_seconds_per_token"] * tokens
    mtp_leg_s = score["mtp_seconds_per_token"] * tokens
    # Both legs pay the same seed prologue; it is in the timed leg but is not
    # decode, and it dilutes the reported ratio.
    prologue_s = [(b["build_us"] + b["eval_wall_us"]) / 1e6 for b in begins]
    serial_prologue = prologue_s[0] if prologue_s else 0.0
    mtp_prologue = prologue_s[1] if len(prologue_s) > 1 else serial_prologue

    serial_decode_s = serial_leg_s - serial_prologue
    mtp_decode_s = mtp_leg_s - mtp_prologue
    serial_forward_us = serial_decode_s / tokens * 1e6

    warm = [r for r in rounds if r["round"] > 1]
    full = defaultdict(list)
    for r in warm:
        if r["accepted"] == r["depth"]:
            full[r["depth"]].append(r["round_us"])

    # round_us = fixed + slope * width, fitted on the widths actually observed
    xs = [d + 1 for d in sorted(full)]
    ys = [st.mean(full[d]) for d in sorted(full)]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    fixed = my - slope * mx

    h = slope / serial_forward_us
    fixed_in_forwards = fixed / serial_forward_us

    best_case = {}
    for d in sorted(full):
        cost = st.mean(full[d])
        best_case[d] = {
            "round_us": cost,
            "tokens": d + 1,
            "raw_speedup_if_all_accepted": (d + 1) * serial_forward_us / cost,
        }

    accepts = defaultdict(int)
    rejects = defaultdict(int)
    for r in rounds:
        for i in range(r["accepted"]):
            accepts[i] += 1
        if r["accepted"] < r["depth"]:
            rejects[r["accepted"]] += 1
    positions = {}
    for i in range(max(list(accepts) + list(rejects)) + 1):
        a, j = accepts[i], rejects[i]
        positions[i] = {
            "accepted": a,
            "rejected": j,
            "observed": a + j,
            "p_hat": a / (a + j) if (a + j) else None,
            "prior_0_98_pow_i": 0.98**i,
        }

    total_round_us = sum(r["round_us"] for r in rounds)
    total_tokens = sum(r["accepted"] + 1 for r in rounds)

    report = {
        "window_tokens": tokens,
        "reported_local_ratio": score["mtp_decode_speedup"],
        "serial_prologue_s": serial_prologue,
        "mtp_prologue_s": mtp_prologue,
        "serial_decode_s": serial_decode_s,
        "mtp_decode_s": mtp_decode_s,
        "decode_only_raw_speedup": serial_decode_s / mtp_decode_s,
        "serial_forward_us": serial_forward_us,
        "round_cost_fit": {
            "fixed_us": fixed,
            "slope_us_per_width": slope,
            "fixed_in_serial_forwards": fixed_in_forwards,
            "h_per_draft_in_serial_forwards": h,
        },
        "best_case_raw_by_depth": best_case,
        "realised": {
            "rounds": len(rounds),
            "tokens": total_tokens,
            "tokens_per_round": total_tokens / len(rounds),
            "mean_round_us": total_round_us / len(rounds),
        },
        "per_position_acceptance": positions,
        "promoted_score": args.promoted_score,
        "promoted_reachable_at_G_equals_1": (serial_decode_s / mtp_decode_s)
        >= args.promoted_score,
    }

    print(f"window                    {tokens} tokens")
    print(f"reported local ratio      {report['reported_local_ratio']:.4f}")
    print(
        f"prologue (both legs)      {serial_prologue:.3f} s / {mtp_prologue:.3f} s"
    )
    print(
        f"decode-only raw speedup   {report['decode_only_raw_speedup']:.4f}"
        "   (G = 1, same build both legs)"
    )
    print(f"one serial forward        {serial_forward_us/1000:.2f} ms")
    print(
        f"round cost fit            {fixed/1000:.2f} ms + {slope/1000:.2f} ms x width"
    )
    print(
        f"  fixed  in forwards      {fixed_in_forwards:.3f}"
        "   (advisor's model assumes 1.0)"
    )
    print(
        f"  h per draft             {h:.3f}"
        "   (advisor's pessimistic estimate was 0.62)"
    )
    print()
    print("best-case raw speedup if every draft is accepted:")
    for d, v in best_case.items():
        print(
            f"  depth {d} (width {d+1}): {v['round_us']/1000:7.1f} ms"
            f" -> {v['raw_speedup_if_all_accepted']:.3f}x"
        )
    print()
    print("realised per-position acceptance vs the 0.98^i prior:")
    print("  i  accepted rejected   p_hat   0.98^i")
    for i, v in positions.items():
        ph = "  n/a " if v["p_hat"] is None else f"{v['p_hat']:.4f}"
        print(
            f"  {i}  {v['accepted']:8d} {v['rejected']:8d}  {ph}"
            f"  {v['prior_0_98_pow_i']:.4f}"
        )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
