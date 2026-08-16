#!/usr/bin/env python3
"""Refit the per-round cost model from a traced run and test the qmv width kink.

Reads the MLX_QWEN_MTP_TRACE stderr trace of one --local-iterate run and
reports, per verify width: eval_wall, the linear fit over the widths below the
suspected kink, the width-9 residual, and the realised microseconds per emitted
token. Also reports what each cap-8 round actually bought over a cap-4 round.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
KV_RE = re.compile(r"(\w+)=([0-9.]+)")


def parse(path):
    out = []
    for line in open(path, errors="replace"):
        m = ROUND_RE.search(line)
        if not m:
            continue
        rest = m.group(4)
        kv = {k: float(v) for k, v in KV_RE.findall(rest) if k != "ema_in"}
        out.append(
            {
                "round": int(m.group(1)),
                "depth": int(m.group(2)),
                "accepted": int(m.group(3)),
                "width": int(m.group(2)) + 1,
                "tokens": int(m.group(3)) + 1,
                **kv,
            }
        )
    return out


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return slope, my - slope * mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--kink-width", type=int, default=9)
    ap.add_argument("--drop-first-round", action="store_true", default=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    rounds = parse(args.trace)
    warm = [r for r in rounds if r["round"] > 1] if args.drop_first_round else rounds

    by_width = defaultdict(list)
    for r in warm:
        by_width[r["width"]].append(r)

    per_width = {}
    for w in sorted(by_width):
        g = by_width[w]
        mean = lambda k: st.mean(x[k] for x in g)
        tok = sum(x["tokens"] for x in g)
        per_width[w] = {
            "n": len(g),
            "eval_wall_us": mean("eval_wall_us"),
            "verify_build_us": mean("verify_build_us"),
            "draft_build_us": mean("draft_build_us"),
            "round_us": mean("round_us"),
            "accepted_mean": mean("accepted"),
            "tokens_total": tok,
            "us_per_token": sum(x["round_us"] for x in g) / tok,
        }

    below = [w for w in per_width if w < args.kink_width]
    kink = None
    if len(below) >= 3 and args.kink_width in per_width:
        for field in ("eval_wall_us", "round_us"):
            slope, icept = linfit(below, [per_width[w][field] for w in below])
            pred = icept + slope * args.kink_width
            actual = per_width[args.kink_width][field]
            kink = kink or {}
            kink[field] = {
                "slope_us_per_width": slope,
                "intercept_us": icept,
                "predicted_us": pred,
                "actual_us": actual,
                "residual_us": actual - pred,
                "residual_pct": 100.0 * (actual - pred) / pred,
            }

    marginal = {}
    ws = sorted(per_width)
    for a, b in zip(ws, ws[1:]):
        marginal[f"{a}->{b}"] = {
            "d_width": b - a,
            "d_eval_wall_us_per_width": (
                per_width[b]["eval_wall_us"] - per_width[a]["eval_wall_us"]
            )
            / (b - a),
            "d_round_us_per_width": (per_width[b]["round_us"] - per_width[a]["round_us"])
            / (b - a),
        }

    cap4 = [r for r in warm if r["depth"] <= 4]
    cap8 = [r for r in warm if r["depth"] > 4]
    shallow_deep = {}
    for name, g in (("depth_le_4", cap4), ("depth_gt_4", cap8)):
        if not g:
            continue
        tok = sum(x["tokens"] for x in g)
        shallow_deep[name] = {
            "rounds": len(g),
            "tokens": tok,
            "total_us": sum(x["round_us"] for x in g),
            "us_per_token": sum(x["round_us"] for x in g) / tok,
            "mean_depth": st.mean(x["depth"] for x in g),
            "mean_accepted": st.mean(x["accepted"] for x in g),
        }

    report = {
        "rounds_total": len(rounds),
        "rounds_warm": len(warm),
        "per_width": per_width,
        "marginal_per_width": marginal,
        "kink_test": kink,
        "shallow_vs_deep": shallow_deep,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as h:
            json.dump(report, h, indent=2)


if __name__ == "__main__":
    main()
