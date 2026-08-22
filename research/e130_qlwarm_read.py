#!/usr/bin/env python3
"""Price the unwarmed SDPA chunk-B shapes from rung-11 traces. Zero GPU.

The session warms sdpa at qL in {1, 5, 4} (Qwen36MTPBlockSession.swift:541,567).
AttentionUtils.swift:125 splits a verify of width qL in 6...9 at row 5, so
chunk A is always qL=5 and chunk B is qL-5. Verify width is d+1.

  width 6 -> chunk B qL=1   warmed
  width 7 -> chunk B qL=2   NOT warmed
  width 8 -> chunk B qL=3   NOT warmed
  width 9 -> chunk B qL=4   warmed (cap 7 means width 9 never fires)

Widths <= 5 run unchunked as a whole qL=width sdpa, so widths 2 and 3 are
also unwarmed.

A Metal JIT compile is paid once per shape per process, so the cost lands on
the FIRST round that reaches each width. Rule 106: measure that first round
against a tail matched on its own width, never against a mixed-width tail.

Controls that can fail:
  negative  width 6 first hit -- also early in the depth ramp, but its
            chunk-B shape IS warmed, so it isolates "early round" effects
            from "cold shape" effects.
  negative  width 5 first hit -- whole verify at the warmed qL=5.
  Round 1 is excluded everywhere: it carries a separate known excess
  (~29-33 ms width-matched) that would contaminate the width-5 control.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
FIELD_RE = re.compile(r"(\w+)=([-\d.]+)")

WARMED_QL = {1, 4, 5}
SPLIT = 5

T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042,
}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    keys = sorted(T_CRIT_95)
    return T_CRIT_95[min(keys, key=lambda k: abs(k - df))]


def shapes_for_width(width: int) -> list[int]:
    """The sdpa qL values a verify of this width dispatches."""
    if 6 <= width <= 9:
        return [SPLIT, width - SPLIT]
    return [width]


def cold_shapes(width: int) -> list[int]:
    return [q for q in shapes_for_width(width) if q not in WARMED_QL]


def parse_leg(path: str) -> list[dict]:
    rounds = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = ROUND_RE.match(line.rstrip("\n"))
            if not m:
                continue
            rec = {
                "round": int(m.group(1)),
                "d": int(m.group(2)),
                "acc": int(m.group(3)),
            }
            for key, val in FIELD_RE.findall(m.group(4)):
                if key in ("round_us", "verify_build_us", "eval_wall_us",
                           "draft_build_us", "readout_us", "commit_us"):
                    try:
                        rec[key] = float(val)
                    except ValueError:
                        pass
            rec["width"] = rec["d"] + 1
            rounds.append(rec)
    return rounds


def first_hit_excess(rounds: list[dict], metric: str) -> dict[int, dict]:
    """Per width: first occurrence minus the median of the same-width tail."""
    by_width: dict[int, list[dict]] = {}
    for rec in rounds:
        if rec["round"] == 1:
            continue
        if metric not in rec:
            continue
        by_width.setdefault(rec["width"], []).append(rec)
    out = {}
    for width, recs in by_width.items():
        recs.sort(key=lambda r: r["round"])
        if len(recs) < 3:
            continue
        head, tail = recs[0], recs[1:]
        tail_med = statistics.median(r[metric] for r in tail)
        out[width] = {
            "n": len(recs),
            "first_round": head["round"],
            "first_us": head[metric],
            "tail_median_us": tail_med,
            "tail_mad_us": statistics.median(
                abs(r[metric] - tail_med) for r in tail),
            "excess_us": head[metric] - tail_med,
            "cold_shapes": cold_shapes(width),
        }
    return out


def agg(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(values)
    if n == 1:
        return {"n": 1, "mean_us": mean, "sd_us": None, "se_us": None}
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    tc = t_crit(n - 1)
    return {
        "n": n,
        "mean_us": mean,
        "sd_us": sd,
        "se_us": se,
        "df": n - 1,
        "t": mean / se if se else float("nan"),
        "ci95_us": [mean - tc * se, mean + tc * se],
        "significant": bool(se and abs(mean / se) > tc),
    }


def selftest() -> int:
    """Plant a known one-time cost and recover it; prove the test can fail."""
    ok = True

    def synth(cold_cost_us: float, seed: int) -> list[dict]:
        rng = [((seed * 7919 + i * 104729) % 1000) / 1000.0 for i in range(400)]
        recs, seen = [], set()
        plan = [4] * 3 + [5] * 4 + [6] * 6 + [7] * 60
        for i, d in enumerate(plan):
            width = d + 1
            base = 100_000.0 + 8_000.0 * rng[i]
            cost = 0.0
            for q in cold_shapes(width):
                if q not in seen:
                    seen.add(q)
                    cost += cold_cost_us
            recs.append({"round": i + 1, "d": d, "acc": d,
                         "width": width, "round_us": base + cost})
        return recs

    legs = [first_hit_excess(synth(30_000.0, s), "round_us") for s in range(12)]
    for width in (7, 8):
        vals = [lg[width]["excess_us"] for lg in legs if width in lg]
        a = agg(vals)
        if not (25_000 < a["mean_us"] < 35_000):
            print(f"FAIL planted width {width}: {a['mean_us']:.0f} us")
            ok = False
        if not a["significant"]:
            print(f"FAIL planted width {width} not significant")
            ok = False
    for width in (6,):
        vals = [lg[width]["excess_us"] for lg in legs if width in lg]
        a = agg(vals)
        if a["significant"]:
            print(f"FAIL warmed control width {width} fired: {a['mean_us']:.0f}")
            ok = False

    null = [first_hit_excess(synth(0.0, s), "round_us") for s in range(12)]
    for width in (6, 7, 8):
        vals = [lg[width]["excess_us"] for lg in null if width in lg]
        a = agg(vals)
        if a["significant"]:
            print(f"FAIL zero-cost width {width} fired: {a['mean_us']:.0f} us")
            ok = False

    if cold_shapes(7) != [2] or cold_shapes(8) != [3]:
        print("FAIL cold shape map")
        ok = False
    if cold_shapes(6) != [] or cold_shapes(5) != [] or cold_shapes(9) != []:
        print("FAIL warmed shape map")
        ok = False
    if cold_shapes(3) != [3] or cold_shapes(2) != [2]:
        print("FAIL short-width shape map")
        ok = False

    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="research/out/e130-r11lad-*/trace.txt")
    ap.add_argument("--metric", default="round_us")
    ap.add_argument("--out", default="research/e130-artifacts/qlwarm.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no traces matched {args.glob}", file=sys.stderr)
        return 2

    legs, hist = [], {}
    for path in paths:
        rounds = parse_leg(path)
        if not rounds:
            continue
        for rec in rounds:
            hist[rec["width"]] = hist.get(rec["width"], 0) + 1
        legs.append({
            "leg": os.path.basename(os.path.dirname(path)),
            "rounds": len(rounds),
            "total_us": sum(r.get(args.metric, 0.0) for r in rounds),
            "excess": first_hit_excess(rounds, args.metric),
        })

    total_rounds = sum(hist.values())
    widths = sorted(hist)
    per_width = {}
    for width in widths:
        vals = [lg["excess"][width]["excess_us"]
                for lg in legs if width in lg["excess"]]
        firsts = [lg["excess"][width]["first_round"]
                  for lg in legs if width in lg["excess"]]
        per_width[width] = {
            "rounds_all_legs": hist[width],
            "round_share_pct": 100.0 * hist[width] / total_rounds,
            "cold_shapes": cold_shapes(width),
            "warmed": not cold_shapes(width),
            "median_first_round": statistics.median(firsts) if firsts else None,
            "excess": agg(vals),
        }

    mean_leg_total = statistics.fmean(lg["total_us"] for lg in legs)
    cold_widths = [w for w in widths if cold_shapes(w)]
    per_leg_cold = []
    for lg in legs:
        tot = sum(lg["excess"][w]["excess_us"]
                  for w in cold_widths if w in lg["excess"])
        if tot:
            per_leg_cold.append(100.0 * tot / lg["total_us"])
    cold_pct = agg(per_leg_cold)

    report = {
        "schema": "e130-qlwarm-1",
        "source": {
            "warm_loop": "Qwen36MTPBlockSession.swift:541,567 -> [1, 5, 4]",
            "split": "AttentionUtils.swift:125 -> split=5 for 6 <= qL <= 9",
            "warmed_ql": sorted(WARMED_QL),
        },
        "legs": len(legs),
        "rounds_total": total_rounds,
        "metric": args.metric,
        "mean_leg_total_us": mean_leg_total,
        "per_width": per_width,
        "cold_total_pct_of_leg": cold_pct,
        "note": ("Round 1 excluded from every excess. Widths 6 and 5 are "
                 "warmed-shape controls at the same point in the depth ramp."),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"legs={len(legs)} rounds={total_rounds} "
          f"mean_leg_total={mean_leg_total / 1e6:.3f} s")
    print()
    print(f"{'width':>5} {'qL dispatched':>14} {'cold':>5} {'rounds':>7} "
          f"{'share%':>7} {'1st rd':>7} {'excess us':>11} {'se':>9} {'sig':>4}")
    def cell(value: float | None, digits: int = 0) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    for width in widths:
        info = per_width[width]
        exc = info["excess"]
        print(f"{width:>5} {str(shapes_for_width(width)):>14} "
              f"{str(info['cold_shapes'] or '-'):>5} "
              f"{info['rounds_all_legs']:>7} {info['round_share_pct']:>7.1f} "
              f"{str(info['median_first_round']):>7} "
              f"{cell(exc.get('mean_us')):>11} "
              f"{cell(exc.get('se_us')):>9} "
              f"{('YES' if exc.get('significant') else 'ns') if exc.get('n') else 'n/a':>4}")
    print()
    if cold_pct.get("n"):
        ci = cold_pct.get("ci95_us")
        print(f"cold-shape one-time total: {cold_pct['mean_us']:.4f} % of leg "
              f"(se {cold_pct['se_us']:.4f}, n={cold_pct['n']}"
              + (f", CI95 [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "") + ")")
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
