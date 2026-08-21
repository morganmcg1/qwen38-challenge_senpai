#!/usr/bin/env python3
"""E106 rung 0b -- split the pooled N=5120 census line into its three tensors.

    usage: research/e106_trace_split.py TAG [TAG ...] [--width 5] [--json OUT]

Input is `research/out/TAG/census.jsonl` from `research/e106_trace_leg.sh`,
which sets `MLX_E58_DISPATCH_TRACE=1` on top of the one-dispatch-per-buffer
census. Each trace row is `[round, ordinal, width, shapeID, gpuNs]` for one
command buffer that carried exactly one dispatch, so `gpuNs` is that
dispatch's exclusive GPU time and `ordinal` is its encode position in the
round.

`gdn.out_proj` (K=6144), `fa.o_proj` (K=6144) and `mlp.down` (K=17408) share
one kernel name, one grid and one threadgroup, so the aggregate census pools
them. The encode ordinal separates them: inside one decoder layer the N=5120
projection that follows the layer's input projection is the output projection,
and the one that follows `mlp.gate_up` is `mlp.down`.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

HIDDEN = 5120
GPU_CORES = 20
DRAM_PEAK_GB_S = 273.0

# grid.y of one dispatch -> (label, K). grid.y * 8 is the output width.
MARKERS = {
    2060: ("gdn.in_proj", HIDDEN),      # N = 16480
    1792: ("fa.qkv", HIDDEN),           # N = 14336
    4352: ("mlp.gate_up", HIDDEN),      # N = 34816
    31040: ("lm_head", HIDDEN),         # N = 248320
}
# The N=5120 line, resolved by the marker that preceded it in the layer.
NARROW_GY = 640
NARROW_FROM_MARKER = {
    "gdn.in_proj": ("gdn.out_proj", 6144),
    "fa.qkv": ("fa.o_proj", 6144),
    "mlp.gate_up": ("mlp.down", 17408),
}
CLEAN = ("lm_head", "mlp.gate_up", "gdn.in_proj", "fa.qkv")

SHAPE_RE = re.compile(
    r"^(?P<phase>[^|]+)\|(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")


def affine4(k: int, n: int) -> int:
    return n * k // 2 + 4 * (n * k // 64)


def fit(points):
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((p[1] - mean_y) ** 2 for p in points)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    return intercept, slope, (1.0 - ss_res / ss_tot if ss_tot else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--json")
    args = ap.parse_args()
    payload = {}

    for tag in args.tags:
        path = pathlib.Path("research/out") / tag / "census.jsonl"
        if not path.exists():
            sys.exit(f"e106_trace_split: no census at {path}")

        # round -> [(ordinal, gy, gx, gpu_ns)] for the phase under test
        per_round = collections.defaultdict(list)
        for line in path.open():
            rec = json.loads(line)
            if rec.get("event") != "gputime" or not rec.get("trace"):
                continue
            shapes = rec.get("trace_shapes", [])
            parsed = []
            for shape in shapes:
                match = SHAPE_RE.match(shape)
                parsed.append(match)
            for rnd, ordinal, width, shape_id, gpu_ns in rec["trace"]:
                if width != args.width:
                    continue
                match = parsed[shape_id]
                if match is None or match.group("phase") != args.phase:
                    continue
                if not match.group("kernel").startswith("affine_qmv_fast"):
                    continue
                per_round[rnd].append(
                    (ordinal, int(match.group("gy")), int(match.group("gx")),
                     gpu_ns))

        if not per_round:
            print(f"=== {tag}: no traced qmv dispatches at M={args.width}")
            continue

        # Label every dispatch by walking the round in encode order.
        samples = collections.defaultdict(list)
        unlabelled = 0
        for rnd, rows in sorted(per_round.items()):
            marker = None
            for _ordinal, gy, gx, gpu_ns in sorted(rows):
                if gx != args.width:
                    continue
                if gy in MARKERS:
                    marker = MARKERS[gy][0]
                    samples[marker].append(gpu_ns / 1e3)
                elif gy == NARROW_GY:
                    if marker in NARROW_FROM_MARKER:
                        samples[NARROW_FROM_MARKER[marker][0]].append(
                            gpu_ns / 1e3)
                    else:
                        unlabelled += 1
        rounds = len(per_round)
        print(f"=== {tag}   M={args.width}   traced rounds={rounds}   "
              f"unlabelled N=5120 dispatches={unlabelled}")

        geometry = {
            "lm_head": (HIDDEN, 248_320), "mlp.gate_up": (HIDDEN, 34_816),
            "gdn.in_proj": (HIDDEN, 16_480), "fa.qkv": (HIDDEN, 14_336),
            "gdn.out_proj": (6144, 5120), "fa.o_proj": (6144, 5120),
            "mlp.down": (17_408, 5120),
        }
        stats = {}
        for label, values in samples.items():
            k, n = geometry[label]
            ordered = sorted(values)
            stats[label] = {
                "k": k, "n": n, "count": len(values),
                "per_round": len(values) / rounds,
                "gb": affine4(k, n) / 1e9,
                "mean_us": statistics.fmean(values),
                "median_us": statistics.median(values),
                "min_us": ordered[0], "max_us": ordered[-1],
                "p05_us": ordered[int(0.05 * (len(ordered) - 1))],
                "p25_us": ordered[int(0.25 * (len(ordered) - 1))],
                "p75_us": ordered[int(0.75 * (len(ordered) - 1))],
                "stdev_us": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "x_bytes_per_tg": 2 * args.width * k,
            }

        clean_points = [(stats[c]["gb"], stats[c]["mean_us"])
                        for c in CLEAN if c in stats]
        f_us, s_us, r2 = fit(clean_points)
        print(f"\nrefit on {len(clean_points)} clean families: "
              f"F = {f_us:.2f} us/dispatch   S = {s_us:.1f} us/GB "
              f"= {1e6 / s_us:.1f} GB/s   R^2 = {r2:.8f}")

        order = ("lm_head", "mlp.gate_up", "gdn.in_proj", "fa.qkv",
                 "gdn.out_proj", "fa.o_proj", "mlp.down")
        print(f"\n  {'tensor':<14} {'K':>6} {'N':>7} {'n/rnd':>6} {'GB/disp':>8} "
              f"{'us/disp':>9} {'median':>9} {'min':>8} {'max':>8} {'sd':>6} "
              f"{'law us':>9} {'excess':>8} {'exc %':>7} {'x KB/tg':>8}")
        for label in order:
            if label not in stats:
                continue
            e = stats[label]
            law = f_us + e["gb"] * s_us
            excess = e["mean_us"] - law
            e["law_us"] = law
            e["excess_us"] = excess
            e["excess_pct"] = 100.0 * excess / e["mean_us"]
            e["excess_p05_us"] = e["p05_us"] - law
            e["excess_median_us"] = e["median_us"] - law
            e["gb_per_s"] = e["gb"] / (e["mean_us"] * 1e-6)
            e["gb_per_s_after_f"] = (e["gb"] / ((e["mean_us"] - f_us) * 1e-6)
                                     if e["mean_us"] > f_us else 0.0)
            print(f"  {label:<14} {e['k']:6d} {e['n']:7d} {e['per_round']:6.1f} "
                  f"{e['gb']:8.5f} {e['mean_us']:9.2f} {e['median_us']:9.2f} "
                  f"{e['min_us']:8.2f} {e['max_us']:8.2f} {e['stdev_us']:6.2f} "
                  f"{law:9.2f} {excess:8.2f} {e['excess_pct']:7.2f} "
                  f"{e['x_bytes_per_tg'] / 1024:8.1f}")

        # A floor excess is real per-dispatch work. An excess that appears only
        # above the 5th percentile is interference from the surrounding chain.
        print(f"\n  {'tensor':<14} {'p05':>9} {'p25':>9} {'p50':>9} {'p75':>9} "
              f"{'law':>9} {'exc p05':>9} {'exc p50':>9} {'exc mean':>9}")
        for label in order:
            if label not in stats:
                continue
            e = stats[label]
            print(f"  {label:<14} {e['p05_us']:9.2f} {e['p25_us']:9.2f} "
                  f"{e['median_us']:9.2f} {e['p75_us']:9.2f} {e['law_us']:9.2f} "
                  f"{e['excess_p05_us']:9.2f} {e['excess_median_us']:9.2f} "
                  f"{e['excess_us']:9.2f}")

        # H1 (fixed per dispatch) against H2/H3 (scales with bytes or with K).
        out = stats.get("gdn.out_proj")
        down = stats.get("mlp.down")
        if out and down:
            ratio_k = down["k"] / out["k"]
            ratio_excess = (down["excess_us"] / out["excess_us"]
                            if out["excess_us"] else float("inf"))
            print(f"\n  discriminator: K ratio = {ratio_k:.3f}, "
                  f"bytes ratio = {down['gb'] / out['gb']:.3f}, "
                  f"excess ratio = {ratio_excess:.3f}")
            print(f"    H1 (fixed per dispatch) predicts an excess ratio of "
                  f"1.00 with both near {(down['excess_us'] + out['excess_us']) / 2:.2f} us")
            print(f"    H2/H3 (scales with bytes or K) predicts "
                  f"{ratio_k:.2f}")
        payload[tag] = {"rounds": rounds, "width": args.width,
                        "fit": {"F_us": f_us, "S_us_per_gb": s_us, "r2": r2},
                        "tensors": stats}
        print()

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
