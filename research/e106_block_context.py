#!/usr/bin/env python3
"""E106 rung 0 -- does the host decoder block change the cost of an N=5120 dispatch?

    usage: research/e106_block_context.py TAG [TAG ...] [--width 5] [--json OUT]

The two-factor reading of the N=5120 anomaly says the redundant per-threadgroup
`x` re-read is free while `x` stays resident, and that residency depends both on
the size of `x` and on eviction pressure from whatever ran before it. The
matched control behind that reading is `gdn.out_proj` against `fa.o_proj`: two
byte-identical dispatches that differ only in their predecessor block.

This reducer adds the control that the same trace already contains but that the
tensor-level split throws away. Every decoder layer ends with the same
`mlp.gate_up` -> `mlp.down` pair, and 48 of those layers are Gated DeltaNet
layers while 16 are full-attention layers. So `mlp.down@gdn` and `mlp.down@fa`
are byte-identical, share a kernel, a grid, a threadgroup and an immediate
predecessor, and differ only in the block that ran earlier in the same layer.

That separates two things the tensor-level split cannot:

  * a *block* effect that reaches across `mlp.gate_up`, which would mean the
    pressure source outlives 0.1 GB of intervening streaming; and
  * a *local* effect confined to the dispatch right after the block, which is
    what `gdn.out_proj` minus `fa.o_proj` measures.

`mlp.gate_up@gdn` against `mlp.gate_up@fa` is carried as a negative control. It
is a clean family that lands on the law, so any split there is positional or
thermal drift rather than a property of the N=5120 shape.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import statistics
import sys

HIDDEN = 5120

# grid.y of one dispatch -> label. grid.y * 8 is the output width.
MARKERS = {
    2060: "gdn.in_proj",    # N = 16480
    1792: "fa.qkv",         # N = 14336
    4352: "mlp.gate_up",    # N = 34816
    31040: "lm_head",       # N = 248320
}
BLOCK_MARKERS = {"gdn.in_proj": "gdn", "fa.qkv": "fa"}
NARROW_GY = 640
NARROW_FROM_MARKER = {
    "gdn.in_proj": "gdn.out_proj",
    "fa.qkv": "fa.o_proj",
    "mlp.gate_up": "mlp.down",
}
CLEAN = ("lm_head", "mlp.gate_up", "gdn.in_proj", "fa.qkv")
GEOMETRY = {
    "lm_head": (HIDDEN, 248_320), "mlp.gate_up": (HIDDEN, 34_816),
    "gdn.in_proj": (HIDDEN, 16_480), "fa.qkv": (HIDDEN, 14_336),
    "gdn.out_proj": (6144, 5120), "fa.o_proj": (6144, 5120),
    "mlp.down": (17_408, 5120),
}

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
    return (sy - slope * sx) / n, slope


def describe(values):
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    sem = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, statistics.median(values), sd, sem


def collect(path, width, phase):
    """round -> [(ordinal, gy, gpu_ns)] for one phase at one draft width."""
    per_round = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        for rnd, ordinal, w, shape_id, gpu_ns in rec["trace"]:
            if w != width:
                continue
            match = parsed[shape_id]
            if match is None or match.group("phase") != phase:
                continue
            if not match.group("kernel").startswith("affine_qmv_fast"):
                continue
            if int(match.group("gx")) != width:
                continue
            per_round[rnd].append(
                (ordinal, int(match.group("gy")), gpu_ns))
    return per_round


def label(per_round, width):
    """(tensor, host) -> [us], plus (tensor, host, layer_idx) -> [us]."""
    groups = collections.defaultdict(list)
    by_layer = collections.defaultdict(list)
    pooled = collections.defaultdict(list)
    unlabelled = 0
    for _rnd, rows in sorted(per_round.items()):
        marker = None
        host = None
        layer_idx = 0
        for _ordinal, gy, gpu_ns in sorted(rows):
            us = gpu_ns / 1e3
            if gy in MARKERS:
                marker = MARKERS[gy]
                if marker in BLOCK_MARKERS:
                    host = BLOCK_MARKERS[marker]
                    layer_idx += 1
                pooled[marker].append(us)
                if host is not None:
                    groups[(marker, host)].append(us)
                    by_layer[(marker, host, layer_idx)].append(us)
            elif gy == NARROW_GY:
                tensor = NARROW_FROM_MARKER.get(marker)
                if tensor is None or host is None:
                    unlabelled += 1
                    continue
                pooled[tensor].append(us)
                groups[(tensor, host)].append(us)
                by_layer[(tensor, host, layer_idx)].append(us)
    return groups, by_layer, pooled, unlabelled


def collect_all(path, width, phase):
    """round -> [(ordinal, kernel, gy, us)] including non-qmv kernels."""
    per_round = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        for rnd, ordinal, w, shape_id, gpu_ns in rec["trace"]:
            if w != width:
                continue
            match = parsed[shape_id]
            if match is None or match.group("phase") != phase:
                continue
            per_round[rnd].append((ordinal, match.group("kernel"),
                                   int(match.group("gy")), gpu_ns / 1e3))
    return per_round


def block_gap(per_round, block_kernel):
    """(intervening dispatches, intervening us, block us, victim us) per layer.

    The victim is the N=5120 projection that follows the block. Within one
    width the dispatch count is fixed, but the intervening GPU time and the
    block kernel's own GPU time both vary from layer to layer. That gives two
    dose-response tests with no confound from width: elapsed distance, and how
    much work the block itself did.
    """
    rows_out = []
    for _rnd, rows in sorted(per_round.items()):
        pending = None
        for _ordinal, kernel, gy, us in sorted(rows):
            if block_kernel in kernel:
                pending = [0, 0.0, us]
                continue
            if pending is None:
                continue
            if gy == NARROW_GY:
                rows_out.append((pending[0], pending[1], pending[2], us))
                pending = None
            else:
                pending[0] += 1
                pending[1] += us
    return rows_out


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan")
    r = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    return r, slope


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
            sys.exit(f"e106_block_context: no census at {path}")
        per_round = collect(path, args.width, args.phase)
        if not per_round:
            print(f"=== {tag}: no traced qmv dispatches at M={args.width}")
            continue
        groups, by_layer, pooled, unlabelled = label(per_round, args.width)

        f_us, s_us = fit([(affine4(*GEOMETRY[c]) / 1e9,
                           statistics.fmean(pooled[c]))
                          for c in CLEAN if c in pooled])
        print(f"=== {tag}   M={args.width}   rounds={len(per_round)}   "
              f"unlabelled={unlabelled}")
        print(f"    refit on clean families: F = {f_us:.2f} us   "
              f"S = {s_us:.1f} us/GB = {1e6 / s_us:.1f} GB/s")

        print(f"\n  {'tensor':<14} {'host':<5} {'n':>5} {'mean us':>9} "
              f"{'median':>9} {'sd':>7} {'sem':>6} {'law us':>9} "
              f"{'excess':>8}")
        rows = {}
        for (tensor, host), values in sorted(groups.items()):
            law = f_us + affine4(*GEOMETRY[tensor]) / 1e9 * s_us
            mean, med, sd, sem = describe(values)
            rows[(tensor, host)] = (mean, sem, len(values))
            print(f"  {tensor:<14} {host:<5} {len(values):5d} {mean:9.2f} "
                  f"{med:9.2f} {sd:7.2f} {sem:6.2f} {law:9.2f} "
                  f"{mean - law:8.2f}")

        print("\n  matched controls -- byte-identical pairs that differ only "
              "in host block")
        print(f"  {'pair':<34} {'gdn us':>9} {'fa us':>9} {'diff':>8} "
              f"{'sem':>6} {'sigma':>7}")
        for tensor in ("mlp.down", "mlp.gate_up", "gdn.in_proj"):
            a = rows.get((tensor, "gdn"))
            b = rows.get((tensor, "fa"))
            if not a or not b:
                continue
            diff = a[0] - b[0]
            sem = math.sqrt(a[1] ** 2 + b[1] ** 2)
            sigma = diff / sem if sem else float("nan")
            name = f"{tensor}@gdn vs {tensor}@fa"
            print(f"  {name:<34} {a[0]:9.2f} {b[0]:9.2f} {diff:8.2f} "
                  f"{sem:6.2f} {sigma:7.2f}")
        out = rows.get(("gdn.out_proj", "gdn"))
        o = rows.get(("fa.o_proj", "fa"))
        if out and o:
            diff = out[0] - o[0]
            sem = math.sqrt(out[1] ** 2 + o[1] ** 2)
            name = "gdn.out_proj vs fa.o_proj (effect B)"
            print(f"  {name:<34} {out[0]:9.2f} {o[0]:9.2f} {diff:8.2f} "
                  f"{sem:6.2f} {diff / sem if sem else float('nan'):7.2f}")

        # A positional trend would mean the excess is drift through the round
        # rather than a property of the block.
        print("\n  excess against layer index (quartiles of the 64 layers)")
        print(f"  {'tensor':<14} {'host':<5} {'q1':>8} {'q2':>8} {'q3':>8} "
              f"{'q4':>8} {'q4-q1':>8}")
        for tensor in ("gdn.out_proj", "fa.o_proj", "mlp.down",
                       "mlp.gate_up"):
            for host in ("gdn", "fa"):
                idxs = sorted({k[2] for k in by_layer
                               if k[0] == tensor and k[1] == host})
                if len(idxs) < 4:
                    continue
                law = f_us + affine4(*GEOMETRY[tensor]) / 1e9 * s_us
                quarts = []
                for q in range(4):
                    lo = q * len(idxs) // 4
                    hi = (q + 1) * len(idxs) // 4
                    vals = [v for i in idxs[lo:hi]
                            for v in by_layer[(tensor, host, i)]]
                    quarts.append(statistics.fmean(vals) - law)
                print(f"  {tensor:<14} {host:<5} {quarts[0]:8.2f} "
                      f"{quarts[1]:8.2f} {quarts[2]:8.2f} {quarts[3]:8.2f} "
                      f"{quarts[3] - quarts[0]:8.2f}")

        # Effect B is the excess on the dispatch that follows the block. If the
        # encode distance to that dispatch changes with width, then what looks
        # like width gating may be distance gating instead.
        all_rounds = collect_all(path, args.width, args.phase)
        print("\n  within-width dose-response on the dispatch that follows a block")
        print(f"  {'block':<18} {'dose':<10} {'n':>5} {'disp':>6} {'dose us':>8} "
              f"{'sd':>6} {'victim us':>10} {'r':>7} {'slope':>7}")
        dose_out = {}
        for kernel in ("gated_delta_step", "sdpa_vector"):
            rows_out = block_gap(all_rounds, kernel)
            if not rows_out:
                continue
            victims = [r[3] for r in rows_out]
            counts = [r[0] for r in rows_out]
            for col, dose_name in ((1, "gap"), (2, "block")):
                dose = [r[col] for r in rows_out]
                r, slope = pearson(dose, victims)
                dose_out[f"{kernel}/{dose_name}"] = {
                    "n": len(rows_out),
                    "intervening_dispatches": statistics.fmean(counts),
                    "dose_mean_us": statistics.fmean(dose),
                    "dose_sd_us": statistics.pstdev(dose),
                    "victim_mean_us": statistics.fmean(victims),
                    "pearson_r": r, "slope_us_per_us": slope,
                }
                print(f"  {kernel:<18} {dose_name:<10} {len(rows_out):5d} "
                      f"{statistics.fmean(counts):6.2f} "
                      f"{statistics.fmean(dose):8.2f} "
                      f"{statistics.pstdev(dose):6.2f} "
                      f"{statistics.fmean(victims):10.2f} {r:7.3f} {slope:7.3f}")
                # Quintiles isolate the dose-response from a single outlier.
                ordered = sorted(rows_out, key=lambda t: t[col])
                step = len(ordered) // 5
                if step < 4:
                    continue
                cells = [(statistics.fmean(c[col] for c in chunk),
                          statistics.fmean(c[3] for c in chunk))
                         for chunk in (ordered[q * step:(q + 1) * step]
                                       for q in range(5))]
                print("        dose quintile: " + "  ".join(
                    f"{g:7.2f}" for g, _v in cells))
                print("        victim mean  : " + "  ".join(
                    f"{v:7.2f}" for _g, v in cells))

        payload[tag] = {
            "width": args.width, "rounds": len(per_round),
            "f_us": f_us, "s_us_per_gb": s_us, "dose_response": dose_out,
            "groups": {f"{t}@{h}": {
                "count": n, "mean_us": m, "sem_us": e,
                "law_us": f_us + affine4(*GEOMETRY[t]) / 1e9 * s_us,
                "excess_us": m - (f_us + affine4(*GEOMETRY[t]) / 1e9 * s_us),
            } for (t, h), (m, e, n) in rows.items()},
        }

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
