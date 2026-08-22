#!/usr/bin/env python3
"""E106 rung 0b detail -- the dispatch sequence of one round, and the per-layer
cost of every N=5120 projection.

    usage: research/e106_trace_detail.py TAG [--width 5] [--round N]

`research/e106_trace_split.py` gives the three-tensor split. This gives the
distribution behind it: the verbatim kernel order of one layer, and the cost of
each N=5120 dispatch keyed by the layer it belongs to. Two tensors with the
same K, the same N and the same grid can still differ, and only a per-layer
view shows whether that difference is a context effect or a shape effect.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

SHAPE_RE = re.compile(
    r"^(?P<phase>[^|]+)\|(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")
MARKER_GY = {2060: "gdn.in_proj", 1792: "fa.qkv", 4352: "mlp.gate_up",
             31040: "lm_head"}
NARROW_FROM = {"gdn.in_proj": "gdn.out_proj", "fa.qkv": "fa.o_proj",
               "mlp.gate_up": "mlp.down"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--round", type=int)
    args = ap.parse_args()

    path = pathlib.Path("research/out") / args.tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e106_trace_detail: no census at {path}")
    per_round = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        shapes = [SHAPE_RE.match(s) for s in rec["trace_shapes"]]
        for rnd, ordinal, width, sid, ns in rec["trace"]:
            if width != args.width:
                continue
            match = shapes[sid]
            if match is None or match.group("phase") != args.phase:
                continue
            per_round[rnd].append(
                (ordinal, match.group("kernel"), int(match.group("gx")),
                 int(match.group("gy")), ns / 1e3))

    rounds = sorted(per_round)
    if not rounds:
        sys.exit("e106_trace_detail: no traced dispatches")
    show = args.round if args.round is not None else rounds[len(rounds) // 2]

    print(f"=== {args.tag}   rounds={len(rounds)}   showing round {show}")
    print("\nfirst 34 dispatches of the round, in encode order:")
    for ordinal, kernel, gx, gy, us in sorted(per_round[show])[:34]:
        print(f"  {ordinal:5d} {us:9.2f} us  grid={gx}x{gy}x1  {kernel[:72]}")

    # Per-layer cost of every N=5120 projection.
    per_layer = collections.defaultdict(list)
    order = collections.defaultdict(list)
    for rnd in rounds:
        marker = None
        layer = -1
        for ordinal, kernel, gx, gy, us in sorted(per_round[rnd]):
            if not kernel.startswith("affine_qmv_fast") or gx != args.width:
                continue
            if gy in MARKER_GY:
                name = MARKER_GY[gy]
                if name in ("gdn.in_proj", "fa.qkv"):
                    layer += 1
                marker = name
                continue
            if gy == 640 and marker in NARROW_FROM:
                per_layer[(layer, NARROW_FROM[marker])].append(us)
                order[NARROW_FROM[marker]].append((layer, us))

    print(f"\nper-layer N=5120 cost, mean over {len(rounds)} rounds:")
    print(f"  {'layer':>5} {'kind':<14} {'us':>9} {'sd':>7} {'n':>4}   "
          f"{'layer':>5} {'kind':<14} {'us':>9} {'sd':>7} {'n':>4}")
    keys = sorted(per_layer)
    half = (len(keys) + 1) // 2
    for i in range(half):
        cells = []
        for key in (keys[i], keys[i + half] if i + half < len(keys) else None):
            if key is None:
                cells.append(" " * 44)
                continue
            values = per_layer[key]
            sd = statistics.pstdev(values) if len(values) > 1 else 0.0
            cells.append(f"  {key[0]:5d} {key[1]:<14} "
                         f"{statistics.fmean(values):9.2f} {sd:7.2f} "
                         f"{len(values):4d}")
        print("".join(cells))

    print("\ngrouped by kind:")
    for kind in ("gdn.out_proj", "fa.o_proj", "mlp.down"):
        values = [v for (layer, k), vs in per_layer.items() if k == kind
                  for v in vs]
        if not values:
            continue
        values.sort()
        n = len(values)
        print(f"  {kind:<14} n={n:5d}  mean={statistics.fmean(values):8.2f}  "
              f"p10={values[n // 10]:8.2f}  p50={values[n // 2]:8.2f}  "
              f"p90={values[(9 * n) // 10]:8.2f}  min={values[0]:8.2f}  "
              f"max={values[-1]:8.2f}")

    # mlp.down splits by the layer kind that hosts it.
    gdn_layers = {layer for (layer, kind) in per_layer if kind == "gdn.out_proj"}
    fa_layers = {layer for (layer, kind) in per_layer if kind == "fa.o_proj"}
    for name, layers in (("mlp.down in GDN layers", gdn_layers),
                         ("mlp.down in FA layers", fa_layers)):
        values = [v for (layer, kind), vs in per_layer.items()
                  if kind == "mlp.down" and layer in layers for v in vs]
        if values:
            print(f"  {name:<24} n={len(values):5d}  "
                  f"mean={statistics.fmean(values):8.2f}  "
                  f"sd={statistics.pstdev(values):6.2f}")


if __name__ == "__main__":
    main()
