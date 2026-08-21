#!/usr/bin/env python3
"""E106 -- dump every dispatch of one decode round in encode order.

    usage: research/e106_round_layout.py TAG --width W --phase P [--round N]

The tensor-level reducers label an N=5120 dispatch by the marker that preceded
it. That inference is only safe if the encode sequence really is one marker per
projection. This dumps the raw sequence so the layer structure can be read
directly, including non-qmv kernels, and so a missing family shows up as a gap
instead of being silently folded into a neighbouring label.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

SHAPE_RE = re.compile(
    r"^(?P<phase>[^|]+)\|(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--phase", default="target_forward")
    ap.add_argument("--round", type=int)
    ap.add_argument("--head", type=int, default=60)
    args = ap.parse_args()

    path = pathlib.Path("research/out") / args.tag / "census.jsonl"
    per_round = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        for rnd, ordinal, width, shape_id, gpu_ns in rec["trace"]:
            if width != args.width:
                continue
            match = parsed[shape_id]
            if match is None or match.group("phase") != args.phase:
                continue
            per_round[rnd].append((ordinal, match.group("kernel"),
                                   int(match.group("gx")),
                                   int(match.group("gy")), gpu_ns))
    if not per_round:
        sys.exit(f"no dispatches at width={args.width} phase={args.phase}")

    rnd = args.round if args.round is not None else sorted(per_round)[1]
    rows = sorted(per_round[rnd])
    print(f"round {rnd}: {len(rows)} dispatches at width={args.width} "
          f"phase={args.phase}   (rounds available: {len(per_round)})")
    counts = collections.Counter((k, gx, gy) for _o, k, gx, gy, _n in rows)
    print("\n  per-round kernel census")
    for (kernel, gx, gy), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5d}  grid={gx}x{gy}  {kernel}")
    print(f"\n  first {args.head} dispatches in encode order")
    for ordinal, kernel, gx, gy, gpu_ns in rows[:args.head]:
        short = kernel.replace("affine_", "").replace("_bfloat16_t", "")
        print(f"    {ordinal:5d}  grid={gx:2d}x{gy:<6d} {gpu_ns / 1e3:9.2f} us  "
              f"{short}")


if __name__ == "__main__":
    main()
