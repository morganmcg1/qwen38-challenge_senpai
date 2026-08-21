#!/usr/bin/env python3
"""E106 rung 0 -- print the GPU encode order around each N=5120 projection.

    usage: research/e106_encode_order.py TAG [--width 5] [--round N]

`gdn.out_proj` and `fa.o_proj` have identical geometry (K=6144, N=5120) yet
only `gdn.out_proj` sits above the dispatch law. This dumps the dispatches
that run immediately before each of the three N=5120 projections so the
predecessor traffic can be compared directly.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
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

MARKERS = {2060: "gdn.in_proj", 1792: "fa.qkv", 4352: "mlp.gate_up",
           31040: "lm_head"}
NARROW_FROM_MARKER = {"gdn.in_proj": "gdn.out_proj", "fa.qkv": "fa.o_proj",
                      "mlp.gate_up": "mlp.down"}
NARROW_GY = 640


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--round", type=int)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--window", type=int, default=7)
    args = ap.parse_args()

    path = pathlib.Path("research/out") / args.tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e106_encode_order: no census at {path}")

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
            per_round[rnd].append((ordinal, match, gpu_ns))

    if not per_round:
        sys.exit("e106_encode_order: no traced dispatches for that selection")

    rnd = args.round if args.round is not None else sorted(per_round)[
        len(per_round) // 2]
    rows = sorted(per_round[rnd], key=lambda r: r[0])
    print(f"=== {args.tag}  M={args.width}  round={rnd}  "
          f"dispatches={len(rows)}")

    labels = {}
    marker = None
    for i, (_ordinal, match, _ns) in enumerate(rows):
        if not match.group("kernel").startswith("affine_qmv_fast"):
            continue
        if int(match.group("gx")) != args.width:
            continue
        gy = int(match.group("gy"))
        if gy in MARKERS:
            marker = MARKERS[gy]
            labels[i] = marker
        elif gy == NARROW_GY and marker in NARROW_FROM_MARKER:
            labels[i] = NARROW_FROM_MARKER[marker]

    seen = set()
    for i in sorted(labels):
        name = labels[i]
        if name not in ("gdn.out_proj", "fa.o_proj", "mlp.down"):
            continue
        if name in seen:
            continue
        seen.add(name)
        print(f"\n--- predecessors of {name} (encode #{i}) ---")
        for j in range(max(0, i - args.window), min(len(rows), i + 2)):
            _ordinal, match, gpu_ns = rows[j]
            mark = ">>" if j == i else "  "
            grid = (f"{match.group('gx')}x{match.group('gy')}"
                    f"x{match.group('gz')}")
            tg = (f"{match.group('tx')}x{match.group('ty')}"
                  f"x{match.group('tz')}")
            print(f"{mark} #{j:4d} {match.group('kernel')[:42]:42s} "
                  f"grid={grid:<16s} tg={tg:<10s} "
                  f"{gpu_ns / 1e3:9.2f}us  {labels.get(j, '')}")


if __name__ == "__main__":
    main()
