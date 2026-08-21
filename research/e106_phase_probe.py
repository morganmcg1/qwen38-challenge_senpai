#!/usr/bin/env python3
"""E106 -- list the (width, phase, kernel, grid.x) cells present in a census trace.

    usage: research/e106_phase_probe.py TAG

Used to find out which phase label and kernel family carries a given draft
width, so a reducer filters on the right cell instead of silently returning an
empty set.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

SHAPE_RE = re.compile(
    r"^(?P<phase>[^|]+)\|(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")


def main() -> None:
    tag = sys.argv[1]
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    cells = collections.Counter()
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        for _rnd, _ordinal, width, shape_id, _gpu_ns in rec["trace"]:
            match = parsed[shape_id]
            if match is None:
                continue
            kernel = match.group("kernel")
            if "qmv" not in kernel:
                continue
            cells[(width, match.group("phase"), kernel,
                   int(match.group("gx")), int(match.group("gy")))] += 1
    print(f"{'width':>5} {'phase':<16} {'gx':>3} {'gy':>7} {'n':>7}  kernel")
    for (width, phase, kernel, gx, gy), n in sorted(cells.items()):
        print(f"{width:5d} {phase:<16} {gx:3d} {gy:7d} {n:7d}  {kernel}")


if __name__ == "__main__":
    main()
