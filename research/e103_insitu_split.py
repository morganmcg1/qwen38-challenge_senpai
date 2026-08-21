#!/usr/bin/env python3
"""Read the in-situ SDPA dispatch structure straight out of a census leg.

    usage: research/e103_insitu_split.py TAG ROUNDS [SIGNATURE_SUBSTRING]

`research/e103_census_costs.py` reduces `exclusive_kernels`, which only holds
command buffers that carried exactly one kernel. At verify width 6 the FA
attention op emits two kernels into one buffer, so it disappears from that
view entirely. This reads the `signatures` table instead, which keys every
command buffer by the ordered list of kernels inside it, and so shows both
the partition the trusted `qL * gqa <= 32` cap forces and any copy the
partition drags along.

`buffers` is the number of command buffers with that exact signature,
`dispatches` counts the kernels inside them, and `gpu_ns` is the GPU
interval of the buffer, counted once. A census leg is never a timing leg.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: e103_insitu_split.py TAG ROUNDS [SUBSTRING]")
    tag, rounds = sys.argv[1], float(sys.argv[2])
    pat = sys.argv[3] if len(sys.argv) > 3 else "sdpa"
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    agg: dict[str, collections.Counter] = {}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime":
            continue
        for key, val in rec.get("signatures", {}).items():
            if pat not in key:
                continue
            e = agg.setdefault(key, collections.Counter())
            e["buffers"] += val["buffers"]
            e["dispatches"] += val["dispatches"]
            e["gpu_ns"] += val["gpu_ns"]
    print(f"=== {tag}, {rounds:g} rounds, signatures matching {pat!r} ===")
    for key, e in sorted(agg.items(), key=lambda kv: -kv[1]["gpu_ns"]):
        print(f"  buffers={e['buffers']:6d} ({e['buffers'] / rounds:6.2f}/rd)"
              f"  kernels_per_buffer={e['dispatches'] / e['buffers']:.0f}"
              f"  us/buffer={e['gpu_ns'] / 1e3 / e['buffers']:8.2f}"
              f"  us/round={e['gpu_ns'] / 1e3 / rounds:9.1f}")
        print(f"      {key}")


if __name__ == "__main__":
    main()
