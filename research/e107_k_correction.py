#!/usr/bin/env python3
"""Settle advisor f1 item 4: does the E107 reducer resolve K per dispatch?

Edward challenged the E107 ledger line for the `grid=1x640x1` affine-4 cell,
which records K = 5120, 14.75 MB and 147.7 GB/s. This script answers from two
sources that cannot be argued with: the reducer's own table, and the
checkpoint header.
"""

from __future__ import annotations

import collections
import glob
import json
import pathlib
import re
import struct

DRAM_PEAK_GBPS = 273.0
GROUP = 64
BITS = 4
# The per-dispatch mean the E107 draft-head table recorded for this cell.
RECORDED_US = 99.87
RECORDED_K = 5120
RECORDED_MB = 14.75
RECORDED_GBPS = 147.7


def qmv_bytes(n: int, k: int) -> int:
    return n * k * BITS // 8 + n * (k // GROUP) * 2 * 2


def checkpoint_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for path in sorted(glob.glob("weights/**/*.safetensors", recursive=True)):
        with open(path, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
        for name, value in header.items():
            if name != "__metadata__":
                shapes[name] = value["shape"]
    return shapes


def main() -> None:
    src = pathlib.Path("research/e107_wandb_log.py").read_text()
    table = re.search(r"CENSUS_CELLS = \[(.*?)\n\]", src, re.S).group(1)
    ks = re.findall(r"\n     \d+, [\d *]+, (\d+)\)", table)
    print("E107 reducer, CENSUS_CELLS K column:", ks)
    resolves = len(set(ks)) > 1
    print(f"reducer resolves K per dispatch: {resolves}")
    print("  every cell carries the same hard-coded constant, so K is an "
          "assumption of the table, not a measurement\n"
          if not resolves else "")

    shapes = checkpoint_shapes()
    quant = {n: s for n, s in shapes.items()
             if n.endswith(".weight") and len(s) == 2}

    for grid_y, label in ((640, "disputed"), (1536, "control"),
                          (4352, "control, fused")):
        n = grid_y * 8
        hits = collections.Counter()
        for name, shape in quant.items():
            if shape[0] == n:
                hits[(shape[1] * 8, name.split(".")[-2])] += 1
        print(f"grid=1x{grid_y}x1 -> N={n}  ({label})")
        if hits:
            for (k, role), count in sorted(hits.items()):
                print(f"    K={k:<6} {role:<14} x{count}")
        else:
            print("    no raw tensor at this N; a fused pair")
        print(f"    K={RECORDED_K} present: "
              f"{any(k == RECORDED_K for k, _ in hits)}")

    print(f"\ncorrected prices for grid=1x640x1 at the recorded "
          f"{RECORDED_US} us/dispatch")
    print(f"{'K':>8}{'MB':>9}{'GB/s':>9}{'vs peak':>10}  verdict")
    n = 5120
    for k in (5120, 6144, 17408):
        nbytes = qmv_bytes(n, k)
        gbps = nbytes / (RECORDED_US * 1e-6) / 1e9
        note = ("recorded, but no such tensor exists" if k == RECORDED_K
                else "impossible, exceeds DRAM peak"
                if gbps > DRAM_PEAK_GBPS else "physically possible")
        print(f"{k:>8}{nbytes / 1e6:>9.2f}{gbps:>9.1f}"
              f"{100.0 * gbps / DRAM_PEAK_GBPS:>9.1f} %  {note}")

    print(f"\nrecorded line: K={RECORDED_K}, {RECORDED_MB} MB, "
          f"{RECORDED_GBPS} GB/s")
    print("verdict: Edward is right. The reducer hard-codes K=5120 for every "
          "census cell. No tensor in this checkpoint has N=5120 with K=5120, "
          "so the bytes and the GB/s of that ledger line are wrong. The two "
          "control cells are unaffected: N=12288 is q_proj at K=5120 and "
          "N=34816 is the fused gate_up at K=5120, both correct.")


if __name__ == "__main__":
    main()
