#!/usr/bin/env python3
"""E130 rung 2 positive control: prove that the register ladder moves occupancy.

The QMV ladder holds the executed instruction stream fixed and walks the entry
register count. If its timing is flat, there are two readings:

  (a) occupancy changes and this kernel does not care, or
  (b) occupancy does not change with the register count at all.

Only (a) closes the occupancy axis. (b) would mean the instrument is blind and
the floor law is unverified, because `resident_simdgroups` is DERIVED from the
register count in every census this campaign has run, not measured.

This control separates them. It builds a deliberately latency-bound kernel: one
dependent fma chain per thread, so instruction-level parallelism is exactly 1
and only thread-level parallelism can hide the fma latency. Its throughput must
scale with resident simdgroups. The same dead-ballast trick then walks its
register count across the same treads. If that kernel's time moves on the
staircase while the QMV's does not, the ladder is real and the QMV is saturated.

  python3 research/e130_occupancy_control.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)

BUDGET = {"applegpu_g16s": 3072, "applegpu_g17s": 3968}

HEADER = """#include <metal_stdlib>
using namespace metal;

// One dependent fma chain per thread. ILP is 1 by construction, so the only
// way to hide the fma result latency is to hold more simdgroups resident.
// `ballast` is initialised from device memory with a thread-dependent address
// and consumed under a condition the compiler cannot fold, so it stays live
// across the chain without adding any chain work.
"""

KERNEL = """
[[kernel]] void e130_lat_b%(b)d(
    device float* y [[buffer(0)]],
    const device float* seed [[buffer(1)]],
    const constant int& iters [[buffer(2)]],
    const constant int& sink [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]]) {
  float ballast[%(bmax)d];
  for (int i = 0; i < %(b)d; i++) {
    ballast[i] = seed[(gid * 7u + uint(i) * 131u) & 4095u];
  }
  float acc = seed[gid & 4095u];
  for (int k = 0; k < iters; k++) {
    acc = fma(acc, 1.0000001f, 1e-7f);
    if (sink < 0) {
      for (int i = 0; i < %(b)d; i++) {
        ballast[i] = fma(ballast[i], 1.0000001f, acc);
      }
    }
  }
  float tail = 0.0f;
  for (int i = 0; i < %(b)d; i++) {
    tail += ballast[i];
  }
  y[gid] = acc + (sink < 0 ? tail : 0.0f);
}
"""


def source(widths: list[int]) -> str:
    bmax = max(widths) if widths else 1
    text = HEADER
    for b in widths:
        text += KERNEL % {"b": b, "bmax": max(bmax, 1)}
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-b", type=int, default=104)
    args = ap.parse_args()

    widths = list(range(0, args.max_b + 1, 1))
    text = source(widths)
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out / "e130_lat.metal"
    src.write_text(text)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = build_metallib(text, workdir / "lat")
        cells: dict = {}
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for name, record in translate(lib, arch, workdir / "lat").items():
                if not name.startswith("e130_lat_b"):
                    continue
                b = int(name[len("e130_lat_b"):])
                cells.setdefault(b, {})[arch] = {
                    "registers": record["registers"],
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                    "resident_simdgroups_derived":
                        BUDGET[arch] // record["registers"],
                }

    manifest = {"source": str(src), "cells": cells,
                "register_budget": BUDGET}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)
                                            + "\n")

    print("%-6s %-34s %-34s" % ("B", LOCAL_ARCH, RANKED_ARCH))
    for b in sorted(cells):
        a = cells[b][LOCAL_ARCH]
        r = cells[b][RANKED_ARCH]
        print("%-6d %4dr %4dsp %3dsg %7dt        %4dr %4dsp %3dsg %7dt"
              % (b, a["registers"], a["spill_bytes"],
                 a["resident_simdgroups_derived"], a["text_bytes"],
                 r["registers"], r["spill_bytes"],
                 r["resident_simdgroups_derived"], r["text_bytes"]))
    print("\nwrote %s" % (args.out / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
