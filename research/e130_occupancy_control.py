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

`--run` adds the stronger instrument. The same ballast walk is applied to a
kernel that COUNTS its own resident threadgroups through a device atomic, so
peak concurrency is measured rather than derived. The floor law predicts the
ratio between two register counts, and a ratio needs no shader-core count, so
the law becomes falsifiable on this host.

  python3 research/e130_occupancy_control.py --out DIR
  python3 research/e130_occupancy_control.py --out DIR --run --iters 20000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)

REPO = pathlib.Path(__file__).resolve().parent.parent

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


OCC_HEADER = """#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Peak-residency census. Thread 0 of each threadgroup registers the group in a
// device atomic on entry and records the live count it saw, then the group
// spins on the same dependent fma chain long enough for a whole wave to
// overlap, then deregisters. The host takes the maximum over groups, which is
// the peak number of threadgroups the device held resident at once.
"""

OCC_KERNEL = """
[[kernel]] void e130_occ_b%(b)d(
    device uint* observed [[buffer(0)]],
    device atomic_uint* live [[buffer(1)]],
    const device float* seed [[buffer(2)]],
    const constant int& iters [[buffer(3)]],
    const constant int& sink [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]]) {
  threadgroup uint entry_live;
  float ballast[%(bmax)d];
  for (int i = 0; i < %(b)d; i++) {
    ballast[i] = seed[(gid * 7u + uint(i) * 131u) & 4095u];
  }
  if (tid == 0) {
    entry_live =
        atomic_fetch_add_explicit(live, 1u, memory_order_relaxed) + 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc = seed[gid & 4095u];
  for (int k = 0; k < iters; k++) {
    acc = fma(acc, 1.0000001f, 1e-7f);
    if (sink > 0) {
      for (int i = 0; i < %(b)d; i++) {
        ballast[i] = fma(ballast[i], 1.0000001f, acc);
      }
    }
  }
  float tail = 0.0f;
  for (int i = 0; i < %(b)d; i++) {
    tail += ballast[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    // `tail` is finite for every seed, so the guard never fires, but the
    // compiler cannot prove it and must keep the ballast live across the chain.
    observed[tgid] = (acc + tail) > 1e30f ? 0u : entry_live;
    atomic_fetch_sub_explicit(live, 1u, memory_order_relaxed);
  }
}
"""


def source(widths: list[int]) -> str:
    bmax = max(widths) if widths else 1
    text = HEADER
    for b in widths:
        text += KERNEL % {"b": b, "bmax": max(bmax, 1)}
    return text


def occ_source(widths: list[int]) -> str:
    bmax = max(widths) if widths else 1
    text = OCC_HEADER
    for b in widths:
        text += OCC_KERNEL % {"b": b, "bmax": max(bmax, 1)}
    return text


def census(text: str, prefix: str, workdir: pathlib.Path) -> dict:
    lib = build_metallib(text, workdir / prefix)
    cells: dict = {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for name, record in translate(lib, arch, workdir / prefix).items():
            if not name.startswith(prefix):
                continue
            b = int(name[len(prefix):])
            cells.setdefault(b, {})[arch] = {
                "registers": record["registers"],
                "spill_bytes": record["spill_bytes"],
                "text_bytes": record["text_bytes"],
                "resident_simdgroups_derived":
                    BUDGET[arch] // record["registers"],
            }
    return cells


def run_probe(out: pathlib.Path, text: str, cells: dict, iters: int) -> dict:
    """Measure peak resident threadgroups, then test the floor law's ratio."""
    src = out / "e130_occ.metal"
    src.write_text(text)
    binary = out / "e130_occ_probe"
    build = subprocess.run(
        ["clang", "-fobjc-arc", "-O2", "-framework", "Metal",
         "-framework", "Foundation", "research/e130_occupancy_probe.m",
         "-o", str(binary)],
        cwd=REPO, capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit("probe build failed:\n" + build.stderr)

    # One kernel per distinct LOCAL register count: the law can only be tested
    # where it predicts a different answer.
    by_registers: dict[int, int] = {}
    for b in sorted(cells):
        by_registers.setdefault(cells[b][LOCAL_ARCH]["registers"], b)
    chosen = [by_registers[r] for r in sorted(by_registers)]
    names = ["e130_occ_b%d" % b for b in chosen]

    proc = subprocess.run([str(binary), str(src), str(iters), *names],
                          cwd=REPO, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit("probe run failed:\n" + proc.stderr)
    measured = json.loads(proc.stdout.split("\nJSON ", 1)[1])

    rows = []
    for cell in measured["cells"]:
        b = int(cell["function"][len("e130_occ_b"):])
        local = cells[b][LOCAL_ARCH]
        rows.append({
            "ballast": b,
            "registers": local["registers"],
            "spill_bytes": local["spill_bytes"],
            "resident_simdgroups_derived": local["resident_simdgroups_derived"],
            "peak_concurrent_threadgroups":
                cell["peak_concurrent_threadgroups"],
            "max_total_threads_per_threadgroup":
                cell["max_total_threads_per_threadgroup"],
            "seconds_min": cell["seconds_min"],
        })
    rows.sort(key=lambda r: r["registers"])

    reference = rows[0]
    for row in rows:
        predicted = (row["resident_simdgroups_derived"]
                     / reference["resident_simdgroups_derived"])
        observed = (row["peak_concurrent_threadgroups"]
                    / reference["peak_concurrent_threadgroups"])
        row["floor_law_predicted_ratio"] = predicted
        row["measured_ratio"] = observed
        row["ratio_error"] = observed - predicted
    measured["floor_law_test"] = {
        "reference_registers": reference["registers"],
        "rows": rows,
        "max_abs_ratio_error": max(abs(r["ratio_error"]) for r in rows),
        "concurrency_moved":
            len({r["peak_concurrent_threadgroups"] for r in rows}) > 1,
    }
    return measured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-b", type=int, default=104)
    ap.add_argument("--run", action="store_true",
                    help="measure peak resident threadgroups on this host")
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    widths = list(range(0, args.max_b + 1, 1))
    text = source(widths)
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out / "e130_lat.metal"
    src.write_text(text)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        cells = census(text, "e130_lat_b", workdir)

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

    if not args.run:
        return 0

    occ_text = occ_source(widths)
    with tempfile.TemporaryDirectory() as tmp:
        occ_cells = census(occ_text, "e130_occ_b", pathlib.Path(tmp))
    measured = run_probe(args.out, occ_text, occ_cells, args.iters)
    measured["register_budget"] = BUDGET
    measured["census"] = occ_cells
    (args.out / "occupancy-measured.json").write_text(
        json.dumps(measured, indent=1, sort_keys=True) + "\n")

    test = measured["floor_law_test"]
    print("\n%-6s %5s %5s %11s %11s %9s"
          % ("regs", "sg", "peakTG", "predicted", "measured", "error"))
    for row in test["rows"]:
        print("%-6d %5d %6d %11.4f %11.4f %9.4f"
              % (row["registers"], row["resident_simdgroups_derived"],
                 row["peak_concurrent_threadgroups"],
                 row["floor_law_predicted_ratio"], row["measured_ratio"],
                 row["ratio_error"]))
    print("\nconcurrency moved with the register count: %s"
          % test["concurrency_moved"])
    print("max absolute ratio error against the floor law: %.4f"
          % test["max_abs_ratio_error"])
    print("wrote %s" % (args.out / "occupancy-measured.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
