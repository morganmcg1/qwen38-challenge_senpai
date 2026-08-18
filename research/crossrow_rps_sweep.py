#!/usr/bin/env python3
"""Compile the E32 NA x rows_per_simd grid and report the register/spill gate.

One kernel per translation unit, so each cell's AIR is attributable to that cell
and not to whatever else the compiler inlined alongside it.

The spill gate is deliberately narrow. `alloca > 0` is NOT spilling: the shipped
crossrow bodies keep a small fixed private array, and the `_m` dispatch form
shows 2 allocas simply because it inlines two bodies. What indicates the register
allocator gave up on the accumulators is an alloca whose TYPE is the accumulator
array, `[rows_per_simd x <NA x float>]`. That is the type this gate matches, and
research/crossrow_rps_sweep.py --self-test proves it fires on a known-bad cell
and stays quiet on a known-good one.

  python3 research/crossrow_rps_gen.py
  python3 research/crossrow_rps_sweep.py --out research/e32-rps-grid.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from air_kernel_stats import ALLOCA, peak_live_registers  # noqa: E402

PROBE = pathlib.Path("research/crossrow_rps_probe.metal")
INCLUDE = pathlib.Path("Vendor/mlx-swift/Source/Cmlx/mlx")
NA_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
R_VALUES = [1, 2, 3, 4]
# `[4 x <5 x float>]` and friends: the accumulator array itself landing in
# private memory. Threadgroup/private scratch of any other shape is not a spill.
ACC_ALLOCA = re.compile(r"^\[\s*(\d+)\s+x\s+<\s*(\d+)\s+x\s+float\s*>\s*\]$")
THREADGROUP = re.compile(r"addrspace\(3\)")


def compile_cell(cell: dict, workdir: pathlib.Path) -> dict:
    name = cell["name"]
    ll = workdir / f"{name}.ll"
    ll_o3 = workdir / f"{name}.o3.ll"
    air = workdir / f"{name}.air"
    defines = [f"-D{k}={v}" if v is not None else f"-D{k}" for k, v in cell["defines"].items()]
    base = [
        "xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2",
        *defines, "-I", str(INCLUDE), str(PROBE),
    ]
    out = dict(cell)
    emit = subprocess.run([*base, "-S", "-o", str(ll)], capture_output=True, text=True)
    if emit.returncode != 0:
        out["status"] = "compile_failed"
        out["error"] = emit.stderr.strip().splitlines()[-6:]
        return out

    # E27 measured its ladder after this extra pass, so the anchor cells only
    # reproduce its numbers if the probe runs the same pipeline.
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)],
        capture_output=True, text=True,
    )
    if opt.returncode != 0:
        out["status"] = "metal_opt_failed"
        out["error"] = opt.stderr.strip().splitlines()[-6:]
        return out

    body = []
    inside = False
    for line in ll_o3.read_text().splitlines():
        if line.startswith("define ") and f"@{name}(" in line:
            inside = True
        elif inside and line == "}":
            inside = False
        elif inside:
            body.append(line)
    if not body:
        out["status"] = "kernel_not_found"
        return out

    allocas = [ALLOCA.search(line).group(1) for line in body if ALLOCA.search(line)]
    acc_allocas = [a for a in allocas if ACC_ALLOCA.match(a)]
    peak_regs, peak_vals = peak_live_registers(body)
    obj = subprocess.run([*base, "-c", "-o", str(air)], capture_output=True, text=True)
    out.update(
        status="ok",
        allocas=len(allocas),
        alloca_types=sorted(set(allocas)),
        acc_spill=len(acc_allocas) > 0,
        acc_alloca_types=sorted(set(acc_allocas)),
        peak_live_regs=peak_regs,
        peak_live_values=peak_vals,
        air_lines=len(body),
        device_loads=sum(1 for line in body if "addrspace(1)" in line and "= load" in line),
        threadgroup_refs=sum(1 for line in body if THREADGROUP.search(line)),
        air_object=str(air) if obj.returncode == 0 else None,
        pipeline="metal -O2 -S | metal-opt -passes=default<O3>",
    )
    return out


def cells() -> list[dict]:
    out = []
    for na in NA_VALUES:
        for r in R_VALUES:
            out.append({
                "name": f"xrps_na{na}_r{r}", "arm": "grid_relaxed", "na": na, "r": r,
                "defines": {"PROBE_CELL_RPS": None, "PROBE_NAME": f"xrps_na{na}_r{r}",
                            "PROBE_NA": na, "PROBE_R": r},
            })
        for r in (1, 2, 4):
            out.append({
                "name": f"xrb_na{na}_r{r}", "arm": "coverage_preserving", "na": na, "r": r,
                "defines": {"PROBE_CELL_RB": None, "PROBE_NAME": f"xrb_na{na}_r{r}",
                            "PROBE_NA": na, "PROBE_R": r},
            })
    for na in (2, 3, 4, 5):
        out.append({
            "name": f"xship_na{na}", "arm": "shipped_anchor", "na": na, "r": 4,
            "defines": {"PROBE_CELL_SHIPPED": None, "PROBE_NAME": f"xship_na{na}",
                        "PROBE_NA": na},
        })
    for na, r in ((4, 4), (9, 2), (16, 4)):
        out.append({
            "name": f"xctl_spill_na{na}_r{r}", "arm": "negative_control", "na": na, "r": r,
            "defines": {"PROBE_CELL_FORCED_SPILL": None,
                        "PROBE_NAME": f"xctl_spill_na{na}_r{r}",
                        "PROBE_NA": na, "PROBE_R": r},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e32-rps-grid.json")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep-air", default="")
    args = ap.parse_args()

    subprocess.run([sys.executable, "research/crossrow_rps_gen.py", "--check"], check=True)

    workdir = pathlib.Path(args.keep_air) if args.keep_air else pathlib.Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    todo = cells()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda c: compile_cell(c, workdir), todo))

    results.sort(key=lambda r: (r["arm"], r["na"], r["r"]))
    pathlib.Path(args.out).write_text(json.dumps({"workdir": str(workdir), "cells": results}, indent=2))

    print(f"{'cell':22} {'arm':20} {'NA':>3} {'r':>2} {'NAxr':>5} {'regs':>5} "
          f"{'allocas':>7} {'accSpill':>8} {'status'}")
    for row in results:
        if row["status"] != "ok":
            print(f"{row['name']:22} {row['arm']:20} {row['na']:>3} {row['r']:>2} "
                  f"{row['na'] * row['r']:>5} {'-':>5} {'-':>7} {'-':>8} {row['status']}")
            continue
        print(f"{row['name']:22} {row['arm']:20} {row['na']:>3} {row['r']:>2} "
              f"{row['na'] * row['r']:>5} {row['peak_live_regs']:>5} {row['allocas']:>7} "
              f"{str(row['acc_spill']):>8} ok")
    print(f"\nAIR objects in {workdir}")


if __name__ == "__main__":
    main()
