#!/usr/bin/env python3
"""E46 step 1: kernel-wide register max for the four contrast builds. No GPU.

Stop rule 3 is "any arm's measured register max exceeds 108". That number is not
a per-NA cell figure: E32/E36/E41 measured `..._wide<T, NA>` alone at
62/83/104/125, and the campaign's 108 (NA<=4 table) / 129 (NA<=5 table) are the
max over the seven WIDTH cases `..._m<T, M, IPG, true>`, each of which carries
its full group and tail cells. That max is what this script reports as
`kernel_wide_reg_max`, so an arm is comparable to the advisor's 108 rather than
to a number from a different probe.

A second, strictly larger readout compiles the real `affine_qmv_fast` entry
template, into which every width case plus `qmv_fast_impl` inline. The textual
liveness heuristic over-counts across a switch, so its absolute level is not
comparable to 108 -- but it is the same over-count in every arm, so a MOVE in it
is still evidence that an arm perturbed the shared allocation.

An arm is a patched copy of quantized.h in a shadow include directory searched
before the vendored tree. The working tree is never modified, so this runs
against a clean tree and its result is a property of a named rev.

  python3 research/e46_reg_census.py --out research/e46-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from air_kernel_stats import (  # noqa: E402
    ALLOCA,
    DEVICE_LOAD,
    peak_live_registers,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER_REL = "mlx/backend/metal/kernels/quantized.h"
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx" / HEADER_REL
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e46_entry_probe.metal"
ENTRIES = (
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_1",
)
CELL = "e46_cell"
WIDTHS = list(range(3, 10))
# The tip's kernel-wide max over width cases. Stop rule 3 fires above this.
CEILING = 108

SHIPPED_M6 = "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>"
SHIPPED_M8 = "qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>"
ARM_M6 = "qmv_fast_crossrow_affine4_g64_m<T, 6, 4, true>"
ARM_M8 = "qmv_fast_crossrow_affine4_g64_m<T, 8, 3, true>"

# `[4 x <3 x float>]` and friends: the accumulator array in private memory.
ACC_ALLOCA = re.compile(r"^\[\s*(\d+)\s+x\s+<\s*(\d+)\s+x\s+float\s*>\s*\]$")
# crossrow_na_occupancy.swift pads its name column to 18 and TRUNCATES longer
# names, so an entry-point name runs into maxThreads with no separator. Same
# anchor E32 used: the three trailing integers.
OCC_ROW = re.compile(r"^(?P<name>.*?)(?P<max>\d{3,4})\s+(?P<width>\d+)\s+(?P<tgmem>\d+)\s*$")

ARMS = [
    {"name": "A1_m6_ipg3", "contrast": "A", "arm": "shipped", "treated_m": 6,
     "edits": []},
    {"name": "A2_m6_ipg4", "contrast": "A", "arm": "treated", "treated_m": 6,
     "edits": [(SHIPPED_M6, ARM_M6)]},
    {"name": "B1_m8_ipg4", "contrast": "B", "arm": "shipped", "treated_m": 8,
     "edits": []},
    {"name": "B2_m8_ipg3", "contrast": "B", "arm": "treated", "treated_m": 8,
     "edits": [(SHIPPED_M8, ARM_M8)]},
    # The build that will actually be timed: both treated widths at once, so
    # widths 1,2,3,4,5,7,9 stay byte-identical controls inside one session.
    {"name": "AB_timed_arm", "contrast": "A+B", "arm": "treated", "treated_m": 0,
     "edits": [(SHIPPED_M6, ARM_M6), (SHIPPED_M8, ARM_M8)]},
]


def ipg_table(text: str) -> dict[int, int]:
    out = {}
    for m in WIDTHS:
        hit = re.search(r"qmv_fast_crossrow_affine4_g64_m<T, %d, (\d)" % m, text)
        out[m] = int(hit.group(1)) if hit else 0
    return out


def streams(table: dict[int, int]) -> dict[int, int]:
    return {m: (m + ipg - 1) // ipg if ipg else 0 for m, ipg in table.items()}


def na_cells(m: int, ipg: int) -> list[int]:
    """Every NA the wide helper is instantiated at for one width case."""
    if not ipg:
        return []
    cells = {ipg}
    tail = m % ipg
    if tail:
        cells.add(max(tail, 2))
    return sorted(cells)


def air_stats(body: list[str]) -> dict:
    allocas = [ALLOCA.search(l).group(1) for l in body if ALLOCA.search(l)]
    peak, vals = peak_live_registers(body)
    return {
        "peak_live_regs": peak,
        "peak_live_values": vals,
        "air_lines": len(body),
        "device_loads": sum(1 for l in body if DEVICE_LOAD.search(l)),
        "allocas": len(allocas),
        "alloca_types": sorted(set(allocas)),
        "acc_alloca_types": sorted({a for a in allocas if ACC_ALLOCA.match(a)}),
    }


def compile_probe(shadow: pathlib.Path, tag: str, defines: dict[str, int],
                  wanted: tuple[str, ...]) -> dict:
    ll = shadow / ("%s.ll" % tag)
    ll_o3 = shadow / ("%s.o3.ll" % tag)
    flags = ["-D%s=%d" % (k, v) for k, v in defines.items()]
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", *flags,
         "-I", str(shadow), "-I", str(INCLUDE), "-S", str(PROBE), "-o", str(ll)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        return {"status": "compile_failed",
                "error": emit.stderr.strip().splitlines()[-6:]}
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        return {"status": "metal_opt_failed",
                "error": opt.stderr.strip().splitlines()[-6:]}

    lines = ll_o3.read_text().splitlines()
    found = {}
    for name in wanted:
        body, inside = [], False
        for line in lines:
            if line.startswith("define ") and ("@%s(" % name) in line:
                inside = True
            elif inside and line == "}":
                inside = False
            elif inside:
                body.append(line)
        if not body:
            return {"status": "kernel_not_found", "missing": name}
        found[name] = air_stats(body)
    return {"status": "ok", "functions": found}


def build_occupancy_tool(workdir: pathlib.Path) -> pathlib.Path | None:
    """Compile the existing occupancy reader once; None if it will not build."""
    binary = workdir / "na_occupancy"
    build = subprocess.run(
        ["swiftc", "-O", str(REPO / "research/crossrow_na_occupancy.swift"),
         "-o", str(binary)],
        capture_output=True, text=True)
    return binary if build.returncode == 0 else None


def occupancy(shadow: pathlib.Path, tool: pathlib.Path) -> dict:
    """maxTotalThreadsPerThreadgroup of the real entry point.

    The back end's register allocation caps this, so it is the only NON-textual
    readout available without Xcode's GPU debugger: if two arms agree here, the
    hardware gave them the same budget, whatever the AIR heuristic says. Building
    a pipeline compiles AIR to ISA on the driver; it dispatches no GPU work.
    """
    air = shadow / "entry.air"
    lib = shadow / "entry.metallib"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2",
         "-I", str(shadow), "-I", str(INCLUDE), "-c", str(PROBE), "-o", str(air)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        return {"status": "air_failed"}
    link = subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air),
                           "-o", str(lib)], capture_output=True, text=True)
    if link.returncode != 0:
        return {"status": "metallib_failed"}
    run = subprocess.run([str(tool), str(lib)], capture_output=True, text=True)
    if run.returncode != 0:
        return {"status": "pipeline_failed",
                "error": run.stderr.strip().splitlines()[-4:]}
    out = {"status": "ok", "functions": {}}
    for line in run.stdout.splitlines()[2:]:  # device= line, then the header
        hit = OCC_ROW.match(line)
        if hit:
            out["functions"][hit["name"].strip()] = {
                "max_total_threads_per_threadgroup": int(hit["max"]),
                "thread_execution_width": int(hit["width"]),
                "static_threadgroup_memory_bytes": int(hit["tgmem"]),
            }
    for line in run.stdout.splitlines():
        if line.startswith("device="):
            out["device"] = line.split("=", 1)[1]
    if not out["functions"]:
        out["status"] = "no_rows_parsed"
        out["raw"] = run.stdout.splitlines()[:6]
    return out


def compile_arm(arm: dict, workdir: pathlib.Path,
                occupancy_tool: pathlib.Path | None = None) -> dict:
    shadow = workdir / arm["name"]
    header_dst = shadow / HEADER_REL
    header_dst.parent.mkdir(parents=True, exist_ok=True)
    text = HEADER.read_text()
    for old, new in arm["edits"]:
        if text.count(old) != 1:
            return dict(arm, status="edit_not_unique", pattern=old,
                        occurrences=text.count(old))
        text = text.replace(old, new)
    header_dst.write_text(text)

    table = ipg_table(text)
    out = dict(arm, ipg_table=table, streams=streams(table),
               na_cells={m: na_cells(m, table[m]) for m in WIDTHS})

    cells = {}
    for m in WIDTHS:
        res = compile_probe(shadow, "cell_m%d" % m,
                            {"E46_CELL_M": m, "E46_CELL_IPG": table[m]}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        cells[m] = dict(res["functions"][CELL], ipg=table[m],
                        na_cells=na_cells(m, table[m]))

    entry = compile_probe(shadow, "entry", {}, ENTRIES)
    if entry["status"] != "ok":
        return dict(out, status="entry_%s" % entry["status"],
                    error=entry.get("error"))

    width_max = max(c["peak_live_regs"] for c in cells.values())
    entry_max = max(f["peak_live_regs"] for f in entry["functions"].values())
    out = dict(out, status="ok", width_cells=cells, entry=entry["functions"],
               kernel_wide_reg_max=width_max,
               argmax_width=max(cells, key=lambda m: cells[m]["peak_live_regs"]),
               entry_point_reg_max=entry_max,
               exceeds_ceiling=width_max > CEILING)
    if occupancy_tool is not None:
        out["occupancy"] = occupancy(shadow, occupancy_tool)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e46-reg-census.json")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-occupancy", action="store_true",
                    help="skip the pipeline readout (needs a Metal device)")
    args = ap.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e46-reg-census-"))
    try:
        tool = None if args.no_occupancy else build_occupancy_tool(workdir)
        cells = [compile_arm(arm, workdir, tool) for arm in ARMS]
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    ok = [c for c in cells if c["status"] == "ok"]
    payload = {
        "head": head,
        "dirty_paths": len(dirty.splitlines()) if dirty else 0,
        "ceiling": CEILING,
        "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
        "readouts": {
            "kernel_wide_reg_max": "max over width cases ..._m<T, M, IPG, true>",
            "entry_point_reg_max": "affine_qmv_fast, everything inlined",
        },
        "cells": cells,
        "all_ok": len(ok) == len(cells),
        "kernel_wide_reg_max": {c["name"]: c["kernel_wide_reg_max"] for c in ok},
        "entry_point_reg_max": {c["name"]: c["entry_point_reg_max"] for c in ok},
        "any_exceeds_ceiling": any(c["exceeds_ceiling"] for c in ok),
    }
    payload["all_arms_equal"] = (
        len({c["kernel_wide_reg_max"] for c in ok}) == 1 if ok else False)
    payload["entry_max_equal"] = (
        len({c["entry_point_reg_max"] for c in ok}) == 1 if ok else False)

    occ = {c["name"]: c["occupancy"] for c in ok if c.get("occupancy")}
    if occ:
        threads = {
            name: {fn: v["max_total_threads_per_threadgroup"]
                   for fn, v in o.get("functions", {}).items()}
            for name, o in occ.items()
        }
        payload["occupancy_max_threads"] = threads
        flat = {tuple(sorted(v.items())) for v in threads.values()}
        payload["occupancy_equal_across_arms"] = len(flat) == 1
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("E46 register census   head=%s dirty=%d   ceiling=%d"
          % (head[:8], payload["dirty_paths"], CEILING))
    print("%-14s %-7s %-6s %-6s %-24s %s"
          % ("arm", "status", "wide", "entry", "per-width regs M=3..9", "note"))
    for c in cells:
        if c["status"] != "ok":
            print("%-14s %-7s" % (c["name"], c["status"]), c.get("error", ""))
            continue
        per = " ".join("%d" % c["width_cells"][m]["peak_live_regs"] for m in WIDTHS)
        note = "OVER CEILING" if c["exceeds_ceiling"] else ""
        print("%-14s %-7s %-6d %-6d %-24s %s"
              % (c["name"], c["status"], c["kernel_wide_reg_max"],
                 c["entry_point_reg_max"], per, note))
    for c in ok:
        print("  %-14s streams(M=3..9) = %s   IPG = %s"
              % (c["name"],
                 " ".join(str(c["streams"][m]) for m in WIDTHS),
                 " ".join(str(c["ipg_table"][m]) for m in WIDTHS)))
    if "occupancy_max_threads" in payload:
        print("\nmaxTotalThreadsPerThreadgroup (pipeline, non-textual):")
        for name, fns in payload["occupancy_max_threads"].items():
            print("  %-14s %s" % (name, fns))
        print("  equal across arms: %s"
              % payload["occupancy_equal_across_arms"])
    print("\nany_exceeds_ceiling=%s  all_arms_equal=%s  entry_max_equal=%s"
          % (payload["any_exceeds_ceiling"], payload["all_arms_equal"],
             payload["entry_max_equal"]))
    print("wrote %s" % args.out)
    return 0 if payload["all_ok"] and not payload["any_exceeds_ceiling"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
