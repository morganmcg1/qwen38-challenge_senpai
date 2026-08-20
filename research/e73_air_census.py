#!/usr/bin/env python3
"""E73 rung 0: the static cost of every legal `(M, IPG)` group partition.

Rung 1 asks the GPU what a partition costs. This asks the compiler what a
partition IS, so the timing result can be read against a static census instead
of against intent. For each legal pair it reports, from the compiled AIR of the
SHIPPED wrapper:

  * `groups = ceil(M / IPG)`, the active x-slot count and the weight-stream
    count, and the idle x-slots the frozen grid still launches;
  * live accumulator floats `9 * IPG`, from `acc[4]`, `partial[4]` and `sums`;
  * every alloca WITH its element type and byte size, so an accumulator that
    left the registers is visible as `[4 x <6 x float>]` and not as a count;
  * peak live registers over the real CFG, reusing the E64 liveness pass;
  * the floating-point and device-load counts of the heaviest loop.

`maxTotalThreadsPerThreadgroup` and the other pipeline figures come from
`research/e73_pipeline_probe.m`, which needs a device. This half is offline.

  python3 research/e73_air_census.py --out research/e73-artifacts/rung0-air.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e64_air_census import (  # noqa: E402
    SCORED_FLAGS,
    counts,
    kernel_bodies,
    live_ranges,
    loop_blocks,
    split_blocks,
)
from e73_pairs import (  # noqa: E402
    SHIPPED,
    CROWN,
    bodies,
    groups,
    live_floats,
    name,
    pairs,
    tail,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e73_ipg_probe.metal"

ALLOCA = re.compile(r"=\s*alloca\s+([^,\n]+?)(?:,\s*align\s+\d+)?\s*$", re.M)
VECTOR = re.compile(r"<\s*(\d+)\s+x\s+([\w.]+)\s*>")
ARRAY = re.compile(r"\[\s*(\d+)\s+x\s+(.+)\s*\]$")
SCALAR_BYTES = {"float": 4, "i32": 4, "i16": 2, "bfloat": 2, "half": 2,
                "i8": 1, "i1": 1, "i64": 8, "double": 8, "ptr": 8}

# Apple M4 Pro (Apple 9 family). Apple publishes no per-core register file
# size, so every simdgroup-residency number below is DERIVED from this stated
# assumption and must be read as derived, never as measured.
REGISTER_FILE_BYTES_PER_CORE = 384 * 1024
SIMD_WIDTH = 32


def type_bytes(kind: str) -> int:
    kind = kind.strip()
    array = ARRAY.match(kind)
    if array:
        return int(array.group(1)) * type_bytes(array.group(2))
    vector = VECTOR.fullmatch(kind)
    if vector:
        return int(vector.group(1)) * SCALAR_BYTES.get(vector.group(2), 4)
    return SCALAR_BYTES.get(kind, 4)


def is_accumulator_type(kind: str) -> bool:
    """An array of float vectors is the `acc`/`partial` shape, not scratch."""
    array = ARRAY.match(kind.strip())
    if not array:
        return False
    return bool(VECTOR.fullmatch(array.group(2).strip()))


def cell_stats(body: list[str], m: int, ipg: int) -> dict:
    text = "\n".join(body)
    loop = loop_blocks(body)
    blocks = dict(split_blocks(body))
    loop_text = "\n".join(line for b in loop for line in blocks.get(b, []))
    liveness = live_ranges(body)
    allocas = [kind.strip() for kind in ALLOCA.findall(text)]
    alloca_bytes = sum(type_bytes(kind) for kind in allocas)
    accumulator_allocas = [kind for kind in allocas if is_accumulator_type(kind)]
    peak_loop = max(
        (liveness["blocks"][b]["peak_live"] for b in loop), default=0)
    peak_max = max(entry["peak_live"] for entry in liveness["blocks"].values())
    # Derived only. 32-bit registers per lane -> bytes per simdgroup.
    bytes_per_simdgroup = peak_loop * 4 * SIMD_WIDTH
    return {
        "m": m,
        "ipg": ipg,
        "groups": groups(m, ipg),
        "idle_x_slots": m - groups(m, ipg),
        "tail": tail(m, ipg),
        "inlined_bodies": bodies(m, ipg),
        "live_accumulator_floats": live_floats(ipg),
        "allocas": len(allocas),
        "alloca_types": allocas,
        "alloca_bytes": alloca_bytes,
        "accumulator_allocas": accumulator_allocas,
        "spilled": bool(accumulator_allocas),
        "peak_live_cfg_loop": peak_loop,
        "peak_live_cfg_max": peak_max,
        "derived_bytes_per_simdgroup": bytes_per_simdgroup,
        "derived_resident_simdgroups_per_core": (
            REGISTER_FILE_BYTES_PER_CORE // bytes_per_simdgroup
            if bytes_per_simdgroup else None),
        "loop_blocks": len(loop),
        "loop_fp": {key: counts(loop_text)[key]
                    for key in ("fadd", "fmul", "fma")},
        "loop_device_loads": counts(loop_text)["device_loads"],
        "loop_private_loads": counts(loop_text)["private_loads"],
        "loop_private_stores": counts(loop_text)["private_stores"],
        "instructions": len(body),
        "shipped": SHIPPED.get(m) == ipg,
        "crown": CROWN.get(m) == ipg,
    }


def compile_probe(workdir: pathlib.Path) -> pathlib.Path:
    raw = workdir / "e73.ll"
    optimized = workdir / "e73.o3.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS,
         "-I", str(INCLUDE), "-S", str(PROBE), "-o", str(raw)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        raise SystemExit(f"compile failed:\n{emit.stderr}")
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(raw), "-o", str(optimized)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        raise SystemExit(f"metal-opt failed:\n{opt.stderr}")
    return optimized


def verdict(cells: dict[str, dict]) -> dict:
    """The pre-registered predictions, checked one at a time."""
    checks = {}
    ipg6 = [c for c in cells.values() if c["ipg"] == 6]
    checks["P1_every_ipg6_cell_spills"] = all(c["spilled"] for c in ipg6)
    checks["P1_ipg6_spill_type_is_6_wide"] = all(
        any("6 x float" in kind for kind in c["accumulator_allocas"])
        for c in ipg6)
    checks["P1_no_cell_below_ipg6_spills"] = not any(
        c["spilled"] for c in cells.values() if c["ipg"] < 6)
    checks["P2_m5_ipg5_has_one_alloca"] = cells["m5_ipg5"]["allocas"] == 1
    below = {ipg: max(c["peak_live_cfg_loop"] for c in cells.values()
                      if c["ipg"] == ipg and c["tail"] == 0)
             for ipg in (2, 3, 4, 5)
             if any(c["ipg"] == ipg and c["tail"] == 0
                    for c in cells.values())}
    checks["P3_peak_live_monotone_in_ipg_below_cliff"] = all(
        below[a] <= below[b] for a, b in zip(sorted(below), sorted(below)[1:]))
    predicted = {"m8_ipg6": 3, "m9_ipg6": 3, "m9_ipg5": 2, "m7_ipg5": 2,
                 "m7_ipg4": 2, "m5_ipg3": 2, "m6_ipg4": 2, "m8_ipg3": 2,
                 "m8_ipg5": 2, "m6_ipg6": 2}
    checks["P4_tail_alloca_counts"] = {
        key: {"predicted": value, "observed": cells[key]["allocas"],
              "hit": cells[key]["allocas"] == value}
        for key, value in predicted.items()}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    report = {
        "experiment": "e73",
        "rung": 0,
        "harness": "compile-probe",
        "head": head,
        "dirty_paths": len(dirty.splitlines()),
        "flags": SCORED_FLAGS,
        "probe": str(PROBE.relative_to(REPO)),
        "register_file_bytes_per_core_assumed": REGISTER_FILE_BYTES_PER_CORE,
        "cells": {},
    }
    with tempfile.TemporaryDirectory(prefix="e73-air-") as directory:
        bodies_by_kernel = kernel_bodies(compile_probe(pathlib.Path(directory)))
    for m, ipg in pairs():
        kernel = f"e73_cell_{name(m, ipg)}"
        if kernel not in bodies_by_kernel:
            raise SystemExit(f"missing kernel {kernel} in AIR")
        report["cells"][name(m, ipg)] = cell_stats(
            bodies_by_kernel[kernel], m, ipg)
    report["checks"] = verdict(report["cells"])

    print(f"{'cell':>10} {'grp':>3} {'idle':>4} {'bodies':>7} {'liveF':>5} "
          f"{'alloca':>6} {'bytes':>5} {'spill':>5} {'peakLoop':>8} "
          f"{'peakMax':>7} {'sg/core*':>8}  types")
    for key, cell in report["cells"].items():
        flag = ("S" if cell["shipped"] else "") + ("C" if cell["crown"] else "")
        print(f"{key:>10} {cell['groups']:3d} {cell['idle_x_slots']:4d} "
              f"{','.join(str(b) for b in cell['inlined_bodies']):>7} "
              f"{cell['live_accumulator_floats']:5d} {cell['allocas']:6d} "
              f"{cell['alloca_bytes']:5d} "
              f"{'YES' if cell['spilled'] else 'no':>5} "
              f"{cell['peak_live_cfg_loop']:8d} {cell['peak_live_cfg_max']:7d} "
              f"{cell['derived_resident_simdgroups_per_core']:8d} "
              f" {';'.join(cell['alloca_types'])} {flag}")
    print("* derived from the assumed register file size, not measured")
    for check, value in report["checks"].items():
        if isinstance(value, dict):
            misses = [k for k, v in value.items() if not v["hit"]]
            print(f"{check}: {len(value) - len(misses)}/{len(value)} hit, "
                  f"missed {misses}")
        else:
            print(f"{check}: {'HIT' if value else 'MISS'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
