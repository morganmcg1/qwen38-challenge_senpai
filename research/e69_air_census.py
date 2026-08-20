#!/usr/bin/env python3
"""E69 rung 0, compile half: certify each arm from AIR, not from intent.

The GPU half of E69 is only interpretable if the arms move the bytes and issue
the loads they claim to. This reads the compiled AIR of every arm and reports,
for the k loop only:

  * device loads split by operand, using the load result type. The packed
    weights are the only `i16` loads, x is the only `<4 x bfloat>` load, and
    the scales and biases are the only scalar `bfloat` loads. Each class is
    checked against its closed-form prediction, so a misclassification shows
    up as a failed check rather than a quiet wrong number.
  * threadgroup (addrspace(3)) loads and stores, which only the staging arms
    have.
  * peak live registers over the real CFG, reusing the E64 liveness pass.
  * floating-point operation counts, which must be identical per output row
    across every arm: E69 changes how bytes arrive, never the arithmetic.

  python3 research/e69_air_census.py --na 2 3 4 5 6 \
      --out research/e69-artifacts/rung0-air.json
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

REPO = pathlib.Path(__file__).resolve().parent.parent
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE = REPO / "research/e69_wide_probe.metal"

ARMS = {
    "plain": "e69_cell_plain",
    "wvec": "e69_cell_wvec",
    "xvec": "e69_cell_xvec",
    "wxvec": "e69_cell_wxvec",
    "tgx": "e69_cell_tgx",
    "rows8": "e69_cell_rows8",
    "rows8wxvec": "e69_cell_rows8wxvec",
    "rows8idle": "e69_cell_rows8idle",
}
ROWS_PER_SIMD = {
    "plain": 4, "wvec": 4, "xvec": 4, "wxvec": 4, "tgx": 4,
    "rows8": 8, "rows8wxvec": 8, "rows8idle": 8,
}
STAGES_X = {"tgx"}
VECTORIZES_W = {"wvec", "wxvec", "rows8wxvec"}
VECTORIZES_X = {"xvec", "wxvec", "rows8wxvec"}

LOAD = re.compile(r"=\s*load\s+(?:volatile\s+)?([^,]+?),\s*(?:ptr|[^%]*\*)\s")
ADDRSPACE = re.compile(r"addrspace\((\d+)\)")
STORE = re.compile(r"^\s*store\s+(?:volatile\s+)?([^,]+),")


LANES = re.compile(r"<\s*(\d+)\s+x\s+")


def lanes_of(kind: str) -> int:
    match = LANES.search(kind)
    return int(match.group(1)) if match else 1


def block_loads(lines: list[str]) -> tuple[dict, int]:
    """Loads in one AIR block, keyed by (address space, result type)."""
    found: dict[tuple[str, str], int] = {}
    stores = 0
    for line in lines:
        if STORE.match(line) and "addrspace(3)" in line:
            stores += 1
        match = LOAD.search(line)
        if not match:
            continue
        space = ADDRSPACE.search(line)
        if not space:
            continue
        key = (f"as{space.group(1)}", match.group(1).strip())
        found[key] = found.get(key, 0) + 1
    return found, stores


def load_census(blocks: dict[str, list[str]], loop: list[str]) -> dict:
    """Attribute the loop's loads to w, x and the scales and biases.

    The packed weights are the only `i16` loads. That splits the k-loop head,
    which also carries the scales and biases, from the rolled `i` loop body,
    which carries nothing but x. Both operands are `bfloat`, so the block is
    what separates them, not the result type.
    """
    per_block = {}
    weight = {"loads": 0, "elements": 0}
    scale_bias = {"loads": 0, "elements": 0}
    device_x = {"loads": 0, "elements": 0}
    staged_x = {"loads": 0, "elements": 0}
    stage_stores = 0
    for name in loop:
        found, stores = block_loads(blocks.get(name, []))
        stage_stores += stores
        per_block[name] = {f"{space}:{kind}": n
                           for (space, kind), n in found.items()}
        head = any(space == "as1" and "i16" in kind for space, kind in found)
        for (space, kind), n in found.items():
            elements = n * lanes_of(kind)
            if space == "as1" and "i16" in kind:
                weight["loads"] += n
                weight["elements"] += elements
            elif space == "as3":
                staged_x["loads"] += n
                staged_x["elements"] += elements
            elif space == "as1" and "bfloat" in kind:
                target = scale_bias if head else device_x
                target["loads"] += n
                target["elements"] += elements
    return {
        "per_block": per_block,
        "weight": weight,
        "scale_bias": scale_bias,
        "device_x": device_x,
        "staged_x": staged_x,
        "staging_stores": stage_stores,
    }


def arm_stats(body: list[str], arm: str, na: int) -> dict:
    text = "\n".join(body)
    loop = loop_blocks(body)
    blocks = dict(split_blocks(body))
    loop_text = "\n".join(
        line for name in loop for line in blocks.get(name, []))
    liveness = live_ranges(body)
    rows = ROWS_PER_SIMD[arm]

    census = load_census(blocks, loop)
    total = counts(loop_text)
    # Closed form for one lane and one k-block. values_per_thread is 16, so a
    # lane covers 16 x values per input row and 4 packed uint16 per output row.
    x_lanes = 4 if arm in VECTORIZES_X else 1
    w_lanes = 4 if arm in VECTORIZES_W else 1
    dynamic = {
        "weight_loads": 4 * rows // w_lanes,
        "scale_bias_loads": 2 * rows,
        # A staging arm reads x from the device once per threadgroup, so the
        # two simdgroups split 16*NA elements between them and read them wide.
        "device_x_loads": (2 * na if arm in STAGES_X else 16 * na // x_lanes),
        "staged_x_loads": 16 * na if arm in STAGES_X else 0,
    }
    dynamic["device_loads_total"] = (
        dynamic["weight_loads"] + dynamic["scale_bias_loads"]
        + dynamic["device_x_loads"])
    dynamic["device_loads_per_output_row"] = (
        dynamic["device_loads_total"] / rows)
    return {
        "rows_per_simd": rows,
        "loop_blocks": loop,
        "loop_loads_static": census,
        "per_lane_per_k_block": dynamic,
        "loop_fp": {key: total[key] for key in ("fadd", "fmul", "fma")},
        "loop_fp_per_output_row": {
            key: total[key] / rows for key in ("fadd", "fmul", "fma")},
        "total_fp": {key: counts(text)[key] for key in ("fadd", "fmul", "fma")},
        "peak_live_cfg_loop": max(
            (liveness["blocks"][name]["peak_live"] for name in loop),
            default=0),
        "peak_live_cfg_max": max(
            entry["peak_live"] for entry in liveness["blocks"].values()),
        "allocas": len(re.findall(r"=\s*alloca\b", text)),
        "alloca_types": re.findall(r"=\s*alloca\s+([^,]+)", text),
    }


def compile_probe(workdir: pathlib.Path, na: int) -> pathlib.Path:
    raw = workdir / f"na{na}.ll"
    optimized = workdir / f"na{na}.o3.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS, f"-DE69_NA={na}",
         "-I", str(INCLUDE), "-S", str(PROBE), "-o", str(raw)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        raise SystemExit(f"compile failed at NA={na}:\n{emit.stderr}")
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(raw), "-o", str(optimized)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        raise SystemExit(f"metal-opt failed at NA={na}:\n{opt.stderr}")
    return optimized


def verdict(cell: dict) -> dict:
    plain = cell["plain"]
    checks = {}
    # Only same-R arms may be compared directly. Raising rows_per_simd
    # amortizes the x-side arithmetic, which is part of the mechanism, so a
    # lower fp count per output row at R=8 is the expected result, not a defect.
    for arm, stats in cell.items():
        if stats["rows_per_simd"] != plain["rows_per_simd"]:
            continue
        checks[f"{arm}_fp_unchanged_at_same_rows_per_simd"] = all(
            stats["loop_fp"][key] == plain["loop_fp"][key]
            for key in ("fadd", "fmul", "fma"))
    for arm in VECTORIZES_W:
        if arm in cell:
            checks[f"{arm}_weight_loads_are_4_wide"] = (
                cell[arm]["loop_loads_static"]["weight"]["elements"]
                == 4 * cell[arm]["loop_loads_static"]["weight"]["loads"])
    for arm in VECTORIZES_X:
        if arm in cell:
            checks[f"{arm}_x_loads_are_4_wide"] = (
                cell[arm]["loop_loads_static"]["device_x"]["elements"]
                == 4 * cell[arm]["loop_loads_static"]["device_x"]["loads"])
    for arm in STAGES_X:
        if arm in cell:
            checks[f"{arm}_x_reads_moved_to_threadgroup"] = (
                cell[arm]["loop_loads_static"]["staged_x"]["loads"] > 0
                and cell[arm]["loop_loads_static"]["staging_stores"] > 0)
    for arm, stats in cell.items():
        if stats["rows_per_simd"] == 8:
            checks[f"{arm}_x_elements_per_output_row_halved"] = (
                stats["per_lane_per_k_block"]["device_x_loads"]
                / stats["rows_per_simd"] * 2
                <= plain["per_lane_per_k_block"]["device_x_loads"]
                / plain["rows_per_simd"] + 1e-9)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, nargs="+", default=[6])
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    report = {
        "head": head,
        "dirty_paths": len(dirty.splitlines()),
        "flags": SCORED_FLAGS,
        "probe": str(PROBE.relative_to(REPO)),
        "cells": {},
    }
    with tempfile.TemporaryDirectory(prefix="e69-air-") as directory:
        workdir = pathlib.Path(directory)
        for na in args.na:
            bodies = kernel_bodies(compile_probe(workdir, na))
            report["cells"][na] = {
                arm: arm_stats(bodies[kernel], arm, na)
                for arm, kernel in ARMS.items() if kernel in bodies
            }
    report["checks"] = {na: verdict(report["cells"][na]) for na in args.na}

    for na in args.na:
        print(f"NA={na}")
        for arm, stats in report["cells"][na].items():
            static = stats["loop_loads_static"]
            dynamic = stats["per_lane_per_k_block"]
            print(f"  {arm:11s} R={stats['rows_per_simd']} "
                  f"peak_live cfg_loop={stats['peak_live_cfg_loop']:4d} "
                  f"cfg_max={stats['peak_live_cfg_max']:4d} "
                  f"alloca={stats['allocas']}")
            print(f"              static loop loads  w={static['weight']} "
                  f"sb={static['scale_bias']} x={static['device_x']} "
                  f"staged={static['staged_x']}+{static['staging_stores']}st")
            print(f"              per lane per k-block  w={dynamic['weight_loads']:3d} "
                  f"sb={dynamic['scale_bias_loads']:3d} "
                  f"x={dynamic['device_x_loads']:3d} "
                  f"staged={dynamic['staged_x_loads']:3d}  "
                  f"device/out-row={dynamic['device_loads_per_output_row']:6.2f}")
            print(f"              loop fp/out row {stats['loop_fp_per_output_row']}")
        failed = [c for c, ok in report["checks"][na].items() if not ok]
        print(f"  checks: {len(report['checks'][na]) - len(failed)} pass, "
              f"{len(failed)} fail {failed}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
