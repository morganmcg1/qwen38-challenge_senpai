#!/usr/bin/env python3
"""E64 rung 0c: the compile-only NA=5 vs NA=6 structural differential.

Rung 0b can kill the accumulator hypothesis, but the step in the measured ladder
stays real and unexplained. This dumps the AIR of the shipped wide crossrow body
across the NA ladder and reports EVERY structural difference between two widths,
not only the alloca: opcode mix inside and outside the k loop, `phi` type
widths, k-loop loads and stores by address space, the vector-width histogram,
lane-shuffling work, and how `simd_sum` is emitted.

Differences are ranked by plausible cost with one explicit ruler taken from the
measured ladder itself: NA -> NA+1 adds exactly 4 in-loop vector ops and costs
11.65 ms of one stream in the E63 fit, so one extra in-loop vector op is worth
about 2.91 ms. The ruler prices only what repeats per k block; a difference
outside the loop is reported with cost `once` and never priced with it.

  python3 research/e64_rung0c_diff.py --out research/e64-artifacts/rung0c-diff.json
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
    compile_probe,
    kernel_bodies,
    live_ranges,
    loop_blocks,
    split_blocks,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
ARM = "e64_cell_plain"
# E63 fit: the arithmetic term is 11.65 ms per NA of one stream, and NA -> NA+1
# adds exactly 4 in-loop vector ops (fadd 12,16,20,24,28,32,36,40 for NA=2..9).
MS_PER_LOOP_VECTOR_OP = 11.65 / 4.0
# askeladd's E61 rung 1 single-stream ladder, in ms.
LADDER_MS = {2: 64.40, 3: 72.17, 4: 82.24, 5: 95.48, 6: 122.34, 7: 147.21}

OPCODE = re.compile(r"^\s*(?:%[\w.\-]+\s*=\s*)?([a-z][\w.]*)\b")
CALL = re.compile(r"call\s+[^@]*@([\w.$]+)")
TYPED = re.compile(r"<(\d+) x ([\w]+)>")
ADDRSPACE = re.compile(r"addrspace\((\d+)\)")


def histogram(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        call = CALL.search(line)
        if call:
            name = "call " + re.sub(r"\.(f32|v\d+f32|i32|p0i8)$", "",
                                    call.group(1))
            counts[name] = counts.get(name, 0) + 1
            continue
        match = OPCODE.match(line)
        if not match:
            continue
        opcode = match.group(1)
        if opcode in {"br", "ret"}:
            continue
        counts[opcode] = counts.get(opcode, 0) + 1
    return counts


def memory_by_addrspace(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        kind = None
        if re.match(r"^%[\w.\-]+\s*=\s*load\b", stripped):
            kind = "load"
        elif stripped.startswith("store "):
            kind = "store"
        if not kind:
            continue
        space = ADDRSPACE.search(line)
        key = f"{kind}_addrspace_{space.group(1) if space else '0_private'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def vector_widths(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        if not re.match(r"^\s*%[\w.\-]+\s*=", line):
            continue
        match = TYPED.search(line)
        if match:
            key = f"<{match.group(1)} x {match.group(2)}>"
            counts[key] = counts.get(key, 0) + 1
    return counts


def phi_widths(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        if "= phi " not in line:
            continue
        match = TYPED.search(line)
        key = f"<{match.group(1)} x {match.group(2)}>" if match else "scalar"
        counts[key] = counts.get(key, 0) + 1
    return counts


def cell(body: list[str]) -> dict:
    blocks = split_blocks(body)
    loop = set(loop_blocks(body))
    loop_lines = [line for name, lines in blocks if name in loop
                  for line in lines]
    outside = [line for name, lines in blocks if name not in loop
               for line in lines]
    liveness = live_ranges(body)
    return {
        "air_lines": len(body),
        "loop_blocks": sorted(loop),
        "loop": {
            "opcodes": histogram(loop_lines),
            "memory": memory_by_addrspace(loop_lines),
            "vector_widths": vector_widths(loop_lines),
            "phi_widths": phi_widths(loop_lines),
            "lines": len(loop_lines),
        },
        "outside_loop": {
            "opcodes": histogram(outside),
            "memory": memory_by_addrspace(outside),
            "vector_widths": vector_widths(outside),
            "phi_widths": phi_widths(outside),
            "lines": len(outside),
        },
        "allocas": [line.strip() for line in body if " alloca " in line],
        "peak_live_cfg_loop": max(
            (liveness["blocks"][name]["peak_live"] for name in loop),
            default=0),
    }


def difference(low: dict, high: dict, region: str, key: str) -> list[dict]:
    rows = []
    left = low[region][key]
    right = high[region][key]
    for name in sorted(set(left) | set(right)):
        delta = right.get(name, 0) - left.get(name, 0)
        if delta == 0:
            continue
        rows.append({
            "region": region,
            "kind": key,
            "name": name,
            "low": left.get(name, 0),
            "high": right.get(name, 0),
            "delta": delta,
            "priced_ms": abs(delta) * MS_PER_LOOP_VECTOR_OP
                         if region == "loop" else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--na", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--pair", type=int, nargs=2, default=[5, 6])
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    cells: dict[int, dict] = {}
    with tempfile.TemporaryDirectory(prefix="e64-0c-") as directory:
        workdir = pathlib.Path(directory)
        for na in args.na:
            bodies = kernel_bodies(compile_probe(workdir, na))
            cells[na] = cell(bodies[ARM])

    low, high = args.pair
    rows: list[dict] = []
    for region in ("loop", "outside_loop"):
        for key in ("opcodes", "memory", "vector_widths", "phi_widths"):
            rows += difference(cells[low], cells[high], region, key)
    rows.sort(key=lambda row: (-(row["priced_ms"] or 0.0), row["region"],
                               row["name"]))

    observed = LADDER_MS.get(high, 0.0) - LADDER_MS.get(low, 0.0)
    on_model = MS_PER_LOOP_VECTOR_OP * 4
    report = {
        "flags": SCORED_FLAGS,
        "arm": ARM,
        "ruler_ms_per_loop_vector_op": MS_PER_LOOP_VECTOR_OP,
        "pair": {"low": low, "high": high},
        "ladder_ms": LADDER_MS,
        "observed_step_ms": observed,
        "on_model_arithmetic_ms": on_model,
        "unexplained_step_ms": observed - on_model,
        "cells": {str(na): value for na, value in cells.items()},
        "differences": rows,
        "ladder_opcodes": {
            name: {str(na): cells[na]["loop"]["opcodes"].get(name, 0)
                   for na in args.na}
            for name in sorted({key for na in args.na
                                for key in cells[na]["loop"]["opcodes"]})
        },
        "ladder_peak_live_cfg_loop": {str(na): cells[na]["peak_live_cfg_loop"]
                                      for na in args.na},
        "ladder_allocas": {str(na): cells[na]["allocas"] for na in args.na},
    }

    print(f"NA={low} -> NA={high}: measured step {observed:.2f} ms, "
          f"arithmetic term {on_model:.2f} ms, "
          f"unexplained {observed - on_model:.2f} ms")
    print(f"ruler: 1 extra in-loop vector op = {MS_PER_LOOP_VECTOR_OP:.3f} ms")
    for row in rows:
        priced = f"{row['priced_ms']:8.2f} ms" if row["priced_ms"] is not None \
            else "     once"
        print(f"  {priced}  {row['region']:12s} {row['kind']:14s} "
              f"{row['name']:24s} {row['low']:4d} -> {row['high']:4d} "
              f"({row['delta']:+d})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
