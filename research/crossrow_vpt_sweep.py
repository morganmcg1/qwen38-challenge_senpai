#!/usr/bin/env python3
"""Compile the E36 NA x rows_per_simd x values_per_thread grid.

One kernel per translation unit, so each cell's AIR is attributable to that cell
and not to whatever else the compiler inlined alongside it. Pipeline is E32's,
unchanged: `metal -std=metal3.1 -O2 -S` then `metal-opt -passes=default<O3>`.

TWO measures, because E36 varies an axis E32's gate is blind to.

  acc_spill      accumulator array `[R x <NA x float>]` in private memory. This
                 is E32's gate and E27's calibration; it is a hard verdict.
  private_bytes  total `alloca` bytes. The ONLY structure that grows with
                 values_per_thread is the uint16 staging array
                 `packed[R][V/4]`, which is an alloca in EVERY shipped cell
                 including the clean ones -- so its presence is not a spill, and
                 acc_spill cannot see this axis at all. Reporting its size is
                 the honest substitute for a gate that does not exist.

`air_kernel_stats.ALLOCA` truncates nested array types (`[4 x [4 x i16]]` comes
back as `[4 x [4 x i16]`) because its character class stops at the first `]`.
E32 only compared those strings for equality so it never mattered. E36 has to
size them, so this file parses alloca types with a bracket scan instead and
validates the parser against two cells whose byte counts are known a priori.

  python3 research/crossrow_vpt_gen.py
  python3 research/crossrow_vpt_sweep.py --out research/e36-vpt-grid.json
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
from air_kernel_stats import peak_live_registers  # noqa: E402

PROBE = pathlib.Path("research/crossrow_vpt_probe.metal")
INCLUDE = pathlib.Path("Vendor/mlx-swift/Source/Cmlx/mlx")
NA_VALUES = [2, 3, 4, 5, 6, 7, 8, 9]
MODEL_ONLY_NA = [10, 11, 12]
V_VALUES = [8, 16, 32, 64]
ACC_ALLOCA = re.compile(r"^\[\s*(\d+)\s+x\s+<\s*(\d+)\s+x\s+float\s*>\s*\]$")
STAGE_ALLOCA = re.compile(r"^\[\s*(\d+)\s+x\s+\[\s*(\d+)\s+x\s+i16\s*\]\s*\]$")
SCALAR_BITS = {"float": 32, "half": 16, "bfloat": 16, "double": 64, "ptr": 64}


def alloca_types(body: list[str]) -> list[str]:
    """Alloca operand types, with nested brackets balanced."""
    out = []
    for line in body:
        at = line.find("alloca ")
        if at < 0:
            continue
        rest = line[at + len("alloca ") :].lstrip()
        if not rest.startswith(("[", "<")):
            out.append(rest.split(",")[0].strip())
            continue
        depth, end = 0, None
        for i, ch in enumerate(rest):
            if ch in "[<":
                depth += 1
            elif ch in "]>":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        out.append(rest[: end + 1] if end is not None else rest.split(",")[0].strip())
    return out


def type_bytes(t: str) -> int:
    """Byte size of an AIR aggregate type such as `[4 x [8 x i16]]`."""
    t = t.strip()
    if t.startswith(("[", "<")):
        inner = t[1:-1]
        count, _, elem = inner.partition(" x ")
        return int(count) * type_bytes(elem)
    m = re.fullmatch(r"i(\d+)", t)
    if m:
        return max(1, int(m.group(1)) // 8)
    return SCALAR_BITS.get(t, 32) // 8


def compile_cell(cell: dict, workdir: pathlib.Path) -> dict:
    name = cell["name"]
    ll = workdir / f"{name}.ll"
    ll_o3 = workdir / f"{name}.o3.ll"
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

    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)],
        capture_output=True, text=True,
    )
    if opt.returncode != 0:
        out["status"] = "metal_opt_failed"
        out["error"] = opt.stderr.strip().splitlines()[-6:]
        return out

    body, inside = [], False
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

    allocas = alloca_types(body)
    acc = [a for a in allocas if ACC_ALLOCA.match(a)]
    stage = [a for a in allocas if STAGE_ALLOCA.match(a)]
    peak_regs, peak_vals = peak_live_registers(body)
    out.update(
        status="ok",
        allocas=len(allocas),
        alloca_types=sorted(set(allocas)),
        private_bytes=sum(type_bytes(a) for a in allocas),
        acc_spill=len(acc) > 0,
        acc_alloca_types=sorted(set(acc)),
        acc_bytes=sum(type_bytes(a) for a in acc),
        stage_alloca_types=sorted(set(stage)),
        stage_bytes=sum(type_bytes(a) for a in stage),
        peak_live_regs=peak_regs,
        peak_live_values=peak_vals,
        air_lines=len(body),
        device_loads=sum(1 for line in body if "addrspace(1)" in line and "= load" in line),
        pipeline="metal -O2 -S | metal-opt -passes=default<O3>",
    )
    return out


def cells() -> list[dict]:
    out = []
    for v in V_VALUES:
        for na in NA_VALUES + MODEL_ONLY_NA:
            rs = [1, 2, 3, 4] if na in NA_VALUES else [1, 2]
            for r in rs:
                out.append({
                    "name": f"xvpt_na{na}_r{r}_v{v}", "arm": "grid_relaxed",
                    "na": na, "r": r, "v": v,
                    "defines": {"PROBE_CELL_VPT": None,
                                "PROBE_NAME": f"xvpt_na{na}_r{r}_v{v}",
                                "PROBE_NA": na, "PROBE_R": r, "PROBE_V": v},
                })
            if na in NA_VALUES:
                for r in (1, 2, 4):
                    out.append({
                        "name": f"xvb_na{na}_r{r}_v{v}", "arm": "coverage_preserving",
                        "na": na, "r": r, "v": v,
                        "defines": {"PROBE_CELL_VB": None,
                                    "PROBE_NAME": f"xvb_na{na}_r{r}_v{v}",
                                    "PROBE_NA": na, "PROBE_R": r, "PROBE_V": v},
                    })
    # E27 measured these five cells independently: NA=2..5 spill-free at
    # 62/83/104/125 registers, NA=6 spilled. If they do not reproduce, the probe
    # is wrong and nothing else in the grid counts.
    for na in (2, 3, 4, 5):
        out.append({
            "name": f"xship_na{na}", "arm": "shipped_anchor", "na": na, "r": 4, "v": 16,
            "expect_acc_spill": False, "expect_peak_live_regs": {2: 62, 3: 83, 4: 104, 5: 125}[na],
            "defines": {"PROBE_CELL_SHIPPED": None, "PROBE_NAME": f"xship_na{na}",
                        "PROBE_NA": na},
        })
    # The historical known-bad. Cannot use the shipped template: its
    # static_assert refuses NA=6. This is the generated body at the cell E27
    # independently measured as spilling.
    out.append({
        "name": "xctl_e27_spill_na6_r4_v16", "arm": "negative_control",
        "na": 6, "r": 4, "v": 16,
        "expect_acc_spill": True, "expect_peak_live_regs": 144,
        "defines": {"PROBE_CELL_VPT": None, "PROBE_NAME": "xctl_e27_spill_na6_r4_v16",
                    "PROBE_NA": 6, "PROBE_R": 4, "PROBE_V": 16},
    })
    # Gate 1 known-BAD: the accumulator array forced into private memory.
    for na, r in ((4, 4), (9, 2), (16, 4)):
        out.append({
            "name": f"xctl_accspill_na{na}_r{r}", "arm": "negative_control",
            "na": na, "r": r, "v": 16,
            "expect_acc_spill": True, "expect_acc_bytes": r * na * 4,
            "defines": {"PROBE_CELL_FORCED_ACC_SPILL": None,
                        "PROBE_NAME": f"xctl_accspill_na{na}_r{r}",
                        "PROBE_NA": na, "PROBE_R": r},
        })
    # Gate 1 known-GOOD at the widest cell the model claims is clean.
    for na, r, v in ((4, 4, 16), (5, 4, 16), (12, 2, 16)):
        out.append({
            "name": f"xctl_clean_na{na}_r{r}_v{v}", "arm": "positive_control",
            "na": na, "r": r, "v": v, "expect_acc_spill": False,
            "defines": {"PROBE_CELL_VPT": None,
                        "PROBE_NAME": f"xctl_clean_na{na}_r{r}_v{v}",
                        "PROBE_NA": na, "PROBE_R": r, "PROBE_V": v},
        })
    # Parser validation: `[R x [V/4 x i16]]` byte counts known a priori. Without
    # this the private_bytes column is an unvalidated number, and E36's whole
    # question lives in that column.
    for r, v in ((4, 16), (4, 64), (2, 32)):
        out.append({
            "name": f"xctl_stage_r{r}_v{v}", "arm": "parser_control",
            "na": 2, "r": r, "v": v,
            "expect_stage_bytes": r * (v // 4) * 2,
            "defines": {"PROBE_CELL_FORCED_STAGE_SPILL": None,
                        "PROBE_NAME": f"xctl_stage_r{r}_v{v}",
                        "PROBE_R": r, "PROBE_V": v},
        })
    return out


def check_expectations(results: list[dict]) -> list[str]:
    failures = []
    for row in results:
        keys = [k for k in row if k.startswith("expect_")]
        if not keys:
            continue
        if row["status"] != "ok":
            failures.append(f"{row['name']}: {row['status']} (expected a usable verdict)")
            continue
        for key in keys:
            got = row[key[len("expect_") :]]
            if got != row[key]:
                failures.append(
                    f"{row['name']}: {key[len('expect_'):]}={got} but expected "
                    f"{row[key]} (allocas={row['alloca_types']})"
                )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e36-vpt-grid.json")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--keep-air", default="")
    args = ap.parse_args()

    subprocess.run([sys.executable, "research/crossrow_vpt_gen.py", "--check"], check=True)

    workdir = pathlib.Path(args.keep_air) if args.keep_air else pathlib.Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    todo = cells()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda c: compile_cell(c, workdir), todo))

    results.sort(key=lambda r: (r["arm"], r["v"], r["na"], r["r"]))
    failures = check_expectations(results)
    pathlib.Path(args.out).write_text(json.dumps(
        {"workdir": str(workdir), "gate_validation_failures": failures, "cells": results},
        indent=2,
    ))

    print(f"{'cell':28} {'arm':20} {'NA':>3} {'r':>2} {'vpt':>4} {'regs':>5} "
          f"{'priv':>5} {'stage':>6} {'accSpill':>8} status")
    for row in results:
        if row["status"] != "ok":
            print(f"{row['name']:28} {row['arm']:20} {row['na']:>3} {row['r']:>2} "
                  f"{row['v']:>4} {'-':>5} {'-':>5} {'-':>6} {'-':>8} {row['status']}")
            continue
        print(f"{row['name']:28} {row['arm']:20} {row['na']:>3} {row['r']:>2} "
              f"{row['v']:>4} {row['peak_live_regs']:>5} {row['private_bytes']:>5} "
              f"{row['stage_bytes']:>6} {str(row['acc_spill']):>8} ok")
    print(f"\nAIR in {workdir}")
    if failures:
        print("\nGATE VALIDATION FAILED -- every verdict above is untrustworthy:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    checked = sum(1 for r in results if any(k.startswith("expect_") for k in r))
    print(f"\ngate validation: {checked}/{checked} control cells matched their expectation")


if __name__ == "__main__":
    main()
