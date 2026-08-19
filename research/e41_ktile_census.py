#!/usr/bin/env python3
"""E41 compile-only census for the K-tile ladder. No GPU, run before any timing.

Three things must hold before a single second of GPU time is spent:

1. NON-PERTURBATION. The incumbent cells must still be 62/83/104/125 registers at
   NA=2..5. The K-tile refactor adds two trip-count-1 loops around the shipped
   body; if that moved the incumbent's codegen, the base build is no longer the
   base and every ratio in the experiment is meaningless.
2. THE LADDER IS ONE MECHANISM. KT=all/4/2/1 at fixed (NA, R) must agree on
   peak_live_regs, device_loads, vector_float_ops and loop_backedges. They differ
   only in the `k_tile` constant, so if any static count moves, the rungs are not
   comparable and the timing cannot attribute a step to re-read distance.
3. NO TIMED CELL SPILLS. peak_live_regs <= 128 and no alloca of accumulator type.
   A spilled cell is a different experiment.

  python3 research/e41_ktile_census.py --out research/e41-ktile-census.json
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
from air_kernel_stats import (  # noqa: E402
    ALLOCA,
    ANY_LOAD,
    DEVICE_LOAD,
    FMA,
    VECTOR_OP,
    peak_live_registers,
)

PROBE = pathlib.Path("research/e41_ktile_probe.metal")
INCLUDE = pathlib.Path("Vendor/mlx-swift/Source/Cmlx/mlx")
REG_WALL = 128
# values_per_thread * SIMD_SIZE in the wide crossrow body.
BLOCK_SIZE = 16 * 32
# max K over research/qmv_cost_curve.py SCORED_SHAPES (mlp.down: 17408 x 5120).
WIDEST_K = 17408
KT_ALL_BLOCKS = 64

# `[4 x <5 x float>]` and friends: the accumulator array itself landing in
# private memory. Scratch of any other shape is not an accumulator spill.
ACC_ALLOCA = re.compile(r"^\[\s*(\d+)\s+x\s+<\s*(\d+)\s+x\s+float\s*>\s*\]$")
THREADGROUP = re.compile(r"addrspace\(3\)")

# E32 (research/e32-rps-grid.json) and E36 (research/e36-vpt-grid.json) both
# measured these on this toolchain, before the K-tile parameters existed.
SHIPPED_REGS = {2: 62, 3: 83, 4: 104, 5: 125}
SHIPPED_DEVICE_LOADS = {2: 32, 3: 36, 4: 40, 5: 44}
# E32's coverage-preserving row-blocked cells, from a generated copy of the same
# body. Directional cross-check only: a generated copy is not the shipped
# template, so a mismatch is reported, not fatal.
E32_RB_REGS = {(3, 1): 51, (3, 2): 66, (3, 4): 83,
               (4, 1): 63, (4, 2): 83, (4, 4): 104,
               (6, 1): 87, (6, 2): 117, (6, 4): 144}

# Which cells the arm build actually times. Everything else is priced only.
TIMED = {
    "xrb_na3_r2",          # M=6 anchor: E38 arm(a), the +10.54 % tax
    "xkt_na3_r2_kt1",      # M=3
    "xkt_na4_r2_kt64",     # M=4: no locality
    "xkt_na4_r2_kt4",      # M=8: DISCRIMINATOR
    "xkt_na4_r2_kt1",      # M=7: total-recovery bound
}
LADDERS = [(3, 2), (4, 2), (6, 2), (6, 1)]
# KT=64 is 32768 values, wider than the widest scored K (17408), so it visits all
# of K in one tile -- the "no locality" rung. It exists because KT=0 spells the
# same thing as `k_tile = in_vec_size`, which the compiler proves is trip count 1
# and deletes, dropping a loop level (loop_backedges 4 -> 3). KT=0 would have
# made the top of the ladder a different program from its other rungs, so the
# step could not be attributed to re-read distance alone. KT=0 is still compiled
# below to document that, but it is not a rung and it is not timed.
KT_RUNGS = [1, 2, 4, 64]
KT_PRICED_ONLY = [0]
INVARIANT_KEYS = ("peak_live_regs", "device_loads", "vector_float_ops",
                  "loop_backedges", "allocas")


def compile_cell(cell: dict, workdir: pathlib.Path) -> dict:
    name = cell["name"]
    ll = workdir / f"{name}.ll"
    ll_o3 = workdir / f"{name}.o3.ll"
    defines = [f"-D{k}={v}" if v is not None else f"-D{k}"
               for k, v in cell["defines"].items()]
    base = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2",
            *defines, "-I", str(INCLUDE), str(PROBE)]
    out = dict(cell)
    emit = subprocess.run([*base, "-S", "-o", str(ll)], capture_output=True, text=True)
    if emit.returncode != 0:
        out["status"] = "compile_failed"
        out["error"] = emit.stderr.strip().splitlines()[-6:]
        return out
    # E27/E32/E36 all measured their ladders after this extra pass; the anchor
    # cells only reproduce their numbers through the same pipeline.
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

    allocas = [ALLOCA.search(line).group(1) for line in body if ALLOCA.search(line)]
    acc_allocas = [a for a in allocas if ACC_ALLOCA.match(a)]
    peak_regs, peak_vals = peak_live_registers(body)
    out.update(
        status="ok",
        allocas=len(allocas),
        alloca_types=sorted(set(allocas)),
        acc_spill=len(acc_allocas) > 0,
        acc_alloca_types=sorted(set(acc_allocas)),
        peak_live_regs=peak_regs,
        peak_live_values=peak_vals,
        air_lines=len(body),
        device_loads=sum(1 for line in body if DEVICE_LOAD.search(line)),
        loads=sum(1 for line in body if ANY_LOAD.search(line)),
        vector_float_ops=sum(
            1 for line in body
            if VECTOR_OP.search(line)
            and (FMA.search(line) or re.search(r"=\s*f(mul|add)\s", line))
        ),
        loop_backedges=sum(1 for line in body if "!llvm.loop" in line),
        threadgroup_refs=sum(1 for line in body if THREADGROUP.search(line)),
        fits_reg_wall=peak_regs <= REG_WALL,
        timed=name in TIMED,
        pipeline="metal -O2 -S | metal-opt -passes=default<O3>",
    )
    return out


def cells() -> list[dict]:
    out = []
    for na, regs in SHIPPED_REGS.items():
        out.append({
            "name": f"xship_na{na}", "arm": "incumbent_anchor", "na": na, "r": 4, "kt": None,
            "expect_regs": regs, "expect_device_loads": SHIPPED_DEVICE_LOADS[na],
            "expect_acc_spill": False,
            "defines": {"PROBE_CELL_SHIPPED": None, "PROBE_NAME": f"xship_na{na}",
                        "PROBE_NA": na},
        })
    for na in (3, 4, 6):
        for r in (1, 2, 4):
            out.append({
                "name": f"xrb_na{na}_r{r}", "arm": "sequential_rowblocks",
                "na": na, "r": r, "kt": None,
                "e32_regs": E32_RB_REGS.get((na, r)),
                "defines": {"PROBE_CELL_RB": None, "PROBE_NAME": f"xrb_na{na}_r{r}",
                            "PROBE_NA": na, "PROBE_R": r},
            })
    for na, r in LADDERS:
        for kt in KT_RUNGS + KT_PRICED_ONLY:
            out.append({
                "name": f"xkt_na{na}_r{r}_kt{kt}",
                "arm": "ktile_ladder" if kt in KT_RUNGS else "ktile_folded_loop",
                "na": na, "r": r, "kt": kt,
                "defines": {"PROBE_CELL_KT": None,
                            "PROBE_NAME": f"xkt_na{na}_r{r}_kt{kt}",
                            "PROBE_NA": na, "PROBE_R": r, "PROBE_KT": kt},
            })
    # Spill-gate validation: the gate must fire here and stay quiet on the
    # incumbent anchors above, or no acc_spill verdict in this file is usable.
    for na, r in ((4, 4), (9, 2), (16, 4)):
        out.append({
            "name": f"xctl_spill_na{na}_r{r}", "arm": "negative_control",
            "na": na, "r": r, "kt": None, "expect_acc_spill": True,
            "defines": {"PROBE_CELL_FORCED_SPILL": None,
                        "PROBE_NAME": f"xctl_spill_na{na}_r{r}",
                        "PROBE_NA": na, "PROBE_R": r},
        })
    return out


def check_anchors(rows: dict[str, dict]) -> list[str]:
    bad = []
    for row in rows.values():
        if row["status"] != "ok":
            if "expect_regs" in row or "expect_acc_spill" in row:
                bad.append(f"{row['name']}: {row['status']} (needed a verdict)")
            continue
        if "expect_regs" in row and row["peak_live_regs"] != row["expect_regs"]:
            bad.append(
                f"NON-PERTURBATION FAILED {row['name']}: peak_live_regs="
                f"{row['peak_live_regs']} but E32/E36 measured {row['expect_regs']}"
            )
        if ("expect_device_loads" in row
                and row["device_loads"] != row["expect_device_loads"]):
            bad.append(
                f"NON-PERTURBATION FAILED {row['name']}: device_loads="
                f"{row['device_loads']} but E32/E36 measured "
                f"{row['expect_device_loads']}"
            )
        if ("expect_acc_spill" in row
                and row["acc_spill"] != row["expect_acc_spill"]):
            bad.append(
                f"SPILL GATE FAILED {row['name']}: acc_spill={row['acc_spill']} "
                f"expected {row['expect_acc_spill']} ({row['alloca_types']})"
            )
    return bad


def check_ladders(rows: dict[str, dict]) -> tuple[list[str], list[dict]]:
    bad, report = [], []
    if WIDEST_K >= KT_ALL_BLOCKS * BLOCK_SIZE:
        bad.append(
            f"KT={KT_ALL_BLOCKS} is {KT_ALL_BLOCKS * BLOCK_SIZE} values but the "
            f"widest scored K is {WIDEST_K}: the top rung would tile K, not span it"
        )
    for na, r in LADDERS:
        got = [rows.get(f"xkt_na{na}_r{r}_kt{kt}") for kt in KT_RUNGS]
        ok = [g for g in got if g and g["status"] == "ok"]
        if len(ok) < 2:
            continue
        entry = {"na": na, "r": r,
                 "rungs": {str(g["kt"]): {k: g[k] for k in INVARIANT_KEYS} for g in ok}}
        drift = {k: sorted({g[k] for g in ok}) for k in INVARIANT_KEYS}
        entry["invariant"] = all(len(v) == 1 for v in drift.values())
        entry["drift"] = {k: v for k, v in drift.items() if len(v) > 1}
        folded = rows.get(f"xkt_na{na}_r{r}_kt0")
        if folded and folded["status"] == "ok":
            entry["folded_kt0"] = {k: folded[k] for k in INVARIANT_KEYS}
            entry["kt0_drops_a_loop"] = (
                folded["loop_backedges"] == ok[0]["loop_backedges"] - 1
            )
        report.append(entry)
        if not entry["invariant"]:
            bad.append(
                f"LADDER NOT ONE MECHANISM na={na} r={r}: {entry['drift']} "
                "-- the rungs differ by more than the re-read distance"
            )
    return bad, report


def check_timed(rows: dict[str, dict]) -> list[str]:
    bad = []
    for name in sorted(TIMED):
        row = rows.get(name)
        if row is None or row["status"] != "ok":
            bad.append(f"TIMED CELL MISSING {name}: {row['status'] if row else 'absent'}")
            continue
        if row["acc_spill"]:
            bad.append(f"TIMED CELL SPILLS {name}: {row['acc_alloca_types']}")
        if row["peak_live_regs"] > REG_WALL:
            bad.append(
                f"TIMED CELL OVER WALL {name}: {row['peak_live_regs']} > {REG_WALL}"
            )
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e41-ktile-census.json")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp())
    todo = cells()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda c: compile_cell(c, workdir), todo))
    rows = {r["name"]: r for r in results}

    anchor_failures = check_anchors(rows)
    ladder_failures, ladder_report = check_ladders(rows)
    timed_failures = check_timed(rows)
    failures = anchor_failures + ladder_failures + timed_failures

    pathlib.Path(args.out).write_text(json.dumps({
        "workdir": str(workdir),
        "reg_wall": REG_WALL,
        "timed_cells": sorted(TIMED),
        "anchor_failures": anchor_failures,
        "ladder_failures": ladder_failures,
        "timed_failures": timed_failures,
        "ladders": ladder_report,
        "cells": results,
    }, indent=2))

    print(f"{'cell':24} {'arm':22} {'NA':>3} {'r':>2} {'KT':>4} {'regs':>5} "
          f"{'dLd':>4} {'vfop':>5} {'back':>5} {'alc':>4} {'spill':>6} T")
    for row in sorted(results, key=lambda r: (r["arm"], r["na"], r["r"],
                                              -1 if r["kt"] is None else r["kt"])):
        kt = "-" if row["kt"] is None else ("all" if row["kt"] == 0 else str(row["kt"]))
        if row["status"] != "ok":
            print(f"{row['name']:24} {row['arm']:22} {row['na']:>3} {row['r']:>2} "
                  f"{kt:>4} {row['status']}")
            continue
        print(f"{row['name']:24} {row['arm']:22} {row['na']:>3} {row['r']:>2} "
              f"{kt:>4} {row['peak_live_regs']:>5} {row['device_loads']:>4} "
              f"{row['vector_float_ops']:>5} {row['loop_backedges']:>5} "
              f"{row['allocas']:>4} {str(row['acc_spill']):>6} "
              f"{'*' if row['timed'] else ''}")

    print("\nE32 row-blocked cross-check (generated copy vs shipped template):")
    for row in results:
        if row.get("e32_regs") and row["status"] == "ok":
            mark = "same" if row["peak_live_regs"] == row["e32_regs"] else "DIFFERS"
            print(f"  {row['name']:20} shipped={row['peak_live_regs']:>4} "
                  f"E32={row['e32_regs']:>4}  {mark}")

    print("\nladder invariance (KT must change nothing static):")
    for entry in ladder_report:
        state = "one mechanism" if entry["invariant"] else f"DRIFT {entry['drift']}"
        print(f"  na={entry['na']} r={entry['r']}: {state}")

    print(f"\nAIR in {workdir}\nwrote {args.out}")
    if failures:
        print("\nCENSUS FAILED -- do not spend GPU time:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nCENSUS OK: incumbent unperturbed, ladder is one mechanism, "
          "no timed cell spills.")


if __name__ == "__main__":
    main()
