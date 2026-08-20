#!/usr/bin/env python3
"""E63 rung 0: is the QMV width cliff a memory-level-parallelism collapse?

No GPU. Compiles `qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, NA, true>` in
isolation for NA = 2..9 with the SCORED math flags and reads the AIR, then
answers one preregistered question: do the weight loads the kernel can hold in
flight fall as NA grows, and does `1 / bw(NA)` track `1 / MLP(NA)` better than
the two models the campaign already refuted?

Readouts per NA:

  regs / peak_live / allocas   register-pressure proxies (air_kernel_stats)
  loads_before_first_consumer  distinct device loads issued in the steady-state
                               block before the first USE of the first of them
  max_loads_in_flight          max simultaneously outstanding device loads
  unroll                       device loads per steady-state block vs the one-k
                               source count, and the loop back-edge count
  spill                        private-memory stores/loads that survive -O3

The shipped header asserts NA <= 5, so a shadow copy of quantized.h relaxes ONLY
that bound. The working tree is never modified.

  python3 research/e63_mlp_census.py --out research/e63-artifacts/rung0.json
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
    FMA,
    peak_live_registers,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER_REL = "mlx/backend/metal/kernels/quantized.h"
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx" / HEADER_REL
INCLUDE = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx"
WIDE_PROBE = REPO / "research/e63_wide_probe.metal"
ENTRY_PROBE = REPO / "research/e46_entry_probe.metal"
CELL = "e63_wide_cell"
ENTRIES = (
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
    "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_1",
)

# Scored math flags. Ledger 196(A): -fno-fast-math is the whole register
# difference and -std is irrelevant, so -std is pinned only for reproducibility.
SCORED_FLAGS = ["-std=metal4.0", "-O2", "-fno-fast-math"]

NA_ASSERT = re.compile(
    r"static_assert\(NA >= 2 && NA <= \d+,\s*"
    r'"wide multi-row QMV supports NA in \[2, \d+\]"\);')
NA_ASSERT_RELAXED = ('static_assert(NA >= 2 && NA <= 9, '
                     '"wide multi-row QMV supports NA in [2, 9]");')

LABEL = re.compile(r"^([\w.]+):")
DEF = re.compile(r"^\s*(%[\w.\-]+)\s*=\s*(.*)$")
SSA_REF = re.compile(r"%[\w.\-]+")
GEP_LIKE = re.compile(r"^\s*(getelementptr|bitcast|addrspacecast|inttoptr)\b")
# This toolchain emits TYPED pointers (`i16 addrspace(1)* %83`), not opaque ones.
LOAD_PTR = re.compile(
    r"=\s*load\s+([^,]+),\s*[^,]*?addrspace\((\d+)\)\*\s*(%[\w.\-]+)")
DEVICE_STORE = re.compile(r"^\s*store\s.*addrspace\(1\)\*")
STORE = re.compile(r"^\s*store\s")
# Arithmetic that actually needs the loaded bits, as opposed to a spill store or
# an address computation. This is what "first consumer" means below.
ARITH = re.compile(
    r"=\s*(fmul|fadd|fsub|fdiv|fpext|fptrunc|uitofp|sitofp|and|or|xor|shl|lshr|"
    r"ashr|mul|add|sub|zext|sext|trunc|call [^@]*@(llvm|air)\.fma)\b")

# askeladd's E61 rung 1 ladder, the sole source of the six bandwidth numbers the
# E63 brief rests on. Stream size 14.412 GB.
ASKELADD_BW = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946, 6: 117.8, 7: 97.9}
STREAM_GB = 14.412
# Ledger 200(C) refuted model: bw x regs is constant. `regs` is the register law
# quoted in the E63 brief, reg = 22 + 20 x NA + 4 x [two distinct group sizes];
# an isolated cell has one group size, so the +4 term is absent.
REG_LAW = {na: 22 + 20 * na for na in range(2, 10)}


def relax_na_assert(text: str) -> str:
    if len(NA_ASSERT.findall(text)) != 1:
        raise SystemExit("e63: NA assert anchor is not unique")
    return NA_ASSERT.sub(NA_ASSERT_RELAXED, text)


def kernel_body(lines: list[str], name: str) -> list[str] | None:
    body, inside = [], False
    for line in lines:
        if line.startswith("define ") and ("@%s(" % name) in line:
            inside = True
        elif inside and line == "}":
            break
        elif inside:
            body.append(line)
    return body or None


def defined_kernels(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("define "):
            hit = re.search(r"@([\w.$]+)\(", line)
            if hit:
                out.append(hit.group(1))
    return out


def arg_names(lines: list[str], name: str) -> list[str]:
    for line in lines:
        if line.startswith("define ") and ("@%s(" % name) in line:
            inner = line[line.index("(") + 1:line.rindex(")")]
            return SSA_REF.findall(inner)
    return []


def blocks(body: list[str]) -> list[tuple[str, list[str]]]:
    """Split a function body into (label, lines) basic blocks."""
    out, label, cur = [], "entry", []
    for line in body:
        hit = LABEL.match(line)
        if hit and not line.startswith(" "):
            out.append((label, cur))
            label, cur = hit.group(1), []
        else:
            cur.append(line)
    out.append((label, cur))
    return [(lab, ls) for lab, ls in out if ls]


def loop_body_stats(body: list[str]) -> dict:
    """Per-iteration cost of the k loop, as opposed to whole-function totals.

    A block is inside a loop when one of its LLVM `preds` appears at or after it
    in file order. Whole-function counters cannot separate a genuine
    per-iteration spill from an alloca that LICM promoted around the loop and
    that therefore costs O(1) per launch, which is the distinction this whole
    experiment turns on.
    """
    order = {}
    for i, (label, _) in enumerate(blocks(body)):
        order[label] = i
    total = {"blocks": 0, "lines": 0, "device_loads": 0, "private_loads": 0,
             "private_stores": 0, "device_stores": 0, "fmul": 0, "fadd": 0,
             "fma": 0, "labels": []}
    for label, lines in blocks(body):
        preds = []
        for line in body:
            if line.startswith(label + ":") and "preds =" in line:
                preds = re.findall(r"%([\w.\-]+)", line.split("preds =")[1])
        if not any(order.get(p, -1) >= order.get(label, 0) for p in preds):
            continue
        total["blocks"] += 1
        total["labels"].append(label)
        total["lines"] += len(lines)
        for line in lines:
            if DEVICE_LOAD.search(line):
                total["device_loads"] += 1
            elif re.search(r"=\s*load\s", line):
                total["private_loads"] += 1
            if STORE.match(line):
                if DEVICE_STORE.match(line):
                    total["device_stores"] += 1
                else:
                    total["private_stores"] += 1
            if re.search(r"=\s*fmul\s", line):
                total["fmul"] += 1
            if re.search(r"=\s*fadd\s", line):
                total["fadd"] += 1
            if FMA.search(line):
                total["fma"] += 1
    return total


def provenance(body: list[str], args: list[str]) -> dict[str, str]:
    """Map every SSA pointer value back to the kernel argument it derives from."""
    rhs: dict[str, str] = {}
    for line in body:
        hit = DEF.match(line)
        if hit:
            rhs.setdefault(hit.group(1), hit.group(2))
    root: dict[str, str] = {}

    def walk(name: str, depth: int = 0) -> str:
        if name in root:
            return root[name]
        if name in args or name not in rhs or depth > 64:
            return name
        expr = rhs[name]
        if not GEP_LIKE.match("  " + expr):
            return name
        refs = SSA_REF.findall(expr)
        answer = walk(refs[0], depth + 1) if refs else name
        root[name] = answer
        return answer

    return {name: walk(name) for name in rhs}


def mlp_stats(body: list[str], args: list[str]) -> dict:
    """In-flight device loads in the steady-state (weight-load-carrying) block."""
    prov = provenance(body, args)
    # Argument order of e63_wide_cell: w, scales, biases, x, y, ...
    role = {}
    for i, a in enumerate(args[:5]):
        role[a] = ["w", "scales", "biases", "x", "y"][i]

    best = None
    for label, lines in blocks(body):
        loads = [(i, l) for i, l in enumerate(lines) if DEVICE_LOAD.search(l)]
        weight = 0
        for _, l in loads:
            hit = LOAD_PTR.search(l)
            if hit and role.get(prov.get(hit.group(3), ""), "") == "w":
                weight += 1
        if best is None or weight > best[0] or (weight == best[0]
                                                and len(loads) > len(best[2])):
            best = (weight, label, loads, lines)
    if best is None:
        return {"status": "no_block"}
    _, label, loads, lines = best

    # Two first-use indices per loaded value. `any_use` counts a spill store to
    # private memory as a use; `arith_use` counts only an instruction that
    # consumes the bits. A kernel whose loads are round-tripped through an
    # alloca has any_use == def + 1 for every load at every NA, which measures
    # the alloca and not the load-to-use distance, so both are reported.
    any_use: dict[str, int] = {}
    arith_use: dict[str, int] = {}
    for i, line in enumerate(lines):
        hit = DEF.match(line)
        name, expr = (hit.group(1), hit.group(2)) if hit else (None, line)
        for ref in SSA_REF.findall(expr):
            any_use.setdefault(ref, i)
            if ARITH.search(line) or (STORE.match(line)
                                      and DEVICE_STORE.match(line)):
                arith_use.setdefault(ref, i)

    records = []
    for i, line in enumerate(lines):
        hit = DEF.match(line)
        if not hit or not DEVICE_LOAD.search(line):
            continue
        ptr = LOAD_PTR.search(line)
        buf = role.get(prov.get(ptr.group(3), ""), "?") if ptr else "?"
        records.append({
            "def": i,
            "use": any_use.get(hit.group(1), len(lines)),
            "arith_use": arith_use.get(hit.group(1), len(lines)),
            "buffer": buf,
            "type": ptr.group(1).strip() if ptr else "?",
        })

    def in_flight(subset: list[dict], key: str) -> int:
        peak = 0
        for i in range(len(lines)):
            live = sum(1 for r in subset if r["def"] <= i < r[key])
            peak = max(peak, live)
        return peak

    def before_first_consumer(subset: list[dict], key: str) -> int:
        if not subset:
            return 0
        first = min(subset, key=lambda r: r["def"])
        return sum(1 for r in subset if r["def"] < first[key])

    weights = [r for r in records if r["buffer"] == "w"]
    xs = [r for r in records if r["buffer"] == "x"]
    sb = [r for r in records if r["buffer"] in ("scales", "biases")]

    out = {
        "status": "ok",
        "block": label,
        "block_lines": len(lines),
        "device_loads": len(records),
        "weight_loads": len(weights),
        "x_loads": len(xs),
        "scale_bias_loads": len(sb),
        "unclassified_loads": len(records) - len(weights) - len(xs) - len(sb),
        "load_types": sorted({r["type"] for r in records}),
    }
    for tag, key in (("", "use"), ("_arith", "arith_use")):
        out.update({
            "max_loads_in_flight%s" % tag: in_flight(records, key),
            "max_weight_loads_in_flight%s" % tag: in_flight(weights, key),
            "max_x_loads_in_flight%s" % tag: in_flight(xs, key),
            "loads_before_first_consumer%s" % tag:
                before_first_consumer(records, key),
            "weight_loads_before_first_consumer%s" % tag:
                before_first_consumer(weights, key),
        })
    return out


def air_stats(body: list[str], args: list[str]) -> dict:
    allocas = [ALLOCA.search(l).group(1) for l in body if ALLOCA.search(l)]
    peak, vals = peak_live_registers(body)
    private_stores = sum(1 for l in body
                         if STORE.match(l) and not DEVICE_STORE.match(l))
    private_loads = sum(1 for l in body
                        if re.search(r"=\s*load\s", l) and not DEVICE_LOAD.search(l))
    out = {
        "peak_live_regs": peak,
        "peak_live_values": vals,
        "air_lines": len(body),
        "allocas": len(allocas),
        "alloca_types": sorted(set(allocas)),
        "device_loads_total": sum(1 for l in body if DEVICE_LOAD.search(l)),
        "private_stores": private_stores,
        "private_loads": private_loads,
        "fma": sum(1 for l in body if FMA.search(l)),
        "fmul": sum(1 for l in body if re.search(r"=\s*fmul\s", l)),
        "fadd": sum(1 for l in body if re.search(r"=\s*fadd\s", l)),
        "loop_backedges": sum(1 for l in body if "!llvm.loop" in l),
    }
    out["mlp"] = mlp_stats(body, args)
    out["loop_body"] = loop_body_stats(body)
    return out


def compile_probe(shadow: pathlib.Path, probe: pathlib.Path, tag: str,
                  defines: dict[str, int], wanted: tuple[str, ...]) -> dict:
    ll = shadow / ("%s.ll" % tag)
    ll_o3 = shadow / ("%s.o3.ll" % tag)
    flags = ["-D%s=%d" % (k, v) for k, v in defines.items()]
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS, *flags,
         "-I", str(shadow), "-I", str(INCLUDE), "-S", str(probe), "-o", str(ll)],
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
        body = kernel_body(lines, name)
        if body is None:
            return {"status": "kernel_not_found", "missing": name}
        found[name] = air_stats(body, arg_names(lines, name))
    return {"status": "ok", "functions": found,
            "defined_kernels": defined_kernels(lines), "ll": str(ll_o3)}


def build_occupancy_tool(workdir: pathlib.Path) -> pathlib.Path | None:
    binary = workdir / "na_occupancy"
    build = subprocess.run(
        ["swiftc", "-O", str(REPO / "research/crossrow_na_occupancy.swift"),
         "-o", str(binary)], capture_output=True, text=True)
    return binary if build.returncode == 0 else None


def occupancy(shadow: pathlib.Path, tool: pathlib.Path, na: int) -> dict:
    """maxTotalThreadsPerThreadgroup: the back end's own register verdict.

    The AIR text is a heuristic; this number is capped by the register budget the
    AGX back end actually assigned. Building a pipeline dispatches no GPU work.
    """
    air = shadow / ("occ_na%d.air" % na)
    lib = shadow / ("occ_na%d.metallib" % na)
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", *SCORED_FLAGS,
         "-DE63_NA=%d" % na, "-DE63_DIRECT_NIBBLES=1",
         "-I", str(shadow), "-I", str(INCLUDE), "-c", str(WIDE_PROBE),
         "-o", str(air)], capture_output=True, text=True)
    if emit.returncode != 0:
        return {"status": "air_failed",
                "error": emit.stderr.strip().splitlines()[-4:]}
    link = subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air),
                           "-o", str(lib)], capture_output=True, text=True)
    if link.returncode != 0:
        return {"status": "metallib_failed"}
    run = subprocess.run([str(tool), str(lib), CELL], capture_output=True,
                         text=True)
    if run.returncode != 0:
        return {"status": "pipeline_failed",
                "error": run.stderr.strip().splitlines()[-4:]}
    for line in run.stdout.splitlines():
        if line.startswith(CELL + " "):
            parts = line.split()
            return {"status": "ok",
                    "max_total_threads_per_threadgroup": int(parts[1]),
                    "thread_execution_width": int(parts[2]),
                    "static_threadgroup_memory_bytes": int(parts[3])}
    return {"status": "no_row", "raw": run.stdout.splitlines()[:6]}


def lstsq(cols: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares through the given design columns, no intercept."""
    k, n = len(cols), len(y)
    a = [[sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k)]
         for i in range(k)]
    b = [sum(cols[i][r] * y[r] for r in range(n)) for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, k):
            f = a[r][i] / a[i][i]
            for c in range(i, k):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, k))) / a[i][i]
    return x


def model_comparison(mlp: dict[int, float], peak_live: dict[int, float]) -> dict:
    nas = sorted(n for n in ASKELADD_BW if n in mlp)
    t = [STREAM_GB / ASKELADD_BW[n] * 1000.0 for n in nas]  # ms per stream
    one = [1.0] * len(nas)
    na = [float(n) for n in nas]
    # The spill indicator is not fitted to the timing curve: it is read off the
    # AIR, where a `[4 x <NA x float>]` accumulator alloca appears.
    spill = [1.0 if peak_live.get(n, 0) > 128 else 0.0 for n in nas]
    models = {
        # MLP: bandwidth is proportional to outstanding loads, so time per
        # stream is proportional to 1 / MLP. Degenerate when MLP is constant.
        "mlp": [1.0 / mlp[n] if mlp.get(n) else 0.0 for n in nas],
        # Ledger 199(G): constant cost per weight stream.
        "constant_per_stream": list(one),
        # Ledger 200(C): bw x regs constant, so time is proportional to regs.
        "bw_times_regs_constant": [float(REG_LAW[n]) for n in nas],
        # Lane-weighted arithmetic per weight byte grows linearly in NA, so a
        # compute-bound kernel spends time proportional to NA.
        "alu_work_linear_in_na": list(na),
        "alu_work_plus_offset": None,
        "alu_work_plus_spill_step": None,
    }
    designs = {name: [pred] for name, pred in models.items() if pred is not None}
    designs["alu_work_plus_offset"] = [one, na]
    designs["alu_work_plus_spill_step"] = [one, na, spill]

    out = {"na": nas, "ms_per_stream": t, "spill_indicator": spill,
           "models": {}}
    for name, cols in designs.items():
        if all(v == 0.0 for v in cols[0]):
            out["models"][name] = {"status": "degenerate_all_zero"}
            continue
        coef = lstsq(cols, t)
        fit = [sum(coef[j] * cols[j][r] for j in range(len(coef)))
               for r in range(len(t))]
        resid = [a - b for a, b in zip(t, fit)]
        rel = [r / v for r, v in zip(resid, t)]
        out["models"][name] = {
            "coefficients": coef,
            "fitted_ms": fit,
            "residual_ms": resid,
            "residual_rel": rel,
            "residual_sign": ["+" if r > 0 else "-" if r < 0 else "0"
                              for r in resid],
            "rms_rel": (sum(r * r for r in rel) / len(rel)) ** 0.5,
            "max_abs_rel": max(abs(r) for r in rel),
            "free_parameters": len(coef),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e63-artifacts/rung0.json")
    ap.add_argument("--na", default="2,3,4,5,6,7,8,9")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    nas = [int(v) for v in args.na.split(",")]

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e63-mlp-census-"))
    try:
        shadow = workdir / "relaxed"
        dst = shadow / HEADER_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        source = HEADER.read_text()
        dst.write_text(relax_na_assert(source))

        tool = build_occupancy_tool(workdir)
        cells = {}
        for na in nas:
            res = compile_probe(shadow, WIDE_PROBE, "wide_na%d" % na,
                                {"E63_NA": na, "E63_DIRECT_NIBBLES": 1}, (CELL,))
            cells[na] = (res["functions"][CELL] if res["status"] == "ok"
                         else {"status": res["status"], "error": res.get("error")})
            if tool is not None:
                cells[na]["occupancy"] = occupancy(shadow, tool, na)

        # Single-entry-point check, read from the compiled artifact rather than
        # from prose: the unmodified header, every width case inlined.
        pristine = workdir / "pristine"
        (pristine / HEADER_REL).parent.mkdir(parents=True, exist_ok=True)
        (pristine / HEADER_REL).write_text(source)
        entry = compile_probe(pristine, ENTRY_PROBE, "entry", {}, ENTRIES)

        mlp = {na: c.get("mlp", {}).get("max_weight_loads_in_flight_arith", 0)
               for na, c in cells.items() if "mlp" in c}
        payload = {
            "head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
            "dirty_paths": len(subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain"],
                capture_output=True, text=True).stdout.strip().splitlines()),
            "flags": SCORED_FLAGS,
            "pipeline": "metal %s -S | metal-opt -passes=default<O3>"
                        % " ".join(SCORED_FLAGS),
            "cells": cells,
            "entry": entry,
            "model_comparison": model_comparison(
                {na: float(v) for na, v in mlp.items()},
                {na: float(c["peak_live_regs"]) for na, c in cells.items()
                 if "peak_live_regs" in c}),
        }
        payload["accumulator_spill"] = {
            na: {"peak_live_regs": c.get("peak_live_regs"),
                 "allocas": c.get("allocas"),
                 "alloca_types": c.get("alloca_types"),
                 "accumulator_in_private_memory":
                     any("float" in a for a in c.get("alloca_types", [])),
                 "private_stores": c.get("private_stores"),
                 "private_loads": c.get("private_loads")}
            for na, c in cells.items() if "peak_live_regs" in c
        }
        base, top = mlp.get(2), mlp.get(6)
        payload["kill_rule"] = {
            "max_weight_loads_in_flight_na2": base,
            "max_weight_loads_in_flight_na6": top,
            "identical": base == top,
            "ratio": (base / top) if base and top else None,
            "advance_requires_ratio_ge": 2.0,
        }

        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")

        print("whole function")
        print("NA  regs peakLive alloc maxThr devLd  wLd wFlight wBefore "
              "privSt privLd allocaTypes")
        for na in nas:
            c = cells[na]
            if "mlp" not in c:
                print("%2d  %s %s" % (na, c.get("status"), c.get("error")))
                continue
            m = c["mlp"]
            occ = c.get("occupancy", {})
            print("%2d %5d %7d %6d %6s %5d %4d %7d %7d %6d %6d %s"
                  % (na, REG_LAW[na], c["peak_live_regs"], c["allocas"],
                     occ.get("max_total_threads_per_threadgroup", "-"),
                     c["device_loads_total"], m["weight_loads"],
                     m["max_weight_loads_in_flight_arith"],
                     m["weight_loads_before_first_consumer_arith"],
                     c["private_stores"], c["private_loads"],
                     c["alloca_types"]))
        print("\nk-loop body only (per iteration)")
        print("NA  blocks lines devLd privLd privSt fmul fadd fma")
        for na in nas:
            c = cells[na]
            if "loop_body" not in c:
                continue
            b = c["loop_body"]
            print("%2d %7d %5d %5d %6d %6d %4d %4d %3d"
                  % (na, b["blocks"], b["lines"], b["device_loads"],
                     b["private_loads"], b["private_stores"], b["fmul"],
                     b["fadd"], b["fma"]))
        print("\nkill rule:", json.dumps(payload["kill_rule"]))
        mc = payload["model_comparison"]
        print("ms per stream:", [round(v, 2) for v in mc["ms_per_stream"]])
        for name, fit in mc["models"].items():
            if "rms_rel" not in fit:
                print("model %-26s %s" % (name, fit.get("status")))
                continue
            print("model %-26s k=%d rms_rel=%.4f max_abs_rel=%.4f signs=%s "
                  "resid%%=%s"
                  % (name, fit["free_parameters"], fit["rms_rel"],
                     fit["max_abs_rel"], "".join(fit["residual_sign"]),
                     [round(100 * r, 2) for r in fit["residual_rel"]]))
        print("wrote", out)
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print("kept", workdir)


if __name__ == "__main__":
    raise SystemExit(main())
