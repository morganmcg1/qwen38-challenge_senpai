#!/usr/bin/env python3
"""E88 rungs 0 and 1: AIR census and two-architecture register census. No GPU.

Rung 0 answers "did the change do what it claims and nothing else?" from the
AIR of the runtime-effective JIT string: device loads must fall, allocas must
fall, and the `@air.fma.v*f32` count must be UNCHANGED. A changed FMA count
means the arithmetic moved, and arm W is then no longer bit-exact by
construction.

Rung 1 answers "does the ranked host pay for it?" with `xcrun metal-tt` through
research/agx_crossarch.py, on `applegpu_g16s` (this Mac) and `applegpu_g17s`
(the ranked M5). The stop rule is a hard gate: if the g17s register count rises
at any live cell, or g17s spill appears where there was none, E88 stops before
any GPU time.

Both rungs read the LIVE dispatch table out of the patched source rather than
assuming it, so a cell that the scored worker cannot reach is never censused.

  python3 research/e88_rung01.py --out research/e88-artifacts/rung01.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402
import e88_arms  # noqa: E402
from air_kernel_stats import (  # noqa: E402
    ALLOCA,
    ANY_LOAD,
    DEVICE_LOAD,
    FMA,
    kernels,
    peak_live_registers,
)
from jit_string_compile import (  # noqa: E402
    PREAMBLE_BODY,
    PREAMBLES,
    preamble,
    template_def,
)

ARCHS = ("applegpu_g16s", "applegpu_g17s")
CEILING = {"applegpu_g16s": 96, "applegpu_g17s": 126}

# The dispatch entries the scored worker JIT-compiles for the affine-4
# group-64 path. Both are built because the same switch lives in each.
COMBINED_CELLS = (
    "affine_qmv_fast<bfloat16_t, 64, 4, false>",
    "affine_qmv_fast<bfloat16_t, 64, 4, true>",
)

SCORED_ARGS = """    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]"""

WIDE_CELL = """
[[kernel]] void e88_cell_m{m}_ipg{ipg}(
%s) {{
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}
""" % SCORED_ARGS

PAIR_CELL = """
[[kernel]] void e88_cell_m{m}_pair(
%s) {{
  qmv_fast_crossrow_affine4_g64<bfloat16_t, {m}>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}
""" % SCORED_ARGS

# `out_vec_size >= 4096` selects the wide branch; every scored family clears it
# (smallest scored out_vec_size is 5120), so the wide switch is the live table.
WIDE_SWITCH = re.compile(
    r"if \(out_vec_size >= 4096\) \{(.*?)\n    \} else \{", re.DOTALL)
# Split on the case labels rather than matching label and call adjacently: the
# M = 8 arm carries a long comment between the two, and an adjacency regex
# silently drops that width from the census.
CASE_SPLIT = re.compile(r"^\s*case (\d+):\s*$", re.MULTILINE)
CALL_WIDE = re.compile(
    r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>")
CALL_PAIR = re.compile(r"qmv_fast_crossrow_affine4_g64<T, (\d+)>")


def live_table(header_text: str) -> dict[int, tuple[str, int]]:
    """Read the reachable {M: (kind, IPG)} table out of the wide dispatch."""
    match = WIDE_SWITCH.search(header_text)
    if not match:
        raise SystemExit("e88_rung01: could not locate the wide dispatch switch")
    chunks = CASE_SPLIT.split(match.group(1))
    table: dict[int, tuple[str, int]] = {}
    for label, body in zip(chunks[1::2], chunks[2::2]):
        m = int(label)
        wide = CALL_WIDE.search(body)
        pair = CALL_PAIR.search(body)
        if wide:
            if int(wide.group(1)) != m:
                raise SystemExit("e88_rung01: case %d dispatches width %s"
                                 % (m, wide.group(1)))
            table[m] = ("wide", int(wide.group(2)))
        elif pair:
            if int(pair.group(1)) != m:
                raise SystemExit("e88_rung01: case %d dispatches width %s"
                                 % (m, pair.group(1)))
            table[m] = ("pair", m)
        else:
            raise SystemExit("e88_rung01: case %d reaches no crossrow kernel" % m)
    if not table:
        raise SystemExit("e88_rung01: the wide dispatch switch is empty")
    return dict(sorted(table.items()))


def cell_name(m: int, kind: str, ipg: int) -> str:
    return "e88_cell_m%d_pair" % m if kind == "pair" else "e88_cell_m%d_ipg%d" % (m, ipg)


def cell_source(m: int, kind: str, ipg: int) -> str:
    template = PAIR_CELL if kind == "pair" else WIDE_CELL
    return template.format(m=m, ipg=ipg)


def arm_source(arm: str, table: dict[int, tuple[str, int]]) -> str:
    """The runtime-effective JIT string for this arm, byte-for-byte."""
    parts = []
    for stem in PREAMBLES:
        if stem == "quantized":
            text = e88_arms.apply_arm(
                e88_arms.base_text(e88_arms.TWIN), arm)
            body = PREAMBLE_BODY.search(text)
            if not body:
                raise SystemExit("e88_rung01: patched twin has no preamble body")
            parts.append(body.group(1) + "\n")
        else:
            parts.append(preamble(stem, None))
    parts += [template_def(cell) for cell in COMBINED_CELLS]
    parts += [cell_source(m, kind, ipg) for m, (kind, ipg) in table.items()]
    return "".join(parts)


def air_census(source: str, workdir: pathlib.Path,
               wanted: list[str]) -> dict[str, dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "air.metal"
    src.write_text(source)
    ll = workdir / "air.ll"
    o3 = workdir / "air.o3.ll"
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(src), "-o", str(ll)],
        check=True, capture_output=True, text=True)
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(o3)],
        check=True, capture_output=True, text=True)

    found = kernels(o3)
    out = {}
    for name in wanted:
        body = found.get(name)
        if body is None:
            raise SystemExit("e88_rung01: kernel %s missing from AIR" % name)
        allocas = [ALLOCA.search(l).group(1) for l in body if ALLOCA.search(l)]
        peak, values = peak_live_registers(body)
        out[name] = {
            "air_lines": len(body),
            "device_loads": sum(1 for l in body if DEVICE_LOAD.search(l)),
            "all_loads": sum(1 for l in body if ANY_LOAD.search(l)),
            "allocas": len(allocas),
            "alloca_types": sorted(set(allocas)),
            "fma_f32": sum(1 for l in body if FMA.search(l)),
            "fmul": sum(1 for l in body if re.search(r"=\s*fmul\s", l)),
            "fadd": sum(1 for l in body if re.search(r"=\s*fadd\s", l)),
            "backedges": sum(1 for l in body if "!llvm.loop" in l),
            "peak_live_regs": peak,
            "peak_live_values": values,
        }
    return out


def census_arm(arm: str, table: dict[int, tuple[str, int]],
               workdir: pathlib.Path) -> dict:
    source = arm_source(arm, table)
    names = [cell_name(m, kind, ipg) for m, (kind, ipg) in table.items()]
    lib = agx_crossarch.build_metallib(source, workdir / arm)
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "cells": names,
        "air": air_census(source, workdir / ("%s-air" % arm), names),
        "by_arch": {arch: agx_crossarch.translate(lib, arch, workdir / arm)
                    for arch in ARCHS},
    }


def verdict(shipped: dict, candidate: dict,
            table: dict[int, tuple[str, int]]) -> dict:
    failures, notes = [], []
    for m, (kind, ipg) in table.items():
        name = cell_name(m, kind, ipg)
        a, b = shipped["air"][name], candidate["air"][name]
        if a["fma_f32"] != b["fma_f32"]:
            failures.append(
                "%s: FMA count moved %d -> %d, the arithmetic changed"
                % (name, a["fma_f32"], b["fma_f32"]))
        if a["fmul"] != b["fmul"] or a["fadd"] != b["fadd"]:
            failures.append(
                "%s: scalar float op counts moved (fmul %d->%d, fadd %d->%d)"
                % (name, a["fmul"], b["fmul"], a["fadd"], b["fadd"]))
        if b["device_loads"] > a["device_loads"]:
            failures.append("%s: device loads rose %d -> %d"
                            % (name, a["device_loads"], b["device_loads"]))
        if b["allocas"] > a["allocas"]:
            failures.append("%s: allocas rose %d -> %d"
                            % (name, a["allocas"], b["allocas"]))
        for arch in ARCHS:
            ra = shipped["by_arch"][arch][name]
            rb = candidate["by_arch"][arch][name]
            if arch == "applegpu_g17s":
                if rb["registers"] > ra["registers"]:
                    failures.append(
                        "STOP RULE: %s ranked registers rose %d -> %d on %s"
                        % (name, ra["registers"], rb["registers"], arch))
                if rb["spill_bytes"] and not ra["spill_bytes"]:
                    failures.append(
                        "STOP RULE: %s ranked spill appeared (%d bytes) on %s"
                        % (name, rb["spill_bytes"], arch))
            if rb["registers"] > CEILING[arch]:
                failures.append("%s: %s registers %d exceed the %d ceiling"
                                % (name, arch, rb["registers"], CEILING[arch]))
            if rb["registers"] < ra["registers"]:
                notes.append("%s: %s registers fell %d -> %d"
                             % (name, arch, ra["registers"], rb["registers"]))
    return {"pass": not failures, "failures": failures, "notes": notes}


def table_row(name: str, shipped: dict, candidate: dict) -> str:
    a, b = shipped["air"][name], candidate["air"][name]

    def arch_pair(arch: str) -> str:
        ra = shipped["by_arch"][arch][name]
        rb = candidate["by_arch"][arch][name]
        return "%3d->%-3d %3d/%-3d" % (ra["registers"], rb["registers"],
                                       ra["spill_bytes"], rb["spill_bytes"])

    return "%-20s %4d->%-4d %2d->%-2d %5d->%-5d %5d->%-5d  %s  %s" % (
        name, a["device_loads"], b["device_loads"], a["allocas"], b["allocas"],
        a["fmul"], b["fmul"], a["fadd"], b["fadd"],
        arch_pair("applegpu_g16s"), arch_pair("applegpu_g17s"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/e88-artifacts/rung01.json"))
    args = ap.parse_args()

    header = e88_arms.base_text(e88_arms.HEADER)
    table = live_table(header)
    print("live dispatch table read from the source: %s" % {
        m: ("pair" if kind == "pair" else "IPG %d" % ipg)
        for m, (kind, ipg) in table.items()})

    with tempfile.TemporaryDirectory(prefix="e88-rung01-") as tmp:
        workdir = pathlib.Path(tmp)
        results = {arm: census_arm(arm, table, workdir) for arm in e88_arms.ARMS}

    checks = {arm: verdict(results["shipped"], results[arm], table)
              for arm in e88_arms.ARMS if arm != "shipped"}

    for arm, check in checks.items():
        print("\n=== arm %s against shipped ===" % arm)
        print("%-20s %-10s %-6s %-12s %-12s  %-15s  %-15s" % (
            "cell", "dev loads", "alloca", "fmul", "fadd",
            "g16s reg spillB", "g17s reg spillB"))
        for m, (kind, ipg) in table.items():
            print(table_row(cell_name(m, kind, ipg),
                            results["shipped"], results[arm]))
        print("notes:")
        for line in check["notes"] or ["  none"]:
            print("  %s" % line)
        print("verdict: %s" % ("PASS" if check["pass"] else "FAIL"))
        for line in check["failures"]:
            print("  %s" % line)

    payload = {
        "experiment": "e88",
        "harness": "local",
        "base_sha": e88_arms.BASE_SHA,
        "archs": list(ARCHS),
        "register_ceiling": CEILING,
        "live_table": {str(m): {"kind": kind, "ipg": ipg}
                       for m, (kind, ipg) in table.items()},
        "arms": results,
        "verdicts": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote %s" % args.out)
    return 0 if any(c["pass"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
