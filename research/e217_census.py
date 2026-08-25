#!/usr/bin/env python3
"""E217 step 1 (RULE 400): register, threadgroup-memory and occupancy census.

E217 asks how much of the `G(m) == 2` per-pass fixed cost an explicitly shared
weight tile in threadgroup memory recovers. Before any GPU second is spent,
the three mappings are priced on the two channels that decide whether a
threadgroup-memory kernel can win at all:

  * registers, because a staged consumer that spills has already lost; and
  * the 32,768-byte threadgroup pool, because the tile is a new claim on it
    and the pool caps residency independently of the register file.

The three mappings:

  split   the shipped kernel. 8 output rows per threadgroup, one column
          group each, 2 simdgroups, `(G*32, (n/8)*2, 1)`.
  coop    both column groups in one threadgroup over 4 output rows, 2
          simdgroups, no threadgroup memory and no barrier,
          `(32, (n/4)*2, 1)`. E216 measured this arm and it lost; it is
          censused here only as the cross-host replication control.
  staged  the shipped 8 output rows per threadgroup over 4 simdgroups (2 row
          halves times 2 column groups), with ONE raw packed 8-row weight tile
          in threadgroup memory, `(32, (n/8)*4, 1)`.

The staged tile holds raw packed weight words, not dequantized values: 8 rows
of 512 4-bit columns is 1024 ushort, 2048 bytes. Threadgroup occupancy is the
named main risk of this experiment, so the census reports the AIR-declared
threadgroup bytes per variant and the residency they imply against the pool.

Every source string is read live out of `Qwen35.swift`, so the census cannot
drift from the kernel the worker builds.

Zero GPU seconds; `xcrun metal-tt` runs the AGX backend for a named
architecture on any Mac.

  python3 research/e217_census.py
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agx_crossarch  # noqa: E402

REPO = HERE.parent
QWEN35 = REPO / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
OUT = HERE / "e217-artifacts" / "e217_census.json"

ARCHES = ("applegpu_g16s", "applegpu_g17s")

TG_POOL_BYTES = 32768
# E77 register file per core in bytes, over the 128 bytes one simdgroup needs
# per register. Local is the g16s host; ranked is the M5.
SIMDGROUPS_LOCAL = 384 * 1024 // 128
SIMDGROUPS_RANKED = 496 * 1024 // 128

SIMDGROUPS_PER_TG = {"split": 2, "coop": 2, "staged": 4}
ROWS_PER_TG = {"split": 8, "coop": 4, "staged": 8}

# The tile holds RAW PACKED weight words: 8 rows of 512 4-bit columns.
TILE_WORDS = 1024

# The `G == 2` widths E217 moves, with their shipped IPG. m = 2..5 are G == 1
# and stay on the shipped mapping; they are the flat control band, not arms.
CELLS = ((6, 3), (7, 4), (8, 4), (9, 5))

# MLX injects `bfloat16_t` through its own kernel preamble; a bare
# `metal_stdlib` build spells the same type `bfloat`.
PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

ENTRY_HEAD = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
"""

BODY = {
    "split": """    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    qwen_e120_qmv_m<{m}, {ipg}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x), qmv_out_row, qmv_lid);
""",
    "coop": """    const int qmv_out_row = int(qmv_tid.y) * 4;
    qwen_e120_qmv_m<{m}, {ipg}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_sgid), qmv_out_row, qmv_lid);
""",
    "staged": """    const int qmv_out_row = int(qmv_tid.y) * 8;
    threadgroup ushort qmv_tile[1024];
    qwen_e217_qmv_staged<{m}, {ipg}, {table}>(
        w, scales, biases, x, xsums, y, qmv_tile,
        qmv_k, qmv_n, qmv_stride,
        qmv_out_row, qmv_sgid, qmv_lid);
""",
}

TG_GLOBAL = re.compile(
    r"^@([\w.$]+)\s*=.*addrspace\(3\) global \[(\d+) x (\w+)\]"
)
AIR_WIDTH = {"float": 4, "half": 2, "bfloat": 2, "int": 4, "uint": 4,
             "i8": 1, "i16": 2, "i32": 4, "uchar": 1, "short": 2,
             "ushort": 2}


def live_header(symbol: str) -> str:
    """The exact source string the worker compiles, read out of Qwen35.swift.

    The access modifier is optional in the pattern because the two headers do
    not agree on it, and a strict pattern would silently census a stale copy.
    """
    text = QWEN35.read_text()
    match = re.search(
        r'(?:private |internal |public )?let ' + symbol + r' = """\n(.*?)\n    """\n',
        text,
        re.S,
    )
    if match is None:
        raise SystemExit(f"could not find {symbol} in Qwen35.swift")
    return match.group(1)


def cell_name(mapping: str, m: int, table: bool) -> str:
    return f"qmv_{mapping}_m{m}_{'tbl' if table else 'rec'}"


def arms() -> list[dict]:
    out = []
    for table in (True, False):
        for m, ipg in CELLS:
            for mapping in ("split", "coop", "staged"):
                out.append({"mapping": mapping, "m": m, "ipg": ipg,
                            "table": table})
    return out


def entry(arm: dict) -> str:
    name = cell_name(arm["mapping"], arm["m"], arm["table"])
    return (
        ENTRY_HEAD.format(name=name)
        + BODY[arm["mapping"]].format(
            m=arm["m"], ipg=arm["ipg"],
            table="true" if arm["table"] else "false")
        + "}\n"
    )


def threadgroup_bytes(source: str, workdir: pathlib.Path) -> dict[str, int]:
    """Bytes of threadgroup memory each entry point reaches, from textual AIR.

    An entry point can declare the allocation and then pass its pointer to a
    template instance that is not inlined at this stage, so the global is
    referenced from a different `define` than the kernel that owns it. The walk
    therefore follows the call graph from each function and sums the DISTINCT
    globals reachable from it; a name match on the declaring function would
    report zero for exactly the staged kernels this census exists to price.
    """
    src = workdir / "tg.metal"
    src.write_text(source)
    ll = workdir / "tg.ll"
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", "-emit-llvm", str(src), "-o", str(ll)],
        check=True, capture_output=True)
    text = ll.read_text()
    size = {
        m.group(1): int(m.group(2)) * AIR_WIDTH[m.group(3)]
        for line in text.splitlines()
        if (m := TG_GLOBAL.match(line))
    }
    globals_of: dict[str, set[str]] = {}
    calls_of: dict[str, set[str]] = {}
    name = None
    for line in text.splitlines():
        # Entry points are `define void @name(`; template instances carry a
        # linkage word as well, so the attribute run before `@` is variable.
        define = re.match(r"^define\s+[^@]*@([\w.$]+)\(", line)
        if define:
            name = define.group(1)
            globals_of[name] = set()
            calls_of[name] = set()
        elif line == "}":
            name = None
        elif name is not None:
            for reference in re.findall(r"@([\w.$]+)", line):
                if reference in size:
                    globals_of[name].add(reference)
            call = re.search(r"call \S+ @([\w.$]+)\(", line)
            if call:
                calls_of[name].add(call.group(1))

    def reachable(root: str) -> set[str]:
        found: set[str] = set()
        stack, seen = [root], {root}
        while stack:
            fn = stack.pop()
            found |= globals_of.get(fn, set())
            for callee in calls_of.get(fn, set()):
                if callee not in seen and callee in globals_of:
                    seen.add(callee)
                    stack.append(callee)
        return found

    return {fn: sum(size[g] for g in reachable(fn)) for fn in globals_of}


def pipeline_probe(lib: pathlib.Path, names: list[str],
                   workdir: pathlib.Path) -> dict[str, dict]:
    """Runtime `MTLComputePipelineState` fields for each censused kernel.

    The static census reads the AIR; the advisor's named risk is the RUNTIME
    `staticThreadgroupMemoryLength`, and `maxTotalThreadsPerThreadgroup` is a
    hard gate: the staged mapping dispatches 128-thread threadgroups, so a
    register allocation that caps the pipeline below 128 threads would make the
    launch illegal rather than slow. This builds no pipeline state that runs;
    it creates the pipeline object and reads its properties.
    """
    probe = workdir / "pipeline_probe"
    subprocess.run(
        ["clang", "-fobjc-arc", "-O2", "-framework", "Metal",
         "-framework", "Foundation", "-o", str(probe),
         str(HERE / "e102_pipeline_probe.m")],
        check=True, capture_output=True)
    proc = subprocess.run([str(probe), str(lib), *names],
                          check=True, capture_output=True, text=True)
    out: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("PIPELINE "):
            continue
        fields = dict(
            part.split("=", 1) for part in line.split(" ")[1:] if "=" in part)
        out[fields["function"]] = {
            "max_total_threads_per_threadgroup": int(
                fields["max_total_threads_per_threadgroup"]),
            "thread_execution_width": int(fields["thread_execution_width"]),
            "static_threadgroup_memory_bytes": int(
                fields["static_threadgroup_memory_bytes"]),
        }
    return out


def occupancy(registers: int, tg_bytes: int, simdgroups_per_tg: int) -> dict:
    by_memory = TG_POOL_BYTES // tg_bytes if tg_bytes else 1 << 20
    result = {"threadgroup_bytes": tg_bytes,
              "threadgroups_by_memory": by_memory}
    for tag, budget in (("local", SIMDGROUPS_LOCAL),
                        ("ranked", SIMDGROUPS_RANKED)):
        by_register = budget // registers if registers else 1 << 20
        groups = min(by_register // simdgroups_per_tg, by_memory)
        result[f"{tag}_simdgroups_by_register"] = by_register
        result[f"{tag}_threadgroups"] = groups
        result[f"{tag}_resident_simdgroups"] = groups * simdgroups_per_tg
        result[f"{tag}_binding"] = (
            "memory" if by_memory <= by_register // simdgroups_per_tg
            else "register")
    return result


def main() -> None:
    header = (live_header("qwen35E120QMVHeader") + "\n"
              + live_header("qwen35E217StagedHeader"))
    plan = arms()
    source = PREAMBLE + header + "\n" + "\n".join(entry(a) for a in plan)

    # FINDING 552: `<NA = 7, ROWS = 2>` must never be instantiated. E217's NA
    # values are IPG and TAIL over the four G == 2 cells, so {3, 4, 5}.
    nas = set()
    for m, ipg in CELLS:
        nas.add(ipg)
        nas.add(ipg if m % ipg == 0 else m % ipg)
    if 7 in nas:
        raise SystemExit(f"FINDING 552 tripwire: NA set is {sorted(nas)}")

    result = {
        "experiment": "e217-staged-dequant",
        "step": "1-rule-400-census",
        "harness": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "source": ("Qwen35.swift qwen35E120QMVHeader + qwen35E217StagedHeader,"
                   " read live"),
        "na_instantiated": sorted(nas),
        "tile_words": TILE_WORDS,
        "declared_tile_bytes": TILE_WORDS * 2,
        "arches": list(ARCHES),
        "cells": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx_crossarch.build_metallib(source, workdir)
        records: dict[str, dict] = {}
        for arch in ARCHES:
            for name, rec in agx_crossarch.translate(lib, arch, workdir).items():
                records.setdefault(name, {})[arch.replace("applegpu_", "")] = rec
        tg = threadgroup_bytes(source, workdir)
        names = [cell_name(a["mapping"], a["m"], a["table"]) for a in plan]
        pipelines = pipeline_probe(lib, names, workdir)

    def tg_for(name: str) -> int:
        if name in tg:
            return tg[name]
        for key, value in tg.items():
            if key.endswith(name) or name in key:
                return value
        return 0

    for arm in plan:
        name = cell_name(arm["mapping"], arm["m"], arm["table"])
        rec = records.get(name, {})
        tg_bytes = tg_for(name)
        out = {
            **{k: arm[k] for k in ("mapping", "m", "ipg", "table")},
            "simdgroups_per_threadgroup": SIMDGROUPS_PER_TG[arm["mapping"]],
            "output_rows_per_threadgroup": ROWS_PER_TG[arm["mapping"]],
            "declared_tile_bytes": (
                TILE_WORDS * 2 if arm["mapping"] == "staged" else 0),
            "air_threadgroup_bytes": tg_bytes,
            "pipeline": pipelines.get(name, {}),
        }
        threads = 32 * SIMDGROUPS_PER_TG[arm["mapping"]]
        cap = out["pipeline"].get("max_total_threads_per_threadgroup")
        if cap is not None and cap < threads:
            raise SystemExit(
                f"{name} needs {threads}-thread threadgroups but the pipeline "
                f"caps at {cap}")
        for arch, r in rec.items():
            regs = r.get("registers")
            out[arch] = {
                "registers": regs,
                "spill_bytes": r.get("spill_bytes"),
                "text_bytes": r.get("text_bytes"),
                "text_sha8": r.get("text_sha8"),
                "occupancy": occupancy(
                    regs or 0, tg_bytes, SIMDGROUPS_PER_TG[arm["mapping"]]),
            }
        result["cells"][name] = out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    for table in (True, False):
        print(f"\n=== use_table={table} ===")
        print(f"  {'cell':24s} {'tgB':>5s} {'pipeB':>5s} {'maxT':>5s}"
              f" {'sg/tg':>5s}"
              f" {'g16 reg/sp':>12s} {'g17 reg/sp':>12s}"
              f" {'g16 sgres':>9s} {'g17 sgres':>9s}")
        for arm in plan:
            if arm["table"] is not table:
                continue
            name = cell_name(arm["mapping"], arm["m"], arm["table"])
            e = result["cells"][name]
            g16, g17 = e.get("g16s", {}), e.get("g17s", {})
            o16 = g16.get("occupancy", {})
            o17 = g17.get("occupancy", {})
            print(
                f"  {name:24s} {e['air_threadgroup_bytes']:5d}"
                f" {e['pipeline'].get('static_threadgroup_memory_bytes', -1):5d}"
                f" {e['pipeline'].get('max_total_threads_per_threadgroup', -1):5d}"
                f" {e['simdgroups_per_threadgroup']:5d}"
                f" {str(g16.get('registers')) + '/' + str(g16.get('spill_bytes')):>12s}"
                f" {str(g17.get('registers')) + '/' + str(g17.get('spill_bytes')):>12s}"
                f" {o16.get('local_resident_simdgroups', '-'):>9}"
                f" {o17.get('ranked_resident_simdgroups', '-'):>9}"
            )
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
