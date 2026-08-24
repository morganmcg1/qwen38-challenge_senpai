#!/usr/bin/env python3
"""E170 F1 item: the occupancy law behind `rows_per_simd`, with register accounting.

The advisor asked for a law, not a percentage. This probe supplies the three
inputs the law needs:

  a. the measured register count per thread at `rows_per_simd` 4 and 2, for
     each NA, taken from the real AGX backend rather than counted in source;
  b. the implied register-limited occupancy in simdgroups per core; and
  c. the register file the ranked M5 would need for the NA = 5 dip to vanish
     at `rows_per_simd = 4`.

Two channels answer (a). `xcrun metal-tt` runs the AGX backend for a named
architecture and reports the per-kernel register count, so it covers both this
host (`applegpu_g16s`) and the ranked runner generation (`applegpu_g17s`).
`maxTotalThreadsPerThreadgroup` from a real local pipeline is an independent
readout of the same allocation, but only for this host, and Apple quantizes it.

Both the `entry` cells and the isolated `wide` cells matter, and they answer
different questions. AGX allocates registers per entry point, taking the
maximum over every branch, so `entry` is what the scored dispatch actually
gets. `ROWS_PER_SIMD` is a top-level template argument, so the shipped build
holds two whole pipelines and the plan picks between them per width; that is
why halving the row block at one width can move occupancy at all. The isolated
`wide` cells then show which NA sets that maximum.

Zero GPU seconds for the metal-tt channel. The local occupancy channel builds
pipelines but dispatches nothing.

  python3 research/e170_register_census.py
"""

from __future__ import annotations

import argparse
import ast
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
OCCUPANCY_SWIFT = REPO / "research/crossrow_na_occupancy.swift"

LOCAL_ARCH = agx_crossarch.LOCAL_ARCH
RANKED_ARCH = agx_crossarch.RANKED_ARCH
ARCHES = (LOCAL_ARCH, RANKED_ARCH)

# Register file per core in bytes, and the 128 bytes one simdgroup needs per
# register. Same constants the earlier censuses used, so the numbers stay
# comparable across the ledger.
REGISTER_FILE = {LOCAL_ARCH: 384 * 1024, RANKED_ARCH: 496 * 1024}

ROWS = (4, 2)

# MLX injects `bfloat16_t` through its own preamble; a bare metal_stdlib build
# spells the same type `bfloat`.
PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

WIDE_ENTRY = """
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
    qwen_e120_qmv_wide<{na}, {table}, {rows}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride, 0,
        qmv_tid.y, qmv_sgid, qmv_lid);
}}
"""

SWITCH_ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_m [[buffer(6)]],
    constant int& qmv_k [[buffer(7)]],
    constant int& qmv_n [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const int qmv_gx = int(qmv_tid.x);
    switch (qmv_m) {{
{cases}
        default:
            break;
    }}
}}
"""

SWITCH_CASE = """        case {m}:
            qwen_e120_qmv_m<{m}, {ipg}, {table}, {rows}>(
                w, scales, biases, x, xsums, y,
                qmv_k, qmv_n, qmv_stride,
                qmv_gx, qmv_tid.y, qmv_sgid, qmv_lid);
            break;
"""


def live_header() -> str:
    text = QWEN35.read_text()
    match = re.search(
        r'private let qwen35E120QMVHeader = """\n(.*?)\n    """\n', text, re.S)
    if match is None:
        raise SystemExit("could not find qwen35E120QMVHeader in Qwen35.swift")
    return match.group(1)


def live_ipg_table() -> dict[int, int]:
    """The shipped (M, IPG) switch, read out of the Swift source builder.

    Hard-coding it here would let the census drift away from the kernel it
    claims to describe, which is the exact failure that voided E135's probe.
    """
    text = QWEN35.read_text()
    match = re.search(r"let cases = (\[\([^\n]*\)\])\n", text)
    if match is None:
        raise SystemExit("could not find the (M, IPG) case list in Qwen35.swift")
    return {m: ipg for m, ipg in ast.literal_eval(match.group(1))}


def wide_name(na: int, rows: int, table: bool) -> str:
    return f"e170_wide_na{na}_r{rows}_{'tbl' if table else 'rec'}"


def entry_name(rows: int, table: bool) -> str:
    return f"e170_entry_r{rows}_{'tbl' if table else 'rec'}"


def probe_source(header: str, ipg_table: dict[int, int], nas: list[int]) -> str:
    parts = [PREAMBLE, header, ""]
    for table in (True, False):
        flag = "true" if table else "false"
        for rows in ROWS:
            for na in nas:
                parts.append(WIDE_ENTRY.format(
                    name=wide_name(na, rows, table), na=na, table=flag,
                    rows=rows))
            cases = "".join(
                SWITCH_CASE.format(m=m, ipg=ipg, table=flag, rows=rows)
                for m, ipg in sorted(ipg_table.items()))
            parts.append(SWITCH_ENTRY.format(
                name=entry_name(rows, table), cases=cases))
    return "\n".join(parts)


def local_occupancy(lib: pathlib.Path, workdir: pathlib.Path) -> dict:
    """`maxTotalThreadsPerThreadgroup` per kernel from a real local pipeline."""
    binary = workdir / "e170_occupancy"
    build = subprocess.run(
        ["swiftc", "-O", str(OCCUPANCY_SWIFT), "-o", str(binary)],
        capture_output=True, text=True)
    if build.returncode != 0:
        return {"error": build.stderr.strip()[-600:]}
    done = subprocess.run([str(binary), str(lib), "e170_"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip()[-600:]}
    found = {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit():
            found[parts[0]] = {
                "max_total_threads_per_threadgroup": int(parts[1]),
                "thread_execution_width": int(parts[2]),
                "static_threadgroup_memory_bytes": int(parts[3]),
            }
    return found


def simdgroups(arch: str, registers: int | None) -> int | None:
    if not registers:
        return None
    return REGISTER_FILE[arch] // (128 * registers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path,
                        default=HERE / "e170-artifacts" / "register_census.json")
    parser.add_argument("--nas", default="2,3,4,5",
                        help="NA values to instantiate in isolation")
    parser.add_argument("--skip-local-occupancy", action="store_true",
                        help="skip the pipeline channel; metal-tt only")
    args = parser.parse_args()

    nas = [int(v) for v in args.nas.split(",") if v.strip()]
    header = live_header()
    ipg_table = live_ipg_table()
    source = probe_source(header, ipg_table, nas)

    result = {
        "experiment": "e170-f1-occupancy-law",
        "harness": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "source": "Qwen35.swift qwen35E120QMVHeader, read live",
        "template_signature":
            "template <int NA, bool USE_TABLE, int ROWS> qwen_e120_qmv_wide",
        "ipg_table_read_live": {str(k): v for k, v in sorted(ipg_table.items())},
        "register_file_bytes": REGISTER_FILE,
        "bytes_per_simdgroup_register": 128,
        "cells": {},
        "local_occupancy": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx_crossarch.build_metallib(source, workdir)
        records: dict[str, dict] = {}
        for arch in ARCHES:
            translated = agx_crossarch.translate(
                lib, arch, workdir, select=lambda n: n.startswith("e170_"))
            for name, record in translated.items():
                records.setdefault(name, {})[arch] = record
        if not args.skip_local_occupancy:
            result["local_occupancy"] = local_occupancy(lib, workdir)

    for name, per_arch in sorted(records.items()):
        result["cells"][name] = {
            arch: {
                "registers": rec.get("registers"),
                "spill_bytes": rec.get("spill_bytes"),
                "text_bytes": rec.get("text_bytes"),
                "text_sha8": rec.get("text_sha8"),
                "resident_simdgroups_derived":
                    simdgroups(arch, rec.get("registers")),
            }
            for arch, rec in per_arch.items()
        }

    law = {}
    for arch in ARCHES:
        per_na = {}
        for na in nas:
            row = {}
            for rows in ROWS:
                cell = result["cells"].get(wide_name(na, rows, True), {})
                row[f"rows{rows}"] = cell.get(arch, {})
            regs4 = row["rows4"].get("registers")
            regs2 = row["rows2"].get("registers")
            sg4 = row["rows4"].get("resident_simdgroups_derived")
            sg2 = row["rows2"].get("resident_simdgroups_derived")
            per_na[f"na{na}"] = {
                "registers_rows4": regs4,
                "registers_rows2": regs2,
                "resident_simdgroups_rows4": sg4,
                "resident_simdgroups_rows2": sg2,
                "occupancy_ratio_rows2_over_rows4":
                    round(sg2 / sg4, 4) if sg4 and sg2 else None,
            }
        entries = {}
        for rows in ROWS:
            cell = result["cells"].get(entry_name(rows, True), {}).get(arch, {})
            entries[f"rows{rows}"] = {
                "registers": cell.get("registers"),
                "resident_simdgroups": cell.get("resident_simdgroups_derived"),
            }
        law[arch] = {"per_na_table_pipeline": per_na, "entry_table_pipeline": entries}

        # Where the register file would have to sit for the NA = 5 dip to
        # vanish at rows_per_simd = 4: NA = 5 would need the residency the
        # widest non-dipping width already gets.
        regs5 = per_na.get("na5", {}).get("registers_rows4")
        reference = [per_na[f"na{na}"]["resident_simdgroups_rows4"]
                     for na in nas if na != 5
                     and per_na[f"na{na}"]["resident_simdgroups_rows4"]]
        if regs5 and reference:
            want = max(reference)
            need = 128 * regs5 * want
            law[arch]["dip_vanishes_if"] = {
                "reference_resident_simdgroups": want,
                "na5_rows4_registers": regs5,
                "required_register_file_bytes": need,
                "actual_register_file_bytes": REGISTER_FILE[arch],
                "required_over_actual": round(need / REGISTER_FILE[arch], 4),
                "equivalent_max_registers_at_actual_file":
                    REGISTER_FILE[arch] // (128 * want),
            }
    result["occupancy_law"] = law

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    for arch in ARCHES:
        block = law[arch]
        print(f"\n=== {arch}  (register file {REGISTER_FILE[arch]} bytes, "
              f"{REGISTER_FILE[arch] // 128} simdgroup register slots) ===")
        print(f"  {'cell':10s} {'regs r4':>8s} {'regs r2':>8s} "
              f"{'sg r4':>6s} {'sg r2':>6s} {'sg ratio':>9s}")
        for na in nas:
            row = block["per_na_table_pipeline"][f"na{na}"]
            print(f"  NA={na:<8d} {str(row['registers_rows4']):>8s} "
                  f"{str(row['registers_rows2']):>8s} "
                  f"{str(row['resident_simdgroups_rows4']):>6s} "
                  f"{str(row['resident_simdgroups_rows2']):>6s} "
                  f"{str(row['occupancy_ratio_rows2_over_rows4']):>9s}")
        entry = block["entry_table_pipeline"]
        print(f"  entry      {str(entry['rows4']['registers']):>8s} "
              f"{str(entry['rows2']['registers']):>8s} "
              f"{str(entry['rows4']['resident_simdgroups']):>6s} "
              f"{str(entry['rows2']['resident_simdgroups']):>6s}")
        threshold = block.get("dip_vanishes_if")
        if threshold:
            print(f"  NA=5 rows4 needs {threshold['required_register_file_bytes']} "
                  f"bytes per core to reach {threshold['reference_resident_simdgroups']} "
                  f"simdgroups; the file is {threshold['actual_register_file_bytes']} "
                  f"({threshold['required_over_actual']}x short). Equivalently the "
                  f"backend would have to fit NA=5 in "
                  f"{threshold['equivalent_max_registers_at_actual_file']} registers.")

    occ = result["local_occupancy"]
    if occ and "error" not in occ:
        print("\n=== local pipeline channel (this host only) ===")
        for name, rec in sorted(occ.items()):
            print(f"  {name:28s} maxThreads={rec['max_total_threads_per_threadgroup']:<5d} "
                  f"execWidth={rec['thread_execution_width']} "
                  f"tgMem={rec['static_threadgroup_memory_bytes']}")
    elif occ:
        print("\nlocal pipeline channel unavailable:", occ["error"][:300])

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
