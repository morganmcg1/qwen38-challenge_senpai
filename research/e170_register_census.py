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
import air_kernel_stats  # noqa: E402

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


def air_vector_census(source: str, workdir: pathlib.Path) -> dict:
    """Per-kernel histogram of fused-multiply-add vector widths in AIR.

    The inner product is written over `vec<float, NA>`, so the width the
    compiler actually emits says whether a given NA maps onto the native
    four-wide lane group or has to be split. AIR is architecture-independent,
    so a split here is a property of the source and the front end, not of the
    host, and it therefore transfers to the ranked generation.
    """
    src = workdir / "air_probe.metal"
    src.write_text(source)
    listing = workdir / "air_probe.ll"
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", "-emit-llvm", str(src), "-o", str(listing)],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip()[-600:]}
    # The Metal front end lowers the inner product to `llvm.fmuladd`, not to
    # the `llvm.fma` that air_kernel_stats looks for. At this stage the probe
    # entry points are thin wrappers, so the arithmetic sits in the mangled
    # template instantiation and has to be keyed off its template arguments.
    muladd = re.compile(r"@llvm\.fmuladd\.(v\d+f32|f32)\b")
    targs = re.compile(r"_Z\d+(qwen_e120_qmv_wide|qwen_e120_qmv_m)"
                       r"ILi(\d+)ELb([01])E(?:Li(\d+)E)?")
    out = {}
    for name, body in air_kernel_stats.kernels(listing).items():
        hit = targs.match(name)
        if not hit:
            continue
        template, first, table, rows = hit.groups()
        label = (f"wide_na{first}_r{rows}" if template.endswith("wide")
                 else f"entry_m{first}_r{rows}")
        label += "_tbl" if table == "1" else "_rec"
        widths: dict[str, int] = {}
        for line in body:
            found = muladd.search(line)
            if found:
                widths[found.group(1)] = widths.get(found.group(1), 0) + 1
        peak, allocas = air_kernel_stats.peak_live_registers(body)
        out[label] = {"fma_widths": widths, "fma_total": sum(widths.values()),
                      "lane_issues": sum(
                          (int(k[1:-3]) if k.startswith("v") else 1) * v
                          for k, v in widths.items()),
                      "air_peak_live_regs": peak, "allocas": allocas,
                      "mangled": name}
    return out


#: Live values the kernel body itself demands, read off the template source.
#: `acc[ROWS]` and `partial[ROWS]` are each ROWS copies of a `vec<float, NA>`
#: and are live together at the accumulate step; `sums` and the four staged
#: activation vectors `a0..a3` are NA wide once; `scale_local[ROWS]`,
#: `bias_local[ROWS]` and `packed[ROWS][4]` are ROWS wide with no NA factor.
ARITH_EXPRESSION = "R_data(NA, ROWS) = 2*ROWS*NA + 5*NA + 6*ROWS"


def arithmetic_registers(na: int, rows: int) -> dict:
    acc_partial = 2 * rows * na
    staged = 5 * na
    per_row_scalars = 6 * rows
    return {
        "acc_plus_partial": acc_partial,
        "sums_plus_a0_a3": staged,
        "scale_bias_packed": per_row_scalars,
        "total": acc_partial + staged + per_row_scalars,
    }


def arithmetic_reconciliation(law: dict, nas: list[int]) -> dict:
    """Compare the source-derived register demand with the compiled count.

    The gap is the budget the backend still has for addressing, unrolling and
    keeping loads in flight. Where that gap collapses, the kernel has stopped
    being able to hide memory latency, which is a different failure from
    simply losing resident simdgroups.
    """
    out = {}
    for arch, block in law.items():
        cells = []
        for na in nas:
            cell = block["per_na_table_pipeline"][f"na{na}"]
            for rows, key in ((4, "registers_rows4"), (2, "registers_rows2")):
                compiled = cell[key]
                if compiled is None:
                    continue
                demand = arithmetic_registers(na, rows)
                cells.append({
                    "na": na, "rows": rows,
                    "data_registers_from_source": demand["total"],
                    "compiled_registers": compiled,
                    "scaffolding_headroom": compiled - demand["total"],
                    "breakdown": demand,
                })
        if not cells:
            continue
        headrooms = [c["scaffolding_headroom"] for c in cells]
        others = [c["scaffolding_headroom"] for c in cells
                  if not (c["na"] == max(nas) and c["rows"] == 4)]
        squeezed = [c for c in cells if c["na"] == max(nas) and c["rows"] == 4]
        out[arch] = {
            "expression": ARITH_EXPRESSION,
            "cells": cells,
            "median_headroom": sorted(headrooms)[len(headrooms) // 2],
            "widest_rows4_headroom":
                squeezed[0]["scaffolding_headroom"] if squeezed else None,
            "other_cells_min_headroom": min(others) if others else None,
        }
    return out


def register_model(law: dict, nas: list[int]) -> dict:
    """Least-squares fit of R = base + per_row * ROWS * NA + per_input * NA.

    `acc` and `partial` are each `ROWS` copies of an `NA`-wide vector, and the
    four staged activation vectors plus `sums` are `NA` wide once. The fit
    checks that arithmetic against the compiled counts instead of asserting it.
    """
    out = {}
    for arch, block in law.items():
        rows_a: list[list[float]] = []
        rhs: list[float] = []
        observed = []
        for na in nas:
            cell = block["per_na_table_pipeline"][f"na{na}"]
            for rows, key in ((4, "registers_rows4"), (2, "registers_rows2")):
                value = cell[key]
                if value is None:
                    continue
                rows_a.append([1.0, float(rows * na), float(na)])
                rhs.append(float(value))
                observed.append({"na": na, "rows": rows, "registers": value})
        n = len(rhs)
        if n < 3:
            continue
        # Normal equations for a three-parameter fit; no numpy dependency.
        ata = [[sum(rows_a[i][p] * rows_a[i][q] for i in range(n))
                for q in range(3)] for p in range(3)]
        atb = [sum(rows_a[i][p] * rhs[i] for i in range(n)) for p in range(3)]
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(ata[r][col]))
            ata[col], ata[pivot] = ata[pivot], ata[col]
            atb[col], atb[pivot] = atb[pivot], atb[col]
            for r in range(3):
                if r == col or ata[col][col] == 0:
                    continue
                factor = ata[r][col] / ata[col][col]
                for c in range(3):
                    ata[r][c] -= factor * ata[col][c]
                atb[r] -= factor * atb[col]
        beta = [atb[i] / ata[i][i] if ata[i][i] else 0.0 for i in range(3)]
        for i, record in enumerate(observed):
            predicted = sum(beta[p] * rows_a[i][p] for p in range(3))
            record["predicted"] = round(predicted, 2)
            record["residual"] = round(record["registers"] - predicted, 2)
        out[arch] = {
            "expression": "R(NA, ROWS) = base + per_row_per_input * ROWS * NA "
                          "+ per_input * NA",
            "base": round(beta[0], 2),
            "per_row_per_input": round(beta[1], 3),
            "per_input": round(beta[2], 3),
            "max_abs_residual": round(
                max(abs(r["residual"]) for r in observed), 2),
            "observed": observed,
        }
    return out


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
        "air_vector_census": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        result["air_vector_census"] = air_vector_census(source, workdir)
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
                "text_bytes_rows4": row["rows4"].get("text_bytes"),
                "text_bytes_rows2": row["rows2"].get("text_bytes"),
                "spill_bytes_rows4": row["rows4"].get("spill_bytes"),
                "spill_bytes_rows2": row["rows2"].get("spill_bytes"),
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
    result["register_model"] = register_model(law, nas)
    result["arithmetic_reconciliation"] = arithmetic_reconciliation(law, nas)

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

    for arch, rec in sorted(result["arithmetic_reconciliation"].items()):
        print(f"\n=== {arch} source register demand vs compiled allocation ===")
        print(f"  {rec['expression']}")
        print(f"  {'NA':>3s} {'ROWS':>5s} {'2*R*NA':>7s} {'5*NA':>5s} {'6*R':>5s}"
              f" {'R_data':>7s} {'compiled':>9s} {'headroom':>9s}")
        for cell in rec["cells"]:
            parts = cell["breakdown"]
            print(f"  {cell['na']:>3d} {cell['rows']:>5d}"
                  f" {parts['acc_plus_partial']:>7d}"
                  f" {parts['sums_plus_a0_a3']:>5d}"
                  f" {parts['scale_bias_packed']:>5d}"
                  f" {cell['data_registers_from_source']:>7d}"
                  f" {cell['compiled_registers']:>9d}"
                  f" {cell['scaffolding_headroom']:>9d}")
        print(f"  widest rows4 headroom={rec['widest_rows4_headroom']}"
              f" vs min headroom elsewhere={rec['other_cells_min_headroom']}"
              f" (median {rec['median_headroom']})")

    for arch, fit in sorted(result["register_model"].items()):
        print(f"\n=== {arch} register arithmetic vs compiled counts ===")
        print(f"  {fit['expression']}")
        print(f"  base={fit['base']}  per_row_per_input={fit['per_row_per_input']}"
              f"  per_input={fit['per_input']}"
              f"  max|residual|={fit['max_abs_residual']}")
        print(f"  {'NA':>3s} {'ROWS':>5s} {'compiled':>9s} {'predicted':>10s}"
              f" {'residual':>9s}")
        for record in fit["observed"]:
            print(f"  {record['na']:>3d} {record['rows']:>5d}"
                  f" {record['registers']:>9d} {record['predicted']:>10.2f}"
                  f" {record['residual']:>9.2f}")

    air = result["air_vector_census"]
    if air and "error" not in air:
        print("\n=== AIR fused-multiply-add width census (architecture free) ===")
        for name, rec in sorted(air.items()):
            widths = " ".join(f"{k}x{v}" for k, v in sorted(rec["fma_widths"].items()))
            print(f"  {name:20s} fma={rec['fma_total']:<4d} lanes="
                  f"{rec['lane_issues']:<5d} [{widths}]"
                  f"  air_peak_live={rec['air_peak_live_regs']}"
                  f"  allocas={rec['allocas']}")
    elif air:
        print("\nAIR census unavailable:", air["error"][:300])

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
