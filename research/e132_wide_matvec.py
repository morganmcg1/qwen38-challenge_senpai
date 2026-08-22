#!/usr/bin/env python3
"""E132: can `qwen_e120_qmv_wide<8>` be made spill-free on `applegpu_g17s`?

The Route B dispatch table makes ONE pass over the weight matrix at M=3,4,5 and
two or three passes at M=6..9. A one-pass table `{6:6, 7:7, 8:8}` is the
campaign's largest open lever, but `qwen_e120_qmv_wide<8>` is the only body in
the prize set that spills on the ranked architecture: `applegpu_g17s` reports
126 registers and 48 spill bytes.

This module reads `Qwen35.swift` READ-ONLY at a git revision, rewrites the
dispatch table and the `qwen_e120_qmv_wide` template IN MEMORY, and censuses the
result through `xcrun metal-tt`. Nothing here writes to a tracked source.
thorfinn owns `Qwen35.swift` this round and alphonse owns `quantized.h`, so
every variant below is a patch PROPOSAL, never an edit.

RULE 89. `simdgroups = floor(BUDGET / registers)` is a MODEL OUTPUT computed
from the register count, not a measurement. Every simdgroup figure is labelled
`derived`. Registers, spill bytes and ISA text sizes ARE measurements: they are
read out of the translated binary.

RULE 92. No variant changes a floating-point operation or its order.
`research/e132_wide_source.py` carries the per-flag argument and proves that the
generator reproduces the shipped kernel byte for byte with every flag off.

  python3 research/e132_wide_matvec.py rung0
  python3 research/e132_wide_matvec.py rung1
  python3 research/e132_wide_matvec.py templating-proof
  python3 research/e132_wide_matvec.py budget
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e120_g17s_census as e120  # noqa: E402
import e131_kernel_sources as ks  # noqa: E402
import e132_wide_source as wide  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)
RANKED = agx.RANKED_ARCH
SIMDGROUP_BUDGET = e120.SIMDGROUP_BUDGET

# The highest register count the allocator will issue, measured by the
# named-scalar pressure ladder in `research/e132_instruction_probe.py`. Every
# demand from 128 to 208 live floats returns exactly these numbers, so they are
# the architectural ceiling rather than a property of any one kernel. This
# corrects F34, which records 124 for `applegpu_g17s`.
SATURATING_CEILING = {"applegpu_g16s": 96, "applegpu_g17s": 126}

SHIPPED_TABLE = ((3, 3), (4, 4), (5, 5), (6, 3), (7, 4), (8, 4), (9, 3))

# Realised routed-round width histogram (assignment). Denominator 308.
HISTOGRAM = {4: 16, 5: 20, 6: 20, 7: 12, 8: 240}

TABLES = {
    "shipped": {},
    "one_pass_6": {6: 6},
    "one_pass_67": {6: 6, 7: 7},
    "one_pass_678": {6: 6, 7: 7, 8: 8},
}

# Candidate letters are the advisor's, in E132-F1 section 5.
VARIANTS = {
    "baseline": ({}, "shipped body"),
    "D_late_meta": ({"late_meta": True},
                    "D: load scales/biases where they are used"),
    "S_sink_sums": ({"sink_sums": True},
                    "read the chunk-sum table after the i-loop"),
    "D_S": ({"late_meta": True, "sink_sums": True}, "D + sink_sums"),
    "G4": ({"msplit": {8: 4}}, "G: m-tiling, MSPLIT=4 at NA=8"),
    "G2": ({"msplit": {8: 2}}, "G: m-tiling, MSPLIT=2 at NA=8"),
    "G4_D": ({"msplit": {8: 4}, "late_meta": True}, "G4 + D"),
    "G2_D": ({"msplit": {8: 2}, "late_meta": True}, "G2 + D"),
    "G4_D_S": ({"msplit": {8: 4}, "late_meta": True, "sink_sums": True},
               "G4 + D + sink_sums"),
    "G_wide": ({"msplit": {6: 3, 8: 4, 9: 3}},
               "G at every composite width above 5"),
    "G_wide_D": ({"msplit": {6: 3, 8: 4, 9: 3}, "late_meta": True},
                 "G at every composite width above 5, + D"),
    "B_rows2": ({"rows2_at": 8}, "B: rows_per_simd 4 -> 2 at NA=8 only"),
    "B_rows2_D": ({"rows2_at": 8, "late_meta": True}, "B + D"),
}

QMV_INPUTS = ks.QMV_INPUTS
QMV_OUTPUTS = ks.QMV_OUTPUTS
CASES = re.compile(r"(let cases = \[)(.*?)(\]\n)", re.DOTALL)


# --------------------------------------------------------------------------
# read-only Swift access and in-memory rewriting
# --------------------------------------------------------------------------

def swift_at(rev: str | None) -> str:
    return ks.swift_text(ks.QWEN35, rev)


def set_table(swift: str, pairs) -> str:
    """Replace Route B's whole dispatch table, in memory."""
    match = CASES.search(swift)
    if match is None:
        raise SystemExit("no Route B dispatch table in the extracted Swift text")
    body = ", ".join("(%d, %d)" % (m, ipg) for m, ipg in pairs)
    return swift[:match.start(2)] + body + swift[match.end(2):]


def table_pairs(override: dict[int, int]) -> tuple:
    return tuple((m, override.get(m, ipg)) for m, ipg in SHIPPED_TABLE)


def header_at(swift: str, **flags) -> str:
    header = ks.named_literal(swift, "qwen35E120QMVHeader")
    if not flags:
        wide.assert_faithful(header)
        return header
    return wide.patched_header(header, **flags)


def pipeline_body(swift: str, pairs, table: bool) -> str:
    """The Route B entry-point body restricted to `pairs`, rebuilt from Swift."""
    return ks.e120_qmv_body(set_table(swift, pairs), table)


def na_body(na: int, table: bool, rows_per_simd: int = 4) -> str:
    """An entry point holding exactly `qwen_e120_qmv_wide<NA>`.

    `qwen_e120_qmv_m<NA, NA, FLAG>` has `TAIL == 0`, so it inlines the single
    `qwen_e120_qmv_wide<NA>` body and nothing else. This is the per-body census
    thorfinn's register table is built from.
    """
    sums = "xsums" if table else "qmv_null_sums"
    flag = "USE_TABLE" if table else "false"
    null_decl = "" if table else (
        "\n    const device float* qmv_null_sums = nullptr;")
    return """    const int qmv_m = x_shape[x_ndim - 2];
    const int qmv_k = x_shape[x_ndim - 1];
    const int qmv_n = w_shape[0];
    const int qmv_stride = qmv_m <= 8 ? 8 : 16;
    const uint3 qmv_tid = threadgroup_position_in_grid;
    const uint qmv_lid = thread_index_in_simdgroup;
    const uint qmv_sgid = simdgroup_index_in_threadgroup;
    const int qmv_out_row = int(qmv_tid.y) * %d + int(qmv_sgid) * %d;
    const int qmv_gx = int(qmv_tid.x);%s
    qwen_e120_qmv_m<%d, %d, %s>(
        w, scales, biases, x, %s, y,
        qmv_k, qmv_n, qmv_stride,
        qmv_gx, qmv_out_row, qmv_lid);""" % (
        2 * rows_per_simd, rows_per_simd, null_decl, na, na, flag, sums)


def entry_point(name: str, body: str, table: bool,
                extra_template: list | None = None) -> str:
    inputs = QMV_INPUTS + ([("xsums", "float")] if table else [])
    template = [("bool", "USE_TABLE", "true")] if table else None
    if extra_template:
        template = (template or []) + extra_template
    return e120.generate(name, inputs, QMV_OUTPUTS, body, template)


def library(header: str, parts: list[str]) -> str:
    return "\n".join([e120.PRELUDE, header, ""] + parts) + "\n"


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------

def census(source: str, names: list[str], workdir: pathlib.Path,
           tag: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    wanted = set(names)
    lib = agx.build_metallib(source, workdir / tag)
    for arch in ARCHES:
        found = agx.translate(lib, arch, workdir / tag,
                              select=lambda n: n in wanted)
        for name in names:
            record = found[name]
            registers = record["registers"]
            rows.setdefault(name, {})[arch] = {
                "registers": registers,
                "spill_bytes": record["spill_bytes"],
                "text_bytes": record["text_bytes"],
                "text_sha8": record["text_sha8"],
                "simdgroups_derived": SIMDGROUP_BUDGET[arch] // registers,
            }
    return rows


def weighted_residency(per_width: dict[int, int]) -> float:
    total = sum(HISTOGRAM.values())
    return sum(HISTOGRAM[m] * per_width[m] for m in HISTOGRAM) / total


def bodies_behind(m: int, ipg: int) -> list[int]:
    """The `qwen_e120_qmv_wide<NA>` bodies `qwen_e120_qmv_m<m, ipg>` inlines."""
    tail = m % ipg
    return [ipg] if tail == 0 else sorted({ipg, max(tail, 2)})


def toolchain() -> str:
    done = subprocess.run(["xcrun", "metal", "--version"], capture_output=True,
                          text=True, check=True)
    lines = [l for l in (done.stdout + done.stderr).splitlines() if l.strip()]
    return lines[0] if lines else "unknown"


def receipt_header(base: str) -> dict:
    return {
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "harness": "compile_only",
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "base_ref": base,
        "base_sha": subprocess.run(["git", "rev-parse", base], cwd=str(ROOT),
                                   capture_output=True, text=True,
                                   check=True).stdout.strip(),
        "toolchain": toolchain(),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "histogram": {str(k): v for k, v in HISTOGRAM.items()},
        "histogram_total": sum(HISTOGRAM.values()),
    }


# --------------------------------------------------------------------------
# rung 0
# --------------------------------------------------------------------------

def rung0(base: str, out: pathlib.Path) -> int:
    started = time.time()
    swift = swift_at(base)
    header = header_at(swift)

    distinct = sorted({pair for override in TABLES.values()
                       for pair in table_pairs(override)})
    na_widths = tuple(range(2, 10))
    cells: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for table in (True, False):
            arm = "sumtable" if table else "notable"
            parts, names = [], []
            for name, override in TABLES.items():
                kernel = "switch_%s" % name
                parts.append(entry_point(
                    kernel, pipeline_body(swift, table_pairs(override), table),
                    table))
                names.append(kernel)
            for m, ipg in distinct:
                kernel = "pipeline_m%d_ipg%d" % (m, ipg)
                parts.append(entry_point(
                    kernel, pipeline_body(swift, ((m, ipg),), table), table))
                names.append(kernel)
            for na in na_widths:
                kernel = "body_na%d" % na
                parts.append(entry_point(kernel, na_body(na, table), table))
                names.append(kernel)
            cells[arm] = census(library(header, parts), names, workdir, arm)

    receipt = receipt_header(base)
    receipt.update({
        "tool": "research/e132_wide_matvec.py rung0",
        "rung": 0,
        "tables": {k: [list(p) for p in table_pairs(v)]
                   for k, v in TABLES.items()},
        "cells": cells,
        "matrix": [],
    })

    for arm in ("sumtable", "notable"):
        for table_name, override in TABLES.items():
            pairs = table_pairs(override)
            for structure in ("shared_switch", "per_width"):
                for arch in ARCHES:
                    if structure == "shared_switch":
                        entry = cells[arm]["switch_%s" % table_name][arch]
                        pipelines = 1
                        per_width = {m: entry["simdgroups_derived"]
                                     for m in HISTOGRAM}
                        registers = entry["registers"]
                        spill = entry["spill_bytes"]
                        text = entry["text_bytes"]
                        sgs = entry["simdgroups_derived"]
                    else:
                        pipelines = len(pairs)
                        per = {m: cells[arm]["pipeline_m%d_ipg%d" % (m, ipg)][arch]
                               for m, ipg in pairs}
                        per_width = {m: per[m]["simdgroups_derived"]
                                     for m in HISTOGRAM}
                        registers = max(v["registers"] for v in per.values())
                        spill = max(v["spill_bytes"] for v in per.values())
                        text = sum(v["text_bytes"] for v in per.values())
                        sgs = min(v["simdgroups_derived"] for v in per.values())
                    bodies = {na: cells[arm]["body_na%d" % na][arch]
                              for m, ipg in pairs
                              for na in bodies_behind(m, ipg)}
                    receipt["matrix"].append({
                        "arm": arm,
                        "use_table": arm == "sumtable",
                        "table": table_name,
                        "structure": structure,
                        "arch": arch,
                        "entry_registers": registers,
                        "entry_spill_bytes": spill,
                        "entry_simdgroups_derived": sgs,
                        "isa_text_bytes": text,
                        "distinct_pipelines": pipelines,
                        "per_width_ipg": dict(pairs),
                        "per_width_simdgroups_derived": per_width,
                        "body_registers": {
                            str(na): {"registers": v["registers"],
                                      "spill_bytes": v["spill_bytes"],
                                      "simdgroups_derived":
                                          v["simdgroups_derived"]}
                            for na, v in sorted(bodies.items())},
                        "weighted_simdgroups_derived":
                            round(weighted_residency(per_width), 4),
                    })

    receipt["runtime_seconds"] = round(time.time() - started, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print_rung0(receipt)
    print("\nwrote %s in %.1f s" % (out, receipt["runtime_seconds"]))
    return 0


def print_rung0(receipt: dict) -> None:
    print("E132 rung 0 census   base %s   %s"
          % (receipt["base_sha"][:8], receipt["toolchain"]))
    print("simdgroups are DERIVED from registers (Rule 89): "
          "floor(3072/regs) g16s, floor(3968/regs) g17s")
    print()
    head = ("%-9s %-13s %-14s %-15s %6s %6s %5s %8s %6s %8s"
            % ("arm", "table", "structure", "arch", "regs", "spill", "sg",
               "isa_B", "pipes", "wtd_sg"))
    print(head)
    print("-" * len(head))
    for row in receipt["matrix"]:
        print("%-9s %-13s %-14s %-15s %6d %6d %5d %8d %6d %8.2f"
              % (row["arm"], row["table"], row["structure"], row["arch"],
                 row["entry_registers"], row["entry_spill_bytes"],
                 row["entry_simdgroups_derived"], row["isa_text_bytes"],
                 row["distinct_pipelines"],
                 row["weighted_simdgroups_derived"]))
    print()
    print("per-body census of qwen_e120_qmv_wide<NA>: registers / spill bytes "
          "/ derived simdgroups")
    head = "%-4s %-24s %-24s %-24s %-24s" % (
        "NA", "g16s sumtable", "g17s sumtable", "g16s notable", "g17s notable")
    print(head)
    print("-" * len(head))
    for na in range(2, 10):
        parts = []
        for arm in ("sumtable", "notable"):
            for arch in ARCHES:
                cell = receipt["cells"][arm]["body_na%d" % na][arch]
                parts.append("%3d / %-3d  sg %-3d" % (
                    cell["registers"], cell["spill_bytes"],
                    cell["simdgroups_derived"]))
        print("%-4d %-24s %-24s %-24s %-24s"
              % (na, parts[0], parts[1], parts[2], parts[3]))


# --------------------------------------------------------------------------
# rung 1
# --------------------------------------------------------------------------

def rung1(base: str, out: pathlib.Path, widths=(4, 5, 6, 7, 8, 9)) -> int:
    started = time.time()
    swift = swift_at(base)
    results: dict = {}
    deltas: dict = {}
    plain = header_at(swift)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for name, (flags, _) in VARIANTS.items():
            header = header_at(swift, **flags) if flags else plain
            deltas[name] = len(header) - len(plain)
            for table in (True, False):
                arm = "sumtable" if table else "notable"
                parts, names = [], []
                for na in widths:
                    kernel = "body_na%d" % na
                    rows = 2 if flags.get("rows2_at") == na else 4
                    parts.append(entry_point(
                        kernel, na_body(na, table, rows), table))
                    names.append(kernel)
                results.setdefault(name, {})[arm] = census(
                    library(header, parts), names, workdir,
                    "%s_%s" % (name, arm))

    receipt = receipt_header(base)
    receipt.update({
        "tool": "research/e132_wide_matvec.py rung1",
        "rung": 1,
        "primary_metric": "e132_na8_spill_bytes_applegpu_g17s",
        "primary_metric_direction": "minimize",
        "primary_metric_baseline": 48,
        "widths": list(widths),
        "generator_faithful_sha8": wide.SHIPPED_SHA8,
        "variants": {name: {"note": note, "flags": flags,
                            "source_delta_bytes": deltas[name],
                            "cells": results[name]}
                     for name, (flags, note) in VARIANTS.items()},
    })

    # A variant only ships if it leaves every currently routed width alone.
    base_text = {arm: {na: results["baseline"][arm]["body_na%d" % na]
                       for na in widths} for arm in ("sumtable", "notable")}
    frontier = []
    for name, (flags, note) in VARIANTS.items():
        unchanged, changed = [], []
        for arm in ("sumtable", "notable"):
            for na in widths:
                same = all(
                    results[name][arm]["body_na%d" % na][arch]["text_sha8"]
                    == base_text[arm][na][arch]["text_sha8"]
                    for arch in ARCHES)
                (unchanged if same else changed).append("%s NA=%d" % (arm, na))
        row = {"variant": name, "note": note,
               "source_delta_bytes": deltas[name],
               "machine_code_unchanged": unchanged,
               "machine_code_changed": changed}
        for arm in ("sumtable", "notable"):
            cell = results[name][arm]["body_na8"][RANKED]
            row["na8_%s_g17s" % arm] = {
                "registers": cell["registers"],
                "spill_bytes": cell["spill_bytes"],
                "simdgroups_derived": cell["simdgroups_derived"],
                "isa_text_bytes": cell["text_bytes"]}
        frontier.append(row)
    receipt["frontier"] = frontier
    receipt["runtime_seconds"] = round(time.time() - started, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")

    print("E132 rung 1 spill search   base %s" % receipt["base_sha"][:8])
    print("primary metric e132_na8_spill_bytes_applegpu_g17s, baseline 48, "
          "MINIMIZE.  cells are registers/spill_bytes")
    for arm in ("sumtable", "notable"):
        print()
        head = "%-13s %s" % ("variant [%s]" % arm, " ".join(
            "%-16s" % ("NA=%d g16s|g17s" % na) for na in widths))
        print(head)
        print("-" * len(head))
        for name in VARIANTS:
            cols = []
            for na in widths:
                row = results[name][arm]["body_na%d" % na]
                cols.append("%-16s" % ("%3d/%-3d %3d/%-3d" % (
                    row[ARCHES[0]]["registers"], row[ARCHES[0]]["spill_bytes"],
                    row[ARCHES[1]]["registers"], row[ARCHES[1]]["spill_bytes"])))
            print("%-13s %s" % (name, " ".join(cols)))
    print()
    print("wrote %s in %.1f s" % (out, receipt["runtime_seconds"]))
    return 0


# --------------------------------------------------------------------------
# rung 1b: the instruction-count table E132-F2 asks for
# --------------------------------------------------------------------------

# An AGX ALU instruction is exactly 4 bytes on both architectures and a device
# load with its address arithmetic is 24 (g16s) / 20 (g17s). Both constants
# come from `research/e132_instruction_probe.py calibrate`, which fits ladders
# whose instruction delta is known by construction and reports residual 0.
BYTES_PER_ALU_INSTRUCTION = 4
BYTES_PER_MEMORY_INSTRUCTION = {"applegpu_g16s": 24, "applegpu_g17s": 20}

# Bytes of spill code the allocator emits per spilled 4-byte slot, from
# `research/e132_instruction_probe.py spillcost`.
BYTES_PER_SPILL_SLOT = {"applegpu_g16s": 17.77, "applegpu_g17s": 18.49}


# `research/e132_instruction_probe.py onset` steps the pressure ladder by one
# live value across the spill onset on both architectures. Every frame it
# produces is a multiple of 16 bytes, and a kernel that spills nothing reports
# exactly 0, so the frame is allocated on demand and 16-byte aligned. The
# `4 * slots + 16` law inherited from `research/agx_crossarch.py` is therefore
# wrong: it reads a 16-byte frame as no spill when the frame holds 1 to 4
# slots. Frame bytes are the measurement; the slot count is a band.
def slot_band(frame_bytes: int) -> tuple[int, int]:
    """Spilled 4-byte slots consistent with a 16-byte-aligned frame."""
    if frame_bytes <= 0:
        return (0, 0)
    return (max(1, frame_bytes // 4 - 3), frame_bytes // 4)


SHELL_FLAGS = ("msplit", "rows2_at")


def instruction_band(byte_count: int, arch: str) -> dict:
    """Machine instructions in `byte_count` bytes of AGX code.

    AGX is variable length. Every instruction is at least
    `BYTES_PER_ALU_INSTRUCTION`, so that divisor is the upper bound on the
    count, and a device load with address arithmetic is the widest thing the
    QMV body emits, so that divisor is the lower bound.
    """
    return {
        "bytes": byte_count,
        "instructions_upper": round(byte_count / BYTES_PER_ALU_INSTRUCTION, 1),
        "instructions_lower": round(
            byte_count / BYTES_PER_MEMORY_INSTRUCTION[arch], 1),
    }


def f2table(base: str, out: pathlib.Path, widths=(3, 4, 5, 6, 7, 8, 9)) -> int:
    """Static instruction counts per candidate, per architecture.

    E132-F2 makes the instruction count the ranking key and demotes resident
    simdgroups to a tiebreak. `text_bytes` covers the whole kernel, so the
    prologue and the reduction epilogue are measured separately through the
    `shell_only` generator and subtracted. What is left is the k-block body:
    the inner loop the feedback asks about.
    """
    started = time.time()
    swift = swift_at(base)
    plain = header_at(swift)
    full: dict = {}
    shells: dict = {}
    shell_key_of: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for name, (flags, _note) in VARIANTS.items():
            header = header_at(swift, **flags) if flags else plain
            shell_flags = {k: v for k, v in flags.items() if k in SHELL_FLAGS}
            key = json.dumps(shell_flags, sort_keys=True)
            shell_key_of[name] = key
            for table in (True, False):
                arm = "sumtable" if table else "notable"
                parts, names = [], []
                for na in widths:
                    rows = wide.rows_per_simd(na, **flags)
                    parts.append(entry_point(
                        "body_na%d" % na, na_body(na, table, rows), table))
                    names.append("body_na%d" % na)
                full.setdefault(name, {})[arm] = census(
                    library(header, parts), names, workdir,
                    "full_%s_%s" % (name, arm))
                if (key, arm) in shells:
                    continue
                shell_header = wide.patched_header(
                    plain, shell_only=True, **shell_flags)
                shells[(key, arm)] = census(
                    library(shell_header, parts), names, workdir,
                    "shell_%d_%s" % (abs(hash(key)) % 10 ** 8, arm))

    rows: dict = {}
    for name, (flags, note) in VARIANTS.items():
        key = shell_key_of[name]
        entry = {"note": note, "flags": flags, "cells": {}}
        for arm in ("sumtable", "notable"):
            for na in widths:
                kernel = "body_na%d" % na
                oe = wide.output_elements_per_thread(na, **flags)
                for arch in ARCHES:
                    cell = full[name][arm][kernel][arch]
                    shell_bytes = shells[(key, arm)][kernel][arch]["text_bytes"]
                    inner = cell["text_bytes"] - shell_bytes
                    low, high = slot_band(cell["spill_bytes"])
                    spill_code = round(high * BYTES_PER_SPILL_SLOT[arch])
                    band = instruction_band(inner, arch)
                    entry["cells"].setdefault(arm, {}).setdefault(
                        "na%d" % na, {})[arch] = {
                        "registers": cell["registers"],
                        "simdgroups_derived": cell["simdgroups_derived"],
                        "spill_frame_bytes": cell["spill_bytes"],
                        "spill_slots_band": [low, high],
                        "spill_free": cell["spill_bytes"] == 0,
                        "spill_code_bytes_upper": spill_code,
                        "spill_instructions_point": 2 * high,
                        "spill_instructions_band": [
                            instruction_band(spill_code, arch)[k]
                            for k in ("instructions_lower",
                                      "instructions_upper")],
                        "isa_text_bytes": cell["text_bytes"],
                        "shell_text_bytes": shell_bytes,
                        "inner_loop_text_bytes": inner,
                        "inner_loop_instructions_upper":
                            band["instructions_upper"],
                        "inner_loop_instructions_lower":
                            band["instructions_lower"],
                        "output_elements_per_thread": oe,
                        "inner_loop_instructions_per_output_element":
                            round(band["instructions_upper"] / oe, 2),
                        "dynamic": wide.dynamic_ops(
                            na, arm == "sumtable", **flags),
                    }
        rows[name] = entry

    total = sum(HISTOGRAM.values())
    for name, entry in rows.items():
        for arm in ("sumtable", "notable"):
            for arch in ARCHES:
                deleted = 0.0
                for na in widths:
                    if na not in HISTOGRAM:
                        continue
                    mine = entry["cells"][arm]["na%d" % na][arch]
                    theirs = rows["baseline"]["cells"][arm]["na%d" % na][arch]
                    delta = (theirs[
                        "inner_loop_instructions_per_output_element"]
                        - mine[
                        "inner_loop_instructions_per_output_element"])
                    mine["deleted_instructions_per_output_element"] = round(
                        delta, 2)
                    deleted += delta * HISTOGRAM[na] / total
                entry.setdefault("weighted_deleted_ipoe", {}).setdefault(
                    arm, {})[arch] = round(deleted, 3)

    receipt = receipt_header(base)
    receipt.update({
        "tool": "research/e132_wide_matvec.py f2table",
        "rung": "1b",
        "answers": "E132-F2 items 1-4",
        "ranking_key": "deleted_instructions_per_output_element, "
                       "sumtable arm, applegpu_g17s",
        "instruction_channel": {
            "bytes_per_alu_instruction": BYTES_PER_ALU_INSTRUCTION,
            "bytes_per_memory_instruction": BYTES_PER_MEMORY_INSTRUCTION,
            "bytes_per_spill_slot": BYTES_PER_SPILL_SLOT,
            "source": "research/e132_instruction_probe.py calibrate|spillcost",
        },
        "widths": list(widths),
        "histogram": HISTOGRAM,
        "generator_faithful_sha8": wide.SHIPPED_SHA8,
        "variants": rows,
        "dispatch_tables": dispatch_table_cost(rows),
        "runtime_seconds": round(time.time() - started, 2),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print_f2table(receipt)
    print("\nwrote %s in %.1f s" % (out, receipt["runtime_seconds"]))
    return 0


def dispatch_table_cost(rows: dict, variants=("baseline", "D_S")) -> dict:
    """Cost of every candidate Route B dispatch table, per variant.

    A multi-pass pipeline runs one whole k loop per pass, so its cost per
    OUTPUT element is the cost of the widest body it inlines. Residency is the
    worst body's residency, because that body limits the launch.
    """
    total = sum(HISTOGRAM.values())
    report: dict = {}
    for variant in variants:
        cells = rows[variant]["cells"]["sumtable"]
        for label, override in TABLES.items():
            residency, static, dynamic = {}, {}, {}
            for m, ipg in table_pairs(override):
                if m not in HISTOGRAM:
                    continue
                behind = bodies_behind(m, ipg)
                pick = [cells["na%d" % na][RANKED] for na in behind]
                residency[m] = min(c["simdgroups_derived"] for c in pick)
                static[m] = max(
                    c["inner_loop_instructions_per_output_element"]
                    for c in pick)
                dynamic[m] = max(
                    c["dynamic"]["ops_per_output_element"] for c in pick)
            report.setdefault(variant, {})[label] = {
                "per_width_simdgroups_derived": residency,
                "weighted_simdgroups_derived": round(
                    weighted_residency(residency), 2),
                "weighted_static_instructions_per_output_element": round(
                    sum(HISTOGRAM[m] * static[m] for m in HISTOGRAM) / total,
                    2),
                "weighted_dynamic_ops_per_output_element": round(
                    sum(HISTOGRAM[m] * dynamic[m] for m in HISTOGRAM) / total,
                    2),
            }
    return report


# A spilled value costs one store and one reload each time the k block touches
# it. `acc[]` is read and written once per k block; `partial[]` is updated on
# every one of the four `i` iterations. So the charge per slot per k block runs
# from 2 to 8 memory instructions, and the true value depends on which array
# the allocator chose to spill, which no public field reports.
SPILL_CHARGES = (2, 4, 8)
I_TRIP_COUNT = 4


def spill_charged_ops(cell: dict, charge: int) -> tuple[float, float]:
    """Dynamic ops per output element, with spill traffic added in.

    Returns the optimistic and pessimistic ends of the slot band.
    """
    base = cell["dynamic"]["ops_per_output_element"]
    oe = cell["dynamic"]["output_elements_per_thread"]
    low, high = slot_band(cell["spill_frame_bytes"])
    return (base + low * charge / oe, base + high * charge / oe)


M8_BODIES = ("baseline", "D_S", "D_late_meta", "G4_D", "B_rows2")


def m8_decision(rows: dict, bodies=M8_BODIES) -> dict:
    """Price one NA=8 pass against two NA=4 passes, spill included.

    This is the `{8:8}` question. The shipped table routes M=8 to `(8,4)`,
    which runs the NA=4 body twice over the whole weight matrix. `(8,8)` runs
    the NA=8 body once. Both compute the same 8 output columns, so the honest
    comparison is dynamic work per output element with each body's own spill
    charged against it. The two-pass reference is always the shipped NA=4
    body, because that is what `{8:8}` would replace.
    """
    two_pass = rows["baseline"]["cells"]["sumtable"]["na4"][RANKED]
    report = {
        "two_pass_body": "qwen_e120_qmv_wide<4>, run twice at M=8",
        "two_pass_spill_frame_bytes": two_pass["spill_frame_bytes"],
        "two_pass_ops_per_output_element":
            two_pass["dynamic"]["ops_per_output_element"],
        "one_pass_bodies": {},
    }
    for body in bodies:
        one_pass = rows[body]["cells"]["sumtable"]["na8"][RANKED]
        entry = {
            "spill_frame_bytes": one_pass["spill_frame_bytes"],
            "slot_band": list(slot_band(one_pass["spill_frame_bytes"])),
            "charges": {},
        }
        for charge in SPILL_CHARGES:
            a_low, _ = spill_charged_ops(two_pass, charge)
            b_low, b_high = spill_charged_ops(one_pass, charge)
            entry["charges"][charge] = {
                "one_pass_best": round(b_low, 2),
                "one_pass_worst": round(b_high, 2),
                "delta_percent_best": round(100.0 * (b_low / a_low - 1.0), 2),
                "delta_percent_worst": round(100.0 * (b_high / a_low - 1.0),
                                             2),
            }
        report["one_pass_bodies"][body] = entry
    return report


def patch(base: str, out: pathlib.Path) -> int:
    """The recommended body as a unified diff against the shipped template."""
    swift = swift_at(base)
    shipped = header_at(swift)
    candidate = header_at(swift, late_meta=True, sink_sums=True)
    text = "".join(difflib.unified_diff(
        shipped.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile="qwen35E120QMVHeader (shipped)",
        tofile="qwen35E120QMVHeader (D_S: late_meta + sink_sums)"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print("shipped header %d bytes, candidate %d bytes, delta %+d"
          % (len(shipped), len(candidate), len(candidate) - len(shipped)))
    print("wrote %s" % out)
    return 0


def case9(base: str, out: pathlib.Path) -> int:
    """What does `case 9` cost in the shared Route B switch?

    E132-F3 asks whether `case 9` is dead code. No observed histogram in this
    campaign contains an M=9 round, but the router at `Qwen35.swift:1699`
    admits `3 ... 9`, and the switch ends in `default: break`, which writes no
    output at all. So the arm is only removable together with the router
    bound. This measures what removing it would buy.
    """
    started = time.time()
    swift = swift_at(base)
    header = header_at(swift)
    variants = {
        "shipped": SHIPPED_TABLE,
        "no_case9": tuple(p for p in SHIPPED_TABLE if p[0] != 9),
        "one_pass_678": table_pairs({6: 6, 7: 7, 8: 8}),
        "one_pass_678_no9": tuple(p for p in table_pairs({6: 6, 7: 7, 8: 8})
                                  if p[0] != 9),
    }
    parts, names = [], []
    for label, pairs in variants.items():
        for table in (True, False):
            name = "sw_%s_%s" % (label, "t" if table else "n")
            parts.append(entry_point(name, pipeline_body(swift, pairs, table),
                                     table))
            names.append(name)
    with tempfile.TemporaryDirectory() as tmp:
        rows = census(library(header, parts), names, pathlib.Path(tmp), "c9")

    deltas = {}
    for before, after in (("shipped", "no_case9"),
                          ("one_pass_678", "one_pass_678_no9")):
        for table in (True, False):
            arm = "sumtable" if table else "notable"
            suffix = "t" if table else "n"
            x = rows["sw_%s_%s" % (before, suffix)][RANKED]
            y = rows["sw_%s_%s" % (after, suffix)][RANKED]
            deltas["%s/%s" % (before, arm)] = {
                "registers": y["registers"] - x["registers"],
                "spill_frame_bytes": y["spill_bytes"] - x["spill_bytes"],
                "text_bytes": y["text_bytes"] - x["text_bytes"],
            }

    result = receipt_header(base)
    result.update({
        "tool": "research/e132_wide_matvec.py case9",
        "question": "is Route B `case 9` dead code that can be deleted?",
        "router_bound": "Qwen35.swift:1699 `static let widths = 3 ... 9`",
        "shipped_pair_for_m9": "(9, 3): three passes of qwen_e120_qmv_wide<3>",
        "correctness_note": (
            "the switch ends in `default: break`, which writes no output, so "
            "deleting `case 9` without narrowing the router to `3 ... 8` "
            "turns an M=9 round into a silent wrong answer"),
        "rows": rows,
        "deltas_ranked": deltas,
        "runtime_seconds": round(time.time() - started, 2),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("E132-F3   what does `case 9` cost?   arch=%s" % RANKED)
    print("%-18s %-5s %6s %7s %8s %5s"
          % ("table", "arm", "regs", "frameB", "textB", "sg"))
    for label in variants:
        for table in (True, False):
            row = rows["sw_%s_%s" % (label, "t" if table else "n")][RANKED]
            print("%-18s %-5s %6d %7d %8d %5d"
                  % (label, "sum" if table else "no", row["registers"],
                     row["spill_bytes"], row["text_bytes"],
                     row["simdgroups_derived"]))
    print("\ndeleting `case 9`, %s" % RANKED)
    for key, delta in deltas.items():
        print("  %-28s regs %+d   frame %+d   text %+d bytes"
              % (key, delta["registers"], delta["spill_frame_bytes"],
                 delta["text_bytes"]))
    print("\nwrote %s" % out)
    return 0


def charged_table_cost(rows: dict, variant: str, override: dict,
                       charge: int, pessimistic: bool) -> float:
    """Histogram-weighted dynamic ops per output element, spill included."""
    cells = rows[variant]["cells"]["sumtable"]
    total = sum(HISTOGRAM.values())
    weighted = 0.0
    for m, ipg in table_pairs(override):
        if m not in HISTOGRAM:
            continue
        worst = 0.0
        for na in bodies_behind(m, ipg):
            low, high = spill_charged_ops(cells["na%d" % na][RANKED], charge)
            worst = max(worst, high if pessimistic else low)
        weighted += HISTOGRAM[m] * worst
    return weighted / total


def f3analysis(source: pathlib.Path, out: pathlib.Path) -> int:
    """Re-price the F2 table under the corrected frame-to-slot law."""
    receipt = json.loads(source.read_text())
    rows = receipt["variants"]
    decision = m8_decision(rows)

    campaign = {}
    for charge in SPILL_CHARGES:
        base_cost = charged_table_cost(rows, "baseline", {}, charge, True)
        entry = {"shipped_table_ops_per_output_element": round(base_cost, 2)}
        for label, body in (("one_pass_67", "baseline"),
                            ("one_pass_678", "baseline"),
                            ("one_pass_678_with_D_S", "D_S")):
            override = ({6: 6, 7: 7} if label == "one_pass_67"
                        else {6: 6, 7: 7, 8: 8})
            cost = charged_table_cost(rows, body, override, charge, True)
            entry[label] = {
                "ops_per_output_element": round(cost, 2),
                "delta_percent": round(100.0 * (cost / base_cost - 1.0), 2),
            }
        campaign[charge] = entry

    corrected: dict = {}
    for variant in rows:
        for arm in ("sumtable", "notable"):
            for na in (7, 8, 9):
                key = "na%d" % na
                cell = rows[variant]["cells"][arm].get(key, {}).get(RANKED)
                if not cell:
                    continue
                low, high = slot_band(cell["spill_frame_bytes"])
                corrected.setdefault(variant, {}).setdefault(arm, {})[key] = {
                    "spill_frame_bytes": cell["spill_frame_bytes"],
                    "spill_slots_band": [low, high],
                    "spill_free": cell["spill_frame_bytes"] == 0,
                    "registers": cell["registers"],
                    "dynamic_ops_per_output_element":
                        cell["dynamic"]["ops_per_output_element"],
                }

    result = {
        "schema_version": 1,
        "gpu_used": False,
        "timing_valid": False,
        "harness": "compile_only",
        "occupancy_label": "derived",
        "tool": "research/e132_wide_matvec.py f3",
        "source_receipt": str(source),
        "frame_law": ("16-byte aligned, allocated on demand; slots in "
                      "[frame/4 - 3, frame/4]; frame 0 means no spill"),
        "frame_law_evidence": ("research/e132-artifacts/spill-onset.json, "
                               "55-rung ladder stepping by one live value"),
        "m8_decision": decision,
        "campaign_table_cost_charged": campaign,
        "corrected_spill": corrected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("E132-F3   the {8:8} decision, spill charged against it")
    print("  arm=sumtable (the arm M=8 takes)   arch=%s" % RANKED)
    print("  two passes of wide<4> = %.2f ops/oe, frame %d bytes"
          % (decision["two_pass_ops_per_output_element"],
             decision["two_pass_spill_frame_bytes"]))
    print("  one pass of wide<8>, per candidate body, charge = memory "
          "instructions per slot per k block")
    head = ("  %-12s %7s %8s %19s %19s %19s"
            % ("body", "frameB", "slots", "charge 2", "charge 4", "charge 8"))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for body, entry in decision["one_pass_bodies"].items():
        cols = []
        for charge in SPILL_CHARGES:
            row = entry["charges"][charge]
            cols.append("%7.2f%% .. %6.2f%%" % (row["delta_percent_best"],
                                                row["delta_percent_worst"]))
        print("  %-12s %7d %8s %19s %19s %19s"
              % (body, entry["spill_frame_bytes"],
                 "%d-%d" % tuple(entry["slot_band"]), *cols))

    print("\n  whole-histogram QMV work, spill charged pessimistically")
    head = ("  %-8s %14s %14s %14s %14s"
            % ("charge", "shipped", "{6:6,7:7}", "{6,7,8}", "{6,7,8}+D_S"))
    print(head)
    for charge, entry in campaign.items():
        print("  %-8d %14.2f %8.2f %+5.1f%% %8.2f %+5.1f%% %8.2f %+5.1f%%"
              % (charge, entry["shipped_table_ops_per_output_element"],
                 entry["one_pass_67"]["ops_per_output_element"],
                 entry["one_pass_67"]["delta_percent"],
                 entry["one_pass_678"]["ops_per_output_element"],
                 entry["one_pass_678"]["delta_percent"],
                 entry["one_pass_678_with_D_S"]["ops_per_output_element"],
                 entry["one_pass_678_with_D_S"]["delta_percent"]))

    print("\n  corrected spill, scored arm, %s" % RANKED)
    print("  %-13s %4s %7s %12s %8s" % ("variant", "NA", "frameB", "slots",
                                        "free"))
    for variant in ("baseline", "D_S", "G4_D", "B_rows2"):
        for na in (7, 8, 9):
            cell = corrected.get(variant, {}).get("sumtable", {}).get(
                "na%d" % na)
            if not cell:
                continue
            print("  %-13s %4d %7d %12s %8s"
                  % (variant, na, cell["spill_frame_bytes"],
                     "%d-%d" % tuple(cell["spill_slots_band"]),
                     "yes" if cell["spill_free"] else "no"))
    print("\nwrote %s" % out)
    return 0


def print_f2table(receipt: dict) -> None:
    for arm in ("sumtable", "notable"):
        print("\nE132-F2 instruction table   arm=%s   arch=%s   NA=8"
              % (arm, RANKED))
        head = ("%-11s %7s %7s %8s %7s %6s %8s %8s %6s"
                % ("variant", "textB", "shellB", "innerB", "instr",
                   "spillI", "instr/oe", "deleted", "sg"))
        print(head)
        print("-" * len(head))
        order = sorted(
            receipt["variants"],
            key=lambda n: -receipt["variants"][n]["cells"][arm]["na8"][RANKED][
                "deleted_instructions_per_output_element"])
        for name in order:
            c = receipt["variants"][name]["cells"][arm]["na8"][RANKED]
            print("%-11s %7d %7d %8d %7.1f %6d %8.2f %8.2f %6d"
                  % (name, c["isa_text_bytes"], c["shell_text_bytes"],
                     c["inner_loop_text_bytes"],
                     c["inner_loop_instructions_upper"],
                     c["spill_instructions_point"],
                     c["inner_loop_instructions_per_output_element"],
                     c["deleted_instructions_per_output_element"],
                     c["simdgroups_derived"]))
    print("\nhistogram-weighted deleted instructions per output element "
          "(sumtable, %s)" % RANKED)
    for name in sorted(receipt["variants"],
                       key=lambda n: -receipt["variants"][n][
                           "weighted_deleted_ipoe"]["sumtable"][RANKED]):
        print("  %-11s %+8.3f"
              % (name, receipt["variants"][name]["weighted_deleted_ipoe"]
                 ["sumtable"][RANKED]))
    print("\nRoute B dispatch tables (sumtable, %s). `static` is only "
          "comparable\nwithin one width; `dynamic` is comparable across "
          "widths." % RANKED)
    head = "%-9s %-13s %9s %9s %9s" % (
        "variant", "table", "sg(derived)", "static/oe", "dynamic/oe")
    print(head)
    print("-" * len(head))
    for variant, tables in receipt["dispatch_tables"].items():
        for label, row in tables.items():
            print("%-9s %-13s %9.2f %9.2f %9.2f"
                  % (variant, label, row["weighted_simdgroups_derived"],
                     row["weighted_static_instructions_per_output_element"],
                     row["weighted_dynamic_ops_per_output_element"]))


def templating_proof(base: str, out: pathlib.Path) -> int:
    """Does an `MLXFast.metalKernel` template argument give an independent
    pipeline with an independent register allocation?

    `metal_kernel.cpp` builds `kernel_name` from the kernel name AND a hash of
    the template arguments, then `custom_kernel.cpp` calls
    `d.get_library(name_, ...)` and `d.get_kernel(name_, lib)` with that name.
    Each distinct template argument list therefore compiles its own MTLLibrary
    from its own source and owns its own MTLComputePipelineState. This proves
    the consequence from compiled output.
    """
    started = time.time()
    swift = swift_at(base)
    header = header_at(swift)

    per_instantiation, shared = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for m, ipg in ((5, 5), (6, 6), (8, 8)):
            name = "custom_kernel_qmv_M_%d" % m
            source = library(header, [entry_point(
                name, pipeline_body(swift, ((m, ipg),), True), True,
                [("int", "M", str(m))])])
            per_instantiation[m] = census(
                source, [name], workdir, "tmpl_m%d" % m)[name]
        name = "custom_kernel_qmv_shared"
        source = library(header, [entry_point(
            name, pipeline_body(swift, table_pairs({6: 6, 7: 7, 8: 8}), True),
            True)])
        shared = census(source, [name], workdir, "tmpl_shared")[name]

    receipt = receipt_header(base)
    receipt.update({
        "tool": "research/e132_wide_matvec.py templating-proof",
        "question": "does MLXFast.metalKernel templating produce independent "
                    "pipelines with independent register allocation?",
        "source_evidence": [
            "mlx/backend/common/metal_kernel.cpp:289-298 appends a hash of the "
            "template arguments to kernel_name",
            "mlx/backend/common/metal_kernel.cpp:331-338 emits one "
            "[[host_name(kernel_name)]] explicit instantiation per argument "
            "list",
            "mlx/backend/metal/custom_kernel.cpp:59-72 calls "
            "d.get_library(name_) and d.get_kernel(name_, lib) with that same "
            "name, so each argument list compiles its own MTLLibrary and owns "
            "its own MTLComputePipelineState",
        ],
        "instantiations": {str(m): v for m, v in per_instantiation.items()},
        "shared_switch_one_pass_678": shared,
    })
    keys = sorted(per_instantiation)
    receipt["independent_machine_code"] = {
        arch: len({per_instantiation[m][arch]["text_sha8"] for m in keys})
        == len(keys) for arch in ARCHES}
    receipt["independent_register_allocation"] = {
        arch: len({per_instantiation[m][arch]["registers"] for m in keys})
        > 1 for arch in ARCHES}
    # g16s saturates its 96-register ceiling at every width from M=5 up, so all
    # three instantiations report 96 there whatever the allocator does. A
    # differing register count is therefore evidence of independence when it
    # appears, but an equal one at the ceiling is not evidence against it.
    # Distinct machine code on every architecture is the load-bearing test.
    receipt["register_ceiling"] = SATURATING_CEILING
    receipt["register_allocation_saturated"] = {
        arch: all(per_instantiation[m][arch]["registers"]
                  == SATURATING_CEILING[arch] for m in keys)
        for arch in ARCHES}
    receipt["verdict"] = (
        "independent"
        if all(receipt["independent_machine_code"].values())
        and any(receipt["independent_register_allocation"].values())
        else "not proven")
    receipt["runtime_seconds"] = round(time.time() - started, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")

    print("templating proof: %s" % receipt["verdict"])
    for m in keys:
        row = per_instantiation[m]
        for arch in ARCHES:
            print("  M=%d %-15s registers=%-4d spill=%-4d text=%-6d sha=%s"
                  % (m, arch, row[arch]["registers"], row[arch]["spill_bytes"],
                     row[arch]["text_bytes"], row[arch]["text_sha8"]))
    for arch in ARCHES:
        print("  one shared switch {6:6,7:7,8:8} %-15s registers=%-4d spill=%d"
              % (arch, shared[arch]["registers"], shared[arch]["spill_bytes"]))
    print("wrote %s in %.1f s" % (out, receipt["runtime_seconds"]))
    return 0


# --------------------------------------------------------------------------
# question 2: the g17s register budget
# --------------------------------------------------------------------------

def budget(base: str, out: pathlib.Path) -> int:
    """Settle F33/F34: what IS the per-thread register budget on each arch?

    F34 records 124 on `applegpu_g17s`, but `qwen_e120_qmv_wide<7>` translates
    to 125 registers with zero spill, so at most one of the two can be right.
    A fine sweep of kernels whose live float count is known by construction
    resolves it: the budget is the highest register count the backend reaches
    before it opens a spill frame.
    """
    started = time.time()
    rows: dict = {}
    widths = list(range(40, 141, 2))
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        source = ("#include <metal_stdlib>\nusing namespace metal;\n"
                  + "".join(agx.scalar_kernel(n) for n in widths))
        lib = agx.build_metallib(source, workdir)
        for arch in ARCHES:
            found = agx.translate(lib, arch, workdir)
            clean = [(n, found["k_s%d" % n]["registers"]) for n in widths
                     if not found["k_s%d" % n]["spill_bytes"]]
            first_frame = next((n for n in widths
                                if found["k_s%d" % n]["spill_bytes"]), None)
            at, peak = max(clean, key=lambda pair: pair[1])
            ceiling = max(found["k_s%d" % n]["registers"] for n in widths)
            rows[arch] = {
                "sweep": {str(n): {"registers": found["k_s%d" % n]["registers"],
                                   "spill_bytes":
                                       found["k_s%d" % n]["spill_bytes"]}
                          for n in widths},
                "max_registers_without_a_frame": peak,
                "at_live_floats": at,
                "first_live_float_count_with_a_frame": first_frame,
                # The ceiling is the number the allocator never exceeds however
                # much pressure it is given, spilling or not. That is the
                # quantity F34 names a budget.
                "max_registers_any_kernel": ceiling,
                "saturates_at_expected_ceiling":
                    ceiling == SATURATING_CEILING[arch],
            }
    receipt = receipt_header(base)
    receipt.update({
        "tool": "research/e132_wide_matvec.py budget",
        "question": "F33/F34 record 96 on g16s and 124 on g17s; "
                    "qwen_e120_qmv_wide<7> reaches 125 with zero spill on "
                    "g17s. Which figure survives?",
        "arches": rows,
    })
    receipt["runtime_seconds"] = round(time.time() - started, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    for arch, row in rows.items():
        print("%-15s ceiling = %d registers; highest with no spill frame = %d "
              "(at %d live floats); first frame at %s live floats"
              % (arch, row["max_registers_any_kernel"],
                 row["max_registers_without_a_frame"],
                 row["at_live_floats"],
                 row["first_live_float_count_with_a_frame"]))
    print("wrote %s in %.1f s" % (out, receipt["runtime_seconds"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command, default in (
            ("rung0", "research/e132-artifacts/rung0-table-census.json"),
            ("rung1", "research/e132-artifacts/rung1-spill-search.json"),
            ("f2table", "research/e132-artifacts/f2-instruction-table.json"),
            ("templating-proof",
             "research/e132-artifacts/templating-proof.json"),
            ("case9", "research/e132-artifacts/case9-cost.json"),
            ("patch", "research/e132-artifacts/d_s-body.patch"),
            ("budget", "research/e132-artifacts/register-budget.json")):
        parser = sub.add_parser(command)
        parser.add_argument("--base", default="HEAD")
        parser.add_argument("--json", default=default)
    three = sub.add_parser("f3")
    three.add_argument(
        "--source",
        default="research/e132-artifacts/f2-instruction-table.json")
    three.add_argument(
        "--json", default="research/e132-artifacts/f3-spill-price.json")
    args = ap.parse_args()
    out = pathlib.Path(args.json)
    if args.command == "f3":
        return f3analysis(pathlib.Path(args.source), out)
    return {"rung0": rung0, "rung1": rung1, "f2table": f2table,
            "templating-proof": templating_proof, "case9": case9,
            "patch": patch,
            "budget": budget}[args.command](args.base, out)


if __name__ == "__main__":
    raise SystemExit(main())
