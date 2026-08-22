#!/usr/bin/env python3
"""E138 item 2, stage A: the register census of the whole (M, IPG, RPS) plan
surface, compiled offline for both architectures.

Every cell of the plan surface is a template instantiation of the SAME live
Swift kernel body, so the compiler's register and spill answer for a
hypothetical plan is available without a GPU, without a model and without a
timed run. That answer is what decides which cells are worth GPU time and it
is the census CAMPAIGN RULE 82 requires next to any proposed plan.

RULE 89: `resident_simdgroups` is DERIVED from the register count through
`floor(BUDGET / registers)`. Registers, spill bytes and ISA text size are
measurements; the simdgroup count is a model output.

The body is rebuilt from `Qwen35.swift` through the same reader the standing
cliff gate uses, and the reader is checked against the live Swift generator
before any hypothetical cell is emitted, so a census of last month's kernel
cannot be reported as this one.

    usage: research/e138_plan_census.py [--json PATH] [--widths 5,6,7,8,9]
                                        [--rps 1,2,4,8]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e120_g17s_census as e120  # noqa: E402
import e131_kernel_sources as sources  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
XSUMS_INPUT = ("xsums", "float")


def legal(m: int, ipg: int) -> bool:
    """`Qwen35.swift:1545`: a one-input tail group is not built."""
    return 1 <= ipg <= m and m % ipg != 1


def cell_name(m: int, ipg: int, rps: int) -> str:
    return "e138_qmv_m%d_ipg%d_rps%d" % (m, ipg, rps)


def cell_source(text: str, m: int, ipg: int, rps: int) -> str:
    body = sources.e120_qmv_body(text, table=True, plan=[(m, ipg, rps)])
    return e120.generate(
        cell_name(m, ipg, rps), e120.QMV_INPUTS + [XSUMS_INPUT],
        e120.QMV_OUTPUTS, body,
        template=[("bool", "USE_TABLE", "true")])


def census_cells(cells: list[tuple[int, int, int]], workdir: pathlib.Path,
                 text: str, header: str) -> dict:
    """One metallib per cell, so an unbuildable cell is a fact, not a crash."""
    rows: dict[str, dict] = {}
    for m, ipg, rps in cells:
        name = cell_name(m, ipg, rps)
        source = (e120.PRELUDE + header + "\n"
                  + cell_source(text, m, ipg, rps))
        tag = name
        try:
            record = e120.census(source, tag, workdir)
        except subprocess.CalledProcessError as failure:
            rows[name] = {
                "m": m, "ipg": ipg, "rps": rps, "built": False,
                "passes": (m + ipg - 1) // ipg,
                "error": (failure.stderr or "")[-400:],
            }
            continue
        row = {"m": m, "ipg": ipg, "rps": rps, "built": True,
               "passes": (m + ipg - 1) // ipg,
               "launched_columns": (m + ipg - 1) // ipg}
        for arch, kernels in record.items():
            entry = kernels.get(name)
            if entry is None:
                continue
            row[arch] = entry
        rows[name] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    parser.add_argument("--widths", default="3,4,5,6,7,8,9")
    parser.add_argument("--rps", default="1,2,4,8")
    args = parser.parse_args()

    widths = [int(v) for v in args.widths.split(",")]
    rpss = [int(v) for v in args.rps.split(",")]

    text = sources.swift_text(sources.QWEN35, None)
    header = sources.named_literal(text, "qwen35E120QMVHeader")

    cells = [(m, ipg, rps) for m in widths for ipg in range(1, m + 1)
             for rps in rpss if legal(m, ipg)]

    with tempfile.TemporaryDirectory() as tmp:
        rows = census_cells(cells, pathlib.Path(tmp), text, header)

    receipt = {
        "experiment": "e138-plan-surface",
        "stage": "A-offline-register-census",
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True).stdout.strip(),
        "toolchain": e120.__dict__.get("TOOLCHAIN", "")
        or subprocess.run(["xcrun", "metal", "--version"],
                          capture_output=True, text=True).stderr.strip(),
        "simdgroup_budget": e120.SIMDGROUP_BUDGET,
        "occupancy_rule":
            "RULE 89: resident_simdgroups = floor(BUDGET / registers) is "
            "DERIVED from the register count, not measured",
        "shipped_plan": [list(entry) for entry in
                         sources.width_plan(text, sources.default_table(text))],
        "shipped_table": sources.default_table(text),
        "cells": rows,
    }

    built = [row for row in rows.values() if row["built"]]
    print("e138 plan census: %d cells requested, %d built, %d refused"
          % (len(rows), len(built), len(rows) - len(built)))
    print("%-22s %5s %5s %5s | %s" % ("cell", "pass", "g16s", "spil", "g17s"))
    for name in sorted(rows, key=lambda n: (rows[n]["m"], rows[n]["ipg"],
                                            rows[n]["rps"])):
        row = rows[name]
        if not row["built"]:
            print("%-22s   --  refused" % name)
            continue
        g16 = row.get("applegpu_g16s", {})
        g17 = row.get("applegpu_g17s", {})
        print("%-22s %5d %5s %5s | %s regs, %s spill, %s sg"
              % (name, row["passes"], g16.get("registers"),
                 g16.get("spill_bytes"), g17.get("registers"),
                 g17.get("spill_bytes"), g17.get("resident_simdgroups")))

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
