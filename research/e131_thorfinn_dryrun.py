#!/usr/bin/env python3
"""Dry-run the cliff gate against thorfinn's unlanded Route B `(5, 5) -> (5, 3)`.

E131-F3 section 3 says that change is the gate's first live candidate and that a
false failure on it is a defect. This reproduces the candidate without touching
`Qwen35.swift`, which thorfinn owns: the Swift text is extracted read-only at the
base, the one dispatch-table literal is rewritten in memory, and the Route B
library is rebuilt from the rewritten text.

The predicted move is the shipped `.sumTable` pipeline going 102 -> 94 registers
on `applegpu_g17s`, 38 -> 42 resident simdgroups. Under Rule 89 the simdgroup
figures are `derived`; the registers are measured through `xcrun metal-tt`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e131_cliff_gate as gate  # noqa: E402

BASE = "3e0ecfddfb91ea9d359e31912d08b9480dcb95a2"
CASES = re.compile(r"(let cases = \[)(.*?)(\]\n)", re.DOTALL)


def rewrite_case(swift: str, width: int, inner: int) -> str:
    """Set the inner width of one entry of Route B's dispatch table."""
    match = CASES.search(swift)
    if match is None:
        raise SystemExit("no Route B dispatch table in the extracted Swift text")
    pairs = re.findall(r"\((\d+),\s*(\d+)\)", match.group(2))
    if not any(int(m) == width for m, _ in pairs):
        raise SystemExit("width %d is absent from the dispatch table" % width)
    patched = ", ".join(
        "(%s, %s)" % (m, inner if int(m) == width else i) for m, i in pairs)
    return swift[:match.start(2)] + patched + swift[match.end(2):]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--json", default="research/e131-artifacts/rung3-dryrun-route-b-5-3.json")
    args = ap.parse_args()

    started = time.time()
    patch = lambda swift: rewrite_case(swift, args.width, args.inner)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        base_rows = gate.census(gate.side_sources(args.base), workdir, "base")
        cand_rows = gate.census(
            gate.side_sources(args.base, swift_patch=patch), workdir, "cand")
    rows, failures, warnings = gate.compare(base_rows, cand_rows)

    receipt = {
        "tool": "research/e131_thorfinn_dryrun.py",
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "base_ref": args.base,
        "base_sha": gate.git_sha(args.base),
        "candidate": "base + Route B dispatch table (%d, %d) -> (%d, %d), "
                     "applied in memory to the extracted Swift text"
                     % (args.width, args.width, args.width, args.inner),
        "toolchain": gate.toolchain(),
        "simdgroup_budget": gate.SIMDGROUP_BUDGET,
        "cells": rows,
        "failures": failures,
        "warnings": warnings,
        "verdict": "fail" if failures else "pass",
        "runtime_seconds": round(time.time() - started, 2),
    }
    pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.json).write_text(json.dumps(receipt, indent=2) + "\n")

    gate.print_table(rows)
    for warning in warnings:
        print("WARNING: %s" % warning)
    for failure in failures:
        print("FAIL: %s" % failure)
    print("verdict %s in %.2f s -> %s"
          % (receipt["verdict"], receipt["runtime_seconds"], args.json))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
