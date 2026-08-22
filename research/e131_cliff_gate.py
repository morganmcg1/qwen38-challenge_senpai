#!/usr/bin/env python3
"""Pre-submit occupancy-cliff gate: does the candidate lose a resident simdgroup?

`senpai/entry-point-cliff-census.sh` is the entry point. This module does the
work.

Every scored Metal entry point is compiled for `applegpu_g16s` and
`applegpu_g17s` from two sources - a base git ref and the candidate, which is
the working tree by default - and the register counts are compared. The ranked
runner is `applegpu_g17s`, so a candidate that raises the register count past a
floor-division boundary there loses a resident simdgroup at EVERY dispatch of
that entry point. E121 did exactly that: 101 registers to 102, 39 resident
simdgroups to 38, and the submission that carried it scored below the base.

RULE 89. `simdgroups = floor(BUDGET / registers)` is computed from the register
count, so it is a MODEL OUTPUT, not a measurement. Every simdgroup number this
gate prints or records is labelled `derived`. Only a timing response or a
hardware occupancy counter can promote it. The registers, spill bytes and ISA
text sizes ARE measurements: they are read out of the translated binary.

The gate runs the real AGX backend through `xcrun metal-tt`. It never touches
the GPU, never loads the model and never runs the benchmark.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
import e120_g17s_census as e120  # noqa: E402
import e131_kernel_sources as ks  # noqa: E402
from e123_arms import SIMDGROUP_BUDGET, simdgroups  # noqa: E402
from jit_string_compile import assemble, host_name  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)
RANKED = agx.RANKED_ARCH

# The scored entry points, with the source form each one is built from. The
# three forms have three different edit and rebuild paths, so the form is part
# of the receipt.
#
#   jit_twin            MLX concatenates `mlx-generated/*.cpp` preamble strings
#                       and appends one instantiation. Edit the readable
#                       `.metal`/`.h` AND the generated twin.
#   swift_metal_kernel  `MLXFast.metalKernel` hands MLX a body; MLX generates
#                       the signature at dispatch time. Edit the Swift string.
JIT_CELLS = (
    ("affine_qmv_fast<bfloat16_t, 64, 4, false>",
     "wide QMV: every affine-4 g64 target projection MLX still launches"),
    ("affine_qmv_fast<bfloat16_t, 64, 2, false>",
     "affine-2 coarse draft readout, out_vec_size 98336"),
    ("affine_qmv_fast<bfloat16_t, 64, 4, true>",
     "batched affine-4 g64 QMV"),
)


def git_sha(rev: str) -> str:
    return subprocess.run(["git", "rev-parse", rev], cwd=str(ROOT),
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def toolchain() -> str:
    done = subprocess.run(["xcrun", "metal", "--version"], capture_output=True,
                          text=True, check=True)
    lines = [l for l in (done.stdout + done.stderr).splitlines() if l.strip()]
    return lines[0] if lines else "unknown"


def side_sources(rev: str | None, swift_patch=None) -> dict[str, dict]:
    """Every scored entry point at one revision, as compilable Metal sources.

    `rev is None` means the working tree. Each value carries the library the
    kernel lives in, so kernels that share a library are compiled once.
    `swift_patch` rewrites the extracted `Qwen35.swift` text before the Route B
    library is rebuilt, so an unlanded candidate can be censused without an edit
    to a tracked source another student owns.
    """
    cells: dict[str, dict] = {}
    jit = assemble(tuple(cell for cell, _ in JIT_CELLS), rev)
    for cell, role in JIT_CELLS:
        cells[host_name(cell)] = {
            "library": "jit", "source": jit, "source_form": "jit_twin",
            "role": role, "cell": cell}

    swift = ks.swift_text(ks.QWEN35, rev)
    if swift_patch is not None:
        swift = swift_patch(swift)
    try:
        route_b = ks.route_b_library(swift)
    except ks.SourceUnavailable as error:
        route_b = None
        reason = str(error)
    if route_b is not None:
        for name, role in ks.ROUTE_B_ROLES.items():
            cells[name] = {
                "library": "route_b", "source": route_b,
                "source_form": "swift_metal_kernel", "role": role,
                "cell": name}
    else:
        cells["__route_b_unavailable__"] = {"error": reason}
    return cells


def census(cells: dict[str, dict], workdir: pathlib.Path,
           tag: str) -> dict[str, dict]:
    """Register, spill and text census of every cell, on both architectures."""
    rows: dict[str, dict] = {}
    libraries: dict[str, list[str]] = {}
    for name, spec in cells.items():
        if "error" in spec:
            rows[name] = spec
            continue
        libraries.setdefault(spec["library"], []).append(name)
    for library, names in libraries.items():
        source = cells[names[0]]["source"]
        wanted = set(names)
        lib = agx.build_metallib(source, workdir / ("%s_%s" % (tag, library)))
        for arch in ARCHES:
            found = agx.translate(lib, arch, workdir / ("%s_%s" % (tag, library)),
                                  select=lambda n: n in wanted)
            for name in names:
                record = found.get(name)
                if record is None:
                    rows.setdefault(name, {})["error"] = (
                        "%s not emitted by the %s library" % (name, library))
                    continue
                registers = record["registers"]
                rows.setdefault(name, {})[arch] = {
                    "registers": registers,
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                    "text_sha8": record["text_sha8"],
                    "simdgroups_derived": simdgroups(registers, arch),
                }
        for name in names:
            rows[name]["source_form"] = cells[name]["source_form"]
            rows[name]["role"] = cells[name]["role"]
    return rows


def compare(base: dict, candidate: dict) -> tuple[list[dict], list[str], list[str]]:
    """One row per scored entry point, plus the failures and the warnings."""
    rows, failures, warnings = [], [], []
    for name in sorted(set(base) | set(candidate)):
        if name.startswith("__"):
            for side, table in (("base", base), ("candidate", candidate)):
                if name in table:
                    warnings.append("%s: %s" % (side, table[name]["error"]))
            continue
        if name not in base:
            warnings.append("%s: new entry point, absent from the base" % name)
            continue
        if name not in candidate:
            warnings.append("%s: retired, absent from the candidate" % name)
            continue
        row: dict = {"kernel": name,
                     "source_form": candidate[name].get("source_form"),
                     "role": candidate[name].get("role")}
        for arch in ARCHES:
            b, c = base[name].get(arch), candidate[name].get(arch)
            if b is None or c is None:
                warnings.append("%s: no %s record" % (name, arch))
                continue
            change = 100.0 * c["simdgroups_derived"] / b["simdgroups_derived"] - 100.0
            row[arch] = {
                "base_registers": b["registers"],
                "candidate_registers": c["registers"],
                "registers_delta": c["registers"] - b["registers"],
                "base_simdgroups_derived": b["simdgroups_derived"],
                "candidate_simdgroups_derived": c["simdgroups_derived"],
                "simdgroups_delta_derived":
                    c["simdgroups_derived"] - b["simdgroups_derived"],
                "residency_change_pct_derived": round(change, 3),
                "base_spill_bytes": b["spill_bytes"],
                "candidate_spill_bytes": c["spill_bytes"],
                "base_text_bytes": b["text_bytes"],
                "candidate_text_bytes": c["text_bytes"],
                "unchanged": b["text_sha8"] == c["text_sha8"],
            }
            if c["spill_bytes"]:
                warnings.append(
                    "%s %s: candidate spills %d bytes"
                    % (name, arch, c["spill_bytes"]))
        ranked = row.get(RANKED)
        if ranked and ranked["simdgroups_delta_derived"] < 0:
            failures.append(
                "%s %s: %d -> %d registers loses %d resident simdgroup(s) "
                "(%d -> %d, %.2f %% residency, derived)"
                % (name, RANKED, ranked["base_registers"],
                   ranked["candidate_registers"],
                   -ranked["simdgroups_delta_derived"],
                   ranked["base_simdgroups_derived"],
                   ranked["candidate_simdgroups_derived"],
                   ranked["residency_change_pct_derived"]))
        elif ranked and ranked["simdgroups_delta_derived"] > 0:
            warnings.append(
                "%s %s: %d -> %d registers GAINS %d resident simdgroup(s) "
                "(%d -> %d, +%.2f %% residency, derived)"
                % (name, RANKED, ranked["base_registers"],
                   ranked["candidate_registers"],
                   ranked["simdgroups_delta_derived"],
                   ranked["base_simdgroups_derived"],
                   ranked["candidate_simdgroups_derived"],
                   ranked["residency_change_pct_derived"]))
        rows.append(row)
    return rows, failures, warnings


def print_table(rows: list[dict]) -> None:
    header = ("%-46s %-20s %-24s %-24s" %
              ("scored entry point", "source form",
               "applegpu_g16s regs/sg", "applegpu_g17s regs/sg"))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for arch in ARCHES:
            cell = row.get(arch)
            if cell is None:
                cells.append("%-24s" % "unavailable")
                continue
            cells.append("%-24s" % (
                "%d/%d -> %d/%d %s" % (
                    cell["base_registers"], cell["base_simdgroups_derived"],
                    cell["candidate_registers"],
                    cell["candidate_simdgroups_derived"],
                    "" if cell["simdgroups_delta_derived"] == 0
                    else ("%+.2f%%" % cell["residency_change_pct_derived"]))))
        print("%-46s %-20s %s %s" % (row["kernel"][:46],
                                     row["source_form"] or "?", *cells))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="fail when a candidate loses an applegpu_g17s resident "
                    "simdgroup at any scored Metal entry point")
    ap.add_argument("--base", required=True,
                    help="git ref the candidate must not regress against")
    ap.add_argument("--candidate", default=None,
                    help="git ref to censo instead of the working tree")
    ap.add_argument("--json", default="",
                    help="write the JSON receipt here")
    args = ap.parse_args()

    started = time.time()
    base_sha = git_sha(args.base)
    candidate_sha = git_sha(args.candidate) if args.candidate else None

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        base_rows = census(side_sources(args.base), workdir, "base")
        candidate_rows = census(side_sources(args.candidate), workdir, "cand")
    rows, failures, warnings = compare(base_rows, candidate_rows)

    receipt = {
        "tool": "senpai/entry-point-cliff-census.sh",
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89: simdgroups = floor(BUDGET / registers) is "
                          "a model output computed from the register count, "
                          "not a measurement",
        "ranked_arch": RANKED,
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "toolchain": toolchain(),
        "base_ref": args.base,
        "base_sha": base_sha,
        "candidate_ref": args.candidate or "worktree",
        "candidate_sha": candidate_sha,
        "cells": rows,
        "failures": failures,
        "warnings": warnings,
        "verdict": "fail" if failures else "pass",
        "runtime_seconds": round(time.time() - started, 2),
    }
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")

    print("entry-point cliff census   base %s (%s)   candidate %s"
          % (args.base, base_sha[:8], candidate_sha[:8] if candidate_sha
             else "working tree"))
    print("simdgroup counts are DERIVED from registers (Rule 89): "
          "floor(3072/regs) on applegpu_g16s, floor(3968/regs) on applegpu_g17s")
    print()
    print_table(rows)
    print()
    for line in warnings:
        print("WARNING %s" % line)
    for line in failures:
        print("FAIL    %s" % line)
    print("\nverdict: %s   %.2f s   %s"
          % (receipt["verdict"].upper(), receipt["runtime_seconds"],
             args.json or "no receipt requested"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
