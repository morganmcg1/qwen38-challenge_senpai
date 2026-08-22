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
import re
import subprocess
import sys
import tempfile
import textwrap
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

# Realised routed-round width histogram over 308 rounds. M=8 carries 240 of
# them, 77.9 %, so a per-width templated build must be judged on this weight
# and not on any single entry point.
WIDTH_HISTOGRAM = {4: 16, 5: 20, 6: 20, 7: 12, 8: 240}

# A per-width templated Route B build names each pipeline after the width it
# serves. An entry point with no width marker serves every width the marked
# ones do not claim, which is exactly the shipped shared-switch build.
WIDTH_IN_NAME = re.compile(r"(?:^|_)m(\d+)(?:_|$)")

# Route B has two QMV arms. `Qwen35.swift:1691` and `:1729-1730` send every
# routed width at M >= 4 to the chunk-sum-table arm, so that arm is the scored
# one and an unmarked build must be weighed on it, not on whichever name sorts
# first.
SUMTABLE_IN_NAME = re.compile(r"_sums_")

# An AGX ALU instruction is exactly 4 bytes on both architectures, calibrated
# by `research/e132_instruction_probe.py calibrate` at residual 0. A machine
# code delta of X bytes is therefore at most X/4 instructions.
BYTES_PER_ALU_INSTRUCTION = 4

INSTRUMENT_NOTE = (
    "This gate is a REGISTER instrument and is accurate as one. It is NOT a "
    "time instrument. F131 measured the residency-to-time coefficient at "
    "-0.0014 and +0.0105 on two independent designs, both far inside the "
    "0.05 %/% kill gate, and F114 puts deleted instruction count at r = "
    "+0.949 against measured gain. An exit 1 that comes with real deleted "
    "work should usually be discharged by pricing under Rule 87, not obeyed.")


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
    for library, build, roles in (
            ("route_b", ks.route_b_library, ks.ROUTE_B_ROLES),
            ("cluster_qmv", ks.cluster_qmv_library, ks.CLUSTER_QMV_ROLES)):
        try:
            source = build(swift)
        except ks.SourceUnavailable as error:
            cells["__%s_unavailable__" % library] = {"error": str(error)}
            continue
        for name, role in roles.items():
            cells[name] = {
                "library": library, "source": source,
                "source_form": "swift_metal_kernel", "role": role,
                "cell": name}
        # Per-width templating replaces one entry point with several whose
        # names this file cannot predict, so the library is enumerated after
        # it compiles rather than read from a fixed list.
        cells["__discover_%s__" % library] = {
            "library": library, "source": source,
            "source_form": "swift_metal_kernel", "discover": True}
    return cells


def census(cells: dict[str, dict], workdir: pathlib.Path,
           tag: str) -> dict[str, dict]:
    """Register, spill and text census of every cell, on both architectures."""
    rows: dict[str, dict] = {}
    plans: dict[str, dict] = {}
    for name, spec in cells.items():
        if "error" in spec:
            rows[name] = spec
            continue
        plan = plans.setdefault(spec["library"], {
            "source": spec["source"], "declared": set(), "discover": False,
            "source_form": spec["source_form"]})
        if spec.get("discover"):
            plan["discover"] = True
        else:
            plan["declared"].add(name)

    for library, plan in plans.items():
        tagged = workdir / ("%s_%s" % (tag, library))
        lib = agx.build_metallib(plan["source"], tagged)
        wanted = set(plan["declared"])
        if plan["discover"]:
            wanted |= set(agx.kernel_names(lib))
        missing = set(plan["declared"])
        for arch in ARCHES:
            found = agx.translate(lib, arch, tagged,
                                  select=lambda n: n in wanted)
            missing -= set(found)
            for name, record in found.items():
                registers = record["registers"]
                rows.setdefault(name, {})[arch] = {
                    "registers": registers,
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                    "text_sha8": record["text_sha8"],
                    "simdgroups_derived": simdgroups(registers, arch),
                }
        for name in missing:
            rows.setdefault(name, {})["error"] = (
                "%s not emitted by the %s library" % (name, library))
        for name in wanted - missing:
            declared = cells.get(name, {})
            rows[name]["source_form"] = plan["source_form"]
            rows[name]["role"] = declared.get(
                "role", "discovered in the %s library" % library)
            rows[name]["library"] = library
            rows[name]["declared"] = name in plan["declared"]
    return rows


def route_b_pipelines(rows: dict) -> dict[str, str]:
    """Route B QMV entry point serving each routed width.

    The shipped build has one entry point behind `switch (qmv_m)`, so every
    width maps to it and the weighted figure equals that entry point's own.
    A per-width templated build has one entry point per width, and each claims
    its width by name.
    """
    qmv = [name for name, row in rows.items()
           if not name.startswith("__")
           and row.get("library") == "route_b" and "xsums" not in name]
    claimed: dict[int, str] = {}
    fallback = []
    for name in sorted(qmv):
        hit = WIDTH_IN_NAME.search(name)
        if hit and int(hit.group(1)) in WIDTH_HISTOGRAM:
            claimed[int(hit.group(1))] = name
        else:
            fallback.append(name)
    scored = [n for n in fallback if SUMTABLE_IN_NAME.search(n)] or fallback
    serving: dict[str, str] = {}
    for width in WIDTH_HISTOGRAM:
        if width in claimed:
            serving[str(width)] = claimed[width]
        elif scored:
            serving[str(width)] = scored[0]
    return serving


def weighted_residency(rows: dict, serving: dict[str, str],
                       arch: str) -> float | None:
    """Histogram-weighted derived resident simdgroups over the QMV surface."""
    total = 0
    weight = 0
    for width, name in serving.items():
        cell = rows.get(name, {}).get(arch)
        if cell is None:
            return None
        total += WIDTH_HISTOGRAM[int(width)] * cell["simdgroups_derived"]
        weight += WIDTH_HISTOGRAM[int(width)]
    return round(total / weight, 3) if weight else None


def route_b_surface(base: dict, candidate: dict) -> dict:
    """The width-weighted Route B comparison that decides the QMV verdict."""
    report: dict = {
        "base_serving": route_b_pipelines(base),
        "candidate_serving": route_b_pipelines(candidate),
        "note": "width-weighted derived residency over %s; the shipped "
                "shared-switch build maps every width to its single entry "
                "point, so this reproduces the single-entry-point figure"
                % WIDTH_HISTOGRAM,
    }
    for arch in ARCHES:
        before = weighted_residency(base, report["base_serving"], arch)
        after = weighted_residency(candidate, report["candidate_serving"],
                                   arch)
        if before is None or after is None:
            report[arch] = {"error": "no complete Route B QMV surface"}
            continue
        report[arch] = {
            "base_weighted_simdgroups_derived": before,
            "candidate_weighted_simdgroups_derived": after,
            "delta_derived": round(after - before, 3),
            "change_pct_derived": round(100.0 * after / before - 100.0, 3),
            "base_pipelines": len(set(report["base_serving"].values())),
            "candidate_pipelines": len(set(report["candidate_serving"]
                                           .values())),
        }
    return report


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
                     "role": candidate[name].get("role"),
                     "library": candidate[name].get("library")}
        for arch in ARCHES:
            b, c = base[name].get(arch), candidate[name].get(arch)
            if b is None or c is None:
                warnings.append("%s: no %s record" % (name, arch))
                continue
            change = 100.0 * c["simdgroups_derived"] / b["simdgroups_derived"] - 100.0
            text_delta = c["text_bytes"] - b["text_bytes"]
            spill_delta = c["spill_bytes"] - b["spill_bytes"]
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
                "spill_bytes_delta": spill_delta,
                "base_text_bytes": b["text_bytes"],
                "candidate_text_bytes": c["text_bytes"],
                "text_bytes_delta": text_delta,
                "instructions_delta_upper_bound":
                    round(text_delta / BYTES_PER_ALU_INSTRUCTION, 1),
                "unchanged": b["text_sha8"] == c["text_sha8"],
            }
            if c["spill_bytes"]:
                warnings.append(
                    "%s %s: candidate spills %d bytes"
                    % (name, arch, c["spill_bytes"]))
        ranked = row.get(RANKED)
        route_b_qmv = (row["library"] == "route_b" and "xsums" not in name)
        if ranked and ranked["simdgroups_delta_derived"] < 0:
            line = ("%s %s: %d -> %d registers loses %d resident simdgroup(s) "
                    "(%d -> %d, %.2f %% residency, derived); spill %+d bytes, "
                    "machine code %+d bytes (at most %+.1f instructions)"
                    % (name, RANKED, ranked["base_registers"],
                       ranked["candidate_registers"],
                       -ranked["simdgroups_delta_derived"],
                       ranked["base_simdgroups_derived"],
                       ranked["candidate_simdgroups_derived"],
                       ranked["residency_change_pct_derived"],
                       ranked["spill_bytes_delta"],
                       ranked["text_bytes_delta"],
                       ranked["instructions_delta_upper_bound"]))
            if route_b_qmv:
                warnings.append(
                    "sub-report, does not decide the verdict: %s" % line)
            else:
                failures.append(line)
            if ranked["spill_bytes_delta"] < 0 or ranked["text_bytes_delta"] < 0:
                warnings.append(
                    "%s %s: the residency loss comes with deleted work, so it "
                    "is a Rule 87 pricing question rather than a cliff"
                    % (name, RANKED))
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


def print_surface(surface: dict) -> None:
    """The width-weighted Route B report that decides the QMV verdict."""
    print("Route B QMV surface, width-weighted over %s" % WIDTH_HISTOGRAM)
    base_serving = surface["base_serving"]
    cand_serving = surface["candidate_serving"]
    print("  %-7s %-42s %s" % ("width", "base pipeline", "candidate pipeline"))
    for width in sorted(WIDTH_HISTOGRAM, key=int):
        key = str(width)
        print("  %-7s %-42s %s"
              % (width, base_serving.get(key, "unavailable")[:42],
                 cand_serving.get(key, "unavailable")))
    for arch in ARCHES:
        cell = surface.get(arch, {})
        if "error" in cell:
            print("  %s: %s" % (arch, cell["error"]))
            continue
        print("  %s: %.3f -> %.3f derived simdgroups (%+.2f %%), "
              "%d -> %d pipeline(s)"
              % (arch, cell["base_weighted_simdgroups_derived"],
                 cell["candidate_weighted_simdgroups_derived"],
                 cell["change_pct_derived"], cell["base_pipelines"],
                 cell["candidate_pipelines"]))


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

    surface = route_b_surface(base_rows, candidate_rows)
    ranked = surface.get(RANKED, {})
    if "error" in ranked:
        warnings.append("Route B QMV surface: %s" % ranked["error"])
    elif ranked["delta_derived"] < 0:
        failures.append(
            "Route B QMV surface %s: width-weighted derived residency "
            "%.3f -> %.3f (%.2f %%) across %d -> %d pipeline(s)"
            % (RANKED, ranked["base_weighted_simdgroups_derived"],
               ranked["candidate_weighted_simdgroups_derived"],
               ranked["change_pct_derived"], ranked["base_pipelines"],
               ranked["candidate_pipelines"]))
    elif ranked["delta_derived"] > 0:
        warnings.append(
            "Route B QMV surface %s: width-weighted derived residency "
            "%.3f -> %.3f (+%.2f %%)"
            % (RANKED, ranked["base_weighted_simdgroups_derived"],
               ranked["candidate_weighted_simdgroups_derived"],
               ranked["change_pct_derived"]))

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
        "route_b_surface": surface,
        "width_histogram": WIDTH_HISTOGRAM,
        "instrument_note": INSTRUMENT_NOTE,
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
    print_surface(surface)
    print()
    for line in warnings:
        print("WARNING %s" % line)
    for line in failures:
        print("FAIL    %s" % line)
    print("\n%s" % textwrap.fill(INSTRUMENT_NOTE, 78))
    print("\nverdict: %s   %.2f s   %s"
          % (receipt["verdict"].upper(), receipt["runtime_seconds"],
             args.json or "no receipt requested"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
