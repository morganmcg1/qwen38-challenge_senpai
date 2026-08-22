#!/usr/bin/env python3
"""Acceptance suite for the entry-point cliff gate, including templating.

E132 rung 2 widens the gate's Route B verdict from one entry point to the
width-weighted resident simdgroups of every pipeline in the build. The three
E131 proofs must still hold, and two new ones cover the templated case:

  e121_fails         the E121 chunk-sum transplant loses a resident simdgroup
  e126_passes        the E126 revert does not
  revert_clean       the revert against the pre-E121 parent reports no delta
  templated_passes   per-width templating with the shipped table must pass
  wide8_detected     adding {8:8} must be detected as a residency loss

The last proof is named for what the gate measures, not for a verdict on the
change. E132 prices the same {8:8} edit at -9.7 % to -15.2 % QMV instructions
per output element, so its exit 1 is a demand for that price under Rule 87,
not a veto. See INSTRUMENT_NOTE.

The two templated builds are compiled here rather than read from a revision,
because no templated Route B build exists in the tree yet. They are real
libraries built from the shipped kernel header through the same AGX backend,
so the register counts are measurements, not assumptions. Only the pipeline
naming is synthetic, and that is the part of the gate under test.

No GPU, no model, no timing. Every simdgroup figure is derived under Rule 89.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e131_cliff_gate as gate  # noqa: E402
import e132_wide_matvec as wm  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e132-artifacts"

# `git log` for the E121 regression and its revert, from the E131 rung 3 suite.
REVISION_PROOFS = (
    ("e121_fails", "da025231", "5d97175c", "fail"),
    ("e126_passes", "5d97175c", "04171655", "pass"),
    ("revert_clean", "da025231", "04171655", "pass"),
)

# The scored sumtable arm carries the width marker, so `route_b_pipelines`
# resolves each width to its own pipeline.
TEMPLATED_NAME = "qwen35_custom_affine4_g64_qmv_wide_sums_m%d_v1"

MEANINGS = {
    "e121_fails":
        "E131 proof, unchanged by rung 2. These revisions predate the Route B "
        "header, so no Route B surface exists and the JIT cell decides.",
    "e126_passes":
        "E131 proof, unchanged by rung 2.",
    "revert_clean":
        "E131 proof, unchanged by rung 2.",
    "templated_passes":
        "Splitting the shared switch into one pipeline per routed width must "
        "not read as a regression. Residency rises 38.00 -> 41.87 derived "
        "because each pipeline now allocates for its own width only.",
    "wide8_detected":
        "Adding {8:8} costs resident simdgroups and the widened gate sees it. "
        "This is a detection proof, not a verdict on the change: the same "
        "edit deletes 9.7 % to 15.2 % of QMV instructions per output element, "
        "so Rule 87 pricing, not this exit code, decides whether it ships.",
}


def templated_cells(swift: str, table: dict[int, int]) -> dict[str, dict]:
    """A Route B library with one entry point per routed width."""
    header = wm.header_at(swift)
    parts, names = [], []
    for m, ipg in wm.table_pairs(table):
        if m not in gate.WIDTH_HISTOGRAM:
            continue
        name = TEMPLATED_NAME % m
        parts.append(wm.entry_point(
            name, wm.pipeline_body(swift, ((m, ipg),), True), True))
        names.append(name)
    source = wm.library(header, parts)
    return {name: {"library": "route_b", "source": source,
                   "source_form": "swift_metal_kernel",
                   "role": "templated Route B pipeline for M=%s"
                           % gate.WIDTH_IN_NAME.search(name).group(1),
                   "cell": name}
            for name in names}


def run_surface_proof(label: str, base_cells: dict, candidate_cells: dict,
                      expect: str) -> dict:
    """Compare two in-memory builds through the gate's own verdict logic."""
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        base = gate.census(base_cells, workdir, "pbase")
        candidate = gate.census(candidate_cells, workdir, "pcand")
    rows, failures, warnings = gate.compare(base, candidate)
    surface = gate.route_b_surface(base, candidate)
    ranked = surface[gate.RANKED]
    if ranked["delta_derived"] < 0:
        failures.append(
            "Route B QMV surface %s: width-weighted derived residency "
            "%.3f -> %.3f (%.2f %%) across %d -> %d pipeline(s)"
            % (gate.RANKED, ranked["base_weighted_simdgroups_derived"],
               ranked["candidate_weighted_simdgroups_derived"],
               ranked["change_pct_derived"], ranked["base_pipelines"],
               ranked["candidate_pipelines"]))
    verdict = "fail" if failures else "pass"
    return {
        "proof": label,
        "expected": expect,
        "verdict": verdict,
        "held": verdict == expect,
        "route_b_surface": surface,
        "failures": failures,
        "warnings": warnings,
        "cells": rows,
        "runtime_seconds": round(time.time() - started, 2),
    }


def run_revision_proof(label: str, base: str, candidate: str,
                       expect: str) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        base_rows = gate.census(gate.side_sources(base), workdir, "pbase")
        cand_rows = gate.census(gate.side_sources(candidate), workdir, "pcand")
    rows, failures, warnings = gate.compare(base_rows, cand_rows)
    surface = gate.route_b_surface(base_rows, cand_rows)
    ranked = surface.get(gate.RANKED, {})
    if "error" not in ranked and ranked["delta_derived"] < 0:
        failures.append(
            "Route B QMV surface %s: width-weighted derived residency "
            "%.3f -> %.3f (%.2f %%)"
            % (gate.RANKED, ranked["base_weighted_simdgroups_derived"],
               ranked["candidate_weighted_simdgroups_derived"],
               ranked["change_pct_derived"]))
    verdict = "fail" if failures else "pass"
    return {
        "proof": label,
        "expected": expect,
        "verdict": verdict,
        "held": verdict == expect,
        "base_ref": base,
        "candidate_ref": candidate,
        "route_b_surface": surface,
        "failures": failures,
        "warnings": warnings,
        "cells": rows,
        "runtime_seconds": round(time.time() - started, 2),
    }


def main() -> int:
    started = time.time()
    results = []
    for label, base, candidate, expect in REVISION_PROOFS:
        results.append(run_revision_proof(label, base, candidate, expect))

    swift = wm.swift_at("HEAD")
    shipped_switch = gate.side_sources("HEAD")
    # Keep only the Route B library so the templated comparison is not
    # diluted by the unchanged JIT and cluster cells.
    route_b_only = {name: spec for name, spec in shipped_switch.items()
                    if spec.get("library") == "route_b"}
    results.append(run_surface_proof(
        "templated_passes", route_b_only,
        templated_cells(swift, {}), "pass"))
    results.append(run_surface_proof(
        "wide8_detected", templated_cells(swift, {}),
        templated_cells(swift, {8: 8}), "fail"))
    for row in results:
        row["meaning"] = MEANINGS[row["proof"]]

    receipt = {
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "harness": "compile_only",
        "occupancy_label": "derived",
        "occupancy_rule": "Rule 89",
        "tool": "research/e132_gate_proofs.py",
        "toolchain": gate.toolchain(),
        "width_histogram": gate.WIDTH_HISTOGRAM,
        "instrument_note": gate.INSTRUMENT_NOTE,
        "proofs": results,
        "all_held": all(r["held"] for r in results),
        "slowest_proof_seconds": max(r["runtime_seconds"] for r in results),
        "runtime_seconds": round(time.time() - started, 2),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "rung2-gate-proofs.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")

    print("E132 rung 2: cliff gate acceptance suite")
    print("%-18s %-9s %-9s %-7s %s"
          % ("proof", "expected", "verdict", "held", "ranked surface"))
    for row in results:
        ranked = row["route_b_surface"].get(gate.RANKED, {})
        if "error" in ranked:
            detail = ranked["error"]
        else:
            detail = ("%.3f -> %.3f sg derived (%+.2f %%), %d -> %d pipelines"
                      % (ranked["base_weighted_simdgroups_derived"],
                         ranked["candidate_weighted_simdgroups_derived"],
                         ranked["change_pct_derived"],
                         ranked["base_pipelines"],
                         ranked["candidate_pipelines"]))
        print("%-18s %-9s %-9s %-7s %s"
              % (row["proof"], row["expected"], row["verdict"],
                 "yes" if row["held"] else "NO", detail))
    print("\n" + gate.INSTRUMENT_NOTE)
    print("\nwide8_detected: " + MEANINGS["wide8_detected"])
    print("\nslowest proof %.2f s, suite %.2f s, budget 30 s per gate run"
          % (receipt["slowest_proof_seconds"], receipt["runtime_seconds"]))
    print("wrote %s" % path)
    return 0 if receipt["all_held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
