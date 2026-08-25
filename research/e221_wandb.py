#!/usr/bin/env python3
"""Publish the E221 rows_per_simd result to W&B.

    usage: research/e221_wandb.py [--name NAME] [--notes TEXT]

Two stages with DIFFERENT harness labels, both recorded verbatim:

  step 0  static register census. Zero GPU seconds. `xcrun metal`,
          `xcrun metal-opt` and `agx_crossarch.translate` for the local
          `applegpu_g16s` and the ranked `applegpu_g17s`. No timing, no score.
  step 1  standalone kernel microbenchmark. harness=local-microbench. It holds
          no model, so the benchmark wrapper's process lock and 40C cool gate
          never applied and are NOT claimed:
          `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`.

Nothing here is an official or ranked score, and nothing here is a whole-leg
number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e221-artifacts"
REPO = HERE.parent

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e221-na-relief-rows2"
GROUP = "qwen38-r1-e221-na-relief-rows2"

RELIEF_CEILING_MS = 23.066
STOP_RULE_MS = 0.3
NOISE_FLOOR_MS = 0.24


def shell(*argv: str) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return done.stdout.strip()


def identity(rows: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "assignment_pr": 219,
        "assignment_revision": "e221-r0",
        "base_sha": shell("git", "rev-parse", "HEAD"),
        "base_ref": "senpai/qwen38-mtp-r1",
        "assignment_base_sha":
            "1963bb5604e4eeab0c88f4892d969e7875793e40",
        "dirty_candidate_paths": len([
            line for line in shell(
                "git", "status", "--porcelain", "--",
                "Sources", "Vendor", "Package.swift").splitlines() if line]),
        "host": platform.node(),
        "chip": shell("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(shell("sysctl", "-n", "hw.memsize") or 0),
        "os": shell("sw_vers", "-productVersion"),
        "swift": (shell("swift", "--version").splitlines() or [""])[0],
        "metal": (shell("xcrun", "metal", "--version").splitlines() or [""])[0],
        "local_arch": "applegpu_g16s",
        "ranked_arch": "applegpu_g17s",
        # Step 1 harness labels, verbatim.
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "reference_source":
            "self-generated random affine-4/group-64 cells at scored shapes; "
            "no model checkpoint is loaded",
        "blocks": rows.get("blocks"),
        "chains": rows.get("chains"),
        "shipped_rows_per_simd": rows.get("shipped_rows_per_simd"),
        "relief_ceiling_ms_per_round": RELIEF_CEILING_MS,
        "stop_rule_ms_per_round": STOP_RULE_MS,
        "noise_floor_ms_per_round": NOISE_FLOOR_MS,
    }


def register_table(probe: dict, caps: dict) -> wandb.Table:
    """Real AGX registers and spill bytes over the (NA, ROWS, table) grid."""
    arches = ("applegpu_g16s", "applegpu_g17s")
    columns = ["arch", "kernel", "form", "na", "rows", "use_table",
               "accumulators", "registers", "spill_bytes", "spills",
               "register_cap", "air_lane_peak", "bytes_per_row_per_kblock",
               "finding_549_published", "finding_552_miscompiled"]
    rows = []
    for e in probe["kernels"]:
        for arch in arches:
            regs = e.get("%s_registers" % arch)
            if regs is None:
                continue
            spill = e.get("%s_spill_bytes" % arch) or 0
            rows.append([
                arch, e["kernel"], e["form"], e["na"], e["rows"], e["table"],
                e["accumulators"], regs, spill, bool(spill),
                caps.get(arch), e["air_lane_peak"],
                e["bytes_per_row_per_kblock"], e.get("finding_549"),
                e["finding_552_miscompiled"],
            ])
    return wandb.Table(columns=columns, data=rows)


def comparison_table(report: dict) -> wandb.Table:
    columns = ["cell", "na", "groups", "cold", "invocations_per_round",
               "rows4_ms_per_round", "rows4param_ms_per_round",
               "rows2param_ms_per_round",
               "rows2_vs_rows4_delta_ms_per_round",
               "rows2_vs_rows4param_delta_ms_per_round",
               "rows2_vs_rows4param_ratio",
               "rows2_vs_rows4param_ci95_disjoint",
               "rows4param_inert_ci95_overlaps",
               "bytes_per_row_per_kblock_rows4",
               "bytes_per_row_per_kblock_rows2"]
    rows = []
    for c in report["comparisons"]:
        p4 = c.get("rows4param", {})
        p2 = c.get("rows2param", {})
        same = c.get("rows2_vs_rows4param", {})
        rows.append([
            c["cell"], c["na"], c["groups"], c["cold"],
            c["invocations_per_round"], c["rows4_ms_per_round"],
            p4.get("ms_per_round"), p2.get("ms_per_round"),
            p2.get("delta_ms_per_round"), same.get("delta_ms_per_round"),
            same.get("ratio"),
            (not same["ci95_overlaps"]) if "ci95_overlaps" in same else None,
            p4.get("ci95_overlaps_rows4"),
            p4.get("bytes_per_row_per_kblock"),
            p2.get("bytes_per_row_per_kblock"),
        ])
    return wandb.Table(columns=columns, data=rows)


def slope_table(report: dict) -> wandb.Table:
    columns = ["tag", "rows4_ms_per_round_per_na",
               "rows4param_ms_per_round_per_na",
               "rows2param_ms_per_round_per_na", "rows2_slope_is_steeper"]
    rows = []
    for tag, e in sorted(report["na_slope_per_geometry"].items()):
        r4 = e["rows4_ms_per_round_per_na"]
        r2 = e.get("rows2param_ms_per_round_per_na")
        rows.append([tag, r4, e.get("rows4param_ms_per_round_per_na"), r2,
                     (r2 > r4) if r2 is not None else None])
    return wandb.Table(columns=columns, data=rows)


def conversion_table(report: dict) -> wandb.Table:
    """How one extra ISSUED megabyte converts to time, by route."""
    columns = ["tag", "route", "na", "delta_issued_mb",
               "delta_us_per_invocation", "us_per_issued_mb",
               "width_over_geometry_rate_ratio"]
    rows = []
    for tag, e in sorted(report["issued_byte_conversion"].items()):
        ratio = e.get("width_over_geometry_rate_ratio")
        for g in e.get("geometry_route_rows4_to_rows2", []):
            rows.append([tag, "geometry_rows4_to_rows2", g["na"],
                         g["delta_issued_mb"], g["delta_us_per_invocation"],
                         g["us_per_issued_mb"], ratio])
        w = e.get("width_route_na4_to_na5_at_rows4")
        if w:
            rows.append([tag, "width_na4_to_na5_at_rows4", 5,
                         w["delta_issued_mb"], w["delta_us_per_invocation"],
                         w["us_per_issued_mb"], ratio])
    return wandb.Table(columns=columns, data=rows)


def pooled_table(report: dict) -> wandb.Table:
    columns = ["arm", "rows4_ms_per_round", "rows2param_ms_per_round",
               "relief_ms_per_round", "relief_ceiling_ms_per_round",
               "meets_stop_rule", "outside_noise_floor"]
    rows = []
    for tag, v in sorted(report["verdict"].items()):
        p = report["pooled"][tag]
        rows.append([
            tag, p["rows4_ms_per_round"],
            p.get("rows2param", {}).get("ms_per_round"),
            v["relief_ms_per_round"], RELIEF_CEILING_MS,
            v["meets_stop_rule"], v["outside_noise_floor"],
        ])
    return wandb.Table(columns=columns, data=rows)


def exactness_table(sanity: dict) -> wandb.Table:
    columns = ["cell", "m", "rows", "parameterized", "positive_control",
               "elements", "differing", "non_finite"]
    rows = []
    for e in sanity["value_comparisons"]:
        rows.append([e["cell"], e["m"], e["rows"], e["parameterized"],
                     e.get("positive_control", False), e["elements"],
                     e["differing"], e.get("non_finite")])
    return wandb.Table(columns=columns, data=rows)


def coverage_table(sanity: dict) -> wandb.Table:
    columns = ["cell", "n", "rows", "rows_written", "min_row", "max_row",
               "min_writes", "max_writes", "control_rows_written",
               "control_max_row", "control_covers"]
    rows = []
    for e in sanity["write_census"]:
        rows.append([e["cell"], e["n"], e["rows"], e["rows_written"],
                     e["min_row"], e["max_row"], e["min_writes"],
                     e["max_writes"], e["control_rows_written"],
                     e["control_max_row"], e["control_covers"]])
    return wandb.Table(columns=columns, data=rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=pathlib.Path,
                       default=REPO / "research/out/e219/e221-rows")
    parser.add_argument("--name", default="e221-rows-per-simd-relief")
    parser.add_argument("--notes", default=(
        "E221: does rows_per_simd 4 -> 2 relieve the NA slope? Refuted. "
        "rows=2 is slower in 24 of 24 timed cells and the NA slope gets "
        "STEEPER on 2 of 3 scored cells."))
    args = parser.parse_args()

    rows_blob = json.loads((args.session / "rows.json").read_text())
    sanity = json.loads((args.session / "sanity_rows.json").read_text())
    report = json.loads((ARTIFACTS / "e221-rows-report.json").read_text())
    probe = json.loads((ARTIFACTS / "e221_register_probe.json").read_text())
    wall = json.loads((ARTIFACTS / "e221_register_wall.json").read_text())

    caps = {arch: e["max_registers_without_a_frame"]
            for arch, e in wall["budget"].items()}

    config = identity(rows_blob)
    config.update({
        "g16s_register_cap": caps.get("applegpu_g16s"),
        "g17s_register_cap": caps.get("applegpu_g17s"),
        "bytes_per_output_row_per_kblock_formula": "12 + 36*NA/rows",
        "rows_parameterization_control_pass":
            probe["rows_parameterization_control_pass"],
        "finding_549_rows4_control_pass":
            probe["finding_549_rows4_control_pass"],
    })

    run = wandb.init(project=PROJECT, entity=ENTITY, group=GROUP,
                     name=args.name, notes=args.notes, config=config,
                     job_type="kernel-microbench",
                     tags=["e221", "rows-per-simd", "qmv-geometry",
                           "harness-local-microbench", "not-gate-qualified",
                           "negative-result"])

    same = [c for c in report["comparisons"] if "rows2_vs_rows4param" in c]
    n_slower = sum(1 for c in same
                   if c["rows2_vs_rows4param"]["delta_ms_per_round"] > 0)
    n_disjoint = sum(1 for c in same
                     if not c["rows2_vs_rows4param"]["ci95_overlaps"])
    inert = report["parameterized_header_inertness"]
    worst = min(v["relief_ms_per_round"] for v in report["verdict"].values())
    best = max(v["relief_ms_per_round"] for v in report["verdict"].values())

    run.summary.update({
        "verdict": "not useful: rows_per_simd 4 -> 2 costs more than it saves",
        "hypothesis_supported": False,
        "governing_premise":
            "neither: the NA slope is a per-column inner-loop cost, not "
            "register pressure and not activation-byte volume",
        "timed_cells": len(same),
        "cells_where_rows2_is_slower": n_slower,
        "cells_with_disjoint_ci95": n_disjoint,
        "best_pooled_relief_ms_per_round": best,
        "worst_pooled_relief_ms_per_round": worst,
        "relief_ceiling_ms_per_round": RELIEF_CEILING_MS,
        "stop_rule_met": any(v["meets_stop_rule"]
                            for v in report["verdict"].values()),
        "header_inert_ci95_overlap_cells":
            sum(1 for v in inert.values() if v["ci95_overlaps_rows4"]),
        "header_inert_total_cells": len(inert),
        "bitexact_differing_max": max(
            e["differing"] for e in sanity["value_comparisons"]
            if not e.get("positive_control")),
        "positive_control_min_differing": min(
            e["differing"] for e in sanity["value_comparisons"]
            if e.get("positive_control")),
        "write_coverage_ok": all(
            e["rows_written"] == e["n"] and e["min_row"] == 0
            and e["max_row"] == e["n"] - 1 and e["max_writes"] == 1
            for e in sanity["write_census"]),
        "coverage_control_breaks": all(
            not e["control_covers"] for e in sanity["write_census"]
            if e["rows"] != 4),
        "g17s_rows4_spill_onset_na": 7,
        "g17s_rows2_spills_at_any_na": False,
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": False,
    })

    run.log({
        "registers": register_table(probe, caps),
        "timed_comparisons": comparison_table(report),
        "na_slope_per_geometry": slope_table(report),
        "issued_byte_conversion": conversion_table(report),
        "pooled_verdict": pooled_table(report),
        "bitexactness": exactness_table(sanity),
        "write_coverage": coverage_table(sanity),
    })

    artifact = wandb.Artifact("e221-rows-per-simd", type="experiment")
    for path in sorted(ARTIFACTS.glob("*.json")):
        artifact.add_file(str(path))
    census = ARTIFACTS / "qmv-geometry-byte-census.md"
    if census.exists():
        artifact.add_file(str(census))
    for name in ("rows.json", "sanity_rows.json", "sanity.json",
                 "rows.meta.txt", "sanity.meta.txt"):
        path = args.session / name
        if path.exists():
            artifact.add_file(str(path), name="session/%s" % name)
    run.log_artifact(artifact)

    print("run:", run.url)
    print("run id:", run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
