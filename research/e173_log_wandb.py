#!/usr/bin/env python3
"""E173: publish the F-decomposition evidence to W&B.

The E173 instruments are host and static analyses, not training runs, so this
run carries tables and summary scalars rather than a loss curve. The point is
durability: the census must survive a workspace recycle.

harness=local throughout. No ranked claim is made or implied.
"""

from __future__ import annotations

import collections
import json
import pathlib
import subprocess

import wandb

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "e173-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def load(name: str):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT.parent, capture_output=True, text=True, check=True
    ).stdout.strip()


def admission_tables(payload: dict, config: str) -> dict[str, wandb.Table]:
    out: dict[str, wandb.Table] = {}

    routable = wandb.Table(
        columns=[
            "build", "shape", "k", "n", "m", "calls_per_round", "accepted",
            "refusing_guard", "host_ns_per_call", "ms_per_round_all_cells",
        ]
    )
    per_m: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: {"ns": 0.0, "cells": 0.0, "accepted": 0.0}
    )
    for r in payload["routable"]:
        routable.add_data(
            config, r["shape"], r["k"], r["n"], r["m"], r["calls_per_round"],
            r["accepted"], r["refusing_guard"], r["host_ns_per_call"],
            r["host_ns_per_call"] * r["calls_per_round"] / 1e6,
        )
        slot = per_m[r["m"]]
        slot["ns"] += r["host_ns_per_call"] * r["calls_per_round"]
        slot["cells"] += r["calls_per_round"]
        slot["accepted"] += r["calls_per_round"] if r["accepted"] else 0
    out[f"admission/routable_{config}"] = routable

    census = wandb.Table(
        columns=[
            "build", "m", "entries", "accepted", "refused", "host_ns_per_cell",
            "ms_per_round",
        ]
    )
    for m, slot in sorted(per_m.items()):
        census.add_data(
            config, m, slot["cells"], slot["accepted"],
            slot["cells"] - slot["accepted"], slot["ns"] / slot["cells"],
            slot["ns"] / 1e6,
        )
    out[f"admission/census_{config}"] = census

    build = wandb.Table(
        columns=[
            "build", "shape", "m", "routed", "table_path", "calls_per_round",
            "host_us_per_cell", "ms_per_round",
        ]
    )
    per_build_m: dict[int, float] = collections.defaultdict(float)
    for r in payload["routed_entry_point_graph_build"]:
        build.add_data(
            config, r["shape"], r["m"], r["routed"], r["table_path"],
            r["calls_per_round"], r["host_ns_per_call"] / 1e3,
            r["host_ns_per_call"] * r["calls_per_round"] / 1e6,
        )
        per_build_m[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
    out[f"admission/entry_point_build_{config}"] = build

    totals = wandb.Table(columns=["build", "m", "ms_per_round"])
    for m, ns in sorted(per_build_m.items()):
        totals.add_data(config, m, ns / 1e6)
    out[f"admission/entry_point_build_total_{config}"] = totals

    guards = wandb.Table(
        columns=["build", "case", "k", "n", "bits", "group_size", "host_ns_per_call"]
    )
    for r in payload["refusal_counterfactuals"]:
        guards.add_data(
            config, r["case"], r["k"], r["n"], r["bits"], r["group_size"],
            r["host_ns_per_call"],
        )
    out[f"admission/refusal_counterfactuals_{config}"] = guards

    sidecar = wandb.Table(
        columns=["build", "state", "returned_table", "slot_count", "host_ns_per_call"]
    )
    for r in payload["xsums_sidecar"]:
        sidecar.add_data(
            config, r["state"], r["returned_table"], r["slot_count"],
            r["host_ns_per_call"],
        )
    out[f"admission/xsums_sidecar_{config}"] = sidecar
    return out


def entry_point_us_per_round(payload: dict, m: int) -> float:
    return (
        sum(
            r["host_ns_per_call"] * r["calls_per_round"]
            for r in payload["routed_entry_point_graph_build"]
            if r["m"] == m
        )
        / 1e3
    )


def main() -> int:
    release = load("admission.json")
    debug = load("admission-debug.json")
    gpu = load("gpu-line-items.json")
    gpu_floor = load("gpu-line-items-inner8.json")
    tablepays = load("tablepays.json")
    phases = load("trace-phase-split.json")
    inventory = load("f-inventory.json")
    e168 = load("e168-curvature.json")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e173-decompose-per-round-fixed-cost",
        job_type="analysis",
        tags=[
            "e173", "qwen-alphonse", "harness-local", "host-census",
            "no-ranked-claim", "not-gate-qualified",
        ],
        config={
            "experiment": "e173-decompose-the-per-round-fixed-cost",
            "assignment_pr": 172,
            "revision": "r0",
            "harness": "local",
            "base_sha": "589793323fca5d1570870b7a044355944f959fa9",
            "candidate_sha": git("rev-parse", "HEAD"),
            "branch": "qwen-alphonse/e173-decompose-f",
            "host_chip": "Apple M4 Pro",
            "host_gpu_architecture": "applegpu_g16s",
            "host_memory_bytes": 51539607552,
            "routed_cells_per_weight_pass": 257,
            "f_local_ms_target": 11.792,
            "f_local_source": "FINDING 404 fixed term, M4 Pro class",
            "f_ranked_ms_reference_only": 6.899,
            "evaluates_anything_admission_arm": False,
            "holds_model": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "ranked_transfer_claimed": False,
        },
    )

    logs: dict[str, object] = {}
    summary: dict[str, float] = {}

    for payload, label in ((release, "release"), (debug, "debug")):
        if payload:
            logs.update(admission_tables(payload, label))

    if release and debug:
        # The debug-to-release ratio is why the first pass could not be quoted.
        # Keep it visible instead of dropping the debug run.
        ratio = wandb.Table(columns=["item", "m", "debug_us", "release_us", "ratio"])
        for m in (1, 3, 4):
            d = entry_point_us_per_round(debug, m)
            rel = entry_point_us_per_round(release, m)
            ratio.add_data("entry_point_build_per_round", m, d, rel, d / rel)
        logs["admission/debug_release_ratio"] = ratio

    for payload, label in ((gpu, "amortised"), (gpu_floor, "inner8_floor")):
        if not payload:
            continue
        table = wandb.Table(
            columns=[
                "arm", "name", "family", "m", "calls_per_round", "bytes_per_call",
                "us_per_call", "ms_per_round", "implied_gb_per_s", "inner",
            ]
        )
        for i in payload["items"]:
            table.add_data(
                label, i["name"], i["family"], i["m"], i["calls_per_round"],
                i["bytes_per_call"], i["seconds_per_call"] * 1e6, i["ms_per_round"],
                i["bytes_per_call"] / i["seconds_per_call"] / 1e9,
                payload["inner_calls_per_timed_region"],
            )
        logs[f"gpu/line_items_{label}"] = table

    if tablepays:
        table = wandb.Table(
            columns=[
                "m_verify", "n_rounds", "table_path", "median_round_ms",
                "median_verify_build_ms", "median_eval_wall_ms",
                "median_host_tail_ms", "mean_acc", "mean_shortfall",
                "mean_cache_pos",
            ]
        )
        for row in tablepays["per_width"]:
            table.add_data(
                row["m_verify"], row["n"], row["table_path"],
                row["median_round_ms"], row["median_verify_build_ms"],
                row["median_eval_wall_ms"], row["median_host_tail_ms"],
                row["mean_acc"], row["mean_shortfall"], row["mean_cache_pos"],
            )
        logs["tablepays/per_width"] = table

        contrast = tablepays["boundary_contrast"]
        curvature = contrast["point"]["curvature_ms"]
        ci = contrast["curvature_ci95_ms"]
        rows = [
            [f"curvature_m{m}", curvature[m], ci[m][0], ci[m][1]]
            for m in sorted(curvature)
        ]
        rows.append(
            [
                "contrast_m3_minus_m4",
                contrast["point"]["contrast_ms"],
                contrast["contrast_ci95_ms"][0],
                contrast["contrast_ci95_ms"][1],
            ]
        )
        logs["tablepays/contrast"] = wandb.Table(
            columns=["quantity", "value_ms", "ci_low_ms", "ci_high_ms"], data=rows
        )
        summary["e37_contrast_ms"] = contrast["point"]["contrast_ms"]
        summary["e37_contrast_ci_low_ms"] = contrast["contrast_ci95_ms"][0]
        summary["e37_contrast_ci_high_ms"] = contrast["contrast_ci95_ms"][1]

    if e168:
        table = wandb.Table(
            columns=[
                "dataset", "n_rounds", "boundary_curvature_m3_ms", "boundary_ci_low",
                "boundary_ci_high", "control_curvature_m4_ms", "control_ci_low",
                "control_ci_high", "contrast_ms", "contrast_ci_low",
                "contrast_ci_high", "identified",
            ]
        )
        for res in e168["results"]:
            if not res.get("usable"):
                continue
            identified = res["label"].startswith("adaptive")
            table.add_data(
                res["label"], res["n_rounds"], res["boundary_curvature_m3_ms"],
                res["boundary_ci"][0], res["boundary_ci"][1],
                res["control_curvature_m4_ms"], res["control_ci"][0],
                res["control_ci"][1], res["contrast_ms"], res["contrast_ci"][0],
                res["contrast_ci"][1], identified,
            )
            if identified:
                summary["e168_adaptive_contrast_ms"] = res["contrast_ms"]
                summary["e168_adaptive_contrast_ci_low_ms"] = res["contrast_ci"][0]
                summary["e168_adaptive_contrast_ci_high_ms"] = res["contrast_ci"][1]
        logs["e168/curvature_contrast"] = table
        run.summary["e168_coverage_note"] = e168["coverage_note"]

    if phases:
        table = wandb.Table(
            columns=[
                "m", "n_rounds", "round_ms", "verify_build_ms", "eval_wall_ms",
                "draft_build_ms", "readout_ms", "commit_ms", "upkeep_ms",
            ]
        )
        for row in phases["per_width"]:
            table.add_data(
                row["m"], row["n"], row["round_ms"], row["verify_build_ms"],
                row["eval_wall_ms"], row["draft_build_ms"], row["readout_ms"],
                row["commit_ms"], row["upkeep_ms"],
            )
        logs["trace/phase_split"] = table

    if inventory:
        table = wandb.Table(
            columns=["name", "ms_per_round", "side", "method", "mechanism"]
        )
        for row in inventory["rows"]:
            table.add_data(
                row["name"], row["ms_per_round"], row["side"], row["method"],
                row["mechanism"],
            )
        logs["inventory/rows"] = table
        summary.update(
            {
                "inventory_explained_ms": inventory["explained_ms"],
                "inventory_host_ms": inventory["host_ms"],
                "inventory_gpu_ms": inventory["gpu_ms"],
                "inventory_mixed_ms": inventory["mixed_ms"],
                "inventory_residual_ms": inventory["residual_ms"],
            }
        )
        run.summary["inventory_verdict"] = inventory["verdict"]

    if release:
        per_m: dict[int, float] = collections.defaultdict(float)
        for r in release["routable"]:
            per_m[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
        summary["admission_routable_ms_per_round_m4_release"] = per_m[4] / 1e6
        summary["admission_routable_ms_per_round_m1_release"] = per_m[1] / 1e6
        for m in (1, 3, 4):
            summary[f"entry_point_build_ms_per_round_m{m}_release"] = (
                entry_point_us_per_round(release, m) / 1e3
            )

    run.log(logs)
    for key, value in summary.items():
        run.summary[key] = value
    print(f"run_id {run.id}")
    print(f"run_url {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
