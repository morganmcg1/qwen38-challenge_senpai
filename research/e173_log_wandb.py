#!/usr/bin/env python3
"""E173: publish the F-decomposition evidence to W&B.

The E173 instruments are host and static analyses, not training runs, so this
run carries tables and summary scalars rather than a loss curve. The point is
durability: the advisor asked for the census to survive a workspace recycle.

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

    t = wandb.Table(
        columns=[
            "build", "shape", "k", "n", "m", "calls_per_round", "accepted",
            "refusing_guard", "host_ns_per_call", "ms_per_round_all_cells",
        ]
    )
    per_m: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: {"ns": 0.0, "cells": 0, "accepted": 0}
    )
    for r in payload["routable"]:
        t.add_data(
            config, r["shape"], r["k"], r["n"], r["m"], r["calls_per_round"],
            r["accepted"], r["refusing_guard"], r["host_ns_per_call"],
            r["host_ns_per_call"] * r["calls_per_round"] / 1e6,
        )
        slot = per_m[r["m"]]
        slot["ns"] += r["host_ns_per_call"] * r["calls_per_round"]
        slot["cells"] += r["calls_per_round"]
        slot["accepted"] += r["calls_per_round"] if r["accepted"] else 0
    out[f"admission/routable_{config}"] = t

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


def main() -> int:
    release = load("admission.json")
    debug = load("admission-debug.json")
    gpu = load("gpu-line-items.json")
    gpu_floor = load("gpu-line-items-inner8.json")
    tablepays = load("tablepays.json")
    phases = load("trace-phase-split.json")
    inventory = load("f-inventory.json")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e173-decompose-per-round-fixed-cost",
        job_type="analysis",
        tags=[
            "e173", "qwen-alphonse", "harness=local", "host-census",
            "no-ranked-claim", "gate_qualified_for_timing=false",
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
        # The debug-to-release ratio is the reason the first pass could not be
        # quoted. Keep it visible instead of silently dropping the debug run.
        ratio = wandb.Table(columns=["item", "m", "debug_us", "release_us", "ratio"])
        for m in (1, 3, 4):
            for payload, key in ((debug, "debug"), (release, "release")):
                pass
            d = sum(
                r["host_ns_per_call"] * r["calls_per_round"]
                for r in debug["routed_entry_point_graph_build"]
                if r["m"] == m
            ) / 1e3
            rel = sum(
                r["host_ns_per_call"] * r["calls_per_round"]
                for r in release["routed_entry_point_graph_build"]
                if r["m"] == m
            ) / 1e3
            ratio.add_data("entry_point_build_per_round", m, d, rel, d / rel)
        logs["admission/debug_release_ratio"] = ratio

    for payload, label in ((gpu, "amortised"), (gpu_floor, "inner8_floor")):
        if not payload:
            continue
        t = wandb.Table(
            columns=[
                "arm", "name", "family", "m", "calls_per_round", "bytes_per_call",
                "us_per_call", "ms_per_round", "implied_gb_per_s", "inner",
            ]
        )
        for i in payload["items"]:
            t.add_data(
                label, i["name"], i["family"], i["m"], i["calls_per_round"],
                i["bytes_per_call"], i["seconds_per_call"] * 1e6, i["ms_per_round"],
                i["bytes_per_call"] / i["seconds_per_call"] / 1e9,
                payload["inner_calls_per_timed_region"],
            )
        logs[f"gpu/line_items_{label}"] = t

    if tablepays:
        t = wandb.Table(
            columns=["m", "n_rounds", "median_round_ms", "increment_ms"]
        )
        prev = None
        for row in tablepays["per_width"]:
            inc = None if prev is None else row["median_round_ms"] - prev
            t.add_data(row["m"], row["n"], row["median_round_ms"], inc)
            prev = row["median_round_ms"]
        logs["tablepays/per_width"] = t
        contrast = tablepays["boundary_contrast"]
        logs["tablepays/contrast"] = wandb.Table(
            columns=["quantity", "value_ms", "ci_low_ms", "ci_high_ms"],
            data=[
                [
                    "boundary_curvature_m3",
                    contrast["boundary_curvature_ms"],
                    contrast["boundary_ci"][0],
                    contrast["boundary_ci"][1],
                ],
                [
                    "non_boundary_curvature_m4",
                    contrast["non_boundary_curvature_ms"],
                    contrast["non_boundary_ci"][0],
                    contrast["non_boundary_ci"][1],
                ],
                [
                    "contrast",
                    contrast["contrast_ms"],
                    contrast["contrast_ci"][0],
                    contrast["contrast_ci"][1],
                ],
            ],
        )
        summary["tablepays_contrast_ms"] = contrast["contrast_ms"]
        summary["tablepays_contrast_ci_low_ms"] = contrast["contrast_ci"][0]
        summary["tablepays_contrast_ci_high_ms"] = contrast["contrast_ci"][1]

    if phases:
        t = wandb.Table(
            columns=[
                "m", "n_rounds", "round_ms", "verify_build_ms", "eval_wall_ms",
                "draft_build_ms", "readout_ms", "commit_ms", "upkeep_ms",
            ]
        )
        for row in phases["per_width"]:
            t.add_data(
                row["m"], row["n"], row["round_ms"], row["verify_build_ms"],
                row["eval_wall_ms"], row["draft_build_ms"], row["readout_ms"],
                row["commit_ms"], row["upkeep_ms"],
            )
        logs["trace/phase_split"] = t

    if inventory:
        t = wandb.Table(columns=["name", "ms_per_round", "side", "method", "mechanism"])
        for row in inventory["rows"]:
            t.add_data(
                row["name"], row["ms_per_round"], row["side"], row["method"],
                row["mechanism"],
            )
        logs["inventory/rows"] = t
        summary.update(
            {
                "inventory_explained_ms": inventory["explained_ms"],
                "inventory_host_ms": inventory["host_ms"],
                "inventory_gpu_ms": inventory["gpu_ms"],
                "inventory_residual_ms": inventory["residual_ms"],
            }
        )
        run.summary["inventory_verdict"] = inventory["verdict"]

    if release:
        per_m = collections.defaultdict(float)
        for r in release["routable"]:
            per_m[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
        summary["admission_routable_ms_per_round_m4_release"] = per_m[4] / 1e6
        summary["admission_routable_ms_per_round_m1_release"] = per_m[1] / 1e6
        build_m = collections.defaultdict(float)
        for r in release["routed_entry_point_graph_build"]:
            build_m[r["m"]] += r["host_ns_per_call"] * r["calls_per_round"]
        for m, ns in build_m.items():
            summary[f"entry_point_build_ms_per_round_m{m}_release"] = ns / 1e6

    run.log(logs)
    for key, value in summary.items():
        run.summary[key] = value
    print(f"run_id {run.id}")
    print(f"run_url {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
