#!/usr/bin/env python3
"""Publish the E217 staged threadgroup-memory QMV mapping result to W&B.

    usage: research/e217_wandb.py research/e217-artifacts/e217-report.json
                                  [--name NAME] [--notes TEXT]
                                  [--census research/e217-artifacts/e217_census.json]
                                  [--coverage research/e217-artifacts/e217_coverage.json]
                                  [--gate research/out/e217-gate-0ulp/gate.json]

Every number in the document is `harness=local` and ungated: the legs run with
the per-round phase trace on and `MLXFAST_LOCAL_COOL_GATE=0` under the standing
counterbalanced conditions. No number here is an official or ranked score, and
no absolute leg time is a candidate speed claim.

A contrast named `X - off` is the SAVING of mapping `X` against the shipped
`split` mapping, so positive means `X` is faster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e217-staged-dequant"
GROUP = "qwen38-r1-e217-staged-dequant"

CONTRASTS = ["staged_minus_off", "coop_minus_off", "staged_minus_coop"]
PHASES = ["round_us", "eval_wall_us", "verify_build_us"]


def width_table(report: dict) -> wandb.Table:
    columns = ["contrast", "m", "moved_width", "n_rounds", "saving_ms",
               "stdev_ms", "sem_ms", "two_sigma_ms", "ci95_lo_ms", "ci95_hi_ms",
               "excludes_zero", "above_noise_floor", "public_share",
               "pooled_share"]
    public = report["census_public"]
    pooled = report["census_pooled"]
    rows = []
    for name in CONTRASTS:
        by_width = report["contrasts"][name]["round_us"]["by_width"]
        for key, entry in sorted(by_width.items(), key=lambda kv: int(kv[0])):
            m = int(key)
            rows.append([
                name.replace("_minus_", " - "), m, m >= 6, entry["n_rounds"],
                entry["saving_ms"], entry["stdev_ms"], entry["sem_ms"],
                entry["two_sigma_ms"], entry["ci95_ms"][0], entry["ci95_ms"][1],
                entry["excludes_zero"], entry["above_noise_floor"],
                public.get(key), pooled.get(key),
            ])
    return wandb.Table(columns=columns, data=rows)


def phase_table(report: dict) -> wandb.Table:
    columns = ["contrast", "phase", "m", "n_rounds", "saving_ms", "sem_ms",
               "excludes_zero"]
    rows = []
    for name in CONTRASTS:
        for phase in PHASES:
            block = report["contrasts"][name].get(phase)
            if not block:
                continue
            for key, entry in sorted(block["by_width"].items(),
                                     key=lambda kv: int(kv[0])):
                rows.append([
                    name.replace("_minus_", " - "), phase, int(key),
                    entry["n_rounds"], entry["saving_ms"], entry["sem_ms"],
                    entry["excludes_zero"],
                ])
    return wandb.Table(columns=columns, data=rows)


def weighted_table(report: dict) -> wandb.Table:
    columns = ["contrast", "census", "saving_ms_per_round", "sem_ms",
               "ci95_lo_ms", "ci95_hi_ms", "excludes_zero",
               "census_share_covered"]
    rows = []
    for name in CONTRASTS:
        block = report["contrasts"][name]["round_us"]
        for census in ["public", "pooled"]:
            entry = block[f"weighted_{census}_census"]
            rows.append([
                name.replace("_minus_", " - "), census,
                entry["saving_ms_per_round"], entry["sem_ms"],
                entry["ci95_ms"][0], entry["ci95_ms"][1],
                entry["excludes_zero"], entry["census_share_covered"],
            ])
    return wandb.Table(columns=columns, data=rows)


def leg_table(report: dict) -> wandb.Table:
    columns = ["leg", "session", "arm", "mapping", "leg_index",
               "rounds_traced", "gpu_temp_entry_c", "gpu_temp_exit_c",
               "all_tokens_matched", "residual_divergence_count",
               "mtp_seconds_per_token", "serial_seconds_per_token",
               "effective_mean_draft_len", "accepted_draft_rate",
               "worker_sha256_before", "worker_sha256_after",
               "worker_digest_stable", "base_sha",
               "qmv_split_g2", "qmv_coop", "qmv_staged"]
    rows = []
    for leg in report["legs"]:
        witness = leg["mapping_dispatch_witness"]
        rows.append([
            leg["leg"], leg["session"], leg["arm"], leg["mapping"],
            leg["leg_index"], leg["rounds_traced"], leg["gpu_temp_entry_c"],
            leg["gpu_temp_exit_c"], leg["all_tokens_matched"],
            leg["residual_divergence_count"], leg["mtp_seconds_per_token"],
            leg["serial_seconds_per_token"], leg["effective_mean_draft_len"],
            leg["accepted_draft_rate"], leg["worker_sha256_before"],
            leg["worker_sha256_after"], leg["worker_digest_stable"],
            leg["base_sha"], witness.get("qmv_split_g2"),
            witness.get("qmv_coop"), witness.get("qmv_staged"),
        ])
    return wandb.Table(columns=columns, data=rows)


def census_table(path: Path | None) -> wandb.Table | None:
    """RULE 400 static compile probe. `staticThreadgroupMemoryLength` per
    variant is the named main risk of this experiment."""
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    columns = ["mapping", "m", "use_table", "static_threadgroup_memory_bytes",
               "air_threadgroup_bytes", "max_total_threads_per_threadgroup",
               "simdgroups_per_threadgroup", "g16s_registers", "g16s_spill",
               "g17s_registers", "g17s_spill", "g16s_resident_simdgroups",
               "g17s_resident_simdgroups"]
    rows = []
    for cell in data["cells"]:
        rows.append([
            cell.get("mapping"), cell.get("m"), cell.get("use_table"),
            cell.get("static_threadgroup_memory_bytes"),
            cell.get("air_threadgroup_bytes"),
            cell.get("max_total_threads_per_threadgroup"),
            cell.get("simdgroups_per_threadgroup"),
            (cell.get("applegpu_g16s") or {}).get("registers"),
            (cell.get("applegpu_g16s") or {}).get("spill_bytes"),
            (cell.get("applegpu_g17s") or {}).get("registers"),
            (cell.get("applegpu_g17s") or {}).get("spill_bytes"),
            (cell.get("applegpu_g16s") or {}).get("resident_simdgroups"),
            (cell.get("applegpu_g17s") or {}).get("resident_simdgroups"),
        ])
    return wandb.Table(columns=columns, data=rows)


def gate_table(path: Path | None) -> tuple[wandb.Table | None, wandb.Table | None]:
    """Stage-0 0-ULP numerics gate and its positive controls."""
    if path is None or not path.exists():
        return (None, None)
    data = json.loads(path.read_text())
    columns = ["mapping", "cell", "k", "n", "m", "use_table", "elements",
               "differing", "max_ulp", "max_abs_diff", "non_finite"]
    rows = [[c["mapping"], c["cell"], c["k"], c["n"], c["m"], c["use_table"],
             c["elements"], c["differing"], c["max_ulp"], c["max_abs_diff"],
             c["non_finite"]] for c in data["comparisons"]]
    control_columns = ["control", "cell", "m", "use_table", "differing",
                       "max_ulp", "max_abs_diff"]
    control_rows = [[c["mapping"], c["cell"], c["m"], c["use_table"],
                     c["differing"], c["max_ulp"], c["max_abs_diff"]]
                    for c in data["positive_control"]]
    return (wandb.Table(columns=columns, data=rows),
            wandb.Table(columns=control_columns, data=control_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--name", default=EXPERIMENT)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--census", default="research/e217-artifacts/e217_census.json")
    parser.add_argument(
        "--coverage", default="research/e217-artifacts/e217_coverage.json")
    parser.add_argument("--gate", default="research/out/e217-gate-0ulp/gate.json")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    first = report["legs"][0]
    gate_path = Path(args.gate)
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    coverage_path = Path(args.coverage)
    coverage = (json.loads(coverage_path.read_text())
                if coverage_path.exists() else {})
    census_path = Path(args.census)
    census = json.loads(census_path.read_text()) if census_path.exists() else {}

    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "official_score": False,
        "rankable": False,
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "trace_perturbs_timing": True,
        "noise_floor_ms": report["noise_floor_ms"],
        "sessions": ",".join(report["sessions"]),
        "sign_convention": report["sign_convention"],
        "base_sha": first["base_sha"],
        "worker_sha256": first["worker_sha256_before"],
        "head_provenance_sha256": first["head_provenance_sha256"],
        "decode_tokens": first["decode_tokens"],
        "host": first["host"],
        "chip": first["chip"],
        "fixture": "public --local-iterate fixture",
        "reference_source": "candidate-generated reference rows",
        "mechanism": "explicit raw packed weight tile in threadgroup memory",
        "staged_tile_bytes": census.get("declared_tile_bytes"),
        "moved_widths": "6,7,8,9",
        "promotion_rule": report["promotion_gate"]["rule"],
        "promotion_threshold_ms": report["promotion_gate"]["threshold_ms"],
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=args.name, notes=args.notes, config=config,
                     job_type="analysis")

    gate_result = report["promotion_gate"]
    run.summary["promotion_gate/measured_ms"] = gate_result["measured_ms"]
    run.summary["promotion_gate/sem_ms"] = gate_result["sem_ms"]
    run.summary["promotion_gate/threshold_ms"] = gate_result["threshold_ms"]
    run.summary["promotion_gate/met"] = gate_result["met"]

    replication = report["coop_replication"]
    run.summary["coop_replication/measured_pooled_ms"] = \
        replication["measured_pooled_ms"]
    run.summary["coop_replication/finding_565_pooled_ms"] = \
        replication["finding_565_pooled_ms"]
    run.summary["coop_replication/agrees"] = \
        replication["agrees_with_finding_565"]

    for name in CONTRASTS:
        block = report["contrasts"][name]["round_us"]
        label = name.replace("_minus_", "_vs_")
        for census_kind in ["public", "pooled"]:
            entry = block[f"weighted_{census_kind}_census"]
            run.summary[f"{label}/{census_kind}_saving_ms"] = \
                entry["saving_ms_per_round"]
            run.summary[f"{label}/{census_kind}_sem_ms"] = entry["sem_ms"]
        for key, entry in block["by_width"].items():
            run.summary[f"{label}/m{key}_saving_ms"] = entry["saving_ms"]
            run.summary[f"{label}/m{key}_sem_ms"] = entry["sem_ms"]
            run.summary[f"{label}/m{key}_n"] = entry["n_rounds"]

    for key, value in report["integrity"].items():
        if isinstance(value, (bool, int, float, str)):
            run.summary[f"integrity/{key}"] = value

    if gate:
        run.summary["numerics_gate/worst_max_ulp"] = gate["worst_max_ulp"]
        run.summary["numerics_gate/total_differing"] = gate["total_differing"]
        run.summary["numerics_gate/total_elements"] = gate["total_elements"]
        run.summary["numerics_gate/comparisons"] = len(gate["comparisons"])
        run.summary["numerics_gate/controls_all_tripped"] = all(
            c["differing"] > 0 for c in gate["positive_control"])
    if coverage:
        run.summary["coverage_walk/passed"] = coverage["passed"]
        run.summary["coverage_walk/tile_exact_cover"] = \
            coverage["tile_cover"]["exact_cover"]
        run.summary["coverage_walk/output_cells"] = len(coverage["output_cover"])
        run.summary["coverage_walk/controls_rejected"] = (
            len(coverage["output_controls"]) + len(coverage["tile_controls"]))

    logged = {
        "e217/width_savings": width_table(report),
        "e217/weighted_savings": weighted_table(report),
        "e217/phase_savings": phase_table(report),
        "e217/legs": leg_table(report),
    }
    static_census = census_table(census_path)
    if static_census is not None:
        logged["e217/static_census"] = static_census
    comparisons, controls = gate_table(gate_path)
    if comparisons is not None:
        logged["e217/numerics_gate"] = comparisons
        logged["e217/numerics_gate_controls"] = controls
    run.log(logged)
    run.finish()
    print(f"e217: logged {run.url}")


if __name__ == "__main__":
    raise SystemExit(main())
