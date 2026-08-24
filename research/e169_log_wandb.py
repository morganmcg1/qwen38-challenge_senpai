#!/usr/bin/env python3
"""Publish the E169 marginal-verified-row decomposition to W&B.

Every measured number in this run is `harness=local` on g16s and is NOT
gate-qualified: the census times many blocks back to back inside one resident
process, so `cool_gate_passed_real_gate` and `gate_qualified_for_timing` are
false and are logged that way rather than omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def flat(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flat(f"{prefix}/{key}" if prefix else str(key), item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    elif isinstance(value, bool):
        out[prefix] = int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", default="research/out/e169/decomposition.json")
    parser.add_argument("--census", default="research/out/e169/census.json")
    parser.add_argument("--name", default="e169-marginal-verified-row-decomposition")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.decomposition).read_text())
    census_path = Path(args.census)
    census = json.loads(census_path.read_text()) if census_path.exists() else {}

    config = {
        "experiment": "e169-marginal-verified-row-decomposition",
        "assignment_pr": 169,
        "harness": "local",
        "host": "g16s",
        "chip": "Apple M4 Pro",
        "memory_bytes": 51539607552,
        "commit": args.commit,
        "base_sha": "dd844007164e9155f2bc63a69d50d1977bdbdd38",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "identity": census.get("identity", {}),
        "noise_floor_serial_sd_frac": 0.00346,
        "noise_floor_pooled_within_cell_2se_frac": 0.0048,
        "noise_floor_session_drift_frac": 0.0082,
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name, job_type="analysis",
        tags=["e169", "decomposition", "harness-local", "not-gate-qualified"],
        config=config)

    in_situ = report.get("in_situ", {})
    summary = {}
    flat("", report.get("cross_estimator", {}), summary)
    flat("in_situ", in_situ.get("attribution", {}), summary)
    flat("pass_model", in_situ.get("pass_cost_model", {}) or {}, summary)
    flat("bottom_up", report.get("bottom_up", {}).get("families", {}), summary)
    flat("bottom_up_total", {"s": report.get("bottom_up", {}).get(
        "bottom_up_marginal_row_s")}, summary)
    flat("ranked", report.get("ranked_roofline", {}), summary)
    flat("stream_vs_arith", report.get("stream_vs_arithmetic", {}) or {},
         summary)
    flat("arith_efficiency", report.get("arithmetic_efficiency", {}), summary)
    flat("byte_law", report.get("deconfound", {}).get("byte_law_by_width", {}),
         summary)
    static = report.get("static_budget", {})
    flat("static", static.get("per_row", static), summary)
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    pass_model = in_situ.get("pass_cost_model")
    if pass_model:
        table = wandb.Table(columns=[
            "m", "partition", "measured_ms", "predicted_ms", "residual_frac"])
        for row in pass_model["fit"]:
            table.add_data(row["m"], str(row["partition"]),
                           row["measured_ms"], row["predicted_ms"],
                           row["residual_frac"])
        run.log({"pass_cost_fit": table})

        table = wandb.Table(columns=[
            "na", "pass_ms", "gb_per_s", "frac_of_measured_roof",
            "frac_of_spec_roof", "ms_per_row_in_pass"])
        for na, row in pass_model["passes"].items():
            table.add_data(int(na), row["pass_ms"], row["gb_per_s"],
                           row["frac_of_local_measured_roof"],
                           row["frac_of_local_spec_roof"],
                           row["ms_per_row_in_pass"])
        run.log({"weight_pass_cost_by_vector_width": table})

    pairs = report.get("deconfound", {}).get("matched_byte_pairs", [])
    if pairs:
        table = wandb.Table(columns=[
            "mb", "narrow_cell", "wide_cell", "threadgroup_ratio", "m",
            "narrow_us", "wide_us", "wide_over_narrow"])
        for pair in pairs:
            for row in pair["widths"]:
                table.add_data(
                    pair["mb"], pair["narrow_output"]["name"],
                    pair["wide_output"]["name"], pair["threadgroup_ratio"],
                    row["m"], row["narrow_us"], row["wide_us"],
                    row["wide_over_narrow"])
        run.log({"matched_weight_bytes_pairs": table})

    scored = report.get("scored_shapes_vs_byte_law", [])
    if scored:
        table = wandb.Table(columns=[
            "shape", "k", "n", "mb", "m", "measured_us", "predicted_us",
            "residual_frac", "raw_tflops", "fixed_cost_removed_tflops"])
        for record in scored:
            for width, row in record["widths"].items():
                table.add_data(
                    record["name"], record["k"], record["n"], record["mb"],
                    int(width), row["measured_us"], row["predicted_us"],
                    row["residual_frac"], row["raw_tflops"],
                    row["fixed_cost_removed_tflops"])
        run.log({"scored_shapes_out_of_sample": table})

    # Per-arm width curves as a table, so the raw fit inputs stay inspectable
    # next to the fitted slopes rather than only the conclusions.
    arms = report.get("in_situ", {}).get("arms", {})
    if arms:
        table = wandb.Table(columns=[
            "arm", "width", "seconds", "per_row_s", "intercept_s",
            "per_group_s", "r2", "max_pass_spread_frac"])
        for arm, rec in arms.items():
            for width, seconds in rec["seconds_by_width"].items():
                table.add_data(
                    arm, int(width), seconds, rec["per_row_s"], rec["intercept_s"],
                    rec["per_group_s"], rec["r2"], rec["max_pass_spread_frac"])
        run.log({"in_situ_width_curves": table})

    shapes = report.get("bottom_up", {}).get("shapes", {})
    if shapes:
        table = wandb.Table(columns=[
            "shape", "family", "k", "n", "calls_per_verify", "per_call_row_s",
            "per_verify_row_s", "share_of_bottom_up", "r2", "routed_widths"])
        total = report["bottom_up"]["bottom_up_marginal_row_s"]
        for name, rec in shapes.items():
            table.add_data(
                name, rec["family"], rec["k"], rec["n"], rec["calls_per_verify"],
                rec["per_call_row_s"], rec["per_verify_row_s"],
                rec["per_verify_row_s"] / total if total else None,
                rec["r2"], ",".join(str(w) for w in rec["routed_widths"]))
        run.log({"bottom_up_shape_costs": table})

    for path in (args.decomposition, args.census,
                 "research/out/e169/bottom_up.json",
                 "research/out/e169/flop_budget.json"):
        if Path(path).exists():
            run.save(path, policy="now")

    print(f"E169_WANDB_RUN {run.id} {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
