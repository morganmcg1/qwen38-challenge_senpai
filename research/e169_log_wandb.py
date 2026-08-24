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

    summary = {}
    flat("", report.get("cross_estimator", {}), summary)
    flat("in_situ", report.get("in_situ", {}).get("attribution", {}), summary)
    flat("bottom_up", report.get("bottom_up", {}).get("families", {}), summary)
    flat("bottom_up_total", {"s": report.get("bottom_up", {}).get(
        "bottom_up_marginal_row_s")}, summary)
    static = report.get("static_budget", {})
    flat("static", static.get("per_row", static), summary)
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    # Per-arm width curves as a table, so the raw fit inputs stay inspectable
    # next to the fitted slopes rather than only the conclusions.
    arms = report.get("in_situ", {}).get("arms", {})
    if arms:
        table = wandb.Table(columns=[
            "arm", "width", "seconds", "per_row_s", "intercept_s",
            "per_segment_s", "r2", "max_pass_spread_frac"])
        for arm, rec in arms.items():
            for width, seconds in rec["seconds_by_width"].items():
                table.add_data(
                    arm, int(width), seconds, rec["per_row_s"], rec["intercept_s"],
                    rec["per_segment_s"], rec["r2"], rec["max_pass_spread_frac"])
        run.log({"in_situ_width_curves": table})

    shapes = report.get("bottom_up", {}).get("shapes", {})
    if shapes:
        table = wandb.Table(columns=[
            "shape", "family", "k", "n", "calls_per_verify", "per_call_row_s",
            "per_verify_row_s", "share_of_bottom_up", "r2", "routed_to_custom_qmv"])
        total = report["bottom_up"]["bottom_up_marginal_row_s"]
        for name, rec in shapes.items():
            table.add_data(
                name, rec["family"], rec["k"], rec["n"], rec["calls_per_verify"],
                rec["per_call_row_s"], rec["per_verify_row_s"],
                rec["per_verify_row_s"] / total if total else None,
                rec["r2"], int(rec["routed_to_custom_qmv"]))
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
