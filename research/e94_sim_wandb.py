#!/usr/bin/env python3
"""E94: publish the offline ranked-curve analysis to W&B.

usage:
  research/e94_sim_wandb.py --report research/e94-artifacts/rung3-ranked-sim.json
                            --name e94-rung3-ranked-sim [--notes TEXT]

`--report` takes the JSON written by `research/e94_ranked_sim.py --out`. This
run holds NO timing measurement of its own. Every number in it is either an
exact board field, the advisor's fitted ranked cost curve, arithmetic from the
ported scheduler, or a measured rung-3 leg statistic that is logged separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e94-depth-price-cliff-guard"


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--group", default=EXPERIMENT)
    ap.add_argument("--base-sha", default="")
    ap.add_argument("--branch-commit", default="")
    args = ap.parse_args()

    doc = json.loads(Path(args.report).read_text())
    price = doc["price"]
    curve = doc["ranked_curve"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        group=args.group,
        job_type="offline-analysis",
        notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "assignment_pr": 97,
            "student": "qwen-edward",
            "analysis": "research/e94_ranked_sim.py",
            "base_sha": args.base_sha,
            "branch_commit": args.branch_commit,
            "stage_a_windows": doc["windows"],
            "seed": doc["seed"],
            "ranked_curve_a1": curve["a1"],
            "ranked_curve_c1": curve["c1"],
            "ranked_curve_a2": curve["a2"],
            "ranked_curve_c2": curve["c2"],
            "price_ship_marginal": price["ship"]["marginal"],
            "price_m5fit_marginal": price["m5fit"]["marginal"],
            "has_timing_measurement": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        },
    )

    map_cols = ["q", "ship_depth", "m5fit_depth", "best_depth", "ship_us",
                "m5fit_us", "best_us", "delta_pct"]
    run.log({"stage0_ranked_depth_map": table(map_cols, doc["stage_0_ranked"])})
    run.log({"stage0_local_depth_map": table(map_cols, doc["stage_0_local"])})

    run.log({"stage_a_reproduction": table(
        ["target", "accept_obs", "accept_sim", "draft_len_obs",
         "draft_len_sim", "error_pct", "mu"],
        [dict(target=name, **row) for name, row in
         doc["stage_a"]["rows"].items()])})

    stage_b = doc["stage_b"]
    run.log({"stage_b_local_validation": table(
        ["arm", "sim_tokens_per_round", "obs_tokens_per_round",
         "sim_us_per_token", "obs_us_per_token", "error_pct"],
        [dict(arm=arm, **stage_b[arm]) for arm in ("ship", "m5fit")])})

    sweep_rows = []
    cell_rows = []
    for tag, cell in doc["stage_c"].items():
        for row in cell["sweep"]:
            sweep_rows.append(dict(case=tag, prompt=cell["prompt"],
                                   rounds=cell["rounds"], **row))
        cell_rows.append(dict(
            case=tag, **{k: cell[k] for k in
                         ("prompt", "rounds", "accepted_per_round",
                          "accept_rate", "mu", "base_us_per_token",
                          "ship_depth", "m5fit_depth", "best_depth",
                          "best_us_per_token", "ship_us_per_token",
                          "m5fit_us_per_token", "delta_vs_observed_pct",
                          "delta_vs_ship_walk_pct")}))
    run.log({"stage_c_depth_sweep": table(
        ["case", "prompt", "rounds", "depth", "tokens", "round_us",
         "us_per_token"], sweep_rows)})
    run.log({"stage_c_cases": table(list(cell_rows[0].keys()), cell_rows)})

    stage_d = doc["stage_d"]
    run.log({"stage_d_corrected_map": table(
        ["q", "ship", "m5fit", "fixed", "best", "delta_pct"],
        stage_d["rows"])})
    run.log({"stage_d_applied": table(
        ["prompt", "ship_depth", "m5fit_depth", "fixed_depth", "best_depth",
         "ship_us", "fixed_us", "delta_pct", "deep_spread_pct"],
        [dict(prompt=name, **row) for name, row in
         stage_d["applied"].items()])})

    stop = doc["stop_rule"]
    run.summary.update({
        "verdict": stop["verdict"],
        "beagle_pct_vs_ship_walk": stop["beagle_pct"],
        "essays_pct_vs_ship_walk": stop["essays_pct"],
        "published_pct_vs_ship_walk": stop["published_pct"],
        "beagle_pct_vs_observed": doc["published"][
            "delta_vs_observed_pct"]["beagle"],
        "essays_pct_vs_observed": doc["published"][
            "delta_vs_observed_pct"]["essays"],
        "published_pct_vs_observed": doc["published"][
            "delta_vs_observed_pct"]["published_pct"],
        "ranked_optimum_switch_q": doc["ranked_optimum_switch_q"],
        "stage_a_passed": doc["stage_a"]["passed"],
        "stage_a_beagle_error_pct": doc["stage_a"]["rows"]["beagle"]["error_pct"],
        "stage_a_essays_error_pct": doc["stage_a"]["rows"]["essays"]["error_pct"],
        "stage_a_local_error_pct":
            doc["stage_a"]["rows"]["local ship"]["error_pct"],
        "stage_b_obs_delta_pct": stage_b["obs_delta_pct"],
        "stage_b_sim_delta_pct": stage_b["sim_delta_pct"],
        "stage_d_group_factor": stage_d["group_factor"],
        "stage_d_tier_factor": stage_d["tier_factor"],
    })

    print(f"wandb_run_id={run.id}")
    print(f"wandb_run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
