#!/usr/bin/env python3
"""Publish the E134 discrimination and pass-boundary pricing result to W&B.

    usage: research/e134_wandb_log.py [--only RUN ...]

Run from the repository ROOT.

E134 asked whether the per-round margin vector the session already computes
can recover part of the `+8.52 %` oracle discrimination gap. It cannot: every
estimator arm failed leave-one-fixture-out. The oracle decomposition then
showed the gap is not a discrimination gap at all on our own cost curve. It
concentrates at one boundary, and what pays there is the PRICE, not a better
signal.

Runs published here:

  `e134-rung1-incremental-auc`
      Zero GPU. Per-boundary incremental AUC of the new inputs over the
      shipped input set, pooled fit with leave-one-fixture-out validation.
      Carries the secondary metric and the depth-4 sign inversion.
  `e134-rung2-estimator-arms`
      Zero GPU. Every implementable estimator arm, scored as replayed ranked
      median percent with leave-one-prompt-out weight selection.
  `e134-rung3-boundary-decomposition`
      Zero GPU. The oracle gap split by boundary under both cost curves, and
      the price families that follow from it.
  `e134-rung4-pass-boundary-price`
      Zero GPU. The headline. The `pb6` arm, its controls, its placebos, and
      the pre-registered prediction for a ranked receipt.

Every number here is a model output from an offline replayer. No leg in this
experiment is a timing measurement, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim, and nothing here is an official or ranked score. Rule 79 applies:
only a ranked receipt can validate a depth-price change.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e134-oracle-discrimination-at-the-m6-cliff"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
ART = pathlib.Path("research/e134-artifacts")

BASE_SHA = "83e07638b78b562112843b3fbc2325a345bd6232"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 134

COMMON = {
    "experiment": "e134",
    "base_sha": BASE_SHA,
    "advisor_branch": ADVISOR_BRANCH,
    "pr": PR_NUMBER,
    "host_profile": HOST,
    "harness": "ranked-model",
    "timing_valid": False,
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
    "token_window": 512,
    "offered_depth": 8,
    "receipt": "d3c491b5",
}


def load(name: str):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def start(name: str, config: dict):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, name=name,
        job_type=name.split("-")[1], config={**COMMON, **config},
        reinit=True)


def log_rung1() -> None:
    data = load("rung1-incremental-auc.json")
    if data is None:
        print("rung1: no artifact, skipping")
        return
    run = start("e134-rung1-incremental-auc", {
        "scored_surface_changed": False,
        "runs": "e128 forced-depth archive, 12 fixtures, forced_depth=7",
        "validation": "pooled fit, leave-one-fixture-out",
    })
    policy = data["one_global_policy"]
    table = wandb.Table(columns=[
        "boundary", "observations", "fixtures", "shipped_auc_in",
        "shipped_auc_lofo", "all_auc_in", "all_auc_lofo",
        "incremental_lofo", "null_lofo", "net_lofo"])
    for key in sorted(policy, key=int):
        row = policy[key]
        table.add_data(
            int(key), row["observations"], row["fixtures"],
            row["shipped_in_sample"], row["shipped_heldout"],
            row["all_in_sample"], row["all_heldout"],
            row["incremental_heldout"], row["null_floor_heldout"],
            row["net_heldout"])
    run.log({"rung1/per_boundary": table})
    single = policy["4"]["single_incremental_heldout"]
    stable = wandb.Table(columns=["signal", "incremental_auc_lofo"])
    for name, value in sorted(single.items(), key=lambda kv: -kv[1]):
        stable.add_data(name, value)
    run.log({"rung1/single_input_at_depth4": stable})
    top = max(single.values())
    run.summary.update({
        "e134_incremental_auc_at_depth4": top,
        "e134_incremental_auc_at_depth4_best_signal":
            max(single, key=lambda k: single[k]),
        "e134_incremental_auc_at_depth4_all_inputs":
            policy["4"]["incremental_heldout"],
        "stop_rule_threshold": 0.02,
        "stop_rule_passed": top >= 0.02,
        "verdict": "marginal pass; the whole-set incremental AUC at depth 4 "
                   "is NEGATIVE held out and only one single input clears "
                   "the 0.02 bar",
    })
    run.finish()


def log_rung2() -> None:
    data = load("rung2-arms.json")
    if data is None:
        print("rung2: no artifact, skipping")
        return
    run = start("e134-rung2-estimator-arms", {
        "scored_surface_changed": False,
        "shipping_bar_held_out_pct": 0.50,
        "close_axis_bar_held_out_pct": 0.30,
        "validation": "leave-one-prompt-out weight selection, 6 seeds",
    })
    table = wandb.Table(columns=["arm", "in_sample_pct", "held_out_pct",
                                 "held_out_sd", "gap"])
    for name, entry in sorted(data["held_out"].items()):
        table.add_data(name, entry["in_sample"], entry["held_out"],
                       entry["held_out_sd"],
                       entry["held_out"] - entry["in_sample"])
    run.log({"rung2/arms": table})
    controls = wandb.Table(columns=["arm", "mean_pct", "sd"])
    for name, entry in sorted(data["in_sample"].items()):
        controls.add_data(name, entry["mean"], entry["sd"])
    run.log({"rung2/grid_in_sample": controls})
    best = max(v["held_out"] for v in data["held_out"].values())
    run.summary.update({
        "e134_replayed_ranked_median_pct": best,
        "oracle_pct": data["in_sample"]["oracle@0.0"]["mean"],
        "receipt_score": data["receipt_score"],
        "verdict": "CLOSE THE AXIS: no estimator arm clears +0.30 percent "
                   "held out",
    })
    run.finish()


def log_rung3() -> None:
    ours = load("rung3-ours-cliff4.json")
    board = load("rung3-board-decomposition.json")
    if ours is None:
        print("rung3: no artifact, skipping")
        return
    run = start("e134-rung3-boundary-decomposition", {
        "scored_surface_changed": False,
        "curves": "ours (break M>=6) and board (break M>=5)",
    })
    for label, data in (("ours", ours), ("board", board)):
        if data is None:
            continue
        increments = data["increments"]
        total = sum(increments.values())
        table = wandb.Table(columns=["boundary", "increment_pct", "share"])
        for key in sorted(increments, key=int):
            table.add_data(int(key), increments[key],
                           increments[key] / total if total else None)
        run.log({"rung3/increments_%s" % label: table})
        run.summary["oracle_sum_%s" % label] = total
    run.summary.update({
        "our_curve_boundary4_share": 0.262,
        "board_curve_boundary4_share": 0.193,
        "even_split_share": 1.0 / 7.0,
        "verdict": "on the BOARD curve the oracle gap is nearly evenly "
                   "spread, which fires the rung-3 stop rule; on OUR curve "
                   "boundary 4 carries 26 percent and is the only boundary "
                   "where the cost-aware oracle declines an acceptable draft",
    })
    run.finish()


def log_rung4() -> None:
    fine = load("rung3-ours-tier4-fine.json")
    predict = load("rung4-prediction.json")
    if fine is None or predict is None:
        print("rung4: no artifact, skipping")
        return
    run = start("e134-rung4-pass-boundary-price", {
        "scored_surface_changed": True,
        "arm": "pb6",
        "verify_width": predict["width"],
        "tier": predict["tier"],
        "curve": predict["curve"],
        "validation": "leave-one-prompt-out tier selection, 6 seeds",
        "rule_79": "a local timing leg cannot validate a depth price change",
    })
    grid = wandb.Table(columns=["tier", "median_pct", "sd"])
    for name, entry in sorted(fine["arms"].items()):
        if name.startswith("tierprice@"):
            grid.add_data(float(name.split("@")[1]), entry["mean"],
                          entry["sd"])
    run.log({"rung4/tier_grid": grid})

    arms = wandb.Table(columns=["arm", "median_pct", "sd"])
    for name in ("ship", "pb5", "pb6", "pb7", "pbfit", "rankedprice"):
        entry = fine["arms"].get(name)
        if entry:
            arms.add_data(name, entry["mean"], entry["sd"])
    run.log({"rung4/existing_arms": arms})

    prompts = wandb.Table(columns=["prompt", "candidate_time_ratio", "sd",
                                   "ship_raw", "arm_raw",
                                   "median_pct_if_this_prompt_fails"])
    for name, entry in sorted(predict["per_prompt"].items()):
        prompts.add_data(name, entry["ratio"], entry["sd"],
                         entry.get("ship_raw"), entry.get("arm_raw"),
                         predict["jackknife_median_pct"].get(name))
    run.log({"rung4/per_prompt_prediction": prompts})

    lofo = fine.get("cliff_lofo", {})
    run.summary.update({
        "e134_replayed_ranked_median_pct": lofo.get("held_out"),
        "e134_replayed_ranked_median_pct_in_sample": lofo.get("in_sample"),
        "e134_replayed_ranked_median_pct_sd": lofo.get("sd"),
        "predicted_receipt_score": predict["arm_median_raw"],
        "shipped_receipt_score": predict["ship_median_raw"],
        "predicted_published_median_pct":
            predict["published_median_pct"]["mean"],
        "predicted_published_median_z": predict["published_median_pct"]["z"],
        "predicted_unweighted_mean_pct":
            predict["unweighted_mean_pct"]["mean"],
        "predicted_unweighted_mean_z": predict["unweighted_mean_pct"]["z"],
        "worst_single_prompt_jackknife_pct":
            min(predict["jackknife_median_pct"].values()),
        "flat_level_control_pct": -0.2970,
        "placebo_boundary3_held_out_pct": 0.0,
        "placebo_boundary5_held_out_pct": 0.0,
        "board_curve_at_tier_1p50_pct": -1.6469,
        "onepass67_curve_at_tier_1p50_pct": -0.8963,
        "anchor_pbfit_replayed_pct": fine["arms"]["pbfit"]["mean"],
        "anchor_pbfit_measured_gpu_pct": -3.500,
        "anchor_rankedprice_replayed_pct": fine["arms"]["rankedprice"]["mean"],
        "anchor_rankedprice_e128_published_pct": -2.8508,
        "verdict": "pb6 is the implementable proposal. It is fitted to one "
                   "QMV dispatch table: the same arm reads -0.90 percent on "
                   "the curve that follows thorfinn's one-pass table. The "
                   "replayer reproduces E128 rankedprice to 0.25 pp but does "
                   "NOT reproduce E68's measured pbfit result, so the "
                   "instrument is not externally calibrated.",
    })
    run.finish()


RUNS = {"rung1": log_rung1, "rung2": log_rung2, "rung3": log_rung3,
        "rung4": log_rung4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=tuple(RUNS))
    args = ap.parse_args()
    if not ART.exists():
        raise SystemExit("run from the repository root: %s is missing" % ART)
    for name in args.only or tuple(RUNS):
        RUNS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
