#!/usr/bin/env python3
"""Publish the E128 reach-estimator audit to W&B.

    usage: research/e128_wandb_log.py [--only RUN ...]

E128 asks how much of the gap between our realised ranked draft depth and the
static ranked depth optimum is recoverable, and whether the `min(p, conf)`
margin override in `costModelDepth` is what closes it.

Runs published here:

  `e128-rung0-replayer`
      Zero GPU. An exact offline reimplementation of the shipped depth walk,
      validated on archived and fresh 512-token traces at three agreements:
      the `%.6f` walk string, the selected depth, and the forward-carried EMA
      state.
  `e128-rung1-uncensored-acceptance`
      One GPU session. Forced-depth legs that remove the estimator's own
      selection, so per-position acceptance is observed on every round rather
      than only on rounds the estimator already liked. Also carries the margin
      quantization audit and the per-position margin AUC, stratified by
      fixture and never pooled.
  `e128-rung2-counterfactual-pricing`
      Zero GPU. Seven depth policies priced on the board-fitted RANKED round
      cost curve with the uncensored acceptance vectors, recombined into the
      published median exactly per Rule 67.

Every leg here is local and ungated, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim. Rung 1 forces the depth, which changes the work per round, so no
number in this experiment is a timing measurement and none is an official or
ranked score. The ranked prices are model outputs, labelled `harness=ranked`,
computed from the F97 cost curve and never from a local ratio.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e128-reach-estimator-vs-ranked-depth-optimum"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
ART = pathlib.Path("research/e128-artifacts")

BASE_SHA = "526d39739ad76380b56a199a6344d0db02bca765"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 129

COMMON = {
    "experiment": "e128",
    "base_sha": BASE_SHA,
    "advisor_branch": ADVISOR_BRANCH,
    "pr": PR_NUMBER,
    "host_profile": HOST,
    "timing_valid": False,
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
    "token_window": 512,
    "offered_depth": 8,
    "scored_surface_changed": False,
}


def load(name: str):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def start(name: str, config: dict):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, name=name,
        job_type=name.split("-")[1], config={**COMMON, **config},
        reinit=True)


def log_rung0() -> None:
    data = load("rung0-shipped.json")
    fresh = load("rung0-current-base.json")
    if data is None:
        print("rung0: no artifact, skipping")
        return
    run = start("e128-rung0-replayer", {
        "harness": "local",
        "leg_kind": "offline-replay-of-archived-traces",
        "gate": "depth agreement must be 1.000",
    })
    summary = data["summary"]
    run.summary.update({
        "e128_replayer_depth_agreement_frac": summary["depth_agreement"],
        "replayer_sched_agreement_frac": summary["sched_agreement"],
        "replayer_ema_agreement_frac": summary["ema_agreement"],
        "replayer_rounds": summary["rounds"],
        "replayer_legs": summary["legs"],
        "replayer_ema_max_abs_error": summary["ema_max_abs_error"],
    })
    if fresh:
        run.summary.update({
            "current_base_depth_agreement_frac":
                fresh["summary"]["depth_agreement"],
            "current_base_rounds": fresh["summary"]["rounds"],
        })
    table = wandb.Table(columns=[
        "leg", "rounds", "forced_depth", "sched_agreement",
        "depth_agreement", "ema_agreement", "base_sha"])
    for leg in data["legs"] + (fresh["legs"] if fresh else []):
        table.add_data(
            leg["prompt_id"], leg["rounds"], leg["forced_depth"],
            leg["sched_agreement"], leg["depth_agreement"],
            leg["ema_agreement"], leg["base_sha"])
    run.log({"replayer_legs": table})
    run.finish()


def log_rung1() -> None:
    data = load("rung1-forced.json")
    shipped = load("rung1-shipped.json")
    if data is None:
        print("rung1: no artifact, skipping")
        return
    run = start("e128-rung1-uncensored-acceptance", {
        "harness": "local",
        "leg_kind": "forced-depth-uncensored-acceptance",
        "forced_depth": 7,
        "instrument": "research/e128-patches/forced-depth.patch",
        "arm_selector": "DARKBLOOM_E128_FORCE_DEPTH",
    })
    positions = wandb.Table(columns=[
        "fixture", "position", "reached", "accepted", "p", "wilson_low",
        "wilson_high", "margin_auc", "auc_low", "auc_high"])
    legs = wandb.Table(columns=[
        "fixture", "arm", "rounds", "mean_depth", "mean_accepted",
        "accept_rate", "all_tokens_matched", "residual_divergence_count",
        "nil_margin_rounds", "margin_min_gap", "margin_distinct"])
    for leg in data["legs"]:
        for row in leg["positions"]:
            positions.add_data(
                leg["prompt_id"], row["position"], row["reached"],
                row["accepted"], row["p"], row["wilson_low"],
                row["wilson_high"], row["margin_auc"], row["margin_auc_low"],
                row["margin_auc_high"])
        quant = leg["margin_quantization"]
        legs.add_data(
            leg["prompt_id"], "forced7", leg["rounds"], leg["mean_depth"],
            leg["mean_accepted"], leg["accept_rate"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["nil_margin_rounds"], quant.get("min_gap"),
            quant.get("distinct"))
        run.summary.update({
            "forced_accept_rate_%s" % leg["prompt_id"]: leg["accept_rate"],
            "forced_exact_%s" % leg["prompt_id"]: leg["all_tokens_matched"],
        })
    for leg in (shipped["legs"] if shipped else []):
        quant = leg["margin_quantization"]
        legs.add_data(
            leg["prompt_id"], "ship", leg["rounds"], leg["mean_depth"],
            leg["mean_accepted"], leg["accept_rate"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["nil_margin_rounds"], quant.get("min_gap"),
            quant.get("distinct"))
        run.summary.update({
            "ship_accept_rate_%s" % leg["prompt_id"]: leg["accept_rate"],
            "ship_mean_depth_%s" % leg["prompt_id"]: leg["mean_depth"],
        })
    run.log({"per_position_acceptance": positions, "legs": legs})
    run.finish()


def log_rung2() -> None:
    data = load("rung2-pricing.json")
    sensitivity = load("rung2-sensitivity.json")
    if data is None:
        print("rung2: no artifact, skipping")
        return
    run = start("e128-rung2-counterfactual-pricing", {
        "harness": "ranked",
        "leg_kind": "offline-counterfactual-pricing",
        "cost_curve": "F97 board-fitted ranked round cost, two tiers",
        "median_rule": "Rule 67, median recomputed exactly over 8 prompts",
        "receipt": data["receipt"]["id"],
        "receipt_official_score": data["receipt"]["score"],
        "simulation_windows": data["windows"],
    })
    gains = data["median_gain_pct_vs_ship"]
    run.summary.update({
        "e128_recoverable_ranked_median_pct": max(
            gains[arm] for arm in gains if arm != "oracle"),
        "oracle_ranked_median_pct": gains["oracle"],
        "static7_ranked_median_pct": gains["static7"],
        "nomargin_ranked_median_pct": gains["nomargin"],
        "nomargin0_ranked_median_pct": gains["nomargin0"],
        "nomargin1_ranked_median_pct": gains["nomargin1"],
        "recal_ranked_median_pct": gains["recal"],
        "model_reconstructed_base_median": data["base_median"],
    })
    arms = wandb.Table(columns=[
        "arm", "ranked_median", "gain_pct_vs_ship"])
    for arm, median in data["medians"].items():
        arms.add_data(arm, median, gains[arm])
    per_prompt = wandb.Table(columns=["prompt", "arm", "candidate_gain_pct"])
    for arm, entries in data["per_prompt_candidate_gain_pct"].items():
        for prompt, value in entries.items():
            per_prompt.add_data(prompt, arm, value)
    transfer = wandb.Table(columns=[
        "prompt", "fixture", "delta", "target_depth", "simulated_depth",
        "target_accept", "simulated_accept"])
    for prompt, entry in data["transfer"].items():
        transfer.add_data(
            prompt, entry["fixture"], entry["delta"],
            entry["target_depth_f92"], entry["simulated_depth_ship"],
            entry["target_accept_f92"], entry["simulated_accept_ship"])
    payload = {"arms": arms, "per_prompt": per_prompt, "transfer": transfer}
    if sensitivity:
        table = wandb.Table(columns=["variant", "arm", "gain_pct_vs_ship"])
        for variant, entries in sensitivity.items():
            for arm, value in entries.items():
                table.add_data(variant, arm, value)
        payload["sensitivity"] = table
    run.log(payload)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    runs = {"rung0": log_rung0, "rung1": log_rung1, "rung2": log_rung2}
    for name, fn in runs.items():
        if args.only and name not in args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
