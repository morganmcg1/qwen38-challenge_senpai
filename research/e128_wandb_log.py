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
  `e128-hypothesis-j-and-correction-sign`
      Zero GPU, and no simulator. The measured level bias of the walk's
      `expected` is split into three terms that sum to it exactly, and
      hypothesis J's own prediction of the term it owns is formed
      independently and regressed against the measurement. The same run
      carries the SIGN decomposition: the walk consumes the biased estimate
      twice with opposite effect, so `reach` and `expected` are corrected
      separately and together, and each is priced on the ranked cost curve
      from a myopic replay of the recorded rounds. This run is R-free.
  `e128-rung2-counterfactual-pricing`
      Zero GPU. Every depth policy priced on the board-fitted RANKED round
      cost curve with the uncensored acceptance vectors, recombined into the
      published median exactly per Rule 67, and reported as a curve over the
      pinned `R` band. An arm whose sign flips inside that band is recorded as
      sign indeterminate and never reported as a gain.

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


def log_identity() -> None:
    data = load("rung0-identity.json")
    if data is None:
        print("identity: no artifact, skipping")
        return
    run = start("e128-f1-identity-and-r-pinning", {
        "harness": "local",
        "leg_kind": "offline-identity-audit-and-ranked-R-prediction",
        "question": "advisor F1: is the ranked round count R an assumption?",
        "identity": "eff = drafted/R; accept_rate = accepted/drafted; "
                    "declared_rows = R + drafted; R + accepted = 512 + "
                    "[final tail draft accepted]",
    })
    ident = wandb.Table(columns=[
        "leg", "forced_depth", "rounds", "accepted", "drafted", "emitted",
        "declared_rows", "non_drafting_rounds", "rounds_plus_accepted",
        "window_residual", "eff_reported", "eff_recomputed", "eff_residual",
        "accept_rate_reported", "accept_rate_recomputed",
        "accept_rate_residual", "declared_rows_residual", "R_implied",
        "R_counted"])
    max_eff_resid = 0.0
    max_rate_resid = 0.0
    max_rows_resid = 0
    max_r_resid = 0.0
    for leg in data["identity"]:
        ident.add_data(*[leg[k] for k in (
            "leg", "forced_depth", "rounds", "accepted", "drafted", "emitted",
            "declared_rows", "non_drafting_rounds", "rounds_plus_accepted",
            "window_residual", "eff_reported", "eff_recomputed",
            "eff_residual", "accept_rate_reported", "accept_rate_recomputed",
            "accept_rate_residual", "declared_rows_residual", "R_implied",
            "R_counted")])
        max_eff_resid = max(max_eff_resid, abs(leg["eff_residual"]))
        max_rate_resid = max(max_rate_resid, abs(leg["accept_rate_residual"]))
        max_rows_resid = max(max_rows_resid, abs(leg["declared_rows_residual"]))
        max_r_resid = max(max_r_resid, abs(leg["R_implied"] - leg["R_counted"]))
    calib = wandb.Table(columns=[
        "leg", "rounds", "mean_expected", "mean_accepted", "bias", "bias_pct",
        "slope", "pearson_r"])
    worst_bias_pct = 0.0
    for leg in data["calibration"]:
        calib.add_data(*[leg[k] for k in (
            "leg", "rounds", "mean_expected", "mean_accepted", "bias",
            "bias_pct", "slope", "pearson_r")])
        run.summary["walk_expected_bias_pct_%s" % leg["leg"]] = leg["bias_pct"]
        worst_bias_pct = min(worst_bias_pct, leg["bias_pct"])
    manifold = data["manifold"]
    pred = wandb.Table(columns=[
        "prompt", "published_eff", "inside_local_eff_range", "R_predicted",
        "R_band_low", "R_band_high", "R_assumed", "R_floor",
        "R_ratio_predicted_over_assumed", "assumed_inside_band",
        "accept_rate_predicted", "accept_rate_at_assumed_R"])
    inside = 0
    total = 0
    for prompt, entry in manifold["predictions"].items():
        low, high = entry["R_band"]
        ok = None
        if entry["inside_local_eff_range"]:
            ok = low <= entry["R_assumed"] <= high
            total += 1
            inside += bool(ok)
        pred.add_data(
            prompt, entry["eff"], entry["inside_local_eff_range"],
            entry["R_predicted"], low, high, entry["R_assumed"],
            entry["R_floor"], entry["R_ratio_predicted_over_assumed"], ok,
            entry["accept_rate_predicted"], entry["accept_rate_at_assumed_R"])
        run.summary["R_ratio_%s" % prompt] = \
            entry["R_ratio_predicted_over_assumed"]
    run.summary.update({
        "identity_max_abs_eff_residual": max_eff_resid,
        "identity_max_abs_accept_rate_residual": max_rate_resid,
        "identity_max_abs_declared_rows_residual": max_rows_resid,
        "identity_max_abs_R_residual": max_r_resid,
        "identity_legs": len(data["identity"]),
        "walk_expected_worst_bias_pct": worst_bias_pct,
        "R_loo_residual_pstdev_interior":
            manifold["loo_residual_pstdev_interior"],
        "R_loo_residual_max_abs_interior":
            manifold["loo_residual_max_abs_interior"],
        "R_assumed_inside_band_count": inside,
        "R_assumed_checked_count": total,
    })
    run.log({"identity": ident, "walk_calibration": calib,
             "ranked_R_predictions": pred})
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
    r_band = load("rung2-r-band.json")
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
    implementable = {a: v for a, v in gains.items() if a != "oracle"}
    best_arm = max(implementable, key=implementable.get)
    run.summary.update({
        "e128_recoverable_ranked_median_pct": implementable[best_arm],
        "e128_best_implementable_arm": best_arm,
        "oracle_ranked_median_pct": gains["oracle"],
        "model_reconstructed_base_median": data["base_median"],
        "receipt_median_reconstruction_error":
            data["base_median"] - data["receipt"]["score"],
    })
    gate = data.get("validation_gate")
    if gate:
        run.summary.update({
            "validation_gate_passed": gate["passed"],
            "validation_mean_depth_error": gate["mean_depth_error"],
            "validation_max_abs_depth_error": gate["max_abs_depth_error"],
            "validation_mean_accept_error": gate["mean_accept_error"],
            "validation_max_abs_accept_error": gate["max_abs_accept_error"],
            "validation_fixtures": gate["fixtures"],
        })
    if data.get("validation"):
        table = wandb.Table(columns=[
            "fixture", "measured_depth", "simulated_depth", "depth_error",
            "measured_accept", "simulated_accept", "accept_error"])
        for row in data["validation"]:
            table.add_data(
                row["fixture"], row["measured_depth"], row["simulated_depth"],
                row["simulated_depth"] - row["measured_depth"],
                row["measured_accept"], row["simulated_accept"],
                row["simulated_accept"] - row["measured_accept"])
        run.log({"zero_parameter_validation": table})
    run.summary.update({
        "%s_ranked_median_pct" % arm: gains[arm]
        for arm in gains})
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
        for variant, entry in sensitivity["variants"].items():
            for arm, value in entry["median_gain_pct_vs_ship"].items():
                table.add_data(variant, arm, value)
        payload["sensitivity"] = table
    if r_band:
        table = wandb.Table(columns=[
            "scenario", "arm", "gain_pct_vs_ship", "R_beagle", "R_medicine",
            "R_essays"])
        for name, entry in r_band["scenarios"].items():
            for arm, value in entry["median_gain_pct_vs_ship"].items():
                table.add_data(
                    name, arm, value, entry["R"]["beagle"],
                    entry["R"]["medicine"], entry["R"]["essays"])
        payload["r_band"] = table
        spans = r_band["spans"]
        implementable = [a for a in spans if a != "oracle"]
        run.summary.update({
            "r_band_oracle_span_pct": spans["oracle"]["span"],
            "r_band_oracle_min_pct": spans["oracle"]["min"],
            "r_band_oracle_max_pct": spans["oracle"]["max"],
            "r_band_widest_implementable_span_pct":
                max(spans[a]["span"] for a in implementable),
            "r_band_anchor": r_band["anchor"],
            "r_band_sign_indeterminate_arms": sorted(
                a for a, e in spans.items() if e["sign_indeterminate"]),
        })
        run.summary.update({
            "r_band_%s_min_pct" % arm: entry["min"] for arm, entry
            in spans.items()})
        run.summary.update({
            "r_band_%s_max_pct" % arm: entry["max"] for arm, entry
            in spans.items()})
        # Advisor F2: an arm is only a gain if it is positive at every R in
        # the pinned band. The headline metric is reported on that basis.
        safe = {a: spans[a]["min"] for a in implementable
                if not spans[a]["sign_indeterminate"] and spans[a]["min"] > 0}
        run.summary.update({
            "e128_best_implementable_arm_positive_across_R_band":
                max(safe, key=safe.get) if safe else None,
            "e128_recoverable_ranked_median_pct_worst_case_over_R":
                max(safe.values()) if safe else 0.0,
        })
    run.log(payload)
    run.finish()


def log_jensen() -> None:
    data = load("jensen-and-sign.json")
    if data is None:
        print("jensen: no artifact, skipping")
        return
    run = start("e128-hypothesis-j-and-correction-sign", {
        "harness": "local",
        "leg_kind": "offline-myopic-replay",
        "gpu_used": False,
        "acceptance_imputation":
            "exact when the round rejected, conditioned on the uncensored "
            "survival curve when the round saturated",
        "replay_scope":
            "each round keeps its recorded EMA, margin and parent offer; the "
            "counterfactual outcome is NOT propagated into later rounds",
        "cost_curve": "F97 board-fitted ranked round cost, two tiers",
        "r_free": True,
    })
    rows = data["hypothesis_j"]
    tests = data["hypothesis_j_regression"]
    run.summary.update({
        "j_regression_r_measured_on_predicted":
            tests["jensen_predicted_bias"]["r"],
        "j_regression_slope_measured_on_predicted":
            tests["jensen_predicted_bias"]["slope"],
        "j_internal_validation_r":
            tests["jensen_predicts_selection"]["r"],
        "j_internal_validation_slope":
            tests["jensen_predicts_selection"]["slope"],
        "median_gamma": _median([r["gamma"] for r in rows]),
    })
    run.summary.update({
        "j_residual_corr_%s" % k: v
        for k, v in data["hypothesis_j_residual_correlations"].items()})
    split = wandb.Table(columns=[
        "fixture", "rounds", "mean_depth", "mean_expected", "mean_accepted",
        "gamma", "measured_bias", "margin_component", "ema_component",
        "selection_component", "jensen_predicted_bias", "q_slope",
        "eta2_margin", "spearman_margin_capability",
        "serial_corr_capability", "capability_variance"])
    for row in rows:
        assoc = row["association"]
        split.add_data(
            row["prompt_id"], row["rounds"], row["mean_depth"],
            row["mean_expected"], row["mean_accepted"], row["gamma"],
            row["measured_bias"], row["margin_component"],
            row["ema_component"], row["selection_component"],
            row["jensen_predicted_bias"], assoc["q_slope_per_position"],
            assoc["eta2_margin"], assoc["spearman_margin_capability"],
            assoc["serial_corr_capability"], assoc["capability_variance"])
    arms = wandb.Table(columns=[
        "fixture", "arm", "mean_depth", "delta_depth", "us_per_token",
        "ranked_gain_pct"])
    for row in data["sign_decomposition"]:
        arms.add_data(row["prompt_id"], "ship", row["base"]["mean_depth"],
                      0.0, row["base"]["us_per_token"], 0.0)
        for name, entry in row["arms"].items():
            arms.add_data(row["prompt_id"], name, entry["mean_depth"],
                          entry["delta_depth"], entry["us_per_token"],
                          entry["ranked_gain_pct"])
        for name, entry in row["static"].items():
            arms.add_data(row["prompt_id"], name, entry["mean_depth"],
                          entry["delta_depth"], entry["us_per_token"],
                          entry["ranked_gain_pct"])
    grid = wandb.Table(columns=[
        "fixture", "gamma", "mean_depth", "ranked_gain_pct"])
    for row in data["sign_decomposition"]:
        for gamma, entry in row["gamma_grid"].items():
            grid.add_data(row["prompt_id"], float(gamma),
                          entry["mean_depth"], entry["ranked_gain_pct"])
    for name in ("reachonly", "expectedonly", "levelfix", "jensen",
                 "jensen_both", "oracle"):
        gains = [r["arms"][name]["ranked_gain_pct"]
                 for r in data["sign_decomposition"]]
        deltas = [r["arms"][name]["delta_depth"]
                  for r in data["sign_decomposition"]]
        run.summary.update({
            "myopic_%s_median_ranked_pct" % name: _median(gains),
            "myopic_%s_median_delta_depth" % name: _median(deltas),
            "myopic_%s_fixtures_positive" % name: sum(1 for g in gains if g > 0),
        })
    run.log({"bias_split": split, "sign_arms": arms, "gamma_grid": grid})
    run.finish()


def _median(values: list) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return (ordered[n // 2] if n % 2 else
            0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    runs = {"rung0": log_rung0, "identity": log_identity,
            "rung1": log_rung1, "jensen": log_jensen, "rung2": log_rung2}
    for name, fn in runs.items():
        if args.only and name not in args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
