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
  `e128-section5-our-ranked-curve`
      Zero GPU. The board curve belonged to another solver's frontier, so the
      RANKED round cost curve is refitted from our own four receipts and the
      eight per-prompt round counts they imply. The fit is monotone and
      two-tier, it is taken against the expected cost under each prompt's
      depth histogram rather than the cost at the mean depth, and the tier
      step is located empirically and rechecked by leave-one-out. The same run
      re-pins `R` by a one-parameter interpolation sweep between the assumed
      and predicted vectors.
  `e128-rung2-our-curve-pricing`
      Zero GPU. The headline pricing pass. Every depth policy priced on OUR
      fitted RANKED round cost curve with the uncensored acceptance vectors,
      recombined into the published median exactly per Rule 67, and reported
      as a curve over the pinned `R` band. An arm whose sign flips inside that
      band is recorded as sign indeterminate and never reported as a gain.
  `e128-rung2-counterfactual-pricing`
      The same pass on the board-fitted curve, kept as the pre-F3 control so
      the effect of the curve swap on every arm is visible.

Every leg here is local and ungated, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim. Rung 1 forces the depth, which changes the work per round, so no
number in this experiment is a timing measurement and none is an official or
ranked score. The ranked prices are model outputs, labelled `harness=ranked`,
computed from a fitted cost curve and never from a local ratio.
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
REBASE_SHA = "221065c5f16e9118797c15421cbdf0e91269dd5c"
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


CURVE_RUNS = {
    "ours": {
        "run": "e128-rung2-our-curve-pricing",
        "prefix": "rung2-ours-",
        "suffix": "",
        "curve": "E128 section 5, fitted from our own 21 receipts, two tiers",
    },
    "board": {
        "run": "e128-rung2-counterfactual-pricing",
        "prefix": "rung2-",
        "suffix": "_board_curve",
        "curve": "F97 board-fitted ranked round cost, two tiers",
    },
}


def _implementable(gains: dict) -> dict:
    """Counterfactual arms only.

    `oracle` needs the acceptance outcome before the draft is proposed, so it
    is not implementable. `ship` and `price0.18` are the shipped policy itself
    and always score exactly 0.0, so including them would make the recoverable
    gain trivially non-negative.
    """
    excluded = {"oracle", "ship", "price0.18"}
    return {a: v for a, v in gains.items() if a not in excluded}


def log_rung2(curve: str = "ours") -> None:
    spec = CURVE_RUNS[curve]
    pre, suf = spec["prefix"], spec["suffix"]
    data = load(pre + "pricing.json")
    sensitivity = load(pre + "sensitivity.json")
    r_band = load(pre + "r-band.json")
    if data is None:
        print("rung2 %s: no artifact, skipping" % curve)
        return
    run = start(spec["run"], {
        "harness": "ranked",
        "leg_kind": "offline-counterfactual-pricing",
        "cost_curve": spec["curve"],
        "cost_curve_source": curve,
        "median_rule": "Rule 67, median recomputed exactly over 8 prompts",
        "receipt": data["receipt"]["id"],
        "receipt_official_score": data["receipt"]["score"],
        "simulation_windows": data["windows"],
    })
    gains = data["median_gain_pct_vs_ship"]
    implementable = _implementable(gains)
    best_arm = max(implementable, key=implementable.get)
    run.summary.update({
        "e128_recoverable_ranked_median_pct" + suf: implementable[best_arm],
        "e128_best_implementable_arm" + suf: best_arm,
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
        implementable = list(_implementable(spans))
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
            "e128_best_implementable_arm_positive_across_R_band" + suf:
                max(safe, key=safe.get) if safe else None,
            "e128_recoverable_ranked_median_pct_worst_case_over_R" + suf:
                max(safe.values()) if safe else 0.0,
        })
        # The R band uses the accept-rate anchor. That anchor reproduces the
        # published acceptance rate but not the published depth, so its
        # positive arms price against a shipped baseline that was never run.
        # Record both anchors' fidelity so the number above is not read as a
        # gain.
        primary = data["transfer"]
        alt = r_band["scenarios"][r_band.get("primary_scenario", "assumed")]
        depth_err_primary = max(
            abs(e["simulated_depth_ship"] - e["target_depth_f92"])
            for e in primary.values())
        depth_err_alt = max(
            abs(alt["simulated_depth_ship"][p] - e["target_depth_f92"])
            for p, e in primary.items())
        run.summary.update({
            "transfer_depth_anchor_max_abs_depth_error": depth_err_primary,
            "transfer_accept_anchor_max_abs_depth_error": depth_err_alt,
            "transfer_headline_anchor": "depth",
            "r_band_positive_arms_are_credible": False,
            "r_band_credibility_note": (
                "The R band uses the accept-rate anchor, which misses the "
                "published F92 depth by up to %.3f tokens against %.3f for "
                "the depth anchor used by the headline. Its positive arms "
                "price counterfactuals against a shipped baseline that the "
                "published depths rule out." % (depth_err_alt,
                                                depth_err_primary)),
        })
    sweep = load("rung2-curve-sweep.json") if curve == "ours" else None
    if sweep:
        table = wandb.Table(columns=[
            "curve", "breakpoint", "arm", "gain_pct_vs_ship"])
        cost = wandb.Table(columns=["curve", "rows", "round_us"])
        for key, entry in sweep["curves"].items():
            for arm, value in entry["median_gain_pct_vs_ship"].items():
                table.add_data(key, entry["curve"]["breakpoint"], arm, value)
            for rows, value in entry["round_us"].items():
                cost.add_data(key, int(rows), value)
        payload["curve_sweep"] = table
        payload["curve_round_cost"] = cost
        best = {}
        for key, entry in sweep["curves"].items():
            imp = _implementable(entry["median_gain_pct_vs_ship"])
            best[key] = max(imp.values())
        run.summary.update({
            "curve_sweep_best_implementable_pct_%s" % k: v
            for k, v in best.items()})
        positive = sorted(k for k, v in best.items() if v > 0)
        well_fitted = {k: v for k, v in best.items() if k != "predicted"}
        run.summary.update({
            "curve_sweep_any_curve_positive": bool(positive),
            "curve_sweep_positive_curves": positive,
            "curve_sweep_spread_pct": max(best.values()) - min(best.values()),
            # `predicted` is the R vector section 5 rejects: it fits the
            # measured round cost at 867 us RMSE against 91 us for `assumed`,
            # and its fitted curve degenerates to a single slope with no tier
            # step. Report the sweep over the curves the data supports too.
            "curve_sweep_best_implementable_pct_well_fitted_max":
                max(well_fitted.values()),
            "curve_sweep_any_well_fitted_curve_positive":
                any(v > 0 for v in well_fitted.values()),
        })
    scan = load("board-depth-scan.json") if curve == "ours" else None
    if scan:
        table = wandb.Table(columns=[
            "prompt", "f83_weight", "distinct_draft_lens", "shipped_draft_len",
            "best_raw_at_shipped", "best_raw_elsewhere", "shipped_lead_pct",
            "top_row_run_at_shipped"])
        depths = wandb.Table(columns=[
            "prompt", "draft_len", "rows", "best_raw", "median_raw"])
        for prompt, entry in scan["prompts"].items():
            table.add_data(
                prompt, entry["f83_weight"], entry["distinct_draft_lens"],
                entry["shipped_draft_len"], entry["best_raw_at_shipped"],
                entry["best_raw_at_any_other_depth"], entry["shipped_lead_pct"],
                entry["top_row_run_at_shipped_draft_len"])
            for row in entry["by_draft_len"]:
                depths.add_data(prompt, row["draft_len"], row["rows"],
                                row["best_raw"], row["median_raw"])
            run.summary["board_shipped_depth_lead_pct_%s" % prompt] = \
                entry["shipped_lead_pct"]
        payload["board_depth_scan"] = table
        payload["board_depth_detail"] = depths
        weighted = [e for e in scan["prompts"].values() if e["f83_weight"] > 0]
        run.summary.update({
            "board_weighted_prompts_where_shipped_depth_wins":
                sum(1 for e in weighted if e["shipped_lead_pct"] > 0),
            "board_weighted_prompts_scanned": len(weighted),
            "board_min_shipped_lead_pct_weighted":
                min(e["shipped_lead_pct"] for e in weighted),
        })
    run.log(payload)
    run.finish()


def _round_us(fit: dict, rows: int) -> float:
    intercept, slope = (fit["lo"] if rows < fit["breakpoint"] else fit["hi"])
    return intercept + slope * rows


def _marginal_price(fit: dict) -> list:
    """Cost of the d-th extra draft row, normalised by the depth-0 round."""
    unit = _round_us(fit, 1)
    return [(_round_us(fit, d + 2) - _round_us(fit, d + 1)) / unit
            for d in range(8)]


def log_ourcurve() -> None:
    """Section 5: fit the ranked round cost curve from our own receipts."""
    data = load("section5-ourcurve.json")
    if data is None:
        print("ourcurve: no artifact, skipping")
        return
    primary = data["primary_r_scenario"]
    scenarios = data["scenarios"]
    chosen = scenarios[primary]["dist_fit_best"]
    mean_fit = scenarios[primary]["mean_fit_best"]
    board = data["board_curve"]
    run = start("e128-section5-our-ranked-curve", {
        "harness": "ranked",
        "leg_kind": "offline-curve-fit",
        "gpu_used": False,
        "control_points": len(data["control_points"]),
        "fit": "monotone two-tier least squares, exact bounded active set",
        "constraints": "slope >= 0, tier jump >= 0, slope increase >= 0",
        "distribution":
            "per-prompt depth histogram from the E124 fixture traces, "
            "max-entropy exponential tilt onto that prompt's own mean",
        "jensen_control":
            "the curve is fitted against E[c(M)] under the tilted depth "
            "histogram, not against c(E[M])",
        "crown_receipt": data["receipt"]["d3c491b5"]["id"],
        "crown_score": data["receipt"]["d3c491b5"]["score"],
        "primary_r_scenario": primary,
    })
    run.summary.update({
        "ourcurve_breakpoint": chosen["breakpoint"],
        "ourcurve_rmse_us": chosen["rmse"],
        "ourcurve_rmse_pct": 100.0 * data["scenarios"]["r_sweep"]["best"][
            "relative_rmse"],
        "ourcurve_lo_intercept": chosen["lo"][0],
        "ourcurve_lo_slope": chosen["lo"][1],
        "ourcurve_hi_intercept": chosen["hi"][0],
        "ourcurve_hi_slope": chosen["hi"][1],
        "ourcurve_tier_jump_us": chosen["jump_us"],
        "ourcurve_slope_increase_us": chosen["dslope_us"],
        "board_breakpoint": board["breakpoint"],
        "ourcurve_marginal_to_fixed_ratio":
            data["marginal_to_fixed_ratio"]["ours"],
        "board_marginal_to_fixed_ratio":
            data["marginal_to_fixed_ratio"]["board"],
    })
    best = scenarios["r_sweep"]["best"]
    run.summary.update({
        "r_sweep_best_t": best["t"],
        "r_sweep_best_rmse_us": best["rmse"],
        "r_sweep_best_relative_rmse": best["relative_rmse"],
        "r_sweep_best_breakpoint": best["breakpoint"],
        "r_pinning_confirms_assumed": best["t"] == 0.0,
    })
    sweep = wandb.Table(columns=["t", "breakpoint", "rmse_us", "relative_rmse"])
    for row in scenarios["r_sweep"]["grid"]:
        sweep.add_data(row["t"], row["breakpoint"], row["rmse"],
                       row["relative_rmse"])
    grid = wandb.Table(columns=["r_key", "breakpoint", "rmse_us"])
    for r_key, entry in scenarios["rmse_grid_piece"].items():
        for bp, value in entry.items():
            grid.add_data(r_key, int(bp), value)
    loo = wandb.Table(columns=["dropped", "breakpoint", "rmse_us"])
    for row in data["loo_breakpoint"]:
        loo.add_data(row["dropped"], row["breakpoint"], row["rmse"])
    breakpoints = {row["breakpoint"] for row in data["loo_breakpoint"]}
    run.summary.update({
        "loo_breakpoints": sorted(breakpoints),
        "loo_breakpoint_stable": len(breakpoints) == 1,
        "jensen_mean_fit_breakpoint": mean_fit["breakpoint"],
        "jensen_dist_fit_breakpoint": chosen["breakpoint"],
        "jensen_shifts_breakpoint":
            mean_fit["breakpoint"] != chosen["breakpoint"],
    })
    jensen = wandb.Table(columns=[
        "prompt", "mean_depth", "cost_at_mean_us", "cost_over_hist_us",
        "jensen_bias_us", "jensen_bias_pct", "hist_sd_rows"])
    for row in scenarios[primary]["jensen"]:
        jensen.add_data(row["prompt"], row["mbar"], row["cost_at_mean_us"],
                        row["cost_over_hist_us"], row["jensen_bias_us"],
                        row["jensen_bias_pct"], row["hist_sd_rows"])
    prices = wandb.Table(columns=["curve", "depth", "marginal_price"])
    costs = wandb.Table(columns=["curve", "rows", "round_us"])
    for key, fit in (("ours", chosen), ("board", board),
                     ("ours_mean_fit", mean_fit)):
        for depth, value in enumerate(_marginal_price(fit)):
            prices.add_data(key, depth, value)
        for rows in range(1, 9):
            costs.add_data(key, rows, _round_us(fit, rows))
    points = wandb.Table(columns=[
        "prompt", "R", "mean_rows", "measured_round_us", "fitted_round_us",
        "residual_us", "f83_weight", "tokens_per_round"])
    for row in scenarios[primary]["points"]:
        residual = chosen["residuals"][row["prompt"]]
        points.add_data(row["prompt"], row["R"], row["mbar"], row["round_us"],
                        row["round_us"] - residual, residual, row["weight"],
                        row["tokens_per_round"])
    run.log({"r_interpolation_sweep": sweep, "rmse_grid": grid,
             "leave_one_out": loo, "jensen_bias": jensen,
             "marginal_price": prices, "round_cost": costs,
             "control_points": points})
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


def log_followups() -> None:
    """The zero-GPU advisor follow-ups F4 through F9 in one run."""
    hists = load("f4-width-histograms.json")
    slopes = load("f4-slopes.json")
    arms = load("f4-arm-prices.json")
    channel = load("f5-two-channel.json")
    census = load("f6-census.json")
    signals = load("f4-signals.json")
    sweep = load("f4-curve-sweep-admissible.json")
    strata = load("f7-strata.json")
    onepass = load("f7-onepass-repriced.json")
    head = load("f7-head-share.json")
    state = load("f8-state-fe.json")
    f9 = load("f9-state-shape-price.json")
    if census is None or signals is None:
        print("followups: missing artifact, skipping")
        return

    run = start("e128-f4-to-f9-followups", {
        "harness": "ranked",
        "gpu_seconds": 0,
        "rebased_onto": REBASE_SHA,
        "receipt_anchor": "d3c491b5",
        "headline_curve": "passcount_slopeonly@M>=6",
        "pass_boundary_first_two_pass": census["census"]["first_two_pass"],
        "dispatch_table_source": "Qwen35.swift:1565",
    })

    flat = {
        "f6_pooled_best_f_us": census["pooled_best"]["f"],
        "f6_pooled_best_k_us_per_row": census["pooled_best"]["k"],
        "f6_pooled_best_rmse_us": census["pooled_best"]["rmse_us"],
        "f6_pooled_at_census_f_us": census["pooled_fit"]["f"],
        "f6_pooled_at_census_k_us_per_row": census["pooled_fit"]["k"],
        "f6_pooled_at_census_rmse_us": census["pooled_fit"]["rmse_us"],
        "f6_census_us_per_unit": census["census"]["us_per_unit"],
        "f6_arm_f83_weighted_qmv_saving": census["arm_price"][
            "f83_weighted_qmv"],
        "f6_arm_leg_frame": census["arm_price"]["leg_frame"],
        "f6_arm_rule67_median_pct": census["arm_price"]["rule67_median_pct"],
        "f6_m9_mass_max": max(census["m9_mass"].values()),
    }
    for width, row in census["excess_profiles"].items():
        for model, value in row.items():
            flat["f6_excess_pct_%s_M%s" % (model, width)] = value
    for width, row in census["table_design"].items():
        flat["f6_design_f83_share_M%s" % width] = row["f83_share"]
        flat["f6_design_census_weighted_M%s" % width] = row["census_weighted"]
        flat["f6_design_slopeonly_weighted_M%s" % width] = row[
            "slopeonly_weighted"]

    for stratum, row in signals["strata"].items():
        for name, value in row.items():
            flat["f4_auc_%s_%s" % (stratum, name)] = value
    for fixture, row in signals["summary"].items():
        for name, value in row.items():
            flat["f4_auc_fixture_%s_%s" % (fixture, name)] = value

    if hists is not None:
        weighted = hists["f83_weighted"]
        flat["f4_hist_f83_mean_M"] = weighted["mean_M"]
        flat["f4_hist_f83_sd_M"] = weighted["sd_M"]
        flat["f4_hist_f83_p_M_ge_6"] = weighted["p_M_ge_6"]
        for index, value in enumerate(weighted["probs"]):
            flat["f4_hist_f83_p_M%d" % (index + 1)] = value
    if slopes is not None and "admissible_table" in slopes:
        for row in slopes["admissible_table"][:4]:
            key = "%s_b%s" % (row["model"], row["breakpoint"])
            flat["f4_slope_rmse_%s" % key] = row["rmse"]
            flat["f4_slope_aicc_%s" % key] = row["aicc"]
    if arms is not None and "summary" in arms:
        for key, value in arms["summary"].items():
            flat["f4_arm_%s" % key] = value
    if sweep is not None:
        for curve, row in sweep["curves"].items():
            gains = row["median_gain_pct_vs_ship"]
            flat["f4_sweep_%s_oracle" % curve] = gains["oracle"]
            flat["f4_sweep_%s_levelfix105" % curve] = gains["levelfix1.05"]
            flat["f4_sweep_%s_rankedprice" % curve] = gains["rankedprice"]
    if channel is not None and "summary" in channel:
        for key, value in channel["summary"].items():
            if isinstance(value, (int, float)):
                flat["f5_%s" % key] = value

    if strata is not None:
        flat["f7_pass_price_f_us"] = strata["f_hat_us"]
        flat["f7_pass_price_se_us"] = strata["f_se_us"]
        flat["f7_pass_price_ci95_lo_us"] = strata["f_ci95_us"][0]
        flat["f7_pass_price_ci95_hi_us"] = strata["f_ci95_us"][1]
        for key, value in strata["coverage"].items():
            flat["f7_coverage_%s" % key] = value
        for name, row in strata["joint_full_range"].items():
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    flat["f7_joint_%s_%s" % (name, key)] = value
        for name, row in strata["per_stratum_hinge"].items():
            tag = name.replace(",", "")
            for key in ("star", "rmse"):
                if key in row:
                    flat["f7_stratum_%s_%s" % (tag, key)] = row[key]
    if onepass is not None:
        for table, row in onepass["tables"].items():
            tag = table.replace("{", "").replace("}", "").replace(
                ":", "").replace(",", "_")
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    flat["f7_onepass_%s_%s" % (tag, key)] = value
    if head is not None:
        for name, row in head.items():
            tag = name.replace(" ", "_").replace("/", "_").replace("%", "pct")
            for key, value in row.items():
                flat["f7_head_%s_%s" % (tag, key)] = value
    if state is not None:
        s4 = state["section4"]
        flat["f8_six_row_misfit"] = s4["six_row"]["misfit"]
        for sid, row in s4["six_row"]["rows"].items():
            for key, value in row.items():
                flat["f8_six_%s_%s" % (sid, key)] = value
        if s4.get("ours_family"):
            for key, value in s4["ours_family"].items():
                flat["f8_ours_family_%s" % key] = value
        flat["f8_als_relative_misfit"] = s4["als_rel_rmse"]
        flat["f8_hinge_star"] = s4["star"]
        for row in s4["families"]:
            tag = row["family"]
            for key in ("rows", "misfit", "gap_k2_us", "gap_k3_us",
                        "f76_agreement"):
                flat["f8_family%d_%s" % (tag, key)] = row[key]
        for name, row in state["section3"].items():
            for value, label in zip(row["beta"], row["names"]):
                flat["f8_board_%s_%s" % (name, label)] = value
            flat["f8_board_%s_rmse" % name] = row["rmse"]
            flat["f8_board_%s_aicc" % name] = row["aicc"]
        for name, row in state["section2"].items():
            tag = name.replace("+ ", "").replace(" ", "_").replace("*", "")
            flat["f8_ours_%s_break" % tag] = row["best"]["breakpoint"]
            flat["f8_ours_%s_break_rmse" % tag] = row["best"]["rmse"]
            flat["f8_ours_%s_break_bic" % tag] = row["best"]["bic"]
            flat["f8_ours_%s_line_bic" % tag] = row["line"]["bic"]
    if f9 is not None:
        flat["f9_state_us"] = f9["state_us"]
        flat["f9_best_break_known_s"] = f9["section1"]["best_break_known_s"]
        for name, row in f9["section1"]["fits"].items():
            tag = name.replace(" ", "_")
            flat["f9_ours_%s_rmse" % tag] = row["rmse"]
            flat["f9_ours_%s_bic" % tag] = row["bic"]
        for offs, rows in f9["section2"]["our_points"].items():
            otag = offs.replace(" ", "_")
            for name, rmse, bic, params in rows:
                tag = name.replace(" ", "_").replace(">=", "ge").replace(
                    "/", "_").replace("+", "plus")
                flat["f9_shape_%s_%s_rmse" % (otag, tag)] = rmse
                flat["f9_shape_%s_%s_bic" % (otag, tag)] = bic
                flat["f9_shape_%s_%s_params" % (otag, tag)] = params
        for m, value in enumerate(f9["section2"]["u_ours"], start=1):
            flat["f9_u_ours_m%d" % m] = value
        for m, value in enumerate(f9["section2"]["u_board"], start=1):
            flat["f9_u_board_m%d" % m] = value
        if f9.get("section2b"):
            for offs, name, rmse, aicc in f9["section2b"]["board"]:
                tag = "%s_%s" % (offs.replace(" ", "_"),
                                 name.replace(" ", "_").replace("+", "plus")
                                 .replace(".", "p"))
                flat["f9_board_%s_rmse" % tag] = rmse
                flat["f9_board_%s_aicc" % tag] = aicc
        s4 = f9["section4"]
        flat["f9_s_u_price_units"] = s4["s_u"]
        flat["f9_flat_argmax_s0"] = s4["flat_argmax"]["s0"]
        flat["f9_flat_argmax_s_known"] = s4["flat_argmax"]["s_known"]
        for key, value in s4["median"].items():
            flat["f9_median_%s" % key] = value
        flat["f9_median_recoverable"] = s4["median"]["ds"] - s4["median"]["d0"]
        for prompt, row in s4["per_prompt"].items():
            for key in ("phi", "d0", "ds", "dship"):
                flat["f9_%s_%s" % (prompt, key)] = row[key]
        for prompt, row in s4["critical_s"].items():
            for key in ("up", "down"):
                if row[key] is not None:
                    flat["f9_%s_critical_s_%s" % (prompt, key)] = row[key]
        for name, row in s4["cells"].items():
            tag = name.replace(" ", "_").replace(".", "p").replace("+", "plus")
            for s_true, value in row.items():
                flat["f9_cell_%s_s%s" % (tag, s_true.split(".")[0])] = value
        for name, row in f9["section4c"].items():
            tag = name.replace(" ", "_").replace(">=", "ge")
            flat["f9_curve_%s_moved" % tag] = row["moved"]
            flat["f9_curve_%s_delta_median" % tag] = row["delta_median"]
        worst = min(v["min_conf"] for v in f9["section4d"].values())
        flat["f9_min_margin_confidence"] = worst
        flat["f9_entry_gate_rounds"] = sum(v["rounds"]
                                           for v in f9["section4d"].values())

    run.log(flat)
    run.summary.update(flat)
    run.finish()
    print("followups: logged %d scalars" % len(flat))


def log_f10() -> None:
    """F10: the serial lottery, the median carriers, and the row-keyed shape."""
    f10 = load("f10-carrier-and-shape.json")
    if f10 is None:
        print("f10: missing artifact, skipping")
        return

    s0, s1, s2 = f10["section0"], f10["section1"], f10["section2"]
    s3, s3b, s4 = f10["section3"], f10["section3_board"], f10["section4"]

    run = start("e128-f10-carriers-and-row-keyed-shape", {
        "harness": "ranked",
        "gpu_seconds": 0,
        "rebased_onto": REBASE_SHA,
        "receipt_anchor": "d3c491b5",
        "board_rows": f10["n_rows"],
        "pass_price_source": "f7_board_measured_50.4us",
        "state_label_source": "f76_mode_index",
        "alphonse_state_file_present": False,
    })

    flat = {
        "f10_board_rows": f10["n_rows"],
        "f10_serial_run_sd_pct": s0["run_sd_pct"],
        "f10_serial_resid_sd_pct": s0["resid_sd_pct"],
        "f10_serial_resid_over_run": s0["resid_sd_pct"] / s0["run_sd_pct"],
        "f10_approx_mean_err_pct": s0["approx_mean_err_pct"],
        "f10_approx_sd_err_pct": s0["approx_sd_err_pct"],
        "f10_approx_max_abs_err_pct": s0["approx_max_abs_err_pct"],
        "f10_approx_exact_rows": s0["approx_exact_rows"],
    }
    for name, row in s0["serial"].items():
        flat["f10_serial_%s_sd_pct" % name] = row["sd_pct"]
        flat["f10_carrier_share_%s" % name] = s0["carrier_share"][name]
    for i, row in enumerate(s0["common_serial"]):
        flat["f10_rule100_%s_published" % row["id"]] = row["published"]
        flat["f10_rule100_%s_common" % row["id"]] = row["common"]
        flat["f10_rule100_%s_published_rank" % row["id"]] = i

    for arm, row in s1.items():
        tag = arm.replace(".", "p")
        for key in ("median_pct", "beagle_pct", "min4_pct",
                    "replay_mean_pct", "replay_sd_pct"):
            flat["f10_arm_%s_%s" % (tag, key)] = row[key]

    for name, row in s2["ranked"].items():
        for key in ("mean_M", "p_M_eq_8", "p_M_ge_6"):
            flat["f10_width_%s_%s" % (name, key)] = row[key]
    for name, row in s2["aggregates"].items():
        tag = name.split(" ")[0].replace("-", "_")
        for key in ("mean_M", "p_M_eq_8", "p_M_ge_6"):
            flat["f10_width_agg_%s_%s" % (tag, key)] = row[key]
    for name, row in s2["onepass_prices"].items():
        tag = (name.replace("{", "").replace("}", "").replace(":", "")
               .replace(",", "_").replace(" c=", "_c").replace(".", "p"))
        for key in ("ranked_median_pct", "ranked_beagle_pct",
                    "bench_frame_median_pct", "withdrawn_tier_break_pct"):
            flat["f10_onepass_%s_%s" % (tag, key)] = row[key]
        flat["f10_onepass_%s_f_lo_pct" % tag] = row["f_band_pct"][0]
        flat["f10_onepass_%s_f_hi_pct" % tag] = row["f_band_pct"][1]

    for offset, fits in s3.items():
        otag = offset.replace(" ", "").replace("=", "")
        for fit in fits:
            tag = (fit["name"].replace(" ", "_").replace(">=", "ge")
                   .replace("+", "plus"))
            flat["f10_ours_%s_%s_rmse" % (otag, tag)] = fit["rmse"]
            flat["f10_ours_%s_%s_bic" % (otag, tag)] = fit["bic"]
    for fit in s3b:
        otag = fit["offset"].replace(" ", "").replace("=", "")
        tag = (fit["name"].replace(" ", "_").replace(">=", "ge")
               .replace("+", "plus").replace(".", "p"))
        flat["f10_board_%s_%s_rmse" % (otag, tag)] = fit["rmse"]
        flat["f10_board_%s_%s_aicc" % (otag, tag)] = fit["aicc"]

    flat["f10_state_index_variance_explained"] = \
        s4["index_variance_explained"]
    flat["f10_state_largest_step_us"] = s4["span_us"]
    for i, centre in enumerate(s4["index_centres"]):
        flat["f10_state_mode%d_index_centre" % i] = centre
    for fit in s4["fits"]:
        tag = (fit["name"].replace(" ", "_").replace("+", "plus"))
        flat["f10_state_%s_rmse" % tag] = fit["rmse"]
        flat["f10_state_%s_aicc" % tag] = fit["aicc"]
        for nm, b, se in zip(fit["names"], fit["beta"], fit["se"]):
            flat["f10_state_%s_%s" % (tag, nm)] = b
            flat["f10_state_%s_%s_se" % (tag, nm)] = se

    run.log(flat)
    run.summary.update(flat)
    run.finish()
    print("f10: logged %d scalars" % len(flat))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    runs = {"rung0": log_rung0, "identity": log_identity,
            "rung1": log_rung1, "jensen": log_jensen,
            "ourcurve": log_ourcurve,
            "rung2": lambda: log_rung2("ours"),
            "rung2board": lambda: log_rung2("board"),
            "followups": log_followups,
            "f10": log_f10}
    for name, fn in runs.items():
        if args.only and name not in args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
