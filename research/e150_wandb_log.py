#!/usr/bin/env python3
"""E150: publish the per-round discrimination rungs R0, R0.5 and R1/R2.

`harness=local`, `gpu_used=False`. Every rung here is an offline replay over
the E145 measured per-width round cost curve and a recorded acceptance trace.
No leg of this experiment holds the GPU, so nothing published here is a timing
contrast and CAMPAIGN RULE 79 is not engaged.

RULE 144 framing. The ranked frames named by Rule 144 are realised median
pair, unweighted mean of the weighted five, and F83 marginal-weight sum. This
experiment reports in none of them: its frame is the offline replay median
over the eight fixture prompts of the per-prompt mean seconds-per-token ratio
against the shipped schedule. That frame is named on the run as
`replay_median_pct_frame` so no reader can compare it to a ranked number by
accident. Transfer to the board is priced against the lead-frame single
receipt MDE of 0.1547 pp as well as this experiment's own replay noise floor.

Usage:
  python3 e150_wandb_log.py --run-name e150-per-round-discrimination
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wandb  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ARTIFACTS = HERE / "e150-artifacts"

BASE_SHA = "601c137c7fa0291965b59b492615cf402bae82b6"

# Rule 144, lead frame (realised median pair), n = 11 null block, 2 sigma.
# This is a RANKED number and is used only to price transfer, never to judge
# the spread of a replay arm.
RANKED_LEAD_FRAME_MDE_PP = 0.1547

# The published cells every rung is anchored to.
R7_SHIPPED_PCT = 0.1338226619729229
R7_ORACLE_DEPTH_PCT = 6.3508
R1_GATE_PCT = 0.4338


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def log_r0(run, summary: dict, r0: dict) -> None:
    """R0: all 36 E128 arms repriced on the measured curve."""
    for key in ("e150_e128_arms_priced", "e150_e128_best_measured_arm",
                "e150_e128_best_measured_pct",
                "e150_e128_rankings_preserved_frac",
                "e150_e128_rankings_preserved_vs_local_replay_frac",
                "e150_e128_shipped_reference_pct"):
        summary[key] = r0[key]
    summary["e150_e128_arms_beating_shipped"] = \
        ", ".join(r0["e150_e128_arms_beating_shipped"]) or "none"
    summary["e150_e128_best_over_shipped_pp"] = (
        r0["e150_e128_best_measured_pct"]
        - r0["e150_e128_shipped_reference_pct"])

    run.log({"r0_e128_reprice": table(
        ["arm", "e128_published_replayed_pct", "replayed_pct", "measured_pct",
         "measured_pct_sd", "delta_measured_minus_replayed_pp",
         "measured_mean_depth", "measured_width1_share",
         "measured_accept_rate", "implementable"],
        [[row["arm"], row.get("e128_published_replayed_pct"),
          row.get("replayed_pct"), row.get("measured_pct"),
          row.get("measured_pct_sd"),
          row.get("delta_measured_minus_replayed_pp"),
          row.get("measured_mean_depth"), row.get("measured_width1_share"),
          row.get("measured_accept_rate"), row.get("implementable")]
         for row in r0["e150_e128_reprice_table"]])})


def log_r05(run, summary: dict, r05: dict) -> None:
    """R0.5: is the shipped per-round ratio rule the right optimality form?"""
    for key in ("e150_lambda_star",
                "e150_argmax_linearised_pct",
                "e150_argmax_linearised_pct_noclamp",
                "e150_argmax_linearised_pct_clairvoyant",
                "e150_argmax_linearised_pct_clairvoyant_noclamp",
                "e150_policy_form_gain_pp",
                "e150_policy_form_gain_pp_noclamp",
                "e150_policy_form_gain_pp_lopo",
                "e150_policy_form_gain_pp_clairvoyant",
                "e150_policy_form_gain_pp_width5_capped",
                "e150_gain_survives_width_cap",
                "e150_policy_form_recovers_clamp_cost_pp",
                "e150_published_oracle_is_not_the_ceiling",
                "e150_revised_headroom_axis_pp",
                "e150_max_frac_rounds_inadmissible",
                "e150_r05_controls_all_reproduced",
                "e150_r05_lopo_minus_pooled_pp",
                "e150_r05_verdict", "e150_noise_floor_pp"):
        if key in r05:
            summary[key] = r05[key]

    solved = r05.get("e150_lambda_star_solution", {})
    for src, dst in (
            ("mu_star_cost_per_token_normalised", "e150_mu_star"),
            ("lambda_star_tokens_per_us", "e150_lambda_star_tokens_per_us"),
            ("serial_round_us", "e150_r05_serial_round_us"),
            ("bisection_steps", "e150_r05_bisection_steps"),
            ("converged", "e150_r05_bisection_converged")):
        if src in solved:
            summary[dst] = solved[src]

    for name, ok in r05["e150_r05_control_checks"].items():
        summary["e150_r05_check_" + name] = ok

    # The gain is only decision-relevant if it would also be visible on a
    # single ranked receipt in the lead frame.
    for label, key in (("noclamp", "e150_policy_form_gain_pp_noclamp"),
                       ("width5_capped",
                        "e150_policy_form_gain_pp_width5_capped")):
        if key in r05:
            summary["e150_r05_%s_over_ranked_mde" % label] = (
                r05[key] / RANKED_LEAD_FRAME_MDE_PP)

    run.log({"r05_policy_form": table(
        ["cell", "median_pct", "median_pct_sd", "mean_depth", "accept_rate",
         "width1_share", "frac_rounds_inadmissible", "mu_used"],
        [[name, row["median_pct"], row["median_pct_sd"],
          row["weighted_mean_depth"], row["weighted_accept_rate"],
          row["width1_share"], row["frac_rounds_inadmissible"],
          row["mu_used"]]
         for name, row in sorted(r05["e150_r05_table"].items())])})

    run.log({"r05_mu_sensitivity": table(
        ["mu", "median_pct", "mean_depth", "frac_rounds_inadmissible"],
        [[row["mu"], row["median_pct"], row["weighted_mean_depth"],
          row["frac_rounds_inadmissible"]]
         for row in r05["e150_r05_mu_sensitivity"]])})

    # The sweep that separates a genuine policy-form gain from mere access to
    # the width-8 cell, which Rule 143 discounts on transfer anyway.
    run.log({"r05_width_cap_sweep": table(
        ["depth_cap", "max_width", "clamp", "linearised_pct",
         "shipped_rule_same_cap_pct", "policy_form_gain_pp", "width8_share"],
        [[row["depth_cap"], row["max_width"], row["clamp"],
          row["linearised_pct"], row["shipped_rule_same_cap_pct"],
          row["policy_form_gain_pp"],
          row.get("width_histogram", {}).get("8", 0.0)]
         for row in r05["e150_r05_width_cap_sweep"]])})

    lopo = r05.get("e150_r05_lopo_linearised_noclamp") or {}
    fold_mu = lopo.get("fold_mu_star") or {}
    if fold_mu:
        summary["e150_r05_lopo_mu_star_min"] = min(fold_mu.values())
        summary["e150_r05_lopo_mu_star_max"] = max(fold_mu.values())
        run.log({"r05_lopo_mu": table(
            ["held_out_prompt", "mu_star"],
            sorted(fold_mu.items()))})


def log_r1(run, summary: dict, r1: dict) -> None:
    """R1 and R2: the predictability ceiling and the information ladder."""
    for key in ("e150_capturable_pp_at_measured_sigma",
                "e150_capturable_over_shipped_pp",
                "e150_primary_arm",
                "e150_pre_draft_predictor_sigma",
                "e150_pre_draft_predictor_reachable_sigma",
                "e150_pre_draft_predictor_scalar_k_sd",
                "e150_pre_draft_predictor_scalar_k_bias",
                "e150_predictor_sigma_in_sample",
                "e150_predictor_overfit_gap_pp",
                "e150_predictor_overfit_sigma_gap",
                "e150_positive_control_truth_pp",
                "e150_positive_control_truth_clamped_pp",
                "e150_positive_control_constant_pp",
                "e150_positive_control_constant_clamped_pp",
                "e150_control_truth_sigma",
                "e150_controls_all_reproduced",
                "e150_error_distribution_is_gaussian",
                "e150_iid_gaussian_equivalent_pct",
                "e150_iid_gaussian_reachable_equivalent_pct",
                "e150_displacement_equivalent_pct",
                "e150_displacement_wrong_unit_equivalent_pct",
                "e150_iid_ladder_overstates_by_pp",
                "e150_iid_reachable_ladder_overstates_by_pp",
                "e150_displacement_ladder_overstates_by_pp",
                "e150_ratio_over_argmax_pp",
                "e150_ratio_truth_ceiling_pp",
                "e150_ratio_truth_over_argmax_truth_pp",
                "e150_best_constant_lambda",
                "e150_best_constant_lambda_pct",
                "e150_ratio_adaptivity_premium_pp",
                "e150_shipped_information_ratio_pp",
                "e150_rule_only_gain_pp",
                "e150_information_gain_at_fixed_rule_pp",
                "e150_tempered_median_pct",
                "e150_tempered_over_untempered_pp",
                "e150_r1_gate_pct", "e150_r1_gate_cleared",
                "e150_realised_headroom_axis_pp",
                "e150_axis_captured_frac",
                "e150_sequential_information_premium_pp"):
        if key in r1:
            summary[key] = r1[key]

    for name, ok in r1.get("e150_control_checks", {}).items():
        summary["e150_r1_check_" + name] = ok

    primary = r1["e150_capturable_pp_at_measured_sigma"]
    summary["e150_capturable_over_ranked_mde"] = (
        (primary - R7_SHIPPED_PCT) / RANKED_LEAD_FRAME_MDE_PP)
    # The advisor's revised F1 brackets, recorded as a single readable label
    # so the verdict cannot drift from the number.
    if primary < R1_GATE_PCT:
        band = "below +0.4338: axis closed for pre-draft information"
    elif primary < 1.5:
        band = "+0.4338 to +1.5: gap is in the predictor, not the axis"
    elif primary <= 2.5:
        band = "+1.5 to +2.5: at the published norm"
    else:
        band = "above +2.5: better than published, re-examine controls"
    summary["e150_r1_verdict_band"] = band

    run.log({"r1_arms": table(
        ["arm", "median_pct", "median_pct_sd", "mean_depth", "accept_rate",
         "width1_share", "frac_rounds_inadmissible"],
        [[name, row.get("median_pct"), row.get("median_pct_sd"),
          row.get("weighted_mean_depth"), row.get("weighted_accept_rate"),
          row.get("width1_share"), row.get("frac_rounds_inadmissible")]
         for name, row in sorted(r1["e150_predictor_arms"].items())])})

    run.log({"r2_information_ladder": table(
        ["rung", "information", "median_pct", "sigma_out_of_sample",
         "reachable_sigma", "scalar_k_sd", "mean_depth"],
        [[row.get("rung"), row.get("information"), row.get("median_pct"),
          row.get("sigma"), row.get("reachable_sigma"),
          row.get("scalar_k_sd"), row.get("mean_depth")]
         for row in r1["e150_information_ladder_detail"]])})

    # Three ladders in three different error units. Publishing them together
    # is the point: reading the R7-3 ladder in the wrong unit is the single
    # largest error available to this rung.
    run.log({"r1_noise_ladder": table(
        ["sigma", "median_pct", "unit"],
        [[row["sigma"], row["median_pct"], row.get("unit", "all_position")]
         for row in r1["e150_r73_noise_ladder_reindexed"]])})
    run.log({"r1_displacement_ladder": table(
        ["shift", "median_pct", "unit"],
        [[row.get("shift"), row.get("median_pct"),
          row.get("unit", "scalar_k")]
         for row in r1["e150_displacement_ladder"]])})

    reads = r1.get("e150_ladder_conversion_reads", {})
    run.log({"r1_ladder_conversion_reads": table(
        ["read", "implied_pct"],
        [[k, v] for k, v in sorted(reads.items())])})

    run.log({"r1_constant_lambda": table(
        ["fixed_lambda", "median_pct", "mean_depth"],
        [[row["fixed_lambda"], row["median_pct"], row.get("mean_depth")]
         for row in r1["e150_constant_lambda_control"]])})

    dist = r1.get("e150_error_distribution", {})
    for key in ("mean", "sd", "skew", "kurtosis", "q05", "q50", "q95",
                "shapiro_like", "is_gaussian"):
        if key in dist:
            summary["e150_error_dist_" + key] = dist[key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="e150-per-round-discrimination")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    r0 = load("r0.json")
    r05 = load("r05.json")
    r1 = load("r1.json")
    if r0 is None and r05 is None and r1 is None:
        print("no E150 artifacts found under %s" % ARTIFACTS)
        return 1

    anchor = r0 or r05 or r1
    config = {
        "experiment": "E150",
        "rungs_present": [n for n, v in (("R0", r0), ("R0.5", r05),
                                         ("R1/R2", r1)) if v is not None],
        "harness": "local",
        "gpu_used": False,
        "frame": "decode",
        "replay_median_pct_frame": (
            "offline replay median over 8 fixture prompts of the per-prompt "
            "mean seconds-per-token ratio against the shipped schedule"),
        "rule_144_ranked_frame_used_for_transfer_only": (
            "realised median pair"),
        "ranked_lead_frame_mde_pp": RANKED_LEAD_FRAME_MDE_PP,
        "base_sha": BASE_SHA,
        "commit": git_sha(),
        "cost_curve": "e145_r2_live_per_drafting_round measured",
        "receipt": anchor.get("receipt"),
        "receipt_score": anchor.get("receipt_score"),
        "windows": anchor.get("windows"),
        "seeds": anchor.get("seeds"),
        "shipped_reference_pct": R7_SHIPPED_PCT,
        "published_oracle_pct": R7_ORACLE_DEPTH_PCT,
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, name=args.run_name,
                     notes=args.notes, config=config,
                     tags=["e150", "per-round-discrimination", "offline",
                           "no-gpu", "harness-local"])

    summary: dict = {}
    if r0 is not None:
        log_r0(run, summary, r0)
    if r05 is not None:
        log_r05(run, summary, r05)
    if r1 is not None:
        log_r1(run, summary, r1)

    run.summary.update(summary)
    print("run id   %s" % run.id)
    print("run url  %s" % run.url)
    for key in sorted(summary):
        print("  %-56s %s" % (key, summary[key]))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
