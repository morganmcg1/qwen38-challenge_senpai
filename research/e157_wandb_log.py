#!/usr/bin/env python3
"""E157 R0: publish the zero-GPU depth-policy analysis to W&B.

Three record groups, one run.

  attribution   Why the E150 anchor 75a21a4 landed at 3.64907017 instead of
                the 3.68278758 reference. harness=ranked, official scores.
  ranked law    The round-cost law recovered from published receipt rows, and
                the test of whether it identifies a drafted-row price at all.
                harness=ranked.
  local rule    The shipped depth policy, the locally measured price, and the
                depth the marginal rule chooses as a function of acceptance.
                harness=local, never an official or ranked score.

Usage:
  python3 research/e157_wandb_log.py --run-name e157-r0-depth-policy-analysis
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e157-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
BASE_SHA = "c1492927508f91972a57d043d919b612e0895108"
ANCHOR_ID8 = "75a21a47"
REFERENCE_ID8 = "0cf1637e"


def load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise SystemExit(f"missing artifact {path}")
    return json.loads(path.read_text())


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def log_attribution(run, doc: dict) -> None:
    rows = [
        (
            r["prompt"],
            r["edl_candidate"],
            r["edl_reference"],
            r["edl_delta"],
            r["cand_spt_pct_vs_ref"],
            r["serial_pct_vs_ref"],
            r["raw_pct_vs_ref"],
        )
        for r in doc["per_prompt"]
    ]
    run.log(
        {
            "e157/attribution/per_prompt": table(
                [
                    "prompt",
                    "edl_candidate",
                    "edl_reference",
                    "edl_delta",
                    "cand_spt_pct_vs_ref",
                    "serial_pct_vs_ref",
                    "raw_pct_vs_ref",
                ],
                rows,
            )
        }
    )
    run.summary.update(
        {
            "e157_e150_shortfall_attribution": doc["e157_e150_shortfall_attribution"],
            "e157_schedule_transfer_faithful": doc["e157_schedule_transfer_faithful"],
            "e157_schedule_identical_prompt_count": doc[
                "schedule_identical_prompt_count"
            ],
            "e157_max_abs_edl_delta": doc["max_abs_edl_delta"],
            "e157_published_delta_pct": doc["attribution"]["published_delta_pct"],
            "e157_serial_free_delta_pct": doc["attribution"]["serial_free_delta_pct"],
            "e157_serial_draw_share_pct": doc["attribution"]["serial_draw_share_pct"],
            "e157_candidate_leg_mean_pct": doc["attribution"]["candidate_leg_mean_pct"],
            "e157_anchor_official_score": doc["candidate"]["official_score"],
            "e157_reference_official_score": doc["reference"]["official_score"],
        }
    )


def log_ranked_law(run, doc: dict) -> None:
    head = doc["receipts"][0]
    rows = [
        (
            r["prompt"],
            r["rounds"],
            r["verify_width_mean"],
            r["tokens_per_round"],
            r["clean_us_per_round"],
            r["conditional_p_fitted"],
            r["h_marginal_at_run_width_C_quadratic"],
            r["d_star_C_quadratic"],
            r["d_star_A_linear"],
            r["d_star_shipped_flat_price"],
            r["modelled_gain_pct_C_quadratic"],
            r["modelled_gain_pct_A_linear"],
            r["raw_ratio"],
        )
        for r in head["per_prompt"]
    ]
    run.log(
        {
            "e157/ranked_law/per_prompt": table(
                [
                    "prompt",
                    "rounds_recovered",
                    "verify_width_mean",
                    "tokens_per_round",
                    "clean_us_per_round",
                    "conditional_p_fitted",
                    "h_marginal_law_C",
                    "d_star_law_C",
                    "d_star_law_A",
                    "d_star_shipped_flat_price",
                    "modelled_gain_pct_law_C",
                    "modelled_gain_pct_law_A",
                    "raw_ratio",
                ],
                rows,
            ),
            "e157/ranked_law/family": table(
                ["law", "parameters", "relative_rmse_pct", "all_slopes_positive", "beta_us"],
                [
                    (
                        name,
                        law["parameter_count"],
                        100.0 * law["relative_rmse"],
                        law["all_slopes_positive"],
                        json.dumps([round(b, 1) for b in law["beta_us"]]),
                    )
                    for name, law in head["law_family_comparison"].items()
                ],
            ),
            "e157/ranked_law/round_recovery_control": table(
                ["prompt", "recovered_tokens_per_round", "finding_281_tokens_per_round",
                 "relative_difference_pct"],
                [
                    (
                        name,
                        v["recovered_tokens_per_round"],
                        v["finding_281_tokens_per_round_crown"],
                        100.0 * v["relative_difference"],
                    )
                    for name, v in head["round_recovery_positive_control"][
                        "by_prompt"
                    ].items()
                ],
            ),
            "e157/ranked_law/marginal_us_by_width": table(
                ["verify_width", "marginal_us_law_C", "h_law_C"],
                [
                    (
                        int(w),
                        doc["verdict"]["e157_row_price_us"][
                            "law_C_quadratic_marginal_us_at_width"
                        ][w],
                        doc["verdict"]["e157_marginal_rule_breakeven"][
                            "law_C_h_marginal_by_width"
                        ][w],
                    )
                    for w in map(str, range(1, 9))
                ],
            ),
        }
    )
    run.summary.update(
        {
            "e157_ranked_round_fixed_us": head["laws"]["C_quadratic"]["beta_us"][0],
            "e157_ranked_law_A_rmse_pct": 100.0
            * head["laws"]["A_linear"]["relative_rmse"],
            "e157_ranked_law_C_rmse_pct": 100.0
            * head["laws"]["C_quadratic"]["relative_rmse"],
            "e157_ranked_law_E_tokens_rmse_pct": 100.0
            * head["law_family_comparison"]["E_tokens"]["relative_rmse"],
            "e157_round_recovery_worst_control_pct": 100.0
            * head["round_recovery_positive_control"]["worst_relative_difference"],
            "e157_width_versus_tokens_pearson": head["row_versus_token_collinearity"][
                "pearson_width_vs_tokens_per_round"
            ],
            "e157_ranked_row_price_identified": doc["verdict"]["e157_row_price_us"][
                "identified"
            ],
            "e157_local_over_ranked_row_price_at_width_7": doc["verdict"][
                "e157_row_price_us"
            ]["local_over_ranked_at_matched_width_7"],
        }
    )


def log_local_rule(run, doc: dict) -> None:
    positions = doc["e157_measured_acceptance_by_position"]
    run.log(
        {
            "e157/local/acceptance_by_position": table(
                ["position", "measured_acceptance"],
                [(int(k), v) for k, v in sorted(positions.items(), key=lambda kv: int(kv[0]))],
            ),
            "e157/local/rule_tracks_p": table(
                list(doc["e157_rule_tracks_p"][0].keys()),
                [list(r.values()) for r in doc["e157_rule_tracks_p"]],
            ),
        }
    )
    price = doc["e157_row_price_us"]
    run.summary.update(
        {
            "e157_e150_policy_overlap": "already_marginal_priced",
            "e157_local_fixed_term_a_us": price["fixed_term_a_us"],
            "e157_local_row_price_b_us": price["row_price_b_us"],
            "e157_local_rollback_us_per_rejected_draft": price[
                "rollback_us_per_rejected_draft"
            ],
            "e157_local_h_marginal_measured": price["h_marginal_measured"],
            "e157_local_h_shipped": price["h_shipped"],
            "e157_local_underprice_factor": price["underprice_factor"],
            "e157_local_mechanism_visible": doc["e157_local_fixture_power"][
                "mechanism_visible_locally"
            ],
            "e157_local_rounds_analysed": doc["n_rounds"],
        }
    )


def log_law_bracket(run, doc: dict) -> None:
    run.log(
        {
            "e157/law_bracket/per_prompt": table(
                [
                    "prompt",
                    "verify_width_mean",
                    "clean_us_per_round_lower_bound",
                    "clean_us_per_round_upper_bound",
                    "bracket_width_pct",
                    "miss_finding_286_pct",
                    "miss_finding_281_pct",
                    "miss_rule_166_pct",
                ],
                [
                    (
                        r["prompt"],
                        r["verify_width_mean"],
                        r["clean_us_per_round_lower_bound"],
                        r["clean_us_per_round_upper_bound"],
                        r["bracket_width_pct"],
                        100.0 * r["laws"]["finding_286_rows"]["best_case_relative_miss"],
                        100.0 * r["laws"]["finding_281_tokens"]["best_case_relative_miss"],
                        100.0 * r["laws"]["rule_166_from_local"]["best_case_relative_miss"],
                    )
                    for r in doc["per_prompt"]
                ],
            )
        }
    )
    transfer = doc["transfer"]
    run.log(
        {
            "e157/transfer/by_width": table(
                ["verify_width", "ranked_h", "round_total_local_over_ranked"],
                [
                    (
                        int(w),
                        transfer["ranked_h_by_width"][w],
                        transfer["round_total_local_over_ranked_by_width"][w],
                    )
                    for w in map(str, range(1, 10))
                ],
            )
        }
    )
    run.summary.update(
        {
            "e157_finding_286_worst_miss_pct": 100.0
            * doc["law_summary"]["finding_286_rows"]["worst_best_case_relative_miss"],
            "e157_finding_281_worst_miss_pct": 100.0
            * doc["law_summary"]["finding_281_tokens"]["worst_best_case_relative_miss"],
            "e157_rule_166_worst_miss_pct": 100.0
            * doc["law_summary"]["rule_166_from_local"]["worst_best_case_relative_miss"],
            "e157_transfer_local_h_marginal": transfer["local_h_marginal"],
            "e157_transfer_ranked_h_at_width_6": transfer["ranked_h_by_width"]["6"],
            "e157_transfer_ratio_m3_low": transfer[
                "measured_ratio_over_m_three_to_eight"
            ][0],
            "e157_transfer_ratio_m8_high": transfer[
                "measured_ratio_over_m_three_to_eight"
            ][1],
        }
    )


def log_width_sweep(run, doc: dict) -> None:
    run.log(
        {
            "e157/width_sweep/cells": table(
                ["id8", "verify_width", "rounds", "tokens_per_round",
                 "clean_us_per_round"],
                [
                    (
                        c["id8"],
                        c["width"],
                        c["rounds"],
                        c["tokens_per_round"],
                        c["clean_us_per_round"],
                    )
                    for c in doc["sweep_prompt_all_cells"]
                ],
            ),
            "e157/width_sweep/marginal_by_width": table(
                ["verify_width", "ranked_marginal_us"],
                [
                    (int(w), v)
                    for w, v in sorted(
                        doc["fits"]["sweep_prompt_width_quadratic"][
                            "marginal_us_by_width"
                        ].items(),
                        key=lambda kv: int(kv[0]),
                    )
                ],
            ),
        }
    )
    quadratic = doc["fits"]["sweep_prompt_width_quadratic"]
    linear = doc["fits"]["sweep_prompt_width_only"]
    run.summary.update(
        {
            "e157_sweep_cell_count": quadratic["n"],
            "e157_sweep_width_min": linear["width_range"][0],
            "e157_sweep_width_max": linear["width_range"][1],
            "e157_sweep_quadratic_curvature_us": quadratic["beta_us"][2],
            "e157_sweep_quadratic_curvature_stderr": quadratic["curvature_stderr"],
            "e157_sweep_quadratic_r_squared": quadratic["r_squared"],
            "e157_sweep_linear_marginal_us": linear["marginal_us_per_row"],
            "e157_sweep_linear_r_squared": linear["r_squared"],
            "e157_sweep_paired_marginal_us": doc["e157_row_price_us"][
                "marginal_us_per_row"
            ],
            "e157_sweep_paired_marginal_stderr": doc["e157_row_price_us"][
                "marginal_stderr"
            ],
        }
    )


def log_head_confound(run, doc: dict) -> None:
    """FINDING 295's head claim, tested against reports and receipts."""
    local = doc["local_legs_behind_the_local_row_law"]
    price = doc["measured_head_price"]
    strat = doc["ranked_sweep_head_stratification"]
    run.log(
        {
            "e157/head/local_legs": table(
                ["leg", "head_sha256", "head_bytes", "origin"],
                [
                    (leg, v["sha256"], v["bytes"], v["origin"])
                    for leg, v in sorted(local["legs"].items())
                ],
            ),
            "e157/head/ranked_sweep_cells_per_head": table(
                ["head_sha256_prefix", "cell_count", "mean_verify_width"],
                [
                    (h, n, strat["mean_width_by_head"][h])
                    for h, n in sorted(
                        strat["cells_per_head"].items(),
                        key=lambda kv: -kv[1],
                    )
                ],
            ),
            "e157/head/single_head_cells": table(
                ["id8", "verify_width", "rounds", "clean_us_per_round"],
                [
                    (
                        c["id8"],
                        c["width"],
                        c["rounds"],
                        c["clean_us_per_round"],
                    )
                    for c in strat["largest_head_cells"]
                ],
            ),
        }
    )
    clean = strat["single_head_linear"]
    fixed_effect = strat["head_fixed_effect_quadratic"]
    run.summary.update(
        {
            "e157_local_law_head_sha256": local["head_sha256"],
            "e157_local_law_head_bytes": local["head_bytes"],
            "e157_local_law_used_declared_head": local[
                "safetensors_bytes_match_manifest"
            ],
            "e157_local_law_used_pinned_bf16_head": local["is_pinned_bf16_head"],
            "e157_head_draft_step_slope_ratio_pinned_over_declared": price[
                "draft_step_slope_ratio_pinned_over_declared"
            ],
            "e157_head_bytes_ratio_pinned_over_declared": price[
                "head_bytes_ratio_pinned_over_declared"
            ],
            "e157_head_verify_slope_ratio_null_control": price[
                "verify_slope_ratio_pinned_over_declared"
            ],
            "e157_head_round_slope_inflation_if_pinned": price[
                "round_slope_inflation_pinned_over_declared"
            ],
            "e157_head_share_of_round_declared": price[
                "head_share_of_round_declared"
            ],
            "e157_head_share_of_round_pinned": price["head_share_of_round_pinned"],
            "e157_ranked_sweep_distinct_heads": strat["n_distinct_heads"],
            "e157_ranked_sweep_head_homogeneous": strat["head_homogeneous"],
            "e157_single_head_marginal_us": clean["marginal_us_per_row"],
            "e157_single_head_marginal_stderr": clean["marginal_stderr"],
            "e157_single_head_r_squared": clean["r_squared"],
            "e157_single_head_width_max": clean["width_range"][1],
            "e157_curvature_after_head_control_us": fixed_effect[
                "curvature_us_per_row_squared"
            ],
            "e157_curvature_after_head_control_stderr": fixed_effect[
                "curvature_stderr"
            ],
            "e157_curvature_after_head_control_sigma": fixed_effect[
                "curvature_sigma"
            ],
        }
    )


def log_depth_rule(run, doc: dict) -> None:
    """The head-clean ranked depth price and the circularity test."""
    circ = doc["circularity_test"]
    clean = doc["head_clean_price"]
    run.log(
        {
            "e157/depth_rule/per_prompt": table(
                [
                    "prompt",
                    "conditional_p",
                    "live_drafted_mean",
                    "d_star_head_clean",
                    "gain_pct_head_clean",
                    "d_star_curvature_2sigma",
                    "gain_pct_curvature_2sigma",
                    "d_star_shipped_flat",
                    "d_star_finding_286",
                    "depth_error_head_clean",
                ],
                [
                    (
                        r["prompt"],
                        r["conditional_p"],
                        r["live_drafted_mean"],
                        r["d_star_head_clean_ranked_linear"],
                        r["gain_pct_head_clean_ranked_linear"],
                        r["d_star_head_clean_ranked_curvature_2sigma"],
                        r["gain_pct_head_clean_ranked_curvature_2sigma"],
                        r["d_star_shipped_flat_0p18"],
                        r["d_star_finding_286"],
                        r["depth_error_head_clean"],
                    )
                    for r in doc["per_prompt"]
                ],
            )
        }
    )
    run.summary.update(
        {
            "e157_f286_single_divisor_reproduces_both": circ[
                "single_divisor_reproduces_both"
            ],
            "e157_f286_divisor_low": circ["divisor_interval"][0],
            "e157_f286_divisor_high": circ["divisor_interval"][1],
            "e157_f286_depth_price_h": circ["finding_286_depth_price_h"],
            "e157_local_depth_price_h": circ["local_depth_price_h"],
            "e157_f286_h_relative_gap_to_local": circ[
                "depth_price_relative_difference"
            ],
            "e157_rule166_interval_contains_divisor": circ[
                "rule_166_interval_contains_divisor"
            ],
            "e157_head_clean_ranked_row_price_us": clean["row_us"],
            "e157_head_clean_ranked_depth_price_h": clean["depth_price_h"],
            **doc["verdict"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="e157-r0-depth-policy-analysis")
    parser.add_argument("--resume-id", default=None)
    args = parser.parse_args()

    attribution = load(f"e157_attribution_{ANCHOR_ID8}.json")
    ranked = load("e157_ranked_row_law.json")
    local = load("e157_marginal_rule.json")
    bracket = load("e157_rule166_check.json")
    sweep = load("e157_ranked_width_sweep.json")
    confound = load("e157_head_confound.json")
    depth_rule = load("e157_ranked_depth_rule.json")

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        id=args.resume_id,
        resume="allow" if args.resume_id else None,
        name=args.run_name,
        job_type="analysis",
        tags=["e157", "r0", "depth-policy", "zero-gpu", "harness=ranked",
              "harness=local"],
        config={
            "experiment": "e157-draft-depth-is-two-rows-too-deep",
            "rung": "r0",
            "pr": 157,
            "base_sha": BASE_SHA,
            "analysis_commit": git_sha(),
            "anchor_receipt": ANCHOR_ID8,
            "reference_receipt": REFERENCE_ID8,
            "gpu_used": False,
            "per_prompt_metric_source": (
                "Yukon list endpoint /api/benchmarks/"
                "5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?all=true, "
                "officialMetrics.per_prompt, read only"
            ),
            "sign_convention": (
                "a positive d_star minus edl means the marginal rule wants to "
                "draft deeper than the schedule that ran"
            ),
        },
    )
    log_attribution(run, attribution)
    log_ranked_law(run, ranked)
    log_local_rule(run, local)
    log_law_bracket(run, bracket)
    log_width_sweep(run, sweep)
    log_head_confound(run, confound)
    log_depth_rule(run, depth_rule)
    print(run.url, run.id, sep="\n")
    run.finish()


if __name__ == "__main__":
    main()
