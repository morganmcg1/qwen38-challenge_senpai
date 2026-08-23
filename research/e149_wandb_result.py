"""Publish the E149 terminal result to Weights and Biases.

One run for the whole assignment: arm A (gated ABBA, the primary metric), arm
B (cancelled before any GPU work), arm C1 (SDPA query-split cost isolation)
and arm D (the ranked per-width curve from public receipts).

Frame labels follow Rule 144. Every arm A number is `harness=local`; every arm
D number is `harness=ranked` except the H-alt reference curve, which is a
local measurement expressed in ranked units through k = 2.1034.
"""

from __future__ import annotations

import json
import pathlib

import wandb

HERE = pathlib.Path(__file__).resolve().parent

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

PR_NUMBER = 149
BRANCH = "qwen-askeladd/e149-draft-readout-dispatch-reduction-r1"
BASE_SHA = "85b5812aa4649717d2d534d8c101775120aa56bd"
SESSION_COMMIT = "e79ccd981f6c32b437938cf26870e5d8f2e7d8e8"
WORKER_SHA = "55c04ac34e6fd4b9a5b9203919d99a9a3974ee9e23aae3912c0e0886ecc62ec6"

GATED_LEG_SD_PCT = 0.052
F235_US_PER_ROUND_PER_PCT = 515.2
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}

FILES = [
    "e149_armA_abba.sh", "e149_armA_report.py", "e149-armA.json",
    "e149_leaf_witness.sh", "e149-leaf-witness.json",
    "e149_rungD_curve.py", "e149-rungD.json",
    "e149_rungD_robust.py", "e149-rungD-robust.json",
    "e149-null-block-bar-anchor.json",
    "e149_c1_probe.sh", "e149_c1_report.py",
    "e149-c1-sdpa.json", "e149-c1-report.json",
    "e149_wandb_result.py",
]


def load(name: str) -> dict | None:
    path = HERE / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    a = load("e149-armA.json")
    d = load("e149-rungD.json")
    rb = load("e149-rungD-robust.json")
    bar = load("e149-null-block-bar-anchor.json")
    lw = load("e149-leaf-witness.json")
    c1 = load("e149-c1-report.json")
    c1raw = load("e149-c1-sdpa.json")

    dl = d["f6_deliverables"]
    mp = a["medpair_weighted"]

    metrics: dict[str, float | int] = {
        # ---- primary
        "e149_best_arm_local_pct": a["e149_leaf16_local_pct"],
        # ---- arm A, harness=local
        "e149_leaf16_local_pct": a["e149_leaf16_local_pct"],
        "e149_leaf16_local_pct_decode_frame":
            a["e149_leaf16_local_pct_decode_frame"],
        "e149_leaf16_round_cost_us": a["e149_leaf16_round_cost_us"],
        "e149_leaf16_round_cost_us_total_frame":
            a["e149_leaf16_round_cost_us_total_frame"],
        "e149_leaf16_divergences": a["e149_leaf16_divergences"],
        "e149_leaf16_sigma_decode_medpair":
            a["e149_leaf16_local_pct_decode_frame"] / (
                GATED_LEG_SD_PCT * (0.478 ** 2 + 0.522 ** 2) ** 0.5),
        "e149_leaf16_f235_face_pct_of_median": mp["f235_face_pct_of_median"],
        "e149_leaf16_accept_rate_delta_pp": mp["accept_rate_delta_pp"],
        "e149_leaf16_round_count_ratio": sum(
            MEDPAIR[p] * a["per_prompt"][p]["round_count_ratio"] for p in MEDPAIR),
        "e149_leaf16_entry_temp_spread_c": a["entry_temp_c_spread"],
        "e149_leaf16_entry_temp_min_c": a["entry_temp_c_min"],
        "e149_leaf16_entry_temp_max_c": a["entry_temp_c_max"],
        "e149_leaf16_n_timed_legs": a["n_timed_legs"],
        "e149_leaf16_leaves_width8": 12292,
        "e149_leaf16_probes_width8": 3073,
        "e149_leaf16_leaves_width16": 6146,
        "e149_leaf16_probes_width16": 1537,
        "e149_leaf16_probed_rows_width8": 24584,
        "e149_leaf16_probed_rows_width16": 24592,
        "e149_leaf16_fine_pass_row_growth_pct": 100.0 * (24592 - 24584) / 24584,
        "e149_leaf_override_positive_control_passed":
            float(lw["e149_leaf_override_positive_control_passed"]),
        # ---- arm B, cancelled before any GPU work
        "e149_armB_priced_us_per_round": 10.0,
        "e149_armB_priced_pct_of_median": 10.0 / F235_US_PER_ROUND_PER_PCT,
        "e149_armB_dispatches_removed_per_round": 1,
        # ---- arm D, harness=ranked
        "e149_rungD_method_adequate": float(dl["e149_rungD_method_adequate"]),
        "e149_rungD_stop_rule_fired": float(dl["e149_rungD_stop_rule_fired"]),
        "e149_rungD_marginal_5_to_6_us": dl["e149_rungD_marginal_5_to_6_us"],
        "e149_rungD_marginal_6_to_7_us": dl["e149_rungD_marginal_6_to_7_us"],
        "e149_rungD_marginal_7_to_8_us": dl["e149_rungD_marginal_7_to_8_us"],
        "e149_rungD_c1_plutarch_anchor_us":
            dl["e149_rungD_c1_plutarch_anchor_us"],
        "e149_rungD_empirical_reproducibility_floor_pct":
            dl["e149_rungD_empirical_reproducibility_floor_pct"],
        "e149_rungD_7226dc9a_decode_gap_pct":
            dl["e149_rungD_7226dc9a_decode_gap_pct"],
        "e149_rungD_positive_control_recovery_empirical":
            dl["e149_rungD_synthetic_recovery_verdict_correct"][
                "recovery_rate_empirical"],
        "e149_rungD_negative_control_recovery_empirical":
            dl["e149_rungD_negative_control_returns_replayed_answer"][
                "recovery_rate_empirical"],
        "e149_rungD_shape_preference_dchi2_tilt":
            rb["models"]["tilt"]["full"]["delta_chi2_null_minus_alt"],
        "e149_rungD_shape_preference_dchi2_replay":
            rb["models"]["replay"]["full"]["delta_chi2_null_minus_alt"],
        "e149_rungD_shape_preference_robust":
            float(rb["e149_rungD_shape_preference_robust"]),
        "e149_rungD_h_null_rms_pct":
            rb["models"]["tilt"]["full"]["h_null"]["rms_pct"],
        "e149_rungD_h_alt_rms_pct":
            rb["models"]["tilt"]["full"]["h_alt"]["rms_pct"],
        # ---- null block, bar-anchored split
        "e149_null_block_bar_anchored_count":
            bar["e149_null_block_bar_anchored_count"],
    }
    for width in (6, 7, 8):
        metrics[f"e149_rungD_width{width}_admissible_ranked"] = float(
            dl[f"e149_rungD_width{width}_admissible_ranked"])
        metrics[f"e149_rungD_width{width}_confidence"] = dl[
            f"e149_rungD_width{width}_confidence"]
    for width, value in dl["e149_rungD_ranked_curve_us"].items():
        metrics[f"e149_rungD_curve_us_w{width}"] = value
        metrics[f"e149_rungD_cost_per_token_w{width}"] = dl[
            "e149_rungD_cost_per_token"][width]
    for frame, rec in bar["by_frame"].items():
        metrics[f"e149_null_block_sd_all11_{frame}"] = rec["n11_all"]["sd"]
        metrics[f"e149_null_block_sd_nonbar7_{frame}"] = \
            rec["n7_no_bar_anchor"]["sd"]
        metrics[f"e149_null_block_sd_bar4_{frame}"] = \
            rec["n4_bar_anchor_only"]["sd"]
        metrics[f"e149_null_block_mde_2sd_all11_{frame}"] = \
            2.0 * rec["n11_all"]["sd"]
        metrics[f"e149_null_block_mde_2sd_nonbar7_{frame}"] = \
            2.0 * rec["n7_no_bar_anchor"]["sd"]

    ident = c1["identification"]
    metrics.update({
        "e149_sdpa_split_us_per_round": c1["e149_sdpa_split_us_per_round"],
        "e149_sdpa_split_us_per_round_stderr":
            c1["e149_sdpa_split_us_per_round_stderr"],
        "e149_sdpa_split_us_per_round_raw_upper_bound":
            c1["e149_sdpa_split_us_per_round_raw_upper_bound"],
        "e149_sdpa_split_us_per_round_tinyKV_corrected":
            c1["e149_sdpa_split_us_per_round_tinyKV_corrected"],
        "e149_sdpa_split_pct_of_median":
            c1["frames"]["frameB_per_drafting_round_ranked"]["streaming_pct_of_median"],
        "e149_c1_slope_us_per_key_per_layer":
            ident["pooled_slope_us_per_key_per_layer"],
        "e149_c1_slope_stderr": ident["pooled_slope_stderr"],
        "e149_c1_structural_intercept_us_per_layer":
            ident["pooled_intercept_us_per_layer"],
        "e149_c1_fit_resid_rms_us_per_layer":
            ident["pooled_resid_rms_us_per_layer"],
        "e149_c1_implied_effective_bandwidth_gb_per_s":
            ident["implied_effective_bandwidth_gb_per_s"],
        "e149_c1_implied_fraction_of_peak":
            ident["implied_fraction_of_m4_pro_peak"],
        "e149_c1_split_vs_fallback_max_abs":
            c1["exactness"]["split_vs_fallback_max_abs"],
        "e149_c1_split_vs_split_max_abs":
            c1["exactness"]["split_vs_split_max_abs"],
        "e149_c1_positive_control_min_abs":
            c1["exactness"]["positive_control_min_abs"],
        "e149_c1_palindrome_half_spread_pct":
            c1["noise"]["palindrome_half_spread_pct_mean"],
        "e149_c2_opens": float(c1["e149_c2_opens"]),
    })
    for fname, rec in c1["frames"].items():
        short = fname.split("_")[0]
        metrics[f"e149_c1_{short}_streaming_us"] = rec["streaming_us_per_round"]
        metrics[f"e149_c1_{short}_raw_us"] = rec["raw_us_per_round"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e149-terminal-arms-a-b-c-d",
        job_type="analysis",
        tags=["e149", "askeladd", "arm-a", "arm-c", "arm-d", "terminal"],
        config={
            "experiment": "E149",
            "pr_number": PR_NUMBER,
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "session_commit": SESSION_COMMIT,
            "worker_sha256": WORKER_SHA,
            "host": "Apple M4 Pro, Mac mini Mac16,11, 48 GB",
            "ranked_host": "m5-qwen38-27b-mtp (not this machine)",
            "decode_tokens": 512,
            "e149_leaf16_depth_price_arm": a["e149_leaf16_depth_price_arm"],
            "arm_a_estimator": a["estimator"],
            "arm_a_gate": a["gate"],
            "arm_a_gate_qualified": a["all_gate_qualified"],
            "arm_a_all_tokens_matched": a["all_tokens_matched"],
            "e149_armB_status": "cancelled_below_detection_floor",
            "e149_c1_headline_frame": c1["e149_sdpa_split_frame"],
            "e149_c1_width9_status": c1["e149_c1_width9_status"],
            "e149_c2_blocked_reason": c1["e149_c2_blocked_reason"],
            "e149_c1_gate_qualified_for_timing":
                c1["gate_qualified_for_timing"],
            "e149_c1_cool_gate_passed_real_gate":
                c1["cool_gate_passed_real_gate"],
            "e149_c1_kL_sampled": sorted({r["kL"] for r in c1raw["per_round"]}),
            "arm_d_primary_mass_model": dl["primary_mass_model"],
            "arm_d_rows_used": list(d["rows_used"]),
            "arm_d_rows_excluded": d["rows_excluded_with_reason"],
            "arm_d_admissible_set_not_published": dl[
                "e149_rungD_admissible_set"],
            "arm_d_stop_rule_reason": dl["e149_rungD_stop_rule_reason"],
            "e149_rungD_rule138_holds_on_g17s":
                dl["e149_rungD_rule138_holds_on_g17s"],
            "e149_rungD_7226dc9a_matches_e003a86d_decode":
                dl["e149_rungD_7226dc9a_matches_e003a86d_decode"],
            "e149_null_block_bar_anchored_rows":
                bar["e149_null_block_bar_anchored_rows"],
            "local_submit_substitute":
                "HD37: --local-submit cannot complete on 48 GiB; the standing "
                "substitute is a bare 512-token exactness leg",
            "scored_surface_change": "none, all arms are env-gated or analysis",
        },
    )
    run.summary.update(metrics)
    run.log(metrics)

    arm_a = wandb.Table(columns=[
        "prompt", "ranked_weight", "decode_pct", "total_pct", "sigma_decode",
        "decode_round_us_delta", "total_round_us_delta",
        "f235_face_pct_of_median", "accept_rate_shipped", "accept_rate_leaf16",
        "edl_shipped", "edl_leaf16", "rounds_shipped", "rounds_leaf16",
        "within_arm_spread_pct_shipped", "within_arm_spread_pct_leaf16",
        "rule147_detectable_us_per_round"])
    weight = {"beagle_a": 0.478, "essays_montaigne": 0.522, "benchfixture": 0.0}
    rule147 = {"beagle_a": 119.5, "essays_montaigne": 44.5,
               "benchfixture": float("nan")}
    for block in a["per_block"]:
        prompt = block["prompt"]
        arm_a.add_data(
            prompt, weight[prompt], block["decode_pct"], block["total_pct"],
            block["decode_pct"] / GATED_LEG_SD_PCT,
            block["decode_round_us_delta"], block["total_round_us_delta"],
            block["f235_face_pct_of_median"], block["accept_rate_shipped"],
            block["accept_rate_leaf16"], float(block["edl_shipped"][0]),
            float(block["edl_leaf16"][0]), block["round_count_shipped"][0],
            block["round_count_leaf16"][0],
            block["within_arm_spread_pct_shipped"],
            block["within_arm_spread_pct_leaf16"], rule147[prompt])
    run.log({"e149_armA_per_prompt": arm_a})

    hist = wandb.Table(columns=["verify_width", "shipped_rounds",
                                "leaf16_rounds", "identical"])
    for width in sorted(a["width_histogram"]["shipped"], key=int):
        shipped = a["width_histogram"]["shipped"][width]
        leaf16 = a["width_histogram"]["leaf16"][width]
        hist.add_data(int(width), shipped, leaf16, shipped == leaf16)
    run.log({"e149_armA_verify_width_histogram": hist})

    steps = wandb.Table(columns=[
        "step", "fitted_us", "replayed_us", "measured_over_k_us",
        "fitted_over_replayed", "fitted_over_measured",
        "replayed_over_measured"])
    for name, rec in d["results"]["tilt"]["step_marginals"].items():
        steps.add_data(
            name.replace("_to_", "->"), rec["fitted_us"], rec["replayed_us"],
            rec.get("measured_over_k_us"), rec["fitted_over_replayed"],
            rec.get("fitted_over_measured"), rec.get("replayed_over_measured"))
    run.log({"e149_rungD_step_marginals": steps})

    splice = wandb.Table(columns=["mass_model", "curve", "chi2", "rms_pct"])
    for model, rec in rb["models"].items():
        for name, cell in rec["high_width_block"].items():
            splice.add_data(model, name, cell["chi2"], cell["rms_pct"])
    run.log({"e149_rungD_high_width_splice": splice})

    lopo = wandb.Table(columns=[
        "mass_model", "dropped_prompt", "n_rows", "delta_chi2_null_minus_alt",
        "rms_null_pct", "rms_alt_pct", "prefers"])
    for model, rec in rb["models"].items():
        for prompt, cell in rec["leave_one_prompt_out"].items():
            lopo.add_data(model, prompt, cell["n_rows"],
                          cell["delta_chi2_null_minus_alt"],
                          cell["rms_null_pct"], cell["rms_alt_pct"],
                          cell["prefers"])
    run.log({"e149_rungD_leave_one_prompt_out": lopo})

    nulls = wandb.Table(columns=[
        "frame", "subset", "n", "sd_pp", "mde_2sd_pp", "abs_max_pp", "mean_pp"])
    for frame, rec in bar["by_frame"].items():
        for subset in ("n11_all", "n7_no_bar_anchor", "n4_bar_anchor_only"):
            cell = rec[subset]
            nulls.add_data(frame, subset, cell["n"], cell["sd"],
                           2.0 * cell["sd"], cell["absmax"], cell["mean"])
    run.log({"e149_null_block_bar_anchor_split": nulls})

    c1cells = wandb.Table(columns=[
        "kL", "width", "split_us_per_layer", "merged_forecast_us_per_layer",
        "fallback_us_per_layer", "raw_saving_us_per_layer",
        "raw_saving_us_per_round", "tinyKV_corrected_us_per_round",
        "scored_width"])
    for r in c1raw["per_round"]:
        c1cells.add_data(
            r["kL"], r["width"], r["split_us_per_layer"],
            r["merged_forecast_us_per_layer"], r["fallback_us_per_layer"],
            r["upper_bound_saving_us_per_layer"],
            r["upper_bound_saving_us_per_round"],
            r["gpu_attributable_us_per_round"], r["width"] in (6, 7, 8))
    run.log({"e149_c1_sdpa_split_cells": c1cells})

    c1frames = wandb.Table(columns=[
        "frame", "mass_applied", "raw_us_per_round", "streaming_us_per_round",
        "streaming_stderr_us", "tinyKV_us_per_round", "streaming_pct_of_median",
        "raw_pct_of_median"])
    for fname, rec in c1["frames"].items():
        c1frames.add_data(
            fname, rec["mass_applied"], rec["raw_us_per_round"],
            rec["streaming_us_per_round"], rec["streaming_us_per_round_stderr"],
            rec["tinyKV_us_per_round"], rec["streaming_pct_of_median"],
            rec["raw_pct_of_median"])
    run.log({"e149_c1_frames": c1frames})

    c1exact = wandb.Table(columns=[
        "kL", "width", "split_vs_split_max_abs", "split_vs_fallback_max_abs",
        "positive_control_max_abs"])
    for e in c1raw["exactness"]:
        c1exact.add_data(e["kL"], e["width"], e["split_vs_split_max_abs"],
                         e["split_vs_fallback_max_abs"],
                         e["positive_control_max_abs"])
    run.log({"e149_c1_exactness": c1exact})

    art = wandb.Artifact("e149-terminal", type="analysis")
    for name in FILES:
        path = HERE / name
        if path.exists():
            art.add_file(str(path))
    run.log_artifact(art)

    print(run.id)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
