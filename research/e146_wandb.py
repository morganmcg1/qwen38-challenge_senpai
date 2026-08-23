"""E146: publish the census, the power analysis and the local floor to W&B.

Every scalar is prefixed with the harness it belongs to. `ranked/` values come
from published receipts and price an official submission. `local/` values come
from the R-C fixed-binary session on this host and price a local iterate. They
are never combined.
"""

import json
import os
import subprocess
import sys

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT).decode().strip()


def main():
    modes = load("e146-modes.json")
    classify = load("e146-classify.json")
    power = load("e146-power.json")
    pairs = load("e146-pairs.json")
    family = load("e146-family.json")
    cohort = load("e146-cohort.json")
    prefill = load("e146-prefill.json")
    ledger305 = load("e146-ledger305.json")
    for name, payload in (("e146-modes.json", modes), ("e146-classify.json", classify),
                          ("e146-power.json", power), ("e146-pairs.json", pairs),
                          ("e146-family.json", family), ("e146-cohort.json", cohort),
                          ("e146-prefill.json", prefill),
                          ("e146-ledger305.json", ledger305)):
        if payload is None:
            print("missing %s" % name)
            return 1

    ranked = power["ranked"]
    local = power["local"]
    head = classify["headline"]
    ours = cohort["our_row"]
    aucs = {a["feature"]: a for a in cohort["auc_in_sample"]}
    aucs_out = {a["feature"]: a for a in cohort["auc_out_of_sample"]}

    offschedule = load("e146-offschedule.json")

    resume_id = os.environ.get("E146_WANDB_RUN_ID") or None
    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        id=resume_id, resume="must" if resume_id else None,
        name="e146-nuisance-floor-census",
        job_type="measurement",
        tags=["e146", "nuisance", "instrument", "ranked", "local"],
        config={
            "experiment": "E146",
            "pr": 146,
            "base_sha": "d983cb5a992498411f385e4acca1472fc0f0fe37",
            "commit": git("rev-parse", "HEAD"),
            "board_path": modes["board_path"],
            "board_rows_scored": cohort["board_rows_scored"],
            "pure_nuisance_pairs": modes["pure_nuisance_pairs"],
            "step_cut_us": 300.0,
            "headline_anchor": "572b2cc4",
            "headline_row": "1db9d63e",
            "local_prompt": "beagle_a",
            "local_tokens": 512,
            "local_offered_depth": 8,
            "local_cool_gate": "real 40C gate, benchmark.sh --local-cool-gate-only",
        },
    )

    summary = {
        # --- the headline the assignment asked for ------------------------
        "ranked/e146_1db9d63e_mode_corrected_candidate_mean_pct": head["value_pct"],
        "ranked/e146_1db9d63e_mode_corrected_candidate_mean_decode_pct":
            head["value_decode_pct"],
        "ranked/e146_1db9d63e_raw_candidate_mean_pct": head["raw_candidate_mean8_pct"],
        "ranked/e146_1db9d63e_k_us_per_drafting_round": head["k_us_per_drafting_round"],
        "ranked/e146_1db9d63e_row_steps_above_family_flat":
            ours["row_steps_above_family_flat"],
        "ranked/e146_1db9d63e_anchor_steps_above_family_flat":
            ours["anchor_steps_above_family_flat"],
        "ranked/e146_1db9d63e_bar_frame_decode_corrected_pct":
            ours["bar_frame_decode_corrected_pct"],
        "ranked/e146_replay_of_advisor_step_decode_pct":
            ledger305["e146_replay_of_advisor_step_decode_pct"],
        "ranked/e146_advisor_agreement_pp":
            ledger305["e146_advisor_agreement_pp"],
        "ranked/e146_headline_step_sensitivity_total_low_pct":
            ledger305["e146_headline_step_sensitivity_total_low_pct"],
        "ranked/e146_headline_step_sensitivity_total_high_pct":
            ledger305["e146_headline_step_sensitivity_total_high_pct"],
        # --- F2 reproduction: the bar cohort ------------------------------
        "ranked/e146_cohort_size": cohort["cohort_size"],
        "ranked/e146_cohort_n_high": cohort["n_high"],
        "ranked/e146_cohort_n_low": cohort["n_low"],
        "ranked/e146_cohort_p_high": cohort["p_high"],
        "ranked/e146_residual_inflation_high_over_main": cohort["residual_inflation"],
        "ranked/e146_step_us_zero_delta_only": cohort["step_us_zero_delta_only"],
        "ranked/e146_step_se_us": cohort["step_se_us"],
        "ranked/e146_step_pct_of_total_mean8": cohort["step_pct_of_total_mean8"],
        "ranked/e146_step_pct_of_decode_mean8": cohort["step_pct_of_decode_mean8"],
        "ranked/e146_f220_survives_state_correction":
            cohort["e146_f220_survives_state_correction"],
        # --- F2 stop rule (a): does the residual separate the modes? ------
        "ranked/e146_auc_residual_in_sample": aucs["fit_residual_sd_pp"]["auc"],
        "ranked/e146_auc_residual_in_sample_p":
            aucs["fit_residual_sd_pp"]["p_value_one_sided"],
        "ranked/e146_auc_residual_in_sample_null_p95":
            aucs["fit_residual_sd_pp"]["null_p95"],
        # The advisor's requested name for the out-of-sample residual AUC.
        "ranked/e146_classifier_auc_on_null_pairs":
            aucs_out["fit_residual_sd_pp"]["auc"],
        "ranked/e146_auc_residual_out_of_sample":
            aucs_out["fit_residual_sd_pp"]["auc"],
        "ranked/e146_auc_residual_out_of_sample_p":
            aucs_out["fit_residual_sd_pp"]["p_value_one_sided"],
        "ranked/e146_auc_residual_out_of_sample_null_p95":
            aucs_out["fit_residual_sd_pp"]["null_p95"],
        "ranked/e146_auc_k_out_of_sample":
            aucs_out["k_us_per_drafting_round"]["auc"],
        "ranked/e146_auc_prefill_negative_control":
            aucs["NEGATIVE CONTROL prefill_mean8_pct"]["auc"],
        "ranked/e146_auc_abs_prefill_negative_control":
            aucs["NEGATIVE CONTROL abs prefill_mean8_pct"]["auc"],
        # --- F227 / F228: prefill is charged, but the state does not touch it
        "ranked/e146_prefill_prompt_invariance_cv_pct":
            prefill["e146_prefill_prompt_invariance_cv_pct"],
        "ranked/e146_prefill_implied_seconds_mean":
            prefill["e146_prefill_implied_seconds_mean"],
        "ranked/e146_prefill_delta_vs_advisor_seconds":
            prefill["e146_prefill_delta_vs_advisor_seconds"],
        "ranked/e146_prefill_share_of_candidate_leg_pct":
            prefill["e146_prefill_share_of_candidate_leg_pct"],
        "ranked/e146_prefill_median_within_row_cv_pct":
            prefill["within_row_cv"]["prefill"]["median_within_row_cv_pct"],
        "ranked/e146_published_leg_precision_total_sd_pct":
            ranked["published_leg_precision"]["total"]["sd_pct"],
        "ranked/e146_published_leg_precision_total_flat_sd_pct":
            ranked["published_leg_precision"]["total"]["flat_sd_pct"],
        "ranked/e146_published_leg_precision_decode_sd_pct":
            ranked["published_leg_precision"]["decode"]["sd_pct"],
        "ranked/e146_published_leg_precision_decode_flat_sd_pct":
            ranked["published_leg_precision"]["decode"]["flat_sd_pct"],
        "ranked/e146_published_leg_precision_serial_sd_pct":
            ranked["published_leg_precision"]["serial"]["sd_pct"],
        "ranked/e146_published_leg_precision_prefill_sd_pct":
            ranked["published_leg_precision"]["prefill"]["sd_pct"],
        "ranked/e146_published_leg_precision_prefill_flat_sd_pct":
            ranked["published_leg_precision"]["prefill"]["flat_sd_pct"],
        # --- R-A ----------------------------------------------------------
        "ranked/e146_p_high_from_replicates": modes["e146_p_high_from_replicates"],
        "ranked/e146_p_high_ci95_low": modes["p_high_ci95"][0],
        "ranked/e146_p_high_ci95_high": modes["p_high_ci95"][1],
        "ranked/e146_pure_nuisance_pairs": modes["pure_nuisance_pairs"],
        "ranked/e146_step_us_per_drafting_round": modes["step_us_per_drafting_round"],
        "ranked/e146_step_ci95_low": modes["step_ci95"][0],
        "ranked/e146_step_ci95_high": modes["step_ci95"][1],
        "ranked/e146_crown_family_step_us": modes["crown_family_step_us"],
        "ranked/e146_flat_band_sd_us": modes["flat_band_sd_us"],
        "ranked/e146_dip_p_value": modes["dip_p_value"],
        "ranked/e146_gap_bootstrap_p": modes["gap_above_zero_bootstrap_p"],
        "ranked/e146_gap_above_zero_us": modes["gap_above_zero"]["width"],
        "ranked/e146_bic_1_component": modes["mixture_bic"][0]["bic"],
        "ranked/e146_bic_2_components": modes["mixture_bic"][1]["bic"],
        "ranked/e146_bic_3_components": modes["mixture_bic"][2]["bic"],
        "ranked/e146_transitivity_residual_rms_us":
            modes["transitivity"]["residual_rms_us"],
        "ranked/e146_transitivity_null_rms_us":
            modes["transitivity"]["null_residual_rms_us"],
        "ranked/e146_edges_near_allowed_value":
            modes["transitivity"]["edges_near_allowed_value"],
        "ranked/e146_transitivity_edges": modes["transitivity"]["edges"],
        "ranked/e146_candidate_to_serial_move_ratio":
            modes["leg_response"]["candidate_to_serial_ratio"],
        "ranked/e146_shape_r2_vs_flat_percent":
            modes["shape_regression"]["r2_vs_flat_percent"],
        "ranked/e146_shape_slope_us": modes["shape_regression"]["slope_us_per_drafting_round"],
        "ranked/e146_classifier_accuracy":
            classify["contrast_classifier"]["family_placement_coverage"]["accuracy"],
        "ranked/e146_classifier_coverage":
            classify["contrast_classifier"]["family_placement_coverage"]["covered"]
            / classify["contrast_classifier"]["family_placement_coverage"]["total"],
        "ranked/e146_band_gap_in_flat_sd":
            classify["contrast_classifier"]["band_gap_in_flat_sd"],
        # --- R-B ----------------------------------------------------------
        "ranked/e146_contrast_sd_pct": ranked["contrast_sd_pct"],
        "ranked/e146_mode_share_of_variance": ranked["mode_share_of_variance"],
        "ranked/e146_step_pct_of_candidate_mean8": ranked["step_pct_of_candidate_mean8"],
        "ranked/e146_flat_pair_sd_pct": ranked["flat_pair_sd_pct"],
        "ranked/e146_incumbent_floor_understatement": ranked["understatement_factor"],
        "ranked/e146_type1_at_zero": ranked["e146_type1_at_zero"],
        "ranked/e146_type2_at_030pct": ranked["e146_type2_at_030pct"],
        "ranked/e146_wrong_sign_rate_at_030pct": ranked["wrong_sign_rate_at_030pct"],
        "ranked/e146_contrasts_needed_for_030pct": ranked["contrasts_needed"]["0.30"],
        "ranked/e146_contrasts_needed_for_010pct": ranked["contrasts_needed"]["0.10"],
        "ranked/e146_mde_2sigma_pct_paired_replicate":
            ranked["e146_mde_2sigma_pct_paired_replicate"],
        "ranked/e146_mde_2sigma_pct_mode_classified":
            ranked["e146_mde_2sigma_pct_mode_classified"],
        "ranked/e146_within_leg_sd_pct": ranked["within_leg_sd_pct"],
    }
    if local:
        summary.update({
            "local/e146_local_floor_excludes_wired_residency":
                local["e146_local_floor_excludes_wired_residency"],
            "local/e146_local_mde_2sigma_n1_pct": local["mde_2sigma_by_n"]["1"],
            "local/e146_local_mde_2sigma_n4_pct": local["mde_2sigma_by_n"]["4"],
            "local/e146_local_mde_2sigma_n8_pct": local["mde_2sigma_by_n"]["8"],
            "local/e146_local_mde_2sigma_steady_n8_pct":
                local["mde_2sigma_steady_by_n"]["8"],
            "local/e146_local_fixed_binary_sd_pct":
                local["e146_local_fixed_binary_sd_pct"],
            "local/e146_local_fixed_binary_sd_excl_first_pct":
                local["e146_local_fixed_binary_sd_excl_first_pct"],
            "local/e146_local_drift_slope_pct_per_leg":
                local["e146_local_drift_slope_pct_per_leg"],
            "local/e146_local_blocks_only_sd_pct":
                local["e146_local_blocks_only_sd_pct"],
            "local/e146_local_serial_sd_pct": local["e146_local_serial_sd_pct"],
            "local/e146_local_round_count_unique":
                local["e146_local_round_count_unique"],
            "local/e146_local_to_ranked_candidate_noise_ratio":
                local["e146_local_to_ranked_candidate_noise_ratio"],
            "local/e146_local_mtp_legs": local["n_mtp_legs"],
            "local/e146_local_serial_legs": local["n_serial_legs"],
            "local/e146_local_mean_seconds_per_token":
                local["mean_seconds_per_token"],
            "local/e146_local_mde_n1_pct": local["mde_by_n"]["1"],
            "local/e146_local_mde_n2_pct": local["mde_by_n"]["2"],
            "local/e146_local_mde_n4_pct": local["mde_by_n"]["4"],
            "local/e146_local_mde_n8_pct": local["mde_by_n"]["8"],
        })
    if offschedule:
        cal = offschedule["control_calibration"]
        summary.update({
            "ranked/e146_offschedule_state_readout_possible":
                offschedule["e146_offschedule_state_readout_possible"],
            "ranked/e146_offschedule_decisive_fraction":
                offschedule["e146_offschedule_decisive_fraction"],
            "ranked/e146_offschedule_control_separation_margin_auc":
                offschedule["e146_offschedule_control_separation_margin_auc"],
            "ranked/e146_offschedule_high_anchor_auc_max":
                cal["high_anchor_auc_max"],
            "ranked/e146_offschedule_main_anchor_auc_min":
                cal["main_anchor_auc_min"],
            "ranked/e146_offschedule_three_channel_r2":
                offschedule["channel_regression"]["r2_on_serial_and_prefill"],
            "ranked/e146_offschedule_pb6_auc": next(
                t["auc"] for t in offschedule["anchor_inversion_panel"]
                if t["anchor"] == "e003a86d"),
            "ranked/e146_offschedule_pb6_state_share_pct":
                offschedule["pb6_two_basis"]["state_share_pct_of_decode"],
            "ranked/e146_offschedule_pb6_schedule_share_pct":
                offschedule["pb6_two_basis"]["schedule_share_pct_of_decode"],
        })
    run.summary.update(summary)

    pair_table = wandb.Table(columns=[
        "replicate", "target", "solver", "created", "k_us_per_drafting_round",
        "cand_mean8_pct", "serial_mean8_pct", "residual_sd_pp", "heads_match",
        "age_hours", "band"])
    for p in pairs["pairs"]:
        serial8 = sum(p["serial_pct"].values()) / 8.0
        k = p["k_us_per_drafting_round"]
        band = "+step" if k > 300 else ("-step" if k < -300 else "flat")
        pair_table.add_data(p["replicate"], p["target"], p["solver"], p["created"],
                            k, p["cand_mean8_pct"], serial8, p["residual_sd_pp"],
                            p["heads_match"], p["age_hours"], band)
    run.log({"ranked/pure_nuisance_pairs": pair_table})

    shape = wandb.Table(columns=["prompt", "drafting_share", "high_mean_pct",
                                 "flat_mean_pct"])
    for prompt, values in modes["per_prompt"].items():
        shape.add_data(prompt, values["drafting_share"], values["high_mean_pct"],
                       values["flat_mean_pct"])
    run.log({"ranked/step_shape_by_prompt": shape})

    risk = wandb.Table(columns=["row", "anchor", "finding", "k_us_per_drafting_round",
                                "raw_cand_mean8_pct", "steps_above_family_flat",
                                "mode_call", "corrected_cand_mean8_pct",
                                "raw_total_mean8_pct", "corrected_total_mean8_pct",
                                "same_schedule_family", "residual_sd_pp"])
    for entry in classify["e146_findings_at_risk"]:
        risk.add_data(entry["row"], entry["anchor"], entry["finding"],
                      entry["k_us_per_drafting_round"], entry["raw_cand_mean8_pct"],
                      entry["steps_above_family_flat"], entry["mode_call"],
                      entry["corrected_cand_mean8_pct"],
                      entry["raw_total_mean8_pct"],
                      entry["corrected_total_mean8_pct"],
                      entry["same_schedule_family"], entry["residual_sd_pp"])
    run.log({"ranked/findings_at_risk": risk})

    designs = wandb.Table(columns=["design", "sd_pct", "mde_n1_pct", "note"])
    for d in ranked["designs"]:
        designs.add_data(d["design"], d["sd_pct"], d["mde_n1_pct"], d["note"])
    run.log({"ranked/mde_designs": designs})

    bar_cohort = wandb.Table(columns=[
        "id", "solver", "created", "score", "group", "k_us_decode", "k_us_total",
        "decode8_pct", "prefill_mean8_pct", "residual_sd_pp"])
    group_of = {}
    for g in cohort["groups"]:
        for row_id in g["ids"]:
            group_of[row_id] = g["name"]
    for r in cohort["rows"]:
        bar_cohort.add_data(r["id"], r["solver"], r["created"], r["score"],
                            group_of.get(r["id"], "?"), r["k_us_decode"],
                            r["k_us_total"], r["decode8_pct"],
                            r["prefill_mean8_pct"], r["residual_sd_pp"])
    run.log({"ranked/bar_cohort_684821ed": bar_cohort})

    corrected = wandb.Table(columns=[
        "id", "k_us", "steps", "raw_decode8_pct", "corrected_decode8_pct",
        "raw_total8_pct", "corrected_total8_pct", "prefill_mean8_pct"])
    for r in cohort["corrected_rows"]:
        corrected.add_data(r["id"], r["k_us"], r["steps"], r["raw_decode8_pct"],
                           r["corrected_decode8_pct"], r["raw_total8_pct"],
                           r["corrected_total8_pct"], r["prefill_mean8_pct"])
    run.log({"ranked/state_corrected_rows": corrected})

    auc_table = wandb.Table(columns=["scheme", "feature", "auc", "p_value",
                                     "null_median", "null_p95", "n_pos", "n_neg"])
    for scheme, entries in (("in_sample", cohort["auc_in_sample"]),
                            ("out_of_sample", cohort["auc_out_of_sample"])):
        for a in entries:
            auc_table.add_data(scheme, a["feature"], a["auc"],
                               a["p_value_one_sided"], a["null_median"],
                               a["null_p95"], a["n_pos"], a["n_neg"])
    run.log({"ranked/mode_separation_auc": auc_table})

    sensitivity = wandb.Table(columns=["step_source", "step_us",
                                       "corrected_decode_pct",
                                       "corrected_total_pct"])
    for entry in ledger305["steps"]:
        sensitivity.add_data(entry["step_source"], entry["step_us"],
                             entry["corrected_decode_pct"],
                             entry["corrected_total_pct"])
    run.log({"ranked/headline_step_sensitivity": sensitivity})

    f220 = wandb.Table(columns=["id", "k", "steps", "raw_decode8_pct",
                                "corrected_decode8_pct", "residual_sd_pp"])
    for r in cohort["f220"]:
        f220.add_data(r["id"], r["k"], r["steps"], r["raw_decode8_pct"],
                      r["corrected_decode8_pct"], r["residual_sd_pp"])
    run.log({"ranked/f220_flush_fold_receipts": f220})

    fam = wandb.Table(columns=["id", "solver", "created", "score",
                               "k_us_per_drafting_round", "cand_mean8_pct",
                               "same_decisions_as_anchor"])
    for entry in family["family"]:
        fam.add_data(entry["id"], entry["solver"], entry["created"], entry["score"],
                     entry["k_us_per_drafting_round"], entry["cand_mean8_pct"],
                     entry["same_decisions_as_anchor"])
    run.log({"ranked/crown_family_against_anchor": fam})

    if offschedule:
        panel = wandb.Table(columns=["anchor", "role", "auc", "null_p95",
                                     "p_value_one_sided", "n_high", "n_main",
                                     "calibrated_call"])
        for entry in offschedule["anchor_inversion_panel"]:
            panel.add_data(entry["anchor"], entry["role"], entry["auc"],
                           entry["null_p95"], entry["p_value_one_sided"],
                           entry["n_high"], entry["n_main"],
                           entry.get("calibrated_call", ""))
        run.log({"ranked/offschedule_anchor_inversion_panel": panel})

    if local:
        legs = wandb.Table(columns=["leg", "kind", "seconds_per_token", "round_count",
                                    "entry_c", "exit_c", "gate_wait_s"])
        path = os.path.join(ROOT, ".mlxfast-private", "e146", "rc", "legs.jsonl")
        for line in open(path):
            leg = json.loads(line)
            legs.add_data(leg["label"], leg["kind"],
                          leg["metrics"]["parent_measured_seconds_per_token"],
                          leg["metrics"]["round_count"], leg["gpu_temp_entry_c"],
                          leg["gpu_temp_exit_c"], leg["cool_gate_wait_seconds"])
        run.log({"local/rc_legs": legs})

    print("run %s" % run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
