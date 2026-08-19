#!/usr/bin/env python3
"""Attach the E55 register census to the headline bracket run's W&B summary.

The 15:02Z retag destroyed a peer's workspace mid-session, so every reading that
matters is pushed off this machine as soon as it exists rather than at the end.

  python3 research/e55_log_census.py --run wxezisvs
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
SENSITIVITY_PCT_PER_F9 = 8.49751
F9_LOCAL = 0.55435
MEASURED_MTP_LEG_PCT = -4.2952
# E49 Arm 2, merged: control-free contrasts bound the shared-register harm.
E49_ARM2_HARM_BOUND_PCT = 0.0876
RANKED_MIXTURES = {"e48": 0.21630, "edward_upper": 0.08900, "edward_lower": 0.04600}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--census", default="research/e55-reg-census.json")
    ap.add_argument("--exactness", default="research/e55-exactness.json")
    args = ap.parse_args()

    c = json.loads(pathlib.Path(args.census).read_text())
    wide = c["kernel_wide_reg_max"]
    b0 = c["entry_batch0"]

    predicted = -SENSITIVITY_PCT_PER_F9 * F9_LOCAL
    shortfall = MEASURED_MTP_LEG_PCT - predicted

    summary = {
        "census/kernel_wide_reg_max_base": wide["base_na4_table"],
        "census/kernel_wide_reg_max_candidate": wide["m9two_candidate"],
        "census/kernel_wide_reg_max_e27": wide["e27_both_cells"],
        "census/entry_batch0_base": b0["base_na4_table"],
        "census/entry_batch0_candidate": b0["m9two_candidate"],
        "census/entry_batch0_e27": b0["e27_both_cells"],
        "census/kernel_wide_reg_max_delta": c["candidate_vs_base"]["kernel_wide_reg_max_delta"],
        "census/entry_batch0_delta": c["candidate_vs_base"]["entry_batch0_delta"],
        "census/candidate_rises_above_base": c["candidate_vs_base"]["candidate_rises_above_base"],
        "census/candidate_carries_full_e27_allocation":
            c["candidate_vs_e27"]["candidate_carries_full_e27_allocation"],
        "census/positive_control_fired": c["positive_control_fired"],
        "census/recorded_value_checks_all_pass": all(
            v[2] for res in c["recorded_value_checks"].values()
            for v in res.values()),
        "census/widths_with_unchanged_reg_count": json.dumps(
            c["widths_unchanged_vs_base"]),
        "census/harm_only_arm_blocked_by": "static_assert(M >= 3 && M <= 9)",
        # E27's case-5 cell is a LONE NA=5 group: streams(M=5, IPG=5) = 1. The
        # candidate's M=9 cell is NA=5 plus an NA=4 sibling. PR #8 measured the
        # collapse on a lone group, so the two are not the same configuration.
        "census/e27_m5_streams": 1,
        "census/candidate_m9_streams": 2,
        "census/e27_m5_reg": 125,
        "pricing/local_predicted_mtp_leg_pct": predicted,
        "pricing/local_measured_mtp_leg_pct": MEASURED_MTP_LEG_PCT,
        "pricing/local_shortfall_pp": shortfall,
        "pricing/e49_arm2_harm_bound_pct": E49_ARM2_HARM_BOUND_PCT,
    }
    for name, f9 in RANKED_MIXTURES.items():
        gain = -SENSITIVITY_PCT_PER_F9 * f9
        summary["pricing/ranked_%s_gain_only_pct" % name] = gain
        summary["pricing/ranked_%s_with_e49_harm_pct" % name] = gain + E49_ARM2_HARM_BOUND_PCT
        # `shortfall` is POSITIVE (a slowdown term), so a width-independent harm
        # ADDS to the gain rather than subtracting from it.
        summary["pricing/ranked_%s_with_full_shortfall_pct" % name] = gain + shortfall
        # E27 essays leg: what case 5 plus the shared harm must be worth for
        # E27's +0.4803 % board result to hold at this mixture.
        summary["pricing/e27_essays_case5_plus_harm_%s_pct" % name] = 0.4803 - gain

    exactness_path = pathlib.Path(args.exactness)
    if exactness_path.exists():
        x = json.loads(exactness_path.read_text())
        v = x["verdicts"]
        # Each key names its OWN row population and reading, so a reader cannot
        # mistake the M=1 golden pass for evidence about the changed dispatch.
        summary.update({
            "exactness/path_a_m1_golden_bitwise_identical":
                v["path_a_m1_golden_bitwise_identical"],
            "exactness/path_a_covers_changed_dispatch": False,
            "exactness/path_b_wide_argmax_trajectory_identical":
                v["path_b_wide_argmax_trajectory_identical"],
            "exactness/path_b_reading": "argmax-level",
            "exactness/path_c_wide_rows_bitwise_identical":
                v["path_c_wide_rows_bitwise_identical"],
            "exactness/path_c_present": x["direct_bitwise_wide_evidence_present"],
            "exactness/wide_rows_covered": x["wide_rows_covered_argmax_level"],
            "exactness/widths_exercised": json.dumps(x["widths_exercised"]),
            "exactness/golden_hash_shared_across_all_arms":
                v["golden_hash_shared_across_all_arms"],
            "exactness/all_correctness_gates_passed":
                v["all_correctness_gates_passed"],
            "exactness/negative_controls_all_fired": v["negative_controls_all_fired"],
            "exactness/negative_control_count": len(x["negative_control"]["cases"]),
            "exactness/hard_stop_tripped": x["hard_stop_tripped"],
            "exactness/verdict_ok": x["verdict_ok"],
        })
        c_path = x.get("path_c_wide_row_ledger")
        if c_path:
            summary["exactness/path_c_max_abs_ulp_top2_logits"] = c_path[
                "max_abs_ulp_top2_logits"]
            summary["exactness/path_c_rows"] = c_path["row_count"][0]
            pv = c_path["provenance"]
            summary["exactness/path_c_arms_provably_distinct_binaries"] = pv[
                "arms_provably_distinct_binaries"]
            summary["exactness/path_c_base_m9_na"] = pv["base_dispatches_m9_na"]
            summary["exactness/path_c_candidate_m9_na"] = pv[
                "candidate_dispatches_m9_na"]
            summary["exactness/path_c_base_worker_sha256"] = pv["base"][
                "worker_sha256"]
            summary["exactness/path_c_candidate_worker_sha256"] = pv["candidate"][
                "worker_sha256"]
        eos = next(iter(x["eos"].values()))
        summary["exactness/eos_position"] = json.dumps(eos["eos_positions"])
        summary["exactness/tokens_after_first_eos"] = eos["tokens_after_first_eos"]
        summary["exactness/window_closed"] = eos["window_closed"]
        # Withdrawn as cross-arm evidence: both sides of that comparison come
        # from the same build, because the verify-block replay runs on the
        # candidate's own binary.
        summary["exactness/max_rejected_tail_logit_delta_is_cross_arm"] = False

    disposition_path = pathlib.Path("research/e55-swift-test-disposition.json")
    if disposition_path.exists():
        s = json.loads(disposition_path.read_text())
        summary.update({
            "swift_test/base_vs_candidate_failing_sets_identical":
                s["base_vs_candidate_failing_sets_identical"],
            "swift_test/runtime_gate_adds_no_failure":
                s["runtime_gate_adds_no_failure"],
            "swift_test/runtime_gate_provably_took_effect":
                s["runtime_gate_provably_took_effect"],
            "swift_test/no_phantom_run_entry": s["no_phantom_run_entry"],
            "swift_test/verdict_ok": s["verdict_ok"],
            "swift_test/issues": s["arms"]["base_twins_no_runtime"]["issues"],
            "swift_test/distinct_failing_tests": len(
                s["arms"]["base_twins_no_runtime"]["distinct_failing_tests"]),
            "swift_test/failing_test_names": json.dumps(
                s["arms"]["base_twins_no_runtime"]["distinct_failing_tests"]),
        })

    ceiling_path = pathlib.Path("research/e55-ceiling-term.json")
    if ceiling_path.exists():
        t = json.loads(ceiling_path.read_text())
        reg = t["register_decomposition"]
        fit = t["measured_vs_two_term"]
        pr = t["ranked_prereg"]
        bs = t["advisor_prize_back_solve"]
        gap = t["width_mixture_gap"]
        anom = t["e27_conversion_anomaly"]
        summary.update({
            "ceiling/hard_gate_shipped_table_reads_108":
                reg["hard_gate_shipped_reads_108"],
            "ceiling/candidate_register_identical_to_e27":
                reg["candidate_is_register_identical_to_e27"],
            "ceiling/e27_ceiling_set_by_widths":
                json.dumps(reg["e27_ceiling_set_by_widths"]),
            "ceiling/e27_case5_cell_below_ceiling":
                reg["e27_case5_is_below_ceiling"],
            "ceiling/widths_where_candidate_differs_from_e27":
                json.dumps(reg["widths_where_candidate_differs_from_e27"]),
            # The second term is a requested correction to a model whose arms had
            # already run, so it is explicitly not a pre-registration.
            "ceiling/two_term_is_pre_registration": False,
            "ceiling/predicted_one_term_pct":
                fit["predicted_pct"]["one_term_cell_only"],
            "ceiling/predicted_two_term_e27_pct":
                fit["predicted_pct"]["two_term_e27_ceiling_magnitude"],
            "ceiling/predicted_two_term_e49_local_pct":
                fit["predicted_pct"]["two_term_e49_local_dose_reading"],
            "ceiling/residual_one_term_pp":
                fit["residual_pp"]["one_term_cell_only"],
            "ceiling/residual_two_term_e27_pp":
                fit["residual_pp"]["two_term_e27_ceiling_magnitude"],
            "ceiling/best_fitting_variant": fit["best_fitting_variant"],
            "ceiling/fit_improvement_factor": fit["fit_improvement_factor"],
            "ranked_prereg/h_ceiling_dscore_pct":
                json.dumps(pr["h_ceiling"]["predicted_ranked_dscore_pct"]),
            "ranked_prereg/h_m5cell_dscore_pct":
                json.dumps(pr["h_m5cell"]["predicted_ranked_dscore_pct"]),
            "ranked_prereg/separation_pct": pr["separation_pct"],
            "ranked_prereg/separation_in_ranked_mde": pr["separation_in_mde"],
            "ranked_prereg/discriminating": pr["discriminating"],
            "prize/implied_ranked_f9_band_pct":
                json.dumps(bs["implied_ranked_f9_band_pct"]),
            "prize/implied_band_exceeds_e48_mixture":
                bs["implied_band_exceeds_e48"],
            "prize/e49_m9_half_implied_f9_pct":
                bs["e49_m9_half_1_3625"]["implied_f9_pct_no_g"],
            "mixture/local_mean_verify_width": gap["local_mean_verify_width"],
            "mixture/central_pair_gap_rows_min": gap["gap_rows_min"],
            "mixture/central_pair_gap_rows_max": gap["gap_rows_max"],
            "mixture/gap_positive_under_both_readings":
                gap["gap_positive_under_both_readings"],
            "anomaly/e27_implied_dscore_per_mtp_leg":
                anom["implied_dscore_per_mtp_leg"],
            "anomaly/e27_conversion_ratio_to_psi_mtp": anom["ratio_to_psi_mtp"],
            "ceiling/negative_controls_all_fire":
                t["negative_controls"]["all_fire"],
            "ceiling/verdict_ok": t["verdict_ok"],
        })

    stream_path = pathlib.Path("research/e55-stream-model.json")
    if stream_path.exists():
        s = json.loads(stream_path.read_text())
        ident = s["identification"]
        preds = s["predictions"]
        be = {row["m"]: row for row in s["break_even"]["rows"]}
        summary.update({
            "stream/r_from_e49_m9_cell": ident["r_estimates"]["e49_arm1_m9_cell"],
            "stream/r_from_e27_m5_cell": ident["r_estimates"]["e27_m5_cell"],
            "stream/r_from_e27_m9_cell": ident["r_estimates"]["e27_m9_cell"],
            "stream/r_primary_mean": ident["r_primary_mean"],
            "stream/r_primary_spread_pct": ident["r_primary_spread_pct"],
            "stream/independent_cells_agree_within_5pct":
                ident["independent_cells_agree_within_5pct"],
            "stream/collapse_is_real": ident["collapse_is_real"],
            "stream/widths_where_na5_removes_a_stream":
                json.dumps(s["stream_table"]["widths_where_na5_removes_a_stream"]),
            "stream/widths_with_a_lone_na5_group":
                json.dumps(s["stream_table"]["widths_with_a_lone_na5_group"]),
            "stream/m7_m8_predicted_ratio_range":
                json.dumps(preds["m7_m8_predicted_ratio_range"]),
            "stream/pr8_range_consistent_with_model":
                preds["pr8_range_consistent_with_model"],
            "stream/break_even_r_m5": be[5]["break_even_r"],
            "stream/break_even_r_m9": be[9]["break_even_r"],
            "stream/break_even_r_m7": be[7]["break_even_r"],
            "stream/m9_transfer_headroom_pct": be[9]["identified_r_headroom_pct"],
            "stream/m7_unprofitable_for_every_r_above_1":
                be[7]["unprofitable_for_every_r_above_1"],
            # Risk 3 offered two answers; the model rejects both.
            "stream/risk3_answer1_different_families":
                s["risk3"]["assignment_answer_1_different_families"],
            "stream/risk3_answer2_isolated_build_artefact":
                s["risk3"]["assignment_answer_2_isolated_build_is_artefact"],
            "stream/pr8_numbers_used_in_fit": s["pr8_numbers_used_in_fit"],
            "stream/uses_only_dimensionless_ratios":
                s["uses_only_dimensionless_ratios"],
            "stream/negative_controls_all_fire": s["negative_controls"]["all_fire"],
            "stream/verdict_ok": s["verdict_ok"],
        })

    api = wandb.Api()
    run = api.run("%s/%s" % (PROJECT, args.run))
    run.summary.update(summary)
    run.update()
    print("updated %s with %d census/pricing keys" % (run.url, len(summary)))
    for k in sorted(summary):
        print("  %-58s %s" % (k, summary[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
