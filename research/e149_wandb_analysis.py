"""Publish the E149 zero-GPU rungs to Weights and Biases.

Zero GPU work. This run is an analysis record for rung 0a (the at-zero block
at n=11), the F2 section 4 per-prompt null sd, and the F2 section 3 frame-clean
tail re-run. No scored surface file is touched.
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

FILES = [
    "e149_rung0a.py", "e149-rung0a.json",
    "e149_null_sd.py", "e149-null-sd.json",
    "e149_tail_frames.py", "e149-tail-frames.json",
    "e149_f3_cliff.py", "e149-f3-cliff.json",
    "e149_wandb_analysis.py",
]

PRIMARY_ALLOCATION = "geometric"


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    a = load("e149-rung0a.json")
    s = load("e149-null-sd.json")
    t = load("e149-tail-frames.json")
    f3 = load("e149-f3-cliff.json")

    frames = a["e149_at_zero_block_frames"]
    ratio = s["beagle_to_paying_four_total"]
    tot = s["per_prompt_total"]
    dec = s["per_prompt_decode"]
    jit = s["implied_per_round_jitter_us_total"]
    acc = s["accounts_total"]
    dec_mp = s["median_pair_variance_decomposition_total"]

    metrics: dict[str, float | int] = {
        # rung 0a
        "e149_165d4ba7_fitted_k_us_per_dr": a["e149_165d4ba7_fitted_k_us_per_dr"],
        "e149_165d4ba7_steps_exact": a["e149_165d4ba7_steps_exact"],
        "e149_165d4ba7_fitted_k_us_per_dr_matched":
            a["e149_165d4ba7_fitted_k_us_per_dr_matched_fit"],
        "e149_165d4ba7_steps_exact_matched":
            a["e149_165d4ba7_steps_exact_matched_fit"],
        "e149_165d4ba7_classifies_at_zero":
            int(bool(a["e149_165d4ba7_classifies_at_zero"])),
        "e149_at_zero_n11_sd_median_pair": frames["median_pair"]["n=11"]["sd_pp"],
        "e149_at_zero_n11_mde_median_pair":
            frames["median_pair"]["n=11"]["mde_2sigma_pp"],
        "e149_at_zero_n11_sd_unweighted": frames["unweighted"]["n=11"]["sd_pp"],
        "e149_at_zero_n11_mde_unweighted":
            frames["unweighted"]["n=11"]["mde_2sigma_pp"],
        "e149_at_zero_n11_sd_f83": frames["f83"]["n=11"]["sd_pp"],
        "e149_at_zero_n11_mde_f83": frames["f83"]["n=11"]["mde_2sigma_pp"],
        "e149_at_zero_n10_mde_unweighted":
            frames["unweighted"]["n=10"]["mde_2sigma_pp"],
        "e149_three_draw_weighted_five_sd_pp":
            a["three_draw_dispersion"]["weighted_five_sd_pp"],
        "e149_three_draw_published_score_sd":
            a["three_draw_dispersion"]["published_score_sd"],
        # F2 section 4
        "e149_beagle_to_paying_four_noise_ratio": ratio["ratio"],
        "e149_beagle_to_paying_four_noise_ratio_ci_lo": ratio["ci95"][0],
        "e149_beagle_to_paying_four_noise_ratio_ci_hi": ratio["ci95"][1],
        "e149_pooled_paying_four_sd_pp": ratio["pooled_four_sd_pp"],
        "e149_null_sd_account_r_total_round_count":
            acc["total_round_count"]["pearson_r"],
        "e149_null_sd_account_r_rounds_per_second":
            acc["total_rounds_per_second_of_leg"]["pearson_r"],
        "e149_null_sd_account_r_state_exposure":
            acc["f239_drafting_state_exposure_pct"]["pearson_r"],
        "e149_null_sd_account_rho_state_exposure":
            acc["f239_drafting_state_exposure_pct"]["spearman_rho"],
        "e149_null_sd_account_r_mean_draft_length":
            acc["mean_draft_length"]["pearson_r"],
        "e149_implied_per_round_jitter_us_mean": jit["mean"],
        "e149_implied_per_round_jitter_us_spread_ratio": jit["spread_ratio"],
        "e149_beagle_share_of_median_pair_variance":
            dec_mp["beagle_variance_share"],
        "e149_median_pair_sd_from_independent_prompts_pp":
            dec_mp["implied_median_pair_sd_pp_if_independent"],
        # F2 section 3
        "e149_tail_priced_rows": t["n_priced"],
        "e149_tail_board_scored_rows": t["n_scored"],
        # F3
        "e149_null_sd_vs_per_step_p_spearman":
            f3["null_sd_vs_per_step_p"]["eight_prompt_spearman"]["observed"],
        "e149_null_sd_vs_per_step_p_perm_p":
            f3["null_sd_vs_per_step_p"]["eight_prompt_spearman"]["two_sided_p"],
        "e149_null_sd_vs_per_step_p_perm_floor_p95":
            f3["null_sd_vs_per_step_p"]["eight_prompt_spearman"]["abs_null_p95"],
        "e149_null_sd_vs_per_step_p_spearman_weighted_five":
            f3["null_sd_vs_per_step_p"]["weighted_five_spearman"]["observed"],
        "e149_null_sd_vs_per_step_p_perm_p_weighted_five":
            f3["null_sd_vs_per_step_p"]["weighted_five_spearman"]["two_sided_p"],
        "e149_driver_confound_p_vs_rounds":
            f3["driver_confound"]["per_step_p_vs_round_count"],
        "e149_driver_confound_rounds_vs_dlen":
            f3["driver_confound"]["round_count_vs_mean_draft_length"],
        "e149_cliff_index_vs_null_sd_spearman":
            f3["account_tests"][PRIMARY_ALLOCATION]["index_vs_null_sd"][
                "observed"],
        "e149_cliff_index_vs_null_sd_perm_p":
            f3["account_tests"][PRIMARY_ALLOCATION]["index_vs_null_sd"][
                "two_sided_p"],
        "e149_cliff_index_beagle":
            f3["e149_cliff_straddle_index"]["beagle"][PRIMARY_ALLOCATION],
        "e149_cliff_index_beagle_over_paying_four":
            f3["beagle_against_paying_four"][PRIMARY_ALLOCATION][
                "cliff_straddle_index"]["ratio"],
        "e149_width_sd_beagle_over_paying_four":
            f3["beagle_against_paying_four"][PRIMARY_ALLOCATION][
                "sd_verify_width"]["ratio"],
        "e149_cliff_crossing_beagle_over_paying_four":
            f3["beagle_against_paying_four"][PRIMARY_ALLOCATION][
                "cliff_crossing_pair_fraction"]["ratio"],
        "e149_ledger_cliff_step_5_to_6_us": f3["ledger_curve_steps_us"][4],
        "e149_ledger_cliff_step_6_to_7_us": f3["ledger_curve_steps_us"][5],
    }
    for prompt, cell in f3["e149_per_prompt_width_dispersion"].items():
        metrics[f"e149_width_sd_{prompt}"] = \
            cell[PRIMARY_ALLOCATION]["sd_verify_width"]
        metrics[f"e149_cliff_crossing_{prompt}"] = \
            cell[PRIMARY_ALLOCATION]["cliff_crossing_pair_fraction"]
        metrics[f"e149_cliff_straddle_index_{prompt}"] = \
            f3["e149_cliff_straddle_index"][prompt][PRIMARY_ALLOCATION]
    for prompt, value in f3["per_step_p"].items():
        metrics[f"e149_per_step_p_{prompt}"] = value
    for prompt, cell in tot.items():
        metrics[f"e149_null_sd_total_{prompt}"] = cell["sd_pp"]
        metrics[f"e149_null_sd_total_upper95_{prompt}"] = cell["sd_upper95_pp"]
        metrics[f"e149_null_sd_receipt_total_{prompt}"] = cell["receipt_sd_pp"]
    for prompt, cell in dec.items():
        metrics[f"e149_null_sd_decode_{prompt}"] = cell["sd_pp"]
    for thresh, cell in t["within_frame_median_pair_n11"].items():
        key = thresh.replace("-", "m").replace(".", "p")
        metrics[f"e149_tail_medianpair_observed_{key}"] = cell["observed"]
        metrics[f"e149_tail_medianpair_expected_n11_{key}"] = cell["expected"]
    for thresh, cell in t["within_frame_unweighted_weighted_five_n11"].items():
        key = thresh.replace("-", "m").replace(".", "p")
        metrics[f"e149_tail_w5_observed_{key}"] = cell["observed"]
        metrics[f"e149_tail_w5_expected_n11_{key}"] = cell["expected"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e149-rung0-analysis",
        job_type="analysis",
        tags=["e149", "askeladd", "zero-gpu", "null-block", "rung0"],
        config={
            "experiment": "E149",
            "rungs": ["0a", "F2-section-4", "F2-section-3"],
            "pr_number": PR_NUMBER,
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "scored_surface_change": "none",
            "gpu_seconds": 0,
            "harness": "ranked",
            "board_path": s["board"],
            "board_rows_scored": s["n_rows"],
            "lead_frame": "realised median pair (Rule 118, Rule 144)",
            "block_members": s["members"],
            "anchor_multiplicity": s["anchor_multiplicity"],
        },
    )
    run.summary.update(metrics)
    run.log(metrics)

    null = wandb.Table(columns=[
        "prompt", "sd_pp_total", "sd_upper95_pp_total", "receipt_sd_pp_total",
        "sd_pp_decode", "mean_pp_total", "abs_max_pp_total",
        "three_draw_sd_pp", "total_rounds", "rounds_per_second_of_leg",
        "f239_state_exposure_pct", "implied_jitter_us_per_round",
        "detectable_us_per_round_2sigma", "per_step_p", "sd_verify_width",
        "cliff_crossing_pair_fraction", "cliff_straddle_index",
    ])
    exposure = acc["f239_drafting_state_exposure_pct"]["values"]
    rps = acc["total_rounds_per_second_of_leg"]["values"]
    snr = s["per_round_mechanism_snr_per_us_total"]
    width = f3["e149_per_prompt_width_dispersion"]
    for prompt in sorted(tot):
        null.add_data(
            prompt, tot[prompt]["sd_pp"], tot[prompt]["sd_upper95_pp"],
            tot[prompt]["receipt_sd_pp"], dec[prompt]["sd_pp"],
            tot[prompt]["mean_pp"], tot[prompt]["abs_max_pp"],
            s["three_draw_cross_check"][prompt]["three_draw_sd_pp"],
            s["round_count_table"][prompt]["total_rounds"], rps[prompt],
            exposure[prompt], jit["per_prompt"][prompt], 2.0 / snr[prompt],
            f3["per_step_p"][prompt],
            width[prompt][PRIMARY_ALLOCATION]["sd_verify_width"],
            width[prompt][PRIMARY_ALLOCATION]["cliff_crossing_pair_fraction"],
            f3["e149_cliff_straddle_index"][prompt][PRIMARY_ALLOCATION])
    run.log({"e149_per_prompt_null_sd_n11": null})

    tail = wandb.Table(columns=[
        "threshold_pct", "frame", "observed", "expected_n11_sigma",
        "sigma_n11_pp"])
    for frame, key, sigma in (
            ("median_pair (LEAD)", "within_frame_median_pair_n11",
             t["sigma_table"]["median_pair"]["n11"]),
            ("unweighted_weighted_five",
             "within_frame_unweighted_weighted_five_n11",
             t["sigma_table"]["unweighted_weighted_five"]["n11"])):
        for thresh, cell in sorted(t[key].items(), key=lambda kv: -float(kv[0])):
            tail.add_data(float(thresh), frame, cell["observed"],
                          cell["expected"], sigma)
    run.log({"e149_rc_tail_by_frame": tail})

    art = wandb.Artifact("e149-rung0-analysis", type="analysis")
    for name in FILES:
        art.add_file(str(HERE / name))
    run.log_artifact(art)

    print(run.url)
    print(run.id)
    run.finish()


if __name__ == "__main__":
    main()
