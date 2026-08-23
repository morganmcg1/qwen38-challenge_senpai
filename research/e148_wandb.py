"""Publish the E148 board-repricing evidence to Weights and Biases.

Zero GPU work. This run is an analysis record: it reads the JSON produced by
research/e148_ra.py .. e148_f241.py and logs the metrics, the two decision
tables, and the source artifact. No scored surface file is touched.
"""

from __future__ import annotations

import ast
import json
import pathlib

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

PR_NUMBER = 148
BRANCH = "qwen-askeladd/e148-state-corrected-board-repricing"
BASE_SHA = "7e92b877dd4e63e06ec3a88d4a483b0559b5c261"
HEAD_SHA = "0434e7b9374bc33e6a4cbc3dd06c06f45e5fff3c"

STEP_US = 879.0
AT_ZERO_THRESHOLD_STEPS = 0.35

FILES = [
    "e148_lib.py",
    "e148_ra.py",
    "e148_rb.py",
    "e148_rc.py",
    "e148_rd.py",
    "e148_rf.py",
    "e148_f241.py",
    "e148_wandb.py",
    "e148-ra.json",
    "e148-rb.json",
    "e148-rc.json",
    "e148-rd.json",
    "e148-rf.json",
    "e148-f241.json",
]


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def hand_table_rows() -> dict[str, dict[str, object]]:
    tree = ast.parse((HERE / "e148_rd.py").read_text())
    for node in tree.body:
        target = getattr(node, "target", None)
        if getattr(target, "id", "") == "HAND":
            return ast.literal_eval(node.value)
    raise SystemExit("HAND table not found in e148_rd.py")


def main() -> None:
    ra = load("e148-ra.json")
    rb = load("e148-rb.json")
    rc = load("e148-rc.json")
    rd = load("e148-rd.json")
    rf = load("e148-rf.json")
    f241 = load("e148-f241.json")

    step = rb["step_reconciliation"]
    lattice = rb["lattice"]
    sens = f241["e148_at_zero_set_step_sensitivity"]

    metrics: dict[str, float | int] = {
        # R-D primary
        "e148_best_unclaimed_corrected_pct": rd["e148_best_unclaimed_corrected_pct"],
        "e148_unclaimed_candidates": rd["e148_unclaimed_candidates"],
        "e148_hand_adjudicated_rows": rd["hand_adjudicated_rows"],
        "e148_rows_considered_at_threshold": rd["rows_considered"],
        "e148_threshold_pct": rd["threshold_pct"],
        "e148_editable_prefix_count": rd["editable_prefix_count"],
        "e148_source_corpus_bytes": rd["source_corpus_bytes"],
        "e148_ledger_bytes": rd["ledger_bytes"],
        "e148_ambiguity_broken_by_score": rd["resolver_failure_modes"][
            "ambiguity_broken_by_score"
        ],
        "e148_names_more_than_one_resolvable_tree": rd["resolver_failure_modes"][
            "names_more_than_one_resolvable_tree"
        ],
        "e148_resolved_same_cohort": rd["resolver_failure_modes"][
            "resolved_same_cohort"
        ],
        "e148_resolved_cross_schedule": rd["resolver_failure_modes"][
            "resolved_cross_schedule"
        ],
        "e148_unresolved_rows": rd["resolver_failure_modes"]["unresolved"],
        # R-A cohorts
        "e148_scored_rows": ra["e148_scored_rows"],
        "e148_cohort_count": ra["e148_cohort_count"],
        "e148_rows_classified": ra["e148_rows_classified"],
        "e148_rows_unclassifiable": ra["e148_rows_unclassifiable"],
        # R-B controls, registered block
        "e148_control_zero_delta_abs_pct_max_as_registered": rb[
            "e148_control_zero_delta_abs_pct_max"
        ],
        "e148_control_zero_delta_abs_pct_max_w5_unrepaired": rb[
            "e148_control_zero_delta_abs_pct_max_w5"
        ],
        "e148_control_zero_delta_abs_pct_max_at_zero_registered": rb[
            "e148_control_zero_delta_abs_pct_max_at_zero"
        ],
        "e148_control_zero_delta_abs_pct_max_nonzero": rb[
            "e148_control_zero_delta_abs_pct_max_nonzero"
        ],
        "e148_control_zero_delta_at_zero_n_registered": rb[
            "e148_control_zero_delta_at_zero_n"
        ],
        "e148_control_zero_delta_at_zero_sd_pp_registered": rb[
            "e148_control_zero_delta_at_zero_sd_pp"
        ],
        "e148_control_zero_delta_gate_pass": int(
            bool(rb["e148_control_zero_delta_gate_pass"])
        ),
        "e148_gate_pp": rb["gate_pp"],
        # R-B controls, operative block widened to n = 10 in R-C
        "e148_control_zero_delta_at_zero_abs_max_pp_operative": rc[
            "at_zero_abs_max_pp"
        ],
        "e148_control_zero_delta_at_zero_sd_pp_operative": rc["at_zero_sd_pp"],
        "e148_control_zero_delta_at_zero_sd_upper_pp_operative": rc[
            "at_zero_sd_upper_pp"
        ],
        "e148_mde_2sigma_pp": 2.0 * rc["at_zero_sd_pp"],
        # R-B sign control
        "e148_control_promoted_sign_correct_frac": rb[
            "e148_control_promoted_sign_correct_frac"
        ],
        "e148_control_promoted_sign_correct_frac_matched": rb[
            "e148_control_promoted_sign_correct_frac_matched"
        ],
        "e148_control_promoted_sign_correct_frac_uncorrected": rb[
            "e148_control_promoted_sign_correct_frac_uncorrected"
        ],
        "e148_control_promoted_sign_correct_frac_at_zero": rb[
            "e148_control_promoted_sign_correct_frac_at_zero"
        ],
        "e148_control_promoted_sign_n": rb["e148_control_promoted_sign_n"],
        "e148_mining_coverage_at_zero_frac": rb["e148_mining_coverage_at_zero_frac"],
        # R-B state step
        "e148_state_step_us_clean": step["e148_state_step_us_clean"]["state_step_us"],
        "e148_state_step_us_clean_se": step["e148_state_step_us_clean"]["se_us"],
        "e148_state_step_us_population": step["e148_state_step_us_population"][
            "state_step_us"
        ],
        "e148_state_step_us_population_se": step["e148_state_step_us_population"][
            "se_us"
        ],
        "e148_state_step_us_population_n": step["e148_state_step_us_population"]["n"],
        "e148_state_step_from_paired_controls_us": rb[
            "e148_state_step_from_paired_controls_us"
        ],
        # R-B lattice
        "e148_lattice_n": lattice["n"],
        "e148_lattice_zero_mode_centre_steps": lattice["zero_mode_centre_steps"],
        "e148_lattice_zero_mode_n": lattice["zero_mode_n"],
        "e148_lattice_one_mode_centre_steps": lattice["one_mode_centre_steps"],
        "e148_lattice_one_mode_centre_us": lattice["one_mode_centre_us"],
        "e148_lattice_one_mode_n": lattice["one_mode_n"],
        "e148_lattice_valley_frac": lattice["valley_frac"],
        # R-C disposition
        "e148_rows_repriced": rc["e148_rows_repriced"],
        "e148_rows_ambiguous": rc["e148_rows_ambiguous"],
        "e148_refusal_rate": rc["e148_refusal_rate"],
        "e148_prefill_noise_sd_pp": rc["prefill_noise_sd_pp"],
        "e148_prefill_noise_abs_max_pp": rc["prefill_noise_abs_max_pp"],
        # R-F byte model
        "e148_m1_gb_per_full_pass": rf["e148_m1_gb_per_full_pass"],
        "e148_m1_implied_gbps_recomputed": rf["e148_m1_implied_gbps_recomputed"],
        "e148_measured_read_gbps": rf["e148_measured_read_gbps"],
        "e148_byte_model_error_factor": rf["e148_byte_model_error_factor"],
        # F241
        "e148_lattice_slots_delta_bic_one_minus_mixture": f241[
            "e148_lattice_slots_population"
        ]["delta_bic_one_minus_mixture"],
        "e148_lattice_slots_band_n": f241["band_n"],
        "e148_lattice_slots_mixture_best_s_us": f241["three_slot_mixture"]["s_us"],
        "e148_lattice_slots_one_gaussian_mu_us": f241["one_gaussian"]["mu_us"],
        "e148_at_zero_step_invariance_window_low_us": sens["invariance_window_us"][0],
        "e148_at_zero_step_invariance_window_high_us": sens["invariance_window_us"][1],
        "e148_at_zero_step_max_rows_changing_class": sens[
            "max_rows_changing_class_vs_879"
        ],
        "e148_at_zero_step_largest_k_at_zero_us": sens["largest_k_at_zero_us"],
        "e148_at_zero_step_smallest_k_refused_us": sens["smallest_k_refused_us"],
    }
    for prompt, weight in rb["median_weight_census"].items():
        metrics[f"e148_median_weight_{prompt}"] = weight
    for threshold, cell in rc["tail_test"].items():
        key = threshold.replace("-", "m").replace(".", "p")
        metrics[f"e148_tail_observed_{key}"] = cell["observed"]
        metrics[f"e148_tail_expected_sd_upper_{key}"] = cell["expected_sd_upper"]
    for name, count in rc["disposition"].items():
        metrics[f"e148_disposition_{name.replace('-', '_')}"] = count
    for name, count in rd["blocker_census"].items():
        metrics[f"e148_blocker_{name.replace('-', '_')}"] = count

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e148-board-repricing",
        job_type="analysis",
        tags=["e148", "askeladd", "board-repricing", "zero-gpu"],
        config={
            "experiment": "E148",
            "pr_number": PR_NUMBER,
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "head_sha": HEAD_SHA,
            "scored_surface_change": "none",
            "gpu_seconds": 0,
            "board_path": rc["board_path"],
            "board_rows_scored": ra["e148_scored_rows"],
            "state_step_us_campaign_constant": STEP_US,
            "at_zero_threshold_steps": AT_ZERO_THRESHOLD_STEPS,
            "price_frame": rc["price_frame"],
            "price_frame_name": rd["frame"],
            "harness": "ranked",
            "prefill_band": rc["prefill_band"],
        },
    )

    run.summary.update(metrics)
    run.log(metrics)

    board = wandb.Table(
        columns=[
            "median_pair_pct",
            "corrected_total_pct",
            "uniform_four_corrected_pct",
            "row",
            "score",
            "parent",
            "parent_score",
            "steps_exact",
            "k_us_per_drafting_round",
            "prefill_pct",
            "solver",
            "promotion",
            "title",
        ]
    )
    ranked = sorted(
        rc["decode_ranked_table"], key=lambda r: r["corrected_total_pct"]
    )[:120]
    for r in ranked:
        board.add_data(
            r["median_pair_pct"],
            r["corrected_total_pct"],
            r.get("uniform_four_corrected_pct"),
            r["row"],
            r["score"],
            r["parent"],
            r["parent_score"],
            r["steps_exact"],
            r["k_us_per_drafting_round"],
            r.get("prefill_pct"),
            r["solver"],
            r["promotion"],
            r["title"],
        )
    run.log({"e148_repriced_board": board})

    priced = {r["row"]: r for r in rc["decode_ranked_table"]}
    refused = {r["row"]: r for r in rc["decode_refused_table"]}
    hand = wandb.Table(
        columns=[
            "row",
            "verdict",
            "mechanism",
            "median_pair_pct",
            "corrected_total_pct",
            "decode_priceable",
            "source",
            "evidence",
            "caveat",
        ]
    )
    for row, entry in hand_table_rows().items():
        record = priced.get(row) or refused.get(row) or {}
        hand.add_data(
            row,
            entry["verdict"],
            entry["mechanism"],
            record.get("median_pair_pct"),
            record.get("corrected_total_pct"),
            bool(record.get("priceable", False)),
            "; ".join(entry["source"]),
            entry["evidence"],
            entry.get("caveat", ""),
        )
    run.log({"e148_hand_adjudication": hand})

    lattice_points = wandb.Table(columns=["row", "parent", "steps_exact"])
    for point in rb["lattice_points"]:
        lattice_points.add_data(point["row"], point["parent"], point["steps_exact"])
    run.log({"e148_lattice_points": lattice_points})

    artifact = wandb.Artifact("e148-board-repricing", type="analysis")
    for name in FILES:
        artifact.add_file(str(HERE / name), name=name)
    run.log_artifact(artifact)

    print(f"run_id={run.id}")
    print(f"run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
