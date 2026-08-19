#!/usr/bin/env python3
"""Log the E34 r2 autopsy to W&B. Analysis only: no GPU, no timed run."""
import json
import pathlib
import subprocess

import wandb

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = json.loads((REPO / "research" / "e34r2-model-autopsy.json").read_text())

rec, sh, rep = DOC["ranked_reconstruction"], DOC["ranked_vs_local_shape"], DOC["replay"]
pm, el, geo, nz = DOC["primary_metric"], DOC["elimination"], DOC["geometry"], rec["near_zero_row"]

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

run = wandb.init(
    project="qwen38-mlx-challenge-senpai",
    entity="wandb-applied-ai-team",
    name="e34-r2-model-autopsy",
    group="e34-ranked-operating-point-depth-cap",
    job_type="analysis",
    tags=["e34", "r2", "analysis", "no-gpu", "cost-model", "ranked-telemetry"],
    config={
        "assignment_id": "qwen38-r1-e34-ranked-operating-point-depth-cap",
        "revision_id": "r2",
        "pr_number": 39,
        "branch": "qwen-edward/ranked-operating-point-depth-cap",
        "base_sha": "abf6d79f92b97e3c47856be9c1d7798e6dc5a6b5",
        "commit_sha": head,
        "gpu_seconds_used": 0,
        "timed_runs": 0,
        "shipped_surface_diff_bytes": 0,
        "decode_tokens": 512,
        "declared_head_sha256_prefix": "559b24eb",
        "local_memory_profile": "low (48 GiB box)",
        "ranked_memory_profile": "full (128 GiB box)",
        "e33_local_ladder_ms": {str(k): v for k, v in DOC["elimination"].items()
                                if k in ("local_ladder_T5_ms", "local_ladder_T6_ms")},
    },
)

metrics = {
    # --- primary metric, withdrawn ---
    "e34/predicted_ranked_central_pair_at_best_cap": pm["candidate_rescaled"],
    "e34/primary_metric_baseline": pm["baseline"],
    "e34/primary_metric_delta_pct": pm["delta_pct"],
    "e34/primary_metric_r1_withdrawn_value": pm["r1_value"],
    "e34/primary_metric_withdrawn": 1,
    "e34/central_prompts_predicted_correctly": int(pm["central_prompts_correct"]),
    "e34/sigma_score_pct": pm["sigma_score_pct"],
    "e34/detection_threshold_2sigma_pct": pm["detection_threshold_2sigma_pct"],

    # --- the advisor's requested direction test ---
    "autopsy/r1_sign_correct": rep["r1_sign_correct_count"],
    "autopsy/r1_sign_correct_on_shallower": rep["r1_sign_correct_on_shallower"],
    "autopsy/corrected_sign_correct": rep["corrected_sign_correct_count"],
    "autopsy/corrected_sign_correct_on_shallower": rep["corrected_sign_correct_on_shallower"],
    "autopsy/rows_compared": rep["n_compared"],
    "autopsy/shallower_rows": rep["n_shallower"],
    "autopsy/calibrated_acceptance_rate": rep["calibrated_acceptance_rate"],
    "autopsy/calibrated_k": rep["calibrated_k"],

    # --- MT1 token credit ---
    "mt1/r1_implied_acceptance_rate": 1.0,
    "mt1/local_fixture_acceptance_min": DOC["semantics"]["measured_acceptance_rate_range"][0],
    "mt1/local_fixture_acceptance_max": DOC["semantics"]["measured_acceptance_rate_range"][1],
    "mt1/identities_hold": int(DOC["semantics"]["all_identities_hold"]),
    "mt1/plutarch_drafting_rounds_implied": DOC["semantics"]["plutarch_impossibility"]["drafting_rounds_implied"],

    # --- MT3 build asymmetry ---
    "mt3/k_lower_bound_from_elimination": DOC["transfer"]["k_lower_bound"],
    "mt3/k_measured_direct": nz["k_measured_mean"],
    "mt3/k_measured_min": nz["k_measured_range"][0],
    "mt3/k_measured_max": nz["k_measured_range"][1],
    "mt3/prompts_violating_transfer": DOC["transfer"]["prompts_violating"],
    "mt3/zero_draft_row_scores": 1.21028,
    "mt3/candidate_width1_ms": nz["candidate_width1_ms_range"][0],
    "mt3/pinned_serial_ms_per_token": 37.9908,

    # --- MT5 ranked reconstruction ---
    "mt5/monotone_reconstructions_surviving": rec["monotone_chains_surviving"],
    "mt5/median_reproduced": rec["median_reproduced"],
    "mt5/official_score": rec["official_score"],
    "mt5/median_matches_official": int(rec["median_matches_official"]),
    "mt5/fit_intercept_ms": rec["fit"]["intercept_ms"],
    "mt5/fit_slope_ms_per_width": rec["fit"]["slope_ms_per_width"],
    "mt5/fit_r_squared": rec["fit"]["r_squared"],

    # --- elimination ---
    "elimination/acceptance_rate_threshold": el["acceptance_rate_threshold"],
    "elimination/ranked_prompts_clearing_threshold": len(el["ranked_prompts_clearing_threshold"]),
    "elimination/central_prompts_clear_threshold": int(el["central_prompts_clear_threshold"]),
    "elimination/max_ranked_step_bound": el["at_calibrated_rate"]["max_ranked_step_T6_over_T5"],

    # --- P1 / P2 verdicts ---
    "p1/ranked_step_T6_over_T5": sh["ranked_step_T6_over_T5"],
    "p1/local_step_T6_over_T5": sh["local_step_T6_over_T5"],
    "p1/local_overstates_step_pct": sh["local_overstates_step_pct"],
    "p1/predicted_min_compression_pct": geo["local_step_exceeds_bound_by_pct"],
    "p1/confirmed": int(sh["local_overstates_step_pct"] > 0),
    "p2/overstatement_narrow_pct": sh["mean_overstatement_narrow_pct"],
    "p2/overstatement_wide_pct": sh["mean_overstatement_wide_pct"],
    "p2/confirmed": int(sh["mean_overstatement_wide_pct"] > sh["mean_overstatement_narrow_pct"]),
    "p1/falsifier_fired": int(sh["ranked_step_T6_over_T5"] >= 1.30),

    # --- MT4 geometry ---
    "mt4/local_calls_per_buffer_M5": geo["calls_per_command_buffer_local"]["M5"],
    "mt4/local_calls_per_buffer_M6": geo["calls_per_command_buffer_local"]["M6"],
    "mt4/ranked_calls_per_buffer_M5": geo["calls_per_command_buffer_ranked"]["M5"],
    "mt4/ranked_calls_per_buffer_M6": geo["calls_per_command_buffer_ranked"]["M6"],

    # --- provenance ---
    "provenance/rows_dropped": 1,
    "provenance/rows_added": len(DOC["added_rows_from_cross_check"]),
    "provenance/rows_verified": 15,
}
for prompt, rate in rec["acceptance_rate_by_prompt"].items():
    metrics[f"ranked_acceptance/{prompt}"] = rate
for row in rec["rows"]:
    metrics[f"ranked_per_round_ms/{row['prompt']}"] = row["per_round_ms"]
    metrics[f"ranked_rounds/{row['prompt']}"] = row["rounds"]

run.log(metrics)
run.summary.update(metrics)

ladder = wandb.Table(columns=["prompt", "mean_width", "raw_p", "rounds", "proposed",
                              "accepted", "acceptance_rate", "per_round_ms"])
for r in rec["rows"]:
    ladder.add_data(r["prompt"], r["mean_width"], r["raw_p"], r["rounds"],
                    r["proposed_total"], r["accepted_total"], r["acceptance_rate"],
                    r["per_round_ms"])
run.log({"ranked_width_ladder": ladder})

replay = wandb.Table(columns=["row", "beagle_n", "measured_raw_p", "measured_vs_ours_pct",
                              "r1_pred", "r1_vs_ours_pct", "r1_sign_ok",
                              "corrected_pred", "corrected_vs_ours_pct", "corrected_sign_ok",
                              "is_shallower"])
for r in rep["rows"]:
    replay.add_data(r["row"], r["beagle_n_proposed"], r["measured_raw_p"],
                    r["measured_vs_ours_pct"], r["r1_predicted_raw_p"],
                    r["r1_predicted_vs_ours_pct"], r["r1_sign_correct"],
                    r["corrected_predicted_raw_p"], r["corrected_vs_ours_pct"],
                    r["corrected_sign_correct"], r["is_shallower_than_ours"])
run.log({"declared_head_replay": replay})

shape = wandb.Table(columns=["prompt", "mean_width", "ranked_per_round_ms",
                             "ranked_shape", "local_shape", "local_overstates_pct"])
for r in sh["rows"]:
    shape.add_data(r["prompt"], r["mean_width"], r["ranked_per_round_ms"],
                   r["ranked_shape"], r["local_shape"], r["local_overstates_pct"])
run.log({"ranked_vs_local_shape": shape})

art = wandb.Artifact("e34r2-model-autopsy", type="analysis")
art.add_file(str(REPO / "research" / "e34r2-model-autopsy.json"))
art.add_file(str(REPO / "research" / "e34r2-results.md"))
art.add_file(str(REPO / "research" / "e34r2_model_autopsy.py"))
run.log_artifact(art)

print("run_id", run.id)
print("url", run.url)
run.finish()
