#!/usr/bin/env python3
"""Log the E43 ranked partial-identification analysis to W&B.

Analysis only: zero GPU seconds, no timed run, no shipped-surface change.
Set WANDB_RESUME_ID to update an existing run in place.
"""
import json
import os
import pathlib
import subprocess

import wandb

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = json.loads((REPO / "research" / "e43-ranked-step.json").read_text())

inp = DOC["inputs"]
rec = DOC["round_recovery"]
enum = rec["enumeration"]
union = rec["union_over_feasible_selections"]
th = DOC["deliverable_a_family_thresholds"]
mc = DOC["deliverable_a_model_comparison"]
lin_tol = DOC["deliverable_a_linear_only_tolerance"]
arm = DOC["deliverable_b_headline_arm"]
brk = DOC["deliverable_b_step_brackets"][arm]["no_T1_bound"]
cal = DOC["deliverable_b_tolerance_calibration"]
phi = DOC["deliverable_c_phi"]
val = DOC["value"]
mde = DOC["mde"]
ver = DOC["verdicts"]

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                               text=True).strip()
resume_id = os.environ.get("WANDB_RESUME_ID") or None

run = wandb.init(
    project="qwen38-mlx-challenge-senpai",
    entity="wandb-applied-ai-team",
    id=resume_id,
    resume="must" if resume_id else None,
    name="e43-ranked-rho-step-vs-linear",
    group="e43-ranked-rho-step-vs-linear",
    job_type="analysis",
    tags=["e43", "r1", "analysis", "no-gpu", "ranked-telemetry",
          "partial-identification", "cost-model"],
    config={
        "assignment_id": "qwen38-r1-e43-ranked-rho-step-vs-linear",
        "revision_id": "r1",
        "pr_number": 48,
        "branch": "qwen-edward/ranked-rho-step-vs-linear",
        "base_sha": "04ad6bf11437c269df85a47e91faa769c74fe6da",
        "commit_sha": head,
        "gpu_seconds_used": 0,
        "timed_runs": 0,
        "shipped_surface_diff_bytes": 0,
        "decode_tokens": 512,
        "declared_head_sha256_prefix": inp["declared_head"],
        "corpus_rows": inp["corpus_rows"],
        "our_row_id": inp["our_row"]["id"],
        "our_row_status": inp["our_row"]["status"],
        "frontier_row_id": inp["frontier_row"]["id"],
        "zero_draft_anchor_row": inp["zero_draft_anchor"]["id"],
        "local_ladder_ms": inp["local_ladder_ms"],
        "local_step_5_6_ms": inp["local_step_5_6_ms"],
        "prereg_tolerance_frac": DOC["prereg"]["tolerance_frac"],
        "prereg_decisive_residual_ratio":
            DOC["prereg"]["decisive_residual_ratio"],
        "prereg_inconclusive_residual_ratio":
            DOC["prereg"]["inconclusive_residual_ratio"],
        "headline_tolerance_arm": arm,
        "headline_tolerance_frac": brk["tol_frac"],
        "generated_utc": DOC["generated_utc"],
    },
)

metrics = {
    # --- verdicts ---
    "verdict/superlinearity_decisive":
        int(ver["superlinearity"] == "decisive"),
    "verdict/step_vs_smooth_discriminated":
        int(ver["step_vs_smooth_assumption_free"] != "no discrimination"),
    "verdict/local_step_inside_bracket": int(ver["local_step_inside"]),
    "verdict/zero_step_inside_bracket": int(ver["zero_step_inside"]),
    "verdict/e34r2_19pct_discount_supported":
        int(ver["e34r2_claim_supported"]),
    "verdict/reject_cost_explains_curvature":
        int(ver["reject_cost_explains_curvature"]),

    # --- (a) assumption-free family test ---
    "a/threshold_linear_frac": th["linear"]["threshold_frac"],
    "a/threshold_linear_reject_frac": th["linear_reject"]["threshold_frac"],
    "a/threshold_step_frac": th["step"]["threshold_frac"],
    "a/threshold_quadratic_frac": th["quadratic"]["threshold_frac"],
    "a/linear_rejection_factor_assumption_free":
        ver["linear_rejection_factor_assumption_free"],
    "a/linear_rejection_factor_maxent": lin_tol["rejection_factor"],
    "a/linear_required_tol_frac_at_primary_reading":
        lin_tol["required_tol_frac"],

    # --- (a) maxent-rho model comparison ---
    "a/residual_ratio_linear_over_step": mc["ratios"]["linear_over_step"],
    "a/residual_ratio_quadratic_over_step":
        mc["ratios"]["quadratic_over_step"],
    "a/residual_ratio_linear_reject_over_step":
        mc["ratios"]["linear_reject_over_step"],
    "a/step_magnitude_maxent_ms": mc["step_magnitude_ms_maxent"],
    "a/step_magnitude_with_reject_term_ms":
        mc["step_magnitude_ms_with_reject_term"],
    "a/reject_cost_ms_per_reject": mc["reject_cost_ms_per_reject"],
    "a/linear_implied_T1_ms": mc["linear_T1_ms"],
    "a/step_implied_T1_ms": mc["step_T1_ms"],

    # --- round recovery, no T(1) bound ---
    "recovery/cross_product_size": enum["cross_product_size"],
    "recovery/selections_feasible": union["n_selections"],
    "recovery/nodes_visited": enum["nodes_visited"],
    "recovery/prompts_pinned": len(enum["pinned_rounds"]),
    "recovery/primary_matches_monotone_rho":
        int(rec["primary_matches_monotone_rho"]),
    "recovery/monotone_rho_unique": int(rec["monotone_rho_selection"]["unique"]),

    # --- T(1) from the zero-accept row ---
    "T1/upper_ms_assumption_free": inp["T1_bounds_ms"]["kappa_local"]["upper_ms"],
    "T1/lower_ms_kappa_local": inp["T1_bounds_ms"]["kappa_local"]["lower_ms"],
    "T1/kappa_local": inp["T1_bounds_ms"]["kappa_local"]["kappa"],
    "T1/lower_ms_kappa_10": inp["T1_bounds_ms"]["kappa_10"]["lower_ms"],
    "T1/anchor_k_mean": inp["zero_draft_anchor"]["k_mean"],

    # --- (b) step brackets ---
    "b/step_lo_ms": brk["s_lo"],
    "b/step_hi_ms": brk["s_hi"],
    "b/step_bracket_width_ms": ver["bracket_width_ms"],
    "b/step_bracket_width_over_local": ver["bracket_width_over_local"],
    "b/local_step_ms": inp["local_step_5_6_ms"],
    "b/union_step_lo_ms": union.get("s_lo"),
    "b/union_step_hi_ms": union.get("s_hi"),
    "b/calibrated_tol_frac": cal["tol_frac"],
    "b/calibrated_fp_rate": cal["fp"],
    "b/one_sigma_band_fp_rate": cal["scan"][0]["fp"],

    # --- MDE ---
    "mde/mde_80pct_ms": mde["mde_80pct_ms"],
    "mde/false_positive_rate_at_s0": mde["false_positive_rate_at_s0"],
    "mde/draws": mde["draws"],

    # --- value ---
    "value/base_score": val["base_score"],
    "value/crown_gap_pct": val["crown_gap_pct"],
    "value/sigma_score_pct": val["sigma_score_pct"],
}

FAMILIES = ("linear", "linear_reject", "step", "quadratic", "step_reject")
for fam in FAMILIES:
    metrics["a/rms_%s_ms" % fam] = mc[fam]["rms_ms"]
    metrics["a/r2_%s" % fam] = mc[fam]["r2"]
    metrics["a/chi2_per_dof_%s" % fam] = mc[fam]["chi2_per_dof"]

for key, a in val["arms"].items():
    metrics["value/gain_pct_" + key] = a["score_gain_pct"]
for key, f in val["fraction_needed"].items():
    metrics["value/fraction_needed_" + key] = -1.0 if f is None else f

for o in DOC["observations"]:
    n = o["name"]
    metrics["ranked_per_round_ms/" + n] = o["y"]
    metrics["ranked_rounds/" + n] = o["R"]
    metrics["ranked_mean_M/" + n] = o["x"]
    metrics["ranked_alpha/" + n] = o["alpha"]
    metrics["ranked_rejects_per_round/" + n] = o["rejects_per_round"]
    metrics["q_lo/" + n] = o["q_lo"]
    metrics["q_hi/" + n] = o["q_hi"]
    e = brk["excess"][n]
    metrics["excess_lo_ms/" + n] = e["e_lo_ms"]
    metrics["excess_hi_ms/" + n] = e["e_hi_ms"]
    metrics["excess_lo_frac_of_leg/" + n] = e["e_lo_frac"]
    metrics["excess_hi_frac_of_leg/" + n] = e["e_hi_frac"]

for n in ("beagle", "medicine"):
    p = phi["prompts"][n]
    metrics["phi_pass_count_Mge6_lo/" + n] = p["pass_count/M_ge_6"]["lo"]
    metrics["phi_pass_count_Mge6_hi/" + n] = p["pass_count/M_ge_6"]["hi"]
    metrics["phi_pass_count_Meq6_lo/" + n] = p["pass_count/M_eq_6"]["lo"]
    metrics["phi_pass_count_Meq6_hi/" + n] = p["pass_count/M_eq_6"]["hi"]
    metrics["phi_row_share_Mge6_lo/" + n] = p["row_share_M_ge_6"]["lo"]
    metrics["phi_row_share_Mge6_hi/" + n] = p["row_share_M_ge_6"]["hi"]
    metrics["phi_pass_plus_row_Mge6_lo/" + n] = p["pass_plus_row/M_ge_6"]["lo"]
    metrics["phi_pass_plus_row_Mge6_hi/" + n] = p["pass_plus_row/M_ge_6"]["hi"]

metrics["ladder_split/F_ms_per_row"] = phi["ladder_split"]["F_ms_per_row"]
metrics["ladder_split/S_ms_per_extra_pass"] = \
    phi["ladder_split"]["S_ms_per_extra_pass"]
metrics["ladder_split/r2"] = phi["ladder_split"]["r2"]
metrics["ladder_split/implied_step_5_to_6_ms"] = \
    phi["ladder_split"]["implied_step_5_to_6"]

run.log(metrics)
run.summary.update(metrics)

obs_tbl = wandb.Table(columns=[
    "prompt", "mean_M", "rounds", "accepted", "proposed", "alpha", "raw_p",
    "per_round_ms", "rho1", "q_lo", "q_hi", "excess_lo_ms", "excess_hi_ms",
    "excess_lo_pct_of_leg", "excess_hi_pct_of_leg", "readings_surviving"])
for o in DOC["observations"]:
    e = brk["excess"][o["name"]]
    obs_tbl.add_data(
        o["name"], o["x"], o["R"], o["A"], o["D"], o["alpha"], o["ratio"],
        o["y"], o["rho1"], o["q_lo"], o["q_hi"], e["e_lo_ms"], e["e_hi_ms"],
        100 * e["e_lo_frac"], 100 * e["e_hi_frac"],
        len(enum["readings_surviving"][o["name"]]))
run.log({"ranked_admissible_set": obs_tbl})

fam_tbl = wandb.Table(columns=["family", "params", "threshold_pct_slack",
                               "x_measured_pair_noise", "rms_maxent_ms",
                               "r2_maxent", "chi2_per_dof_maxent"])
for fam in FAMILIES:
    t = th.get(fam, {}).get("threshold_frac")
    fam_tbl.add_data(fam, mc[fam]["params"], None if t is None else 100 * t,
                     None if t is None else t / 0.00281,
                     mc[fam]["rms_ms"], mc[fam]["r2"], mc[fam]["chi2_per_dof"])
run.log({"family_comparison": fam_tbl})

brk_tbl = wandb.Table(columns=["arm", "tol_pct", "T1_bound", "s_lo_ms",
                               "s_hi_ms", "width_ms", "contains_zero",
                               "contains_local_32_850"])
for label, entry in DOC["deliverable_b_step_brackets"].items():
    for bound, b in entry.items():
        brk_tbl.add_data(label, 100 * b["tol_frac"], bound, b["s_lo"],
                         b["s_hi"], b["s_hi"] - b["s_lo"],
                         b["s_lo"] <= 0.0,
                         b["s_lo"] <= 32.850 <= b["s_hi"])
brk_tbl.add_data("union_over_42_readings", 100 * union["tol_frac"], "none",
                 union.get("s_lo"), union.get("s_hi"),
                 union.get("s_hi", 0) - union.get("s_lo", 0),
                 union.get("s_lo", 1) <= 0.0,
                 union.get("s_lo", 1) <= 32.850 <= union.get("s_hi", 0))
run.log({"step_brackets": brk_tbl})

phi_tbl = wandb.Table(columns=["prompt", "weighting", "cell", "phi_lo",
                               "phi_hi"])
for n, p in phi["prompts"].items():
    for key, v in p.items():
        if key in ("round_share_M_ge_6", "row_share_M_ge_6"):
            phi_tbl.add_data(n, key, "M_ge_6", v["lo"], v["hi"])
        else:
            weighting, cell = key.split("/")
            phi_tbl.add_data(n, weighting, cell, v["lo"], v["hi"])
run.log({"phi_brackets": phi_tbl})

mde_tbl = wandb.Table(columns=["s_true_ms", "power"])
for pt in mde["curve"]:
    mde_tbl.add_data(pt["s_true_ms"], pt["power"])
run.log({"mde_curve": mde_tbl})

art = wandb.Artifact("e43-ranked-step", type="analysis")
art.add_file(str(REPO / "research" / "e43-ranked-step.json"))
art.add_file(str(REPO / "research" / "e43-results.md"))
art.add_file(str(REPO / "research" / "e43_ranked_step.py"))
run.log_artifact(art)

print("run_id", run.id)
print("url", run.url)
run.finish()
