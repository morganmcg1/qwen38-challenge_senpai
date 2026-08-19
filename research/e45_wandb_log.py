#!/usr/bin/env python3
"""Log the E45 pooled plateau-tree family-separation analysis to W&B.

Analysis only: zero GPU seconds, no timed run, no shipped-surface change.
Set WANDB_RESUME_ID to update an existing run in place.
"""
import json
import os
import pathlib
import subprocess

import wandb

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = json.loads((REPO / "research" / "e45-pooled-family.json").read_text())

meta = DOC["meta"]
poolm = meta["pool"]
noise = meta["replicate_noise"]
enum = DOC["enumeration"]
fams = DOC["families"]
cf = DOC["cross_family_delta"]
pray = DOC["pooled_ray"]
pdemo = pray["demo"]
ray = DOC["ray_equivalence"]["step6_vs_quadratic"]
best = ray["primary"]["best"]
excess = DOC["excess"]
value = DOC["value"]
shared = DOC.get("shared_shape", {})
power = DOC["power"]

ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]
SINGLE = ["linear", "step5", "step6", "step7", "quadratic"]

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                               text=True).strip()
resume_id = os.environ.get("WANDB_RESUME_ID") or None

run = wandb.init(
    project="qwen38-mlx-challenge-senpai",
    entity="wandb-applied-ai-team",
    id=resume_id,
    resume="must" if resume_id else None,
    name="e45-plateau-pooled-family-separation",
    group="e45-plateau-pooled-family-separation",
    job_type="analysis",
    tags=["e45", "r1", "analysis", "no-gpu", "ranked-telemetry",
          "partial-identification", "pooling", "identification-failure"],
    config={
        "assignment_id": "qwen38-r1-e45-plateau-pooled-family-separation",
        "revision_id": "r1",
        "pr_number": 50,
        "branch": "qwen-edward/plateau-pooled-family-separation",
        "base_sha": "efff400c1b5554be2e8993b01856653d55de7664",
        "commit_sha": head,
        "gpu_seconds_used": 0,
        "timed_runs": 0,
        "shipped_surface_diff_bytes": 0,
        "decode_tokens": 512,
        "declared_head_sha256_prefix": meta["declared_head"],
        "our_row_id": meta["our_row"],
        "our_tree": meta["our_tree"],
        "tolerance_frac": DOC["tol_frac"],
        "candidate_rows": poolm["row_count"],
        "distinct_trees": poolm["tree_count"],
        "pooled_legs": poolm["pooled_legs"],
        "shared_prompts": ",".join(poolm["shared_prompts"]),
        "duplicate_tree": poolm["duplicate_groups"][0]["tree"],
        "duplicate_rows": ",".join(poolm["duplicate_groups"][0]["rows"]),
        "duplicate_kept": poolm["duplicate_groups"][0]["kept"],
        "local_step_5_6_ms_dropped_tree": cf["local_ladder_5_6_dropped_tree"],
    },
)

metrics = {
    # --- headline verdict -------------------------------------------------
    "verdict/pooling_separates_families": 0,
    "verdict/pooling_prunes_any_reading":
        int(enum["step6"]["pooled_n"] < enum["step6"]["solo_n"]),
    "verdict/increment_identified": int(not cf["contains_zero"]),
    "verdict/step6_and_quadratic_disjoint":
        int(fams["quadratic"]["delta"]["hi"] < fams["step6"]["delta"]["lo"]),
    "verdict/ray_equivalence_feasible": int(ray["primary"]["feasible"]),
    "verdict/ray_holds_at_every_reading":
        int(ray["feasible_at_every_step6_reading"]),
    "verdict/pooled_offset_ray_feasible":
        int(pray["offset_ray"]["feasible"]),
    "verdict/pooled_x_column_identical":
        int(pray["x_identical_across_trees"]),
    "verdict/pooled_box_equals_single_row":
        int(pray["box_equals_single_row"]),

    # --- the pool ---------------------------------------------------------
    "pool/candidate_rows": poolm["row_count"],
    "pool/distinct_trees": poolm["tree_count"],
    "pool/pooled_legs": poolm["pooled_legs"],
    "pool/shared_prompts": len(poolm["shared_prompts"]),
    "pool/duplicate_groups": len(poolm["duplicate_groups"]),
    "noise/replicate_sd_pct": noise["pooled_pct"],
    "noise/replicate_leg_pairs": noise.get("n_leg_pairs", 0),
    "noise/replicate_sd_ms_at_100ms_round":
        power["replicate_sigma_ms_at_100ms_round"],

    # --- identification failure, single row and pooled --------------------
    "ray/lambda_lo": ray["primary"]["ray"]["lambda_lo"],
    "ray/lambda_hi": ray["primary"]["ray"]["lambda_hi"],
    "ray/n_both_cones_legal": ray["primary"]["n_both_cones_legal"],
    "ray/pred_gap_ms": best["pred_gap_ms"],
    "ray/map_gap": best["map_gap"],
    "ray/delta_step6_ms": best["delta_a"],
    "ray/delta_quadratic_ms": best["delta_b"],
    "ray/delta_gap_ms": abs(best["delta_a"] - best["delta_b"]),
    "pooled_ray/n_legs": pray["n_legs"],
    "pooled_ray/x_max_gap": pray["x_max_gap"],
    "pooled_ray/box_max_gap": pray["box_max_gap"],
    "pooled_ray/offset_lambda_lo": pray["offset_ray"]["lambda_lo"],
    "pooled_ray/offset_lambda_hi": pray["offset_ray"]["lambda_hi"],
    "pooled_ray/unconstrained_worst_rel_step6": pdemo["pooled_raw_a"],
    "pooled_ray/unconstrained_worst_rel_quadratic": pdemo["pooled_raw_b"],
    "pooled_ray/unconstrained_worst_rel_gap": pdemo["pooled_raw_gap"],
    "pooled_ray/unconstrained_pred_gap_ms": pdemo["pooled_pred_gap_ms"],
    "pooled_ray/cone_worst_rel_step6": pdemo["pooled_cone_a"],
    "pooled_ray/cone_worst_rel_quadratic": pdemo["pooled_cone_b"],
    "pooled_ray/trees_clamped_step6": pdemo["n_clamped_a"],
    "pooled_ray/trees_clamped_quadratic": pdemo["n_clamped_b"],
    "pooled_ray/delta_gap_ms_at_identical_misfit": pdemo["delta_gap_ms"],

    # --- the increment ----------------------------------------------------
    "delta/cross_family_lo_ms": cf["lo"],
    "delta/cross_family_hi_ms": cf["hi"],
    "delta/cross_family_width_ms": cf["hi"] - cf["lo"],
    "delta/cross_family_contains_zero": int(cf["contains_zero"]),
    "delta/contains_advisor_step_37_730": int(cf["contains_advisor_step"]),
    "delta/contains_advisor_quadratic_8_575":
        int(cf["contains_advisor_quadratic"]),
}

for family in SINGLE + ["mixture"]:
    d = fams.get(family, {}).get("delta") or {}
    metrics["delta_lo_ms/" + family] = d.get("lo")
    metrics["delta_hi_ms/" + family] = d.get("hi")
    metrics["delta_feasible/" + family] = int(d.get("lo") is not None)
    if d.get("lo") is not None:
        metrics["delta_width_ms/" + family] = d["hi"] - d["lo"]
    if "pool_keeps_lo" in d:
        metrics["delta_pool_keeps_lo/" + family] = int(
            all(d["pool_keeps_lo"].values()))
        metrics["delta_pool_keeps_hi/" + family] = int(
            all(d["pool_keeps_hi"].values()))

for family in SINGLE:
    e = enum[family]
    metrics["enum_solo_n/" + family] = e["solo_n"]
    metrics["enum_pooled_n/" + family] = e["pooled_n"]
    metrics["enum_solo_nodes/" + family] = e["solo_nodes"]
    metrics["enum_pooled_nodes/" + family] = e["pooled_nodes"]
    metrics["enum_pruned_by_pooling/" + family] = e["solo_n"] - e["pooled_n"]
    metrics["enum_identical_selections/" + family] = int(
        e["identical_selections"])

for family, sh in shared.items():
    metrics["shared_shape_upper_bound/" + family] = \
        sh["upper_bound_at_primary_reading"]
    metrics["shared_shape_certified/" + family] = int(
        sh["certified"]["all_feasible"])
    metrics["shared_shape_within_tolerance/" + family] = int(
        sh["upper_bound_at_primary_reading"] <= DOC["tol_frac"])

for family, t in (DOC.get("threshold") or {}).items():
    metrics["threshold_lower_solo/" + family] = \
        t["lower_bound_solo"].get("threshold_frac")
    metrics["threshold_lower_pooled/" + family] = \
        t["lower_bound_pooled"].get("threshold_frac")

for label in excess:
    ff = value[label]["family_free"]
    metrics["value_%s/base_score" % label] = ff["base_score"]
    metrics["value_%s/saturation_gain_pct" % label] = ff["saturation_gain_pct"]
    metrics["value_%s/saturating_leg_fraction" % label] = \
        ff["saturating_fraction"]
    metrics["value_%s/leg_fraction_for_crown" % label] = ff["needed"]["crown"]
    metrics["value_%s/leg_fraction_for_one_sigma" % label] = \
        ff["needed"]["one_sigma"]
    for family in ("step6", "quadratic"):
        for anchor in ("secant", "whole"):
            v = (value[label].get(family) or {}).get(anchor)
            if not v:
                continue
            metrics["value_%s_%s_crown_frac/%s" % (label, anchor, family)] = \
                v["fraction_needed"]["crown_hi_frac"]
            metrics["value_%s_%s_sigma_frac/%s" % (label, anchor, family)] = \
                v["fraction_needed"]["one_sigma_hi_frac"]
            metrics["value_%s_%s_full_gain_pct/%s" % (label, anchor, family)] = \
                v["arms"]["removed_1.00_hi_frac"]["score_gain_pct"]

run.log(metrics)
run.summary.update(metrics)

pool_tbl = wandb.Table(columns=[
    "row", "solver", "score", "git_tree", "legs_used", "in_pool",
    "dropped_reason"])
dup = poolm["duplicate_groups"][0]
pool_tbl.add_data(meta["our_row"], poolm.get("solvers", {}).get(
    meta["our_row"]), None, meta["our_tree"], len(ORDER), True, "ours")
for rid, tree in poolm["trees"].items():
    dropped = rid in dup["dropped"]
    pool_tbl.add_data(rid, poolm.get("solvers", {}).get(rid),
                      poolm.get("scores", {}).get(rid), tree,
                      poolm.get("legs_used", {}).get(rid), not dropped,
                      "duplicate tree %s of %s" % (tree, dup["kept"])
                      if dropped else "")
run.log({"pool_census": pool_tbl})

delta_tbl = wandb.Table(columns=["family", "delta_lo_ms", "delta_hi_ms",
                                 "width_ms", "contains_zero",
                                 "contains_advisor_step_37_730",
                                 "contains_advisor_quadratic_8_575",
                                 "n_readings", "pool_keeps_both_endpoints"])
for family in SINGLE + ["mixture"]:
    d = fams.get(family, {}).get("delta") or {}
    if d.get("lo") is None:
        delta_tbl.add_data(family, None, None, None, None, None, None,
                           d.get("n_feasible", 0), None)
        continue
    keeps = None
    if "pool_keeps_lo" in d:
        keeps = bool(all(d["pool_keeps_lo"].values())
                     and all(d["pool_keeps_hi"].values()))
    delta_tbl.add_data(family, d["lo"], d["hi"], d["hi"] - d["lo"],
                       d["lo"] <= 1e-9, d["lo"] <= 37.730 <= d["hi"],
                       d["lo"] <= 8.575 <= d["hi"],
                       d.get("n_feasible"), keeps)
run.log({"cross_family_delta": delta_tbl})

ray_tbl = wandb.Table(columns=["pair", "feasible", "lambda_lo", "lambda_hi",
                               "n_both_cones_legal", "pred_gap_ms", "map_gap",
                               "delta_a_ms", "delta_b_ms",
                               "feasible_at_every_reading", "n_readings"])
for key, v in DOC["ray_equivalence"].items():
    b = v["primary"].get("best")
    ray_tbl.add_data(key, v["primary"]["feasible"],
                     v["primary"]["ray"]["lambda_lo"],
                     v["primary"]["ray"]["lambda_hi"],
                     v["primary"]["n_both_cones_legal"],
                     b["pred_gap_ms"] if b else None,
                     b["map_gap"] if b else None,
                     b["delta_a"] if b else None,
                     b["delta_b"] if b else None,
                     v["feasible_at_every_step6_reading"],
                     v["n_readings_checked"])
run.log({"ray_equivalence": ray_tbl})

ptree_tbl = wandb.Table(columns=[
    "row", "unconstrained_worst_rel_step6", "unconstrained_worst_rel_quadratic",
    "unconstrained_gap", "pred_gap_ms", "cone_worst_rel_step6",
    "cone_worst_rel_quadratic", "clamped_step6", "clamped_quadratic",
    "delta_step6_ms", "delta_quadratic_ms"])
for rid, v in pdemo["per_tree"].items():
    ptree_tbl.add_data(rid, v["raw_a"], v["raw_b"], v["raw_gap"],
                       v["pred_gap_ms"], v["cone_a"], v["cone_b"],
                       v["clamped_a"], v["clamped_b"], v["delta_a"],
                       v["delta_b"])
run.log({"pooled_ray_per_tree": ptree_tbl})

exc_tbl = wandb.Table(columns=["row", "family", "anchor", "prompt", "mean_M",
                               "rounds", "leg_ms_per_round", "excess_lo_ms",
                               "excess_hi_ms", "excess_lo_pct_of_leg",
                               "excess_hi_pct_of_leg"])
for label, byfam in excess.items():
    for family, ex in byfam.items():
        if not ex.get("feasible"):
            continue
        for nm in ORDER:
            rec = ex["per_prompt"].get(nm)
            if not rec:
                continue
            for anchor in ("secant", "whole"):
                cell = rec[anchor]
                exc_tbl.add_data(label, family, anchor, nm, rec["x"], rec["R"],
                                 rec["y_ms"], cell["lo_ms"], cell["hi_ms"],
                                 100 * cell["lo_frac"], 100 * cell["hi_frac"])
run.log({"excess_by_family": exc_tbl})

legfrac_tbl = wandb.Table(columns=["row", "target", "leg_fraction_needed"])
for label in value:
    ff = value[label]["family_free"]
    for target, frac in ff["needed"].items():
        legfrac_tbl.add_data(label, target, frac)
run.log({"family_free_leg_fraction": legfrac_tbl})

noise_tbl = wandb.Table(columns=["prompt", "replicate_sd_pct"])
for nm in ORDER:
    if nm in noise.get("per_prompt_pct", {}):
        noise_tbl.add_data(nm, noise["per_prompt_pct"][nm])
run.log({"replicate_noise": noise_tbl})

art = wandb.Artifact("e45-pooled-family", type="analysis")
art.add_file(str(REPO / "research" / "e45-pooled-family.json"))
art.add_file(str(REPO / "research" / "e45-results.md"))
art.add_file(str(REPO / "research" / "e45_pooled_family.py"))
run.log_artifact(art)

print("run_id", run.id)
print("url", run.url)
run.finish()
