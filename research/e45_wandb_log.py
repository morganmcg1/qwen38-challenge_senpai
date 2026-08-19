#!/usr/bin/env python3
"""Log the E45 r2 ranked-board weight-stream A/B analysis to W&B.

Analysis only: zero GPU seconds, no timed run, no shipped-surface change. The one
compiler readout (section 5 register probe) is compile-only and its numbers are
logged as config, not as timings.

Set WANDB_RESUME_ID to update an existing run in place.
"""
import json
import math
import os
import pathlib
import subprocess

import wandb

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = json.loads((REPO / "research" / "e45-stream-ab.json").read_text())

prov = DOC["provenance"]
rep = DOC["replication_floor"]
dose = DOC["dose_summary"]
refit = DOC["streams_refit"]
ident = DOC["identifiability"]
klass = DOC["pair_classification"]

# Compile-only readouts from research/e45_stream_register_probe.sh. Hard-coded
# because the probe is a shell script whose output is a report, not a JSON
# artifact; the numbers are reproduced by rerunning it.
REGISTER_PROBE = {
    "s2_streams": 2, "s2_ipg": 4, "s2_peak_live_regs": 163, "s2_allocas": 55,
    "s3_streams": 3, "s3_ipg": 3, "s3_peak_live_regs": 164, "s3_allocas": 56,
    "s2_max_threads_per_tg": 1024, "s3_max_threads_per_tg": 1024,
    "alloca_type_set_identical": True,
    "occupancy_equal": True,
    "air_lines_changed": 25345,
    "device": "Apple M4 Pro",
    "ranked_device": "M5",
}


def pooled_leg(leg, only=None, exclude=None):
    """Dof-weighted pooled relative sd of one leg across replication sets."""
    num = den = 0.0
    for s in rep["sets"]:
        for pp in s["per_prompt"]:
            if only and pp["sha"] != only:
                continue
            if exclude and pp["sha"] == exclude:
                continue
            st = pp[leg]
            if st["n"] < 2:
                continue
            num += (st["n"] - 1) * st["rel_sd"] ** 2
            den += st["n"] - 1
    return math.sqrt(num / den) if den else float("nan"), den


mtp_sd, dof = pooled_leg("mtp")
serial_sd, _ = pooled_leg("serial")
ratio_sd, _ = pooled_leg("ratio")
ctrl_mtp, _ = pooled_leg("mtp", only="c1ec5866")
ctrl_ratio, _ = pooled_leg("ratio", only="c1ec5866")
draft_mtp, _ = pooled_leg("mtp", exclude="c1ec5866")

d8, d4 = dose["8"], dose["4"]
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
    tags=["e45", "r2", "analysis", "no-gpu", "ranked-telemetry",
          "replication-floor", "weight-streams", "dispatch-table",
          "identification-failure", "underpowered-null"],
    config={
        "assignment_id": "qwen38-r1-e45-plateau-pooled-family-separation",
        "revision_id": "r2",
        "pr_number": 50,
        "branch": "qwen-edward/plateau-pooled-family-separation",
        "base_sha": "378ecc31c684e2d347e5eaff6b9350418fe0d41d",
        "analysis_base_sha": prov["base_sha"],
        "commit_sha": head,
        "gpu_seconds_used": 0,
        "timed_runs": 0,
        "shipped_surface_diff_bytes": 0,
        "decode_tokens": 512,
        # Board corpus provenance.
        "trees_scanned": prov["trees_scanned"],
        "trees_with_table": prov["trees_with_table"],
        "distinct_fingerprints": prov["distinct_fingerprints"],
        "board_rows_total": prov["board_rows_total"],
        "board_rows_with_metrics": prov["board_rows_with_metrics"],
        "distinct_trees_with_metrics": prov["distinct_trees_with_metrics"],
        "ab_header_assignment_expected": "653/476/370/10",
        "ab_header_observed": "{}/{}/{}/{}".format(
            prov["trees_scanned"], prov["trees_with_table"],
            prov["distinct_fingerprints"], 10),
        "ab_header_discrepancy_cause": "10 new submissions landed; all carry a table",
        # Identifiability.
        "n_8_observable": ident["n_8_observable"],
        "identifiability_note": ident["note"],
        "per_prompt_metric_keys": ",".join(sorted(ident["per_prompt_metric_keys"])),
        # HEAD dispatch table.
        "head_ipg_table": json.dumps(refit["head_table"], sort_keys=True),
        "head_streams": json.dumps(refit["head_streams"], sort_keys=True),
        "ladder_tree": refit["ladder_tree"],
        "ladder_tree_streams": json.dumps(refit["ladder_tree_streams"], sort_keys=True),
        "shipped_table_stream_minimal_all_widths": all(
            v["is_minimal"] for v in refit["stream_minimality"].values()),
        **{f"register_probe_{k}": v for k, v in REGISTER_PROBE.items()},
    },
    notes=(
        "Ranked-board weight-stream A/B. Primary finding is the board's own "
        "replication floor (0.7353% score rel sd from byte-identical trees "
        "resubmitted by different solvers), which is 7.5x the recorded "
        "SIGMA_SCORE_PCT and larger than both the crown gap and the stream "
        "effect. Stream cost at width 8 is +0.4910% +/- 0.3315% (t=1.48, 4/7 "
        "groups positive): an underpowered null, not an absence. Negative "
        "control passes 14/14."
    ),
)

metrics = {
    # --- primary: the replication floor ---
    "replication/pooled_score_rel_sd_pct": rep["pooled_rel_sd"] * 100,
    "replication/median_set_rel_sd_pct": rep["median_rel_sd"] * 100,
    "replication/max_set_rel_range_pct": rep["max_rel_range"] * 100,
    "replication/recorded_sigma_score_pct": rep["recorded_sigma_score_pct"],
    "replication/understatement_factor": (
        rep["pooled_rel_sd"] * 100 / rep["recorded_sigma_score_pct"]),
    "replication/crown_gap_pct": rep["crown_gap_pct"],
    "replication/crown_gap_in_sd": rep["crown_gap_pct"] / (rep["pooled_rel_sd"] * 100),
    "replication/n_sets": rep["n_sets"],
    "replication/n_rows": rep["n_rows"],
    "replication/n_sets_covariate_invariant": rep["n_sets_all_covariates_invariant"],
    # leg decomposition: is the noise common-mode or differential?
    "replication/pooled_mtp_rel_sd_pct": mtp_sd * 100,
    "replication/pooled_serial_rel_sd_pct": serial_sd * 100,
    "replication/pooled_ratio_rel_sd_pct": ratio_sd * 100,
    "replication/ratio_over_mtp_sd": ratio_sd / mtp_sd,
    "replication/leg_dof": dof,
    "replication/control_mtp_rel_sd_pct": ctrl_mtp * 100,
    "replication/control_ratio_over_mtp_sd": ctrl_ratio / ctrl_mtp,
    "replication/drafting_only_mtp_rel_sd_pct": draft_mtp * 100,
    # --- stream cost, width 8 (the assignment's target) ---
    "stream8/cluster_mean_pct": d8["cluster_mean_d_mtp_rel"] * 100,
    "stream8/cluster_se_pct": d8["cluster_se"] * 100,
    "stream8/cluster_sd_pct": d8["cluster_sd"] * 100,
    "stream8/cluster_t": d8["cluster_t"],
    "stream8/n_pairs": d8["n_pairs"],
    "stream8/n_groups": d8["n_groups"],
    "stream8/groups_positive": d8["groups_positive"],
    "stream8/drafting_prompts_positive": d8["drafting_prompts_positive"],
    "stream8/sign_test_p": d8["sign_test_two_sided_p"],
    # effect measured against the floor it has to clear
    "stream8/effect_over_replication_sd": (
        d8["cluster_mean_d_mtp_rel"] * 100 / (rep["pooled_rel_sd"] * 100)),
    # --- stream cost, width 4 (confounded by group concentration) ---
    "stream4/cluster_mean_pct": d4["cluster_mean_d_mtp_rel"] * 100,
    "stream4/cluster_se_pct": d4["cluster_se"] * 100,
    "stream4/cluster_t": d4["cluster_t"],
    "stream4/n_pairs": d4["n_pairs"],
    "stream4/n_groups": d4["n_groups"],
    "stream4/drafting_prompts_positive": d4["drafting_prompts_positive"],
    "stream4/sign_test_p": d4["sign_test_two_sided_p"],
    # --- pair admission ---
    "pairs/total": len(DOC["pairs"]),
    "pairs/dispatch_only_admitted": sum(
        v for k, v in klass.items() if k.startswith("dispatch-only")),
    "pairs/confounded_rejected": sum(
        v for k, v in klass.items() if k.startswith("confounded")),
    # --- ladder refit ---
    "refit/max_d1_drop_ms": refit["max_d1_drop"],
    "refit/quadratic_falsified": int(refit["quadratic_falsified_by_nonmonotone_d1"]),
    "refit/step6_equals_ladder_stream_model": int(
        refit["step_at_6_equals_ladder_stream_model"]),
}
for fit in refit["fits"]:
    metrics[f"refit/{fit['name']}/max_abs_resid_ms"] = fit["max_abs_resid"]
    metrics[f"refit/{fit['name']}/rms_resid_ms"] = fit["rms_resid"]
    for coef, val in fit["coef"].items():
        metrics[f"refit/{fit['name']}/coef_{coef}"] = val
for k, v in klass.items():
    metrics[f"pairs/class/{k.replace('|', '_').replace(' ', '')}"] = v
run.log(metrics)

# Replication sets: the evidence for the headline number.
rep_tbl = wandb.Table(columns=[
    "tree", "n_rows", "rel_sd_pct", "rel_range_pct", "mean_score", "min_score",
    "max_score", "covariates_invariant", "solvers"])
for s in sorted(rep["sets"], key=lambda x: -x["score"]["rel_range"]):
    sc = s["score"]
    rep_tbl.add_data(s["tree"][:12], sc["n"], sc["rel_sd"] * 100,
                     sc["rel_range"] * 100, sc["mean"], sc["min"], sc["max"],
                     bool(s["all_covariates_invariant"]),
                     ", ".join(r["solver"] for r in s["rows"]))
run.log({"replication_sets": rep_tbl})

# Per-prompt noise, with the drafting-intensity relationship that makes the
# negative control legitimate.
noise_tbl = wandb.Table(columns=["prompt", "rho", "replication_rel_sd_pct", "dof"])
for sha, v in sorted(rep["per_prompt_noise"].items(), key=lambda kv: -kv[1]["rel_sd"]):
    noise_tbl.add_data(sha, v["rho"], v["rel_sd"] * 100, v["dof"])
run.log({"per_prompt_replication_noise": noise_tbl})

# Per-group stream effect: shows the 4/7 split behind the null.
for width in ("8", "4"):
    g_tbl = wandb.Table(columns=["fingerprint", "n_pairs", "mean_d_mtp_pct"])
    for g in sorted(dose[width]["groups"], key=lambda x: -x["mean_d_mtp_rel"]):
        g_tbl.add_data(g["fp"][:12], g["n_pairs"], g["mean_d_mtp_rel"] * 100)
    run.log({f"stream{width}_groups": g_tbl})

# Per-prompt effect vs the noise that would produce it with no effect at all.
for width in ("8", "4"):
    p_tbl = wandb.Table(columns=[
        "prompt", "rho", "is_control", "mean_d_mtp_pct", "observed_pair_sd_pct",
        "expected_pair_sd_pct", "replication_rel_sd_pct", "excess_over_expected"])
    for p in sorted(dose[width]["per_prompt"], key=lambda x: -x["rho"]):
        p_tbl.add_data(p["sha"], p["rho"], bool(p["is_control"]),
                       p["mean_d_mtp_rel"] * 100, p["sd_d_mtp_rel"] * 100,
                       p["expected_pair_rel_sd"] * 100,
                       p["replication_rel_sd"] * 100,
                       p["sd_d_mtp_rel"] / p["expected_pair_rel_sd"])
    run.log({f"stream{width}_per_prompt": p_tbl})

# Negative control, pair by pair: the check that the effect is localised to
# drafting rounds rather than being a whole-run artefact.
nc_tbl = wandb.Table(columns=[
    "fingerprint", "width", "streams_lo", "streams_hi", "control_abs_d_pct",
    "heavy_mean_abs_d_pct", "ratio_heavy_over_control", "control_rank",
    "control_is_smaller", "control_nondraft_rounds"])
for nc in DOC["negative_control"]["8"]:
    ctrl = nc["control_abs_d_mtp_rel"]
    heavy = nc["heavy_mean_abs_d_mtp_rel"]
    nc_tbl.add_data(nc["fp"][:12], nc["width"], nc["streams"][0], nc["streams"][1],
                    ctrl * 100, heavy * 100,
                    (heavy / ctrl) if ctrl else float("inf"),
                    nc["control_rank"], bool(nc["control_is_smaller"]),
                    nc["control_nondraft"])
run.log({"negative_control_width8": nc_tbl})

# Admitted dispatch-only pairs at width 8: the raw evidence for the estimate.
pair_tbl = wandb.Table(columns=[
    "fingerprint", "lo_tree", "hi_tree", "lo_streams8", "hi_streams8",
    "lo_score", "hi_score", "d_score_rel_pct", "mean_d_mtp_pct",
    "mean_d_ratio_pct", "cell_lines", "lo_solvers", "hi_solvers"])
for p in sorted(DOC["doses"]["8"], key=lambda x: -x["mean_d_mtp_rel"]):
    pair_tbl.add_data(
        p["fp"][:12], p["lo"]["tree"][:12], p["hi"]["tree"][:12],
        p["lo"]["streams"]["8"], p["hi"]["streams"]["8"],
        p["lo"]["score"], p["hi"]["score"], p["d_score_rel"] * 100,
        p["mean_d_mtp_rel"] * 100, p["mean_d_ratio_rel"] * 100, p["cell_lines"],
        ", ".join(p["lo"]["solvers"]), ", ".join(p["hi"]["solvers"]))
run.log({"stream8_dispatch_only_pairs": pair_tbl})

# Stream minimality: why delta_T(8) is a calibration input and not a lever.
min_tbl = wandb.Table(columns=[
    "width_M", "legal_ipgs", "min_streams", "shipped_streams", "is_minimal"])
for m in sorted(refit["stream_minimality"], key=int):
    v = refit["stream_minimality"][m]
    min_tbl.add_data(int(m), ",".join(str(i) for i in v["legal_ipgs"]),
                     v["min_streams"], v["shipped"], bool(v["is_minimal"]))
run.log({"stream_minimality": min_tbl})

# Ladder fits, including the deliberately mismatched HEAD-vector arm that shows
# the ladder does not belong to HEAD's dispatch tree.
fit_tbl = wandb.Table(columns=[
    "fit", "max_abs_resid_ms", "rms_resid_ms", "const", "M", "stream_or_step"])
for fit in sorted(refit["fits"], key=lambda f: f["rms_resid"]):
    c = fit["coef"]
    fit_tbl.add_data(fit["name"], fit["max_abs_resid"], fit["rms_resid"],
                     c.get("const"), c.get("M"),
                     c.get("stream", c.get("step", c.get("M2"))))
run.log({"ladder_fits": fit_tbl})

art = wandb.Artifact("e45-stream-ab", type="analysis")
art.add_file(str(REPO / "research" / "e45-stream-ab.json"))
art.add_file(str(REPO / "research" / "e45-results.md"))
art.add_file(str(REPO / "research" / "e45_stream_ab.py"))
art.add_file(str(REPO / "research" / "e45_stream_register_probe.sh"))
run.log_artifact(art)

print(f"logged {run.url}")
run.finish()
