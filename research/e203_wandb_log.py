#!/usr/bin/env python3
"""Publish the E203 within-prompt lookahead desk result to W&B.

    usage: research/e203_wandb_log.py \
        --stage0a-p7 research/e203-artifacts/stage0a-p7.json \
        --stage0a-adapt research/e203-artifacts/stage0a-adapt.json \
        --stage0b research/e203-artifacts/stage0b-ceiling.json \
        --control research/e203-artifacts/censoring-control.json \
        --report research/e203-artifacts/desk-report.txt

harness=ranked only. Stage 0a reads accepted-length streams from the e168
traces; Stage 0b prices them with the E201 instrument imported unchanged and
anchored on paid receipt A. No GPU work ran, so this run publishes no timing
leg and no local measurement: nothing here can be mistaken for a ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e203-within-prompt-lookahead"
BASE_SHA = "7f10e147f2e270a8893c7c3a4ebe0e05cdc2fdd8"


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True,
                          text=True).stdout.strip()


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def breakeven(psi_curve, base, target):
    xs = sorted(float(k) for k in psi_curve)
    ys = [psi_curve["%.6f" % x] for x in xs]
    want = base * (1.0 + target)
    for i in range(1, len(xs)):
        if ys[i] >= want:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (want - ys[i - 1]) \
                / (ys[i] - ys[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0a-p7", required=True)
    ap.add_argument("--stage0a-adapt", required=True)
    ap.add_argument("--stage0b", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    p7 = load(args.stage0a_p7)
    adapt = load(args.stage0a_adapt)
    b = load(args.stage0b)
    ctl = load(args.control)

    smooth = b["laws"]["smooth"]
    step9 = b["laws"]["step9"]
    cap8 = b["static_cap8"]
    psi_ship = smooth["psi"]["ship"]
    psi1 = b["psi_lag1"]
    psi_any = b["psi_any_lag"]

    def ship_at(psi):
        return psi_ship["%.6f" % psi]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e203-desk-within-prompt-lookahead",
        job_type="desk-analysis",
        tags=["e203", "harness=ranked", "desk-only", "no-gpu",
              "stage0", "verdict=not-useful"],
        config={
            "experiment": "e203-within-prompt-lookahead-desk",
            "assignment_id": "e203-within-prompt-lookahead-desk",
            "revision_id": "e203-r0",
            "harness": "ranked",
            "gpu_used": False,
            "stage_reached": "0b",
            "stage1_entered": False,
            "base_sha": BASE_SHA,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "instrument": "E201 survival-pinned latent-q, imported unchanged",
            "cost_law": "E197 merged smooth-step",
            "corpus": "research/out/e168 p7 (fixed_draft_depth=7) + adapt",
            "corpus_prompts": 9,
            "corpus_rounds_p7": sum(v["n_kept"]
                                    for v in p7["prompts"].values()),
            "corpus_trace_commit": "5a5eb96befc22ad976fe88d5e90e289ba8bacae5",
            "corpus_head_identical_to_base": True,
            "corpus_is_local_prose_not_ranked_prompts": True,
            "permutation_draws": p7["draws"],
            "ar_draws": p7["ar_draws"],
            "control_draws": ctl["draws"],
            "seed": p7["seed"],
            "comparator": "best static global cap 8 (FINDING 523)",
            "receipt_A_median": b["receipt_A"],
            "receipt_channel_sigma_pct": 0.689,
            "stop_rule_not_useful_below_pct": 0.2,
            "stop_rule_stage1_at_or_above_pct": 0.5,
        })

    summary = {
        # --- verdict ------------------------------------------------------
        "verdict": "not useful",
        "decision_quantity": "ceiling_at_measured_psi_minus_cap8_pct",
        "ceiling_at_psi_lag1_vs_cap8_pct": 100.0 * (ship_at(psi1) / cap8 - 1),
        "ceiling_at_psi_anylag_vs_cap8_pct":
            100.0 * (ship_at(psi_any) / cap8 - 1),
        "breakeven_psi_for_plus0.2pct": breakeven(psi_ship, cap8, 0.002),
        "breakeven_psi_for_plus0.5pct": breakeven(psi_ship, cap8, 0.005),
        "shortfall_factor_plus0.5pct_vs_anylag_psi":
            breakeven(psi_ship, cap8, 0.005) / psi_any,

        # --- stage 0a, open-loop p7 arm -----------------------------------
        "p7_rho1": p7["structure_budget"]["rho1"],
        "p7_rho1_null_sd": p7["structure_budget"]["rho1_sd"],
        "p7_rho1_upper95": p7["structure_budget"]["rho1_upper95"],
        "p7_rho_any_lag_upper95":
            p7["structure_budget"]["rho_any_lag_upper95"],
        "p7_acf_max_lag": p7["acf_max"]["lag"],
        "p7_acf_max_value": p7["acf_max"]["pooled"],
        "p7_acf_max_p": p7["acf_max"]["p_greater"],
        "p7_drift_pooled": p7["drift"]["pooled"],
        "p7_drift_p_two_sided": p7["drift"]["p_two_sided"],
        "p7_drift_positive_prompts": p7["drift_sign"]["positive"],
        "p7_drift_sign_binomial_p": p7["drift_sign"]["binomial_p_two_sided"],
        "p7_ema_best_alpha": p7["ema_best"]["alpha"],
        "p7_ema_best_skill": p7["ema_best"]["skill"],
        "p7_ar_best_order": p7["ar_best"]["order"],
        "p7_ar_best_skill": p7["ar_best"]["skill"],
        "p7_shipped_tracker_corr":
            p7["shipped_tracker"]["corr_forecast_vs_realised"],
        "p7_shipped_tracker_null_sd": p7["shipped_tracker"]["sd_under_null"],
        "p7_shipped_tracker_rounds": p7["shipped_tracker"]["n_rounds"],
        "p7_loo_rho1_max_abs": max(abs(v["rho1"])
                                   for v in p7["leave_one_out"].values()),
        "p7_loo_ema_skill_max_abs":
            max(abs(v["ema_skill_at_best_alpha"])
                for v in p7["leave_one_out"].values()),
        "psi_upper95_lag1": psi1,
        "psi_upper95_any_lag": psi_any,

        # --- stage 0a, closed-loop adapt arm (confounded) -----------------
        "adapt_rho1": adapt["structure_budget"]["rho1"],
        "adapt_ema_best_skill": adapt["ema_best"]["skill"],
        "adapt_ar_best_skill": adapt["ar_best"]["skill"],
        "adapt_shipped_tracker_corr":
            adapt["shipped_tracker"]["corr_forecast_vs_realised"],

        # --- positive control: how much of that is manufactured? ----------
        "control_rho1_observed": ctl["pooled"]["rho1_obs"],
        "control_rho1_manufactured_carry": ctl["pooled"]["rho1_carry"],
        "control_rho1_manufactured_discard": ctl["pooled"]["rho1_disc"],
        "control_rho1_manufactured_share_carry":
            ctl["pooled"]["rho1_carry"] / ctl["pooled"]["rho1_obs"],
        "control_tracker_observed": ctl["pooled"]["trk_obs"],
        "control_tracker_manufactured_carry": ctl["pooled"]["trk_carry"],
        "control_tracker_manufactured_discard": ctl["pooled"]["trk_disc"],
        "control_tracker_manufactured_share_carry":
            ctl["pooled"]["trk_carry"] / ctl["pooled"]["trk_obs"],

        # --- stage 0b levels, smooth 9-row law ----------------------------
        "median_receipt_A": b["receipt_A"],
        "median_static_cap8": cap8,
        "median_const_global_best": smooth["const_global"]["median"],
        "const_global_best_depth": smooth["const_global"]["depth"],
        "median_const_per_prompt": smooth["const_per_prompt"]["median"],
        "median_oracle_q": smooth["oracle_q"],
        "median_oracle_a": smooth["oracle_a"],
        "oracle_q_vs_cap8_pct": 100.0 * (smooth["oracle_q"] / cap8 - 1),
        "oracle_a_vs_cap8_pct": 100.0 * (smooth["oracle_a"] / cap8 - 1),
        "cap8_vs_A_pct": 100.0 * (cap8 / b["receipt_A"] - 1),

        # --- stage 0b levels, step9 law (sensitivity) ---------------------
        "step9_median_static_cap8": step9["cap8"],
        "step9_median_oracle_q": step9["oracle_q"],
        "step9_oracle_q_vs_cap8_pct":
            100.0 * (step9["oracle_q"] / step9["cap8"] - 1),
        "step9_breakeven_psi_for_plus0.5pct":
            breakeven(step9["psi"]["ship"], step9["cap8"], 0.005),

        # --- instrument validation gate -----------------------------------
        "gate_pass": b["gate"]["pass"],
        "gate_cap7_in_sample": b["gate"]["cap7"],
        "gate_cap4_out_of_sample": b["gate"]["cap4"],
        "gate_cap5_out_of_sample": b["gate"]["cap5"],
    }
    for cap, v in b["static_global_cap"].items():
        summary["static_cap%s" % cap] = v
    run.summary.update(summary)

    acf = wandb.Table(columns=["lag", "pooled", "null_lo", "null_hi",
                               "z", "p_two_sided"])
    for row in p7["acf"]:
        acf.add_data(row["lag"], row["pooled"], row["null_lo"],
                     row["null_hi"], row["z"], row["p_two_sided"])
    run.log({"p7_acf": acf})

    ema = wandb.Table(columns=["alpha", "skill", "null_mean", "null_hi",
                               "p_greater"])
    for row in p7["ema_skill"]:
        ema.add_data(row["alpha"], row["skill"], row["null_mean"],
                     row["null_hi"], row["p_greater"])
    run.log({"p7_ema_skill": ema})

    psi_tbl = wandb.Table(columns=["psi", "ceiling_smooth",
                                   "vs_cap8_pct_smooth",
                                   "ceiling_step9", "vs_cap8_pct_step9"])
    for k in sorted(psi_ship, key=float):
        psi_tbl.add_data(float(k), psi_ship[k],
                         100.0 * (psi_ship[k] / cap8 - 1),
                         step9["psi"]["ship"][k],
                         100.0 * (step9["psi"]["ship"][k] / step9["cap8"] - 1))
    run.log({"ceiling_vs_psi": psi_tbl})

    ctl_tbl = wandb.Table(columns=["prompt", "input_truncated_fraction",
                                   "rho1_observed", "rho1_sim_discard",
                                   "rho1_sim_carry", "tracker_observed",
                                   "tracker_sim_discard", "tracker_sim_carry"])
    for name, v in ctl["prompts"].items():
        ctl_tbl.add_data(name, v["input_truncated_fraction"],
                         v["observed_adapt"]["rho1"],
                         v["sim"]["disc"]["rho1_mean"],
                         v["sim"]["carry"]["rho1_mean"],
                         v["observed_adapt"]["tracker_corr"],
                         v["sim"]["disc"]["tracker_mean"],
                         v["sim"]["carry"]["tracker_mean"])
    run.log({"censoring_control": ctl_tbl})

    prompts = wandb.Table(columns=["prompt", "rounds_kept", "acc_mean",
                                   "acc_sd", "censored_fraction",
                                   "rho1", "drift", "tracker_corr"])
    for name, v in p7["prompts"].items():
        prompts.add_data(name, v["n_kept"], v["acc_mean"], v["acc_sd"],
                         v["censored_fraction"],
                         p7["acf"][0]["per_prompt"][name],
                         p7["drift"]["per_prompt"][name],
                         p7["shipped_tracker"]["per_prompt"][name])
    run.log({"p7_per_prompt": prompts})

    art = wandb.Artifact("e203-desk-within-prompt-lookahead",
                         type="desk-analysis")
    for f in (args.stage0a_p7, args.stage0a_adapt, args.stage0b,
              args.control, args.report):
        art.add_file(str(pathlib.Path(f)))
    for f in ("research/e203_traces.py", "research/e203_stage0a.py",
              "research/e203_stage0b.py",
              "research/e203_censoring_control.py"):
        art.add_file(f)
    run.log_artifact(art)

    print("run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
