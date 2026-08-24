#!/usr/bin/env python3
"""Publish the E201 online per-prompt depth-cap desk result to W&B.

    usage: research/e201_wandb_log.py \
        --desk research/e201-artifacts/desk-online-cap.json \
        --report research/e201-artifacts/desk-report.txt

harness=ranked only. Every number is an offline replay against the merged E197
ranked round-cost law, anchored on paid receipt A and validated out of sample
against the paid cap-4 and cap-5 receipts (FINDING 520 corrected instrument).
Stage 1 ran no GPU work, so this run publishes no timing leg and no local
measurement: there is nothing here that could be mistaken for a ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e201-online-per-prompt-cap"


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True,
                          text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desk", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    desk = json.loads(pathlib.Path(args.desk).read_text())
    report = pathlib.Path(args.report)

    dec = desk["decisive"]
    ver = desk["verdict"]
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e201-desk-online-cap",
        job_type="desk-analysis",
        tags=["e201", "harness=ranked", "desk-only", "no-gpu",
              "stage1", "verdict=not-useful"],
        config={
            "experiment": "e201-online-per-prompt-cap",
            "assignment_id": "e201-online-per-prompt-cap",
            "revision_id": "e201-r0",
            "harness": "ranked",
            "stage": 1,
            "gpu_used": False,
            "instrument": "survival-pinned latent-q (FINDING 520 corrected)",
            "cost_law": "E197 merged smooth-step",
            "base_sha": "e1afcf3e4320e11cc42dc36e5481673c28407f81",
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "receipt_A_median": desk["base_receipt_A"],
            "crown": desk["crown"],
            "caps_considered": desk["caps"],
            "ema_transient_edl": desk["ema_edl"],
            "r_sweep": sorted({int(k) for k in desk["surface"]}),
            "stop_rule_not_useful_below_pct": 0.2,
            "stop_rule_stage2_at_or_above_pct": 0.5,
        })

    summary = {
        # --- decision quantity -------------------------------------------
        "verdict": "not useful",
        "decision_quantity": "online_minus_best_static_global_cap_pct",
        "online_minus_static_pct": dec["online_vs_static"],
        "oracle_minus_static_pct": dec["oracle_vs_static"],
        "online_vs_A_pct": dec["online_vs_A"],
        # --- the four levels ---------------------------------------------
        "median_receipt_A": dec["receipt_A"],
        "median_best_static": dec["static_best"],
        "best_static_cap": dec["static_best_cap"],
        "median_oracle_per_prompt_cap": dec["oracle"],
        "median_best_online_rule": dec["online"],
        # --- instrument validation ---------------------------------------
        "check_cap7_in_sample": desk["model_check"]["cap7"],
        "check_cap4_out_of_sample": desk["model_check"]["cap4"],
        "check_cap4_paid": desk["model_check"]["cap4_paid"],
        "check_cap5_out_of_sample": desk["model_check"]["cap5"],
        "check_cap5_paid": desk["model_check"]["cap5_paid"],
        # --- mechanism ----------------------------------------------------
        "depth_variance_within_prompt": desk["variance_decomposition"]["within"],
        "depth_variance_between_prompt":
            desk["variance_decomposition"]["between"],
        "depth_variance_within_share":
            desk["variance_decomposition"]["within_share"],
        "mean_score_counterfactual_gain_pct":
            desk["mean_score_counterfactual"]["gain_pct"],
        "pivotal_prompts": ",".join(desk["pivots"]),
        "best_rule_r": desk["rule_best"]["r"],
        "best_rule_c0": desk["rule_best"]["c0"],
        "best_rule_cap_lo": desk["rule_best"]["cap_lo"],
        "best_rule_cap_hi": desk["rule_best"]["cap_hi"],
        "best_rule_is_adaptive": desk["rule_best"]["adaptive"],
        # --- sensitivity ---------------------------------------------------
        "margin_9row_smooth": desk["reoptimised_by_law"]["smooth"]["margin"],
        "margin_9row_step9": desk["reoptimised_by_law"]["step9"]["margin"],
        "loo_margin_max_abs_pct": max(abs(v) for v in desk["loo"].values()),
    }
    for cap, v in desk["static_global_cap"].items():
        summary["static_cap%s_smooth" % cap] = v["smooth"]
        summary["static_cap%s_step9" % cap] = v["step9"]
    run.summary.update(summary)

    # per-prompt raw ratio at every cap, both 9-row treatments
    tbl = wandb.Table(columns=["prompt", "rank_at_cap7", "cap", "raw",
                               "p_depth_reaches_cap"])
    for name, caps in desk["cap_binding"].items():
        for cap in desk["caps"]:
            tbl.add_data(name, desk["order_at_cap7"].index(name) + 1, cap,
                         None, caps[str(cap)])
    run.log({"cap_binding": tbl})

    sweep = wandb.Table(columns=["r", "c0", "cap_lo", "cap_hi", "theta",
                                 "mean_median", "vs_A_pct", "adaptive"])
    for row in desk["rule_sweep_top"]:
        sweep.add_data(row["r"], row["c0"], row["cap_lo"], row["cap_hi"],
                       row["theta"], row["mean"], row["vs_A"], row["adaptive"])
    run.log({"rule_sweep_top": sweep})

    art = wandb.Artifact("e201-desk-online-cap", type="desk-analysis")
    art.add_file(str(pathlib.Path(args.desk)))
    art.add_file(str(report))
    art.add_file("research/e201_online_cap.py")
    run.log_artifact(art)

    print("run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
