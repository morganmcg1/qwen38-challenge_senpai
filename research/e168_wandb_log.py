#!/usr/bin/env python3
"""Publish the E168 margin-clamp calibration evidence to W&B.

    usage: research/e168_wandb_log.py --report research/out/e168/report.json

Everything logged here is COUNTS ONLY. The census legs run with
`MLXFAST_LOCAL_COOL_GATE=0` and with the per-round phase trace ON, which
perturbs round wall time. No number in this run is a timing measurement, a
gate-qualified figure or a score, and the gate flags say so verbatim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e168-margin-clamp-calibration"

BASE_SHA = "dd844007164e9155f2bc63a69d50d1977bdbdd38"
CAMPAIGN_BEST = "5a9f130a"
CAMPAIGN_BEST_SCORE = 3.70784519415395
CROWN = "ec24d591"
CROWN_SCORE = 3.7291100105909
PINNED_HEAD = "559b24eb"

# Shipped clamp temperatures under test.
SHIPPED_T = {"0": 2.0, "1": 3.0}


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty() -> int:
    return len(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    ) - 1


def gate_flags() -> dict:
    return {
        "leg_kind": "e168-counts-only-census",
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "timing_claims_permitted": False,
        "phase_trace_perturbs_round_time": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--host", default="apple-m4-pro-applegpu_g16s-20core-48gib")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    pinned = report.get("pinned", {})
    adapt = report.get("adapt", {})

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="calibration-census",
        name="e168-margin-clamp-calibration",
        config={
            "experiment": "e168-margin-clamp-calibration",
            "question": (
                "is the pending-primary top-2 margin predictive of draft "
                "acceptance at positions 0 and 1, and is "
                "min(EMA, sigmoid(margin/T)) a calibrated way to use it"
            ),
            "base_sha": BASE_SHA,
            "candidate_sha": git_head(),
            "dirty_files": git_dirty(),
            "campaign_best": CAMPAIGN_BEST,
            "campaign_best_score": CAMPAIGN_BEST_SCORE,
            "crown": CROWN,
            "crown_score": CROWN_SCORE,
            "pinned_head": PINNED_HEAD,
            "host": args.host,
            "chip": "Apple M4 Pro",
            "ranked_runner_chip": "M5",
            "decode_tokens": 512,
            "offered_depth": report.get("offered_depth", 8),
            "shipped_clamp_temperature_pos0": SHIPPED_T["0"],
            "shipped_clamp_temperature_pos1": SHIPPED_T["1"],
            "head_step_cost_ratio": 0.18,
            "accept_ema_alpha": 0.15,
            "segmented_verify_depth_cap": 7,
            "pinned_arm_env": "MLX_E159_FIXED_DRAFT_DEPTH=7",
            "reproduce_pinned": (
                "research/e168_collect.sh p7 english medicine natural_history "
                "technical dramatic narrative philosophy travel benchfixture"
            ),
            "reproduce_adapt": (
                "research/e168_collect.sh adapt english medicine "
                "natural_history technical dramatic narrative philosophy "
                "travel benchfixture"
            ),
            "reproduce_analysis": (
                "research/e168_margin_calibration.py "
                "--pinned research/out/e168/p7/* "
                "--adapt research/out/e168/adapt/* "
                "--json research/out/e168/report.json"
            ),
            **gate_flags(),
        },
    )

    summary: dict = {}

    if adapt:
        pooled = adapt["pooled"]
        summary.update(
            {
                "adapt/rounds": pooled["rounds"],
                "adapt/replay_mismatch": pooled["replay_mismatch"],
                "adapt/bind_pos0_fraction": pooled["bind_pos0_fraction"],
                "adapt/bind_pos1_fraction": pooled["bind_pos1_fraction"],
                "adapt/depth_loss_fraction": pooled["depth_loss_fraction"],
                "adapt/mean_depth_removed": pooled["mean_depth_removed"],
                "adapt/pure_loss_fraction": pooled["pure_loss_fraction"],
                "adapt/cap_stop_fraction": pooled["cap_stop_fraction"],
            }
        )
        table = wandb.Table(
            columns=[
                "prompt",
                "rounds",
                "bind_fraction",
                "depth_loss_fraction",
                "mean_rows_removed",
                "cap_stop_fraction",
                "mean_offered_depth",
                "accept_fraction",
            ]
        )
        for leg in adapt["legs"]:
            table.add_data(
                leg["label"],
                leg["rounds"],
                leg["bind_fraction"],
                leg["depth_loss_fraction"],
                leg["mean_depth_loss"],
                leg["cap_stop_fraction"],
                leg["mean_d"],
                leg["accept_fraction"],
            )
        run.log({"adapt/per_prompt": table})

        profile = wandb.Table(
            columns=["position", "q", "ci_low", "ci_high", "n"]
        )
        for row in adapt["profile"]:
            profile.add_data(
                row["position"], row["q"], row["ci_low"], row["ci_high"],
                row["reached"],
            )
        run.log({"adapt/position_profile": profile})

    if pinned:
        table = wandb.Table(
            columns=[
                "prompt",
                "rounds",
                "mean_offered_depth",
                "mean_accepted",
                "accept_fraction",
                "mean_margin",
            ]
        )
        for leg in pinned["legs"]:
            table.add_data(
                leg["label"],
                leg["rounds"],
                leg["mean_d"],
                leg["mean_acc"],
                leg["accept_fraction"],
                leg["margin_mean"],
            )
        run.log({"pinned/per_prompt": table})

        profile = wandb.Table(
            columns=["position", "q", "ci_low", "ci_high", "n"]
        )
        for row in pinned["profile"]:
            profile.add_data(
                row["position"], row["q"], row["ci_low"], row["ci_high"],
                row["reached"],
            )
        run.log({"pinned/position_profile": profile})

        for position, block in pinned["positions"].items():
            summary.update(
                {
                    f"pinned/pos{position}/n": block["n"],
                    f"pinned/pos{position}/base_rate": block["base_rate"],
                    f"pinned/pos{position}/fitted_temperature": block[
                        "fitted_temperature"
                    ],
                    f"pinned/pos{position}/shipped_temperature": block[
                        "shipped_temperature"
                    ],
                    f"pinned/pos{position}/auc_margin": block["auc_margin"],
                }
            )
            for name, scores in block["predictors"].items():
                if scores.get("n"):
                    summary[f"pinned/pos{position}/{name}/log_loss"] = scores[
                        "log_loss"
                    ]
                    summary[f"pinned/pos{position}/{name}/brier"] = scores["brier"]
                    summary[f"pinned/pos{position}/{name}/mean_p"] = scores[
                        "mean_prediction"
                    ]

            calib = wandb.Table(
                columns=[
                    "bin",
                    "n",
                    "mean_margin",
                    "empirical_accept",
                    "ci_low",
                    "ci_high",
                    "shipped_clamp_sigmoid",
                    "mean_ema",
                ]
            )
            for row in block["table"]:
                calib.add_data(
                    row["bin"],
                    row["n"],
                    row["mean_margin"],
                    row["empirical"],
                    row["ci_low"],
                    row["ci_high"],
                    row["clamp_sigmoid"],
                    row["mean_ema"],
                )
            run.log({f"pinned/calibration_pos{position}": calib})

    run.summary.update(summary)
    run.finish()
    print(f"e168_wandb_log: logged {len(summary)} summary metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
