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
import sys

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e168_margin_calibration import QUOTED_LADDER  # noqa: E402


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
            "ranked_round_fixed_ms": 16.1585,
            "ranked_row_ms": 5.3350,
            "ranked_prefill_ms_per_token": 1.0310,
            "ranked_serial_ms_per_token": 37.92,
            "reproduce_census": (
                "research/e168_collect.sh adapt,p7 english medicine "
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

    arms = report.get("arms", {})
    if arms:
        per_prompt = wandb.Table(
            columns=[
                "prompt", "arm", "rounds", "mean_depth", "mean_accepted",
                "accept_fraction", "tokens_per_round", "at_cap_fraction",
                "cap_stop_fraction", "accept_fraction_first_half",
                "accept_fraction_second_half", "ranked_raw", "ranked_round_ms",
            ]
        )
        selection = wandb.Table(
            columns=[
                "prompt", "adapt_accept_fraction", "pinned_accept_fraction",
                "selection_gain", "adapt_mean_depth", "pinned_mean_depth",
                "raw_adapt", "raw_pinned", "raw_ratio",
                "pinned_decay_per_position", "pinned_first_step_decay",
            ]
        )
        depth_bins = wandb.Table(
            columns=[
                "prompt", "arm", "chosen_depth", "rounds", "proposed",
                "accepted", "accept_fraction", "tokens_per_round",
            ]
        )
        profiles = wandb.Table(
            columns=["prompt", "arm", "position", "q", "ci_low", "ci_high", "n"]
        )
        for row in arms["prompts"]:
            for arm, price_key, profile_key in (
                ("adapt", "adapt_price", "adapt_profile"),
                ("pinned", "pinned_price", "pinned_profile"),
            ):
                counts = row[arm]
                price = row[price_key]
                per_prompt.add_data(
                    row["label"], arm, counts["rounds"], counts["mean_depth"],
                    counts["mean_accepted"], counts["accept_fraction"],
                    counts["tokens_per_round"], counts["at_cap_fraction"],
                    counts["cap_stop_fraction"],
                    counts["accept_fraction_first_half"],
                    counts["accept_fraction_second_half"],
                    price["raw"], price["round_ms"],
                )
                for depth, bucket in counts["depth_bins"].items():
                    depth_bins.add_data(
                        row["label"], arm, int(depth), bucket["rounds"],
                        bucket["proposed"], bucket["accepted"],
                        bucket["accept_fraction"], bucket["tokens_per_round"],
                    )
                for entry in row[profile_key]:
                    profiles.add_data(
                        row["label"], arm, entry["position"], entry["q"],
                        entry["ci_low"], entry["ci_high"], entry["reached"],
                    )
            decay = row["pinned_decay"]
            selection.add_data(
                row["label"], row["adapt"]["accept_fraction"],
                row["pinned"]["accept_fraction"], row["selection_gain"],
                row["adapt"]["mean_depth"], row["pinned"]["mean_depth"],
                row["raw_adapt"], row["raw_pinned"], row["raw_ratio"],
                decay.get("decay_per_position"), decay.get("first_step_decay"),
            )
        run.log(
            {
                "arms/per_prompt": per_prompt,
                "arms/selection_and_price": selection,
                "arms/depth_bins": depth_bins,
                "arms/position_profiles": profiles,
            }
        )
        prompts = arms["prompts"]
        summary["arms/prompts"] = len(prompts)
        if prompts:
            summary["arms/median_raw_ratio"] = sorted(
                row["raw_ratio"] for row in prompts
            )[len(prompts) // 2]
            summary["arms/mean_selection_gain"] = sum(
                row["selection_gain"] for row in prompts
            ) / len(prompts)

    splice = report.get("splice", {})
    if splice:
        pooled = splice["pooled"]
        summary.update({f"splice/{key}": value for key, value in pooled.items()})
        if pooled.get("removed_resolved"):
            observed = pooled["removed_accepted"] / pooled["removed_resolved"]
            predicted = pooled["removed_predicted"] / pooled["removed_resolved"]
            summary["splice/pure_loss_rate"] = observed
            summary["splice/position_only_prediction"] = predicted
            summary["splice/clamp_skill"] = observed - predicted

    signals = report.get("signal_auc", {})
    if signals:
        table = wandb.Table(
            columns=["position", "n", "base_rate", "signal", "auc", "ci_low", "ci_high"]
        )
        for position, block in signals.items():
            if not block:
                continue
            for name, score in block["auc"].items():
                table.add_data(
                    int(position),
                    block["n"],
                    block["base_rate"],
                    name,
                    score["auc"],
                    score["ci_low"],
                    score["ci_high"],
                )
                summary[f"signal_auc/pos{position}/{name}"] = score["auc"]
        run.log({"pinned/signal_auc": table})

    oracle = report.get("oracle_ceiling", [])
    if oracle:
        table = wandb.Table(columns=["prompt", "rounds", "mean_depth", "raw"])
        for entry in oracle:
            table.add_data(
                entry["label"], entry["rounds"], entry["mean_depth"], entry["raw"]
            )
        run.log({"pinned/oracle_ceiling": table})
        values = sorted(entry["raw"] for entry in oracle)
        summary["oracle/median_raw"] = values[len(values) // 2]

    sweep = report.get("policy_sweep", {})
    if sweep.get("pooled"):
        table = wandb.Table(
            columns=["policy", "median_raw", "min_raw", "max_raw", "deficit", "prompts"]
        )
        for row in sweep["pooled"]:
            table.add_data(
                row["policy"],
                row["median_raw"],
                row["min_raw"],
                row["max_raw"],
                row["median_deficit_vs_best"],
                row["prompts"],
            )
            summary[f"policy/{row['policy']}"] = row["median_raw"]
        run.log({"policy/median_by_policy": table})
        detail = wandb.Table(
            columns=[
                "prompt",
                "policy",
                "mean_depth",
                "predicted_accepted",
                "raw",
                "deficit",
                "local_decode_ms_per_token",
            ]
        )
        for label, rows in sweep["per_prompt"].items():
            for row in rows:
                detail.add_data(
                    label,
                    row["policy"],
                    row["mean_depth"],
                    row["predicted_accepted"],
                    row["raw"],
                    row["raw_deficit_vs_best"],
                    row["local_decode_ms_per_token"],
                )
        run.log({"policy/per_prompt": detail})

    counter = report.get("clamp_counterfactual", {})
    if counter.get("legs"):
        fields = [
            "rounds",
            "mean_depth_clamped",
            "mean_depth_unclamped",
            "predicted_accepted_clamped",
            "measured_accepted",
            "model_error",
            "raw_clamped",
            "raw_unclamped",
            "raw_gain_from_clamp",
        ]
        table = wandb.Table(columns=["prompt", *fields])
        for entry in counter["legs"]:
            table.add_data(entry["label"], *(entry[field] for field in fields))
        run.log({"policy/clamp_counterfactual": table})
        summary.update(
            {f"clamp_counterfactual/{k}": v for k, v in counter["pooled"].items()}
        )

    value_curve = report.get("auc_value_curve", {})
    if value_curve.get("curve"):
        fields = [
            "auc",
            "raw",
            "gap_recovered",
            "mean_depth",
            "mean_accepted",
            "floor",
            "tau",
            "replicate_spread",
            "break_even_ms_per_round",
            "break_even_wasted_head_steps",
        ]
        table = wandb.Table(columns=fields)
        for row in value_curve["curve"]:
            table.add_data(*(row[field] for field in fields))
        run.log({"value/auc_curve": table})
        for row in value_curve["curve"]:
            key = f"{row['auc']:.3f}".replace(".", "p")
            summary[f"value/auc{key}/raw"] = row["raw"]
            summary[f"value/auc{key}/gap_recovered"] = row["gap_recovered"]
            summary[f"value/auc{key}/budget_ms_per_round"] = row[
                "break_even_ms_per_round"
            ]
        summary["value/best_constant_depth"] = value_curve["best_constant_depth"]
        summary["value/best_constant_raw"] = value_curve["best_constant_raw"]
        summary["value/oracle_raw"] = value_curve["oracle_raw"]
        summary["value/head_step_ms"] = value_curve["head_step_ms"]
        constants = wandb.Table(columns=["depth", "raw"])
        for depth, raw in sorted(value_curve["constants"].items()):
            constants.add_data(int(depth), raw)
        run.log({"value/constant_depths": constants})

    timing = report.get("timing", {})
    if timing.get("legs"):
        fields = [
            "prompt",
            "arm",
            "replicate",
            "started",
            "gpu_temp_entry",
            "gpu_temp_exit",
            "mean_depth",
            "mean_accepted",
            "seconds_per_token",
            "decode_ms_per_token",
            "predicted_decode_ms_per_token",
            "residual_pct",
            "matched",
            "cool_gate_passed_real_gate",
            "gate_qualified_for_timing",
        ]
        table = wandb.Table(columns=fields)
        for leg in timing["legs"]:
            table.add_data(*(leg[field] for field in fields))
        run.log({"timing/legs": table})
        contrast = wandb.Table(
            columns=[
                "prompt",
                "arm_a",
                "arm_b",
                "legs_per_arm",
                "decode_ms_a",
                "decode_ms_b",
                "measured_pct",
                "modelled_pct",
                "entry_temp_spread_c",
                "all_matched",
            ]
        )
        for row in timing["prompts"]:
            first, second = row["arms"]
            contrast.add_data(
                row["prompt"],
                first,
                second,
                row["legs_per_arm"][first],
                row["decode_ms_per_token"][first],
                row["decode_ms_per_token"][second],
                row["measured_pct"],
                row["modelled_pct"],
                row["entry_temp_spread_c"],
                row["all_matched"],
            )
            summary[f"timing/{row['prompt']}/measured_pct"] = row["measured_pct"]
            summary[f"timing/{row['prompt']}/modelled_pct"] = row["modelled_pct"]
        run.log({"timing/contrast": contrast})
        residuals = [abs(leg["residual_pct"]) for leg in timing["legs"]]
        summary["timing/max_abs_round_law_residual_pct"] = max(residuals)
        summary["timing/legs"] = len(timing["legs"])

    ladder = report.get("break_even_ladder")
    if ladder:
        table = wandb.Table(
            columns=["step", "q_star", "quoted", "delta", "survival_threshold"]
        )
        for row in ladder:
            quoted = QUOTED_LADDER.get(row["step"])
            table.add_data(
                row["step"],
                row["q_star_homogeneous"],
                quoted,
                (row["q_star_homogeneous"] - quoted) if quoted else None,
                row["survival_threshold_at_q_star"],
            )
            summary[f"ladder/{row['step']}/q_star"] = row["q_star_homogeneous"]
        run.log({"ladder/break_even": table})
        summary["ladder/quoted_rows_reproduced"] = sum(
            1
            for row in ladder
            if QUOTED_LADDER.get(row["step"]) is not None
            and abs(row["q_star_homogeneous"] - QUOTED_LADDER[row["step"]]) < 1e-4
        )

    go_no_go = report.get("depth_go_no_go")
    if go_no_go:
        fields = [
            "step",
            "p_d",
            "reached",
            "ci_low",
            "ci_high",
            "survival",
            "threshold",
            "margin",
            "verdict",
            "extrapolated",
        ]
        table = wandb.Table(columns=["prompt", *fields])
        for label in sorted(go_no_go):
            for row in go_no_go[label]:
                table.add_data(label, *(row[field] for field in fields))
                if not row["extrapolated"]:
                    summary[f"pd/{label}/{row['step']}"] = row["p_d"]
        run.log({"ladder/go_no_go": table})
        deepest = {
            label: max(
                (row["depth"] for row in rows if row["verdict"] == "go"), default=-1
            )
            for label, rows in go_no_go.items()
        }
        for label, depth in deepest.items():
            summary[f"ladder/{label}/deepest_paying_depth"] = depth + 1

    transfer = report.get("optimism_transfer")
    if transfer:
        table = wandb.Table(
            columns=["scope", "depth", "rounds", "fires", "rate"]
        )
        for depth, cell in transfer["pooled"]["by_depth"].items():
            table.add_data("pooled", depth, cell["rounds"], cell["fires"], cell["rate"])
            summary[f"optimism/pooled/d{depth}/rate"] = cell["rate"]
        for leg in transfer["legs"]:
            for depth, cell in leg["by_depth"].items():
                table.add_data(
                    leg["label"], depth, cell["rounds"], cell["fires"], cell["rate"]
                )
            summary[f"optimism/{leg['label']}/fire_rate"] = leg["fire_rate"]
        run.log({"optimism/transfer": table})
        summary["optimism/pooled_fire_rate"] = transfer["pooled"]["fire_rate"]

    cost = report.get("cap_cost")
    if cost:
        fields = [
            "label",
            "max_p",
            "positions_above_cap",
            "true_depth",
            "true_raw",
            "capped_depth",
            "capped_raw",
            "raw_cost",
        ]
        table = wandb.Table(columns=fields)
        for row in cost["prompts"]:
            table.add_data(*(row[field] for field in fields))
            summary[f"cap/{row['label']}/raw_cost"] = row["raw_cost"]
        run.log({"cap/cost": table})
        summary["cap/binding_prompts"] = cost["binding_prompts"]
        summary["cap/mean_raw_cost_when_binding"] = cost[
            "mean_raw_cost_when_binding"
        ]
        summary["cap/max_raw_cost"] = cost["max_raw_cost"]

    run.summary.update(summary)
    run.finish()
    print(f"e168_wandb_log: logged {len(summary)} summary metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
