#!/usr/bin/env python3
"""E99: publish the offline allocation-bound analysis to W&B.

usage:
  research/e99_wandb.py --report research/e99-artifacts/oracle.json \
                        --name e99-oracle-allocation-bound [--notes TEXT]

`--report` takes the JSON written by `research/e99_oracle.py --out`. This run
holds no timing measurement of its own: every number in it is either a recorded
round from a traced leg, the advisor's fitted ranked cost curve, or arithmetic
over those two. The traced legs carry their own identity and gate labels, which
are logged here as configuration so a reader never has to guess which leg the
rounds came from.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e99-oracle-allocation-bound"


def table(columns, rows):
    made = wandb.Table(columns=columns)
    for row in rows:
        made.add_data(*[row.get(column) for column in columns])
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--rung5", default="",
                        help="JSON from research/e99_rung5_price.py")
    parser.add_argument("--name", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--group", default=EXPERIMENT)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--branch-commit", default="")
    args = parser.parse_args()

    doc = json.loads(Path(args.report).read_text())
    headline_tag = doc["headline"]
    headline = doc["legs"][headline_tag]["ranked"]
    summary = doc["headline_summary"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        group=args.group,
        job_type="offline-analysis",
        notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "assignment_pr": 101,
            "student": "qwen-edward",
            "analysis": "research/e99_oracle.py",
            "base_sha": args.base_sha,
            "branch_commit": args.branch_commit,
            "harness": "local",
            "official_or_ranked_score": False,
            "gate_qualified_for_timing": False,
            "cool_gate_passed_real_gate": False,
            "headline_leg": headline_tag,
            "headline_width": summary["width"],
            "nearest_ranked_prompt": summary["nearest_prompt"],
            "pricing_curve": "ranked M5 two-line fit, research/ranked_cost_curve.py",
            "legs": sorted(doc["legs"]),
        },
    )

    leg_rows, counterfactual_rows, fixed_rows = [], [], []
    for tag, curves in doc["legs"].items():
        ranked = curves["ranked"]
        leg_rows.append(dict(
            leg=tag, offered_cap=ranked["offered_cap"],
            rounds=ranked["actual"]["rounds"],
            width=ranked["score"]["effective_mean_draft_len"] + 1.0,
            accepted_draft_rate=ranked["score"]["accepted_draft_rate"],
            censored_share=ranked["censored_share"],
            measured_round_us=ranked["measured_local_round_us"],
            mtp_seconds_per_token=ranked["score"]["mtp_seconds_per_token"],
            all_tokens_matched=ranked["score"]["all_tokens_matched"]))
        for curve, result in curves.items():
            for treatment, values in result["treatments"].items():
                counterfactual_rows.append(dict(
                    leg=tag, curve=curve, treatment=treatment,
                    actual_us_per_token=values["actual"]["us_per_token"],
                    oracle_us_per_token=values["oracle"]["us_per_token"],
                    oracle_gap_pct=values["oracle_gap_pct"],
                    best_fixed_depth=values["best_fixed"]["depth"],
                    best_fixed_us_per_token=values["best_fixed"]["us_per_token"],
                    fixed_gap_pct=values["fixed_gap_pct"],
                    one_bit_g_us_per_token=values["one_bit_g_truncate"]["us_per_token"],
                    one_bit_g_gap_pct=values["one_bit_g_gap_pct"],
                    one_bit_share_of_oracle=values["one_bit_share_of_oracle"],
                    oracle_mean_depth=values["oracle"]["mean_depth"],
                    actual_mean_depth=values["actual"]["mean_depth"]))
                if curve == "ranked":
                    for row in values["fixed_sweep"]:
                        fixed_rows.append(dict(
                            leg=tag, treatment=treatment, depth=row["depth"],
                            us_per_token=row["us_per_token"],
                            tokens_per_round=row["tokens_per_round"]))

    fold_rows = []
    for key, folds in doc["rung4"].items():
        for fold in folds:
            control = fold.get("random_control") or {}
            fold_rows.append(dict(
                policy_class=fold["policy_class"], treatment=fold["treatment"],
                design=fold["fold"],
                actual_us_per_token=fold["actual_us_per_token"],
                held_out_us_per_token=fold["held_out_us_per_token"],
                oracle_us_per_token=fold["oracle_us_per_token"],
                gain_pct=fold["gain_pct"], oracle_gap_pct=fold["oracle_gap_pct"],
                recovered_share=fold["recovered_share"],
                control_clamped=control.get("clamped"),
                control_mean_gain_pct=control.get("mean_gain_pct"),
                control_p95_gain_pct=control.get("p95_gain_pct"),
                control_max_gain_pct=control.get("max_gain_pct"),
                beats_random_control=control.get("beaten_by_fit"),
                policy="\n".join(fold["tree_text"])))

    run.log({
        "legs": table(list(leg_rows[0]), leg_rows),
        "counterfactuals": table(list(counterfactual_rows[0]),
                                 counterfactual_rows),
        "fixed_depth_sweep": table(list(fixed_rows[0]), fixed_rows),
        "rung4_folds": table(list(fold_rows[0]), fold_rows),
        "feature_ranking": table(["feature", "r_accepted", "r_shallow"],
                                 doc["feature_ranking"]),
        "width_surrogate": table(list(doc["width_surrogate"][0]),
                                 doc["width_surrogate"]),
    })

    transfer = [f for f in doc["rung4"]["one_bit_g|observed"]
                if f["fold"].startswith("leg-out")]
    for treatment, values in headline["treatments"].items():
        run.summary[f"headline/{treatment}/oracle_gap_pct"] = \
            values["oracle_gap_pct"]
        run.summary[f"headline/{treatment}/one_bit_g_gap_pct"] = \
            values["one_bit_g_gap_pct"]
        run.summary[f"headline/{treatment}/fixed_gap_pct"] = \
            values["fixed_gap_pct"]
    run.summary["headline/oracle_gap_pct_min"] = summary["oracle_gap_pct_min"]
    run.summary["headline/oracle_gap_pct_max"] = summary["oracle_gap_pct_max"]
    run.summary["headline/treatments_agree_in_sign"] = \
        summary["treatments_agree_in_sign"]
    run.summary["headline/published_move_at_oracle_pct"] = \
        summary["published_move_at_oracle"]["published_pct"]
    run.summary["realisable/leg_out_gain_pct_median"] = \
        statistics.median([f["gain_pct"] for f in transfer])
    run.summary["realisable/leg_out_gain_pct_min"] = \
        min(f["gain_pct"] for f in transfer)
    run.summary["realisable/leg_out_gain_pct_max"] = \
        max(f["gain_pct"] for f in transfer)
    run.summary["realisable/leg_out_recovered_share_median"] = \
        statistics.median([f["recovered_share"] for f in transfer])

    if args.rung5:
        rung5 = json.loads(Path(args.rung5).read_text())
        measured = rung5["off"] + rung5["on"] + rung5["sweep"]
        run.log({
            "rung5_legs": table(list(measured[0]), measured),
            "rung5_replay": table(list(rung5["replay"][0]), rung5["replay"]),
        })
        for field, values in rung5["contrasts"].items():
            run.summary[f"rung5/{field}/off"] = values["off"]
            run.summary[f"rung5/{field}/on"] = values["on"]
            run.summary[f"rung5/{field}/delta_pct"] = values["delta_pct"]
        run.summary["rung5/exactness_green"] = rung5["exactness_green"]
        run.summary["rung5/dram_floor_violations"] = \
            rung5["dram_floor_violations"]
        run.summary["rung5/fired_share_on"] = \
            statistics.fmean([row["fired_share"] for row in rung5["on"]])

    artifact = wandb.Artifact(name="e99-oracle-report", type="analysis")
    artifact.add_file(args.report)
    if args.rung5:
        artifact.add_file(args.rung5)
    run.log_artifact(artifact)
    print(f"run {run.id} {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
