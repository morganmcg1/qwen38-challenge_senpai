#!/usr/bin/env python3
"""Publish the E208 Stage-0 gate and Stage-1 ABBA session to W&B.

    usage: research/e208_wandb_log.py GATE_JSON ANALYSIS_JSON

Every number here is `harness=local` on an Apple M4 Pro. The Stage-1 session is
ungated (`MLXFAST_LOCAL_COOL_GATE=0`) under the standing counterbalanced mode,
so `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` are
logged verbatim. Nothing here is an official or ranked score.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e208-qmv-9row-ipg5"

# Ranked transfer constants supplied by the assignment. Desk model only.
K_TRANSFER = 0.23731
RECEIPT_A = 3.70784519415395
CROWN = 3.7291100105909


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
    analysis = json.loads(pathlib.Path(sys.argv[2]).read_text())

    m9 = analysis.get("paired_m9_ms_per_round_recovered", {})
    allr = analysis.get("paired_all_rounds_ms_per_round_recovered", {})
    non9 = analysis.get("paired_non_m9_ms_per_round_recovered", {})
    whole = analysis["whole_leg_absolute"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e208-qmv-9row-ipg5-stage0-stage1",
        job_type="local-abba",
        config={
            "experiment": "e208-qmv-9row-ipg5",
            "harness": "local",
            "mechanism": "staged QMV plan entry (9,3) -> (9,5); G(9) 3 -> 2",
            "arm_switch": "DARKBLOOM_E208_QMV_ARM",
            "qmv_plan_witness_off": "selective-m6+ipg9-3",
            "qmv_plan_witness_on": "selective-m6+ipg9-5",
            "campaign_base_sha": analysis["campaign_base_sha"],
            "candidate_sha": analysis["candidate_sha"],
            "branch_head_sha": git("rev-parse", "HEAD"),
            "worktree_clean": git("status", "--porcelain") == "",
            "worker_sha256": analysis["worker_sha256"],
            "metallib_source_fingerprint": analysis[
                "metallib_source_fingerprint"
            ],
            "host": analysis["host"],
            "chip": analysis["chip"],
            "token_window": analysis["tokens"],
            "local_mode": "--local-iterate",
            "depth_cap": 8,
            "session_order": analysis["session_order"],
            "timing_source": analysis["timing_source"],
            "cool_gate_passed_real_gate": analysis[
                "cool_gate_passed_real_gate"
            ],
            "gate_qualified_for_timing": analysis["gate_qualified_for_timing"],
            "official_or_ranked_score": analysis["official_or_ranked_score"],
            "ranked_score_boundary_check": "PASS",
            "assignment_scope_check": "PASS",
            "editable_budget_check": "PASS",
            "twin_audit": "OK; no generated twin and no metallib entry for "
            "this JIT kernel family",
        },
        notes="E208. Stage 0 is a bit-exact numerics gate on the (9,5) group "
        "partition with a positive control. Stage 1 is one ungated, "
        "counterbalanced four-leg ABBA session at 512 tokens on the cap-8 "
        "surface, decided on trusted-parent round endpoints and whole-leg "
        "absolute candidate seconds per token.",
    )

    run.summary["stage0/total_elements"] = gate["total_elements"]
    run.summary["stage0/total_differing"] = gate["total_differing"]
    run.summary["stage0/worst_max_ulp"] = gate["worst_max_ulp"]
    run.summary["stage0/positive_control_tripped"] = all(
        row["differing"] > 0 for row in gate["positive_control"]
    )

    run.log(
        {
            "stage0/cells": wandb.Table(
                columns=[
                    "cell", "k", "n", "m", "use_table",
                    "invocations_per_round", "elements", "differing",
                    "max_ulp", "max_abs_diff", "non_finite",
                    "staged_groups", "wide9_groups",
                ],
                data=[
                    [
                        row["cell"], row["k"], row["n"], row["m"],
                        row["use_table"], row["invocations_per_round"],
                        row["elements"], row["differing"], row["max_ulp"],
                        row["max_abs_diff"], row["non_finite"],
                        row["staged_groups"], row["wide9_groups"],
                    ]
                    for row in gate["cells"]
                ],
            ),
            "stage0/positive_control": wandb.Table(
                columns=["cell", "m", "use_table", "differing", "max_ulp",
                         "max_abs_diff"],
                data=[
                    [row["cell"], row["m"], row["use_table"], row["differing"],
                     row["max_ulp"], row["max_abs_diff"]]
                    for row in gate["positive_control"]
                ],
            ),
        }
    )

    run.log(
        {
            "stage1/legs": wandb.Table(
                columns=[
                    "leg", "arm", "qmv_plan", "mtp_seconds_per_token",
                    "serial_seconds_per_token", "local_ratio",
                    "effective_mean_draft_len", "accepted_draft_rate",
                    "round_count", "decode_seconds", "all_tokens_matched",
                    "residual_divergence_count", "declared_rows_total",
                    "reference_checked_row_total", "gpu_temp_entry_c",
                    "gpu_temp_exit_c",
                ],
                data=[
                    [
                        leg["leg"], leg["arm"], leg["observed_plans"],
                        leg["mtp_seconds_per_token"],
                        leg["serial_seconds_per_token"],
                        leg["mtp_decode_speedup"],
                        leg["effective_mean_draft_len"],
                        leg["accepted_draft_rate"], leg["round_count"],
                        leg["decode_seconds"], leg["all_tokens_matched"],
                        leg["residual_divergence_count"],
                        leg["declared_rows_total"],
                        leg["reference_checked_row_total"],
                        leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
                    ]
                    for leg in analysis["legs"]
                ],
            ),
            "stage1/width_census": wandb.Table(
                columns=["served_width_qL", "rounds", "share"],
                data=[
                    [int(w), n, n / analysis["round_count"]]
                    for w, n in sorted(
                        analysis["width_census"].items(), key=lambda kv: int(kv[0])
                    )
                ],
            ),
        }
    )

    for key, block in (
        ("m9", m9), ("all_rounds", allr), ("non_m9", non9),
        ("drift_control_off", analysis.get("drift_control_off_ms_per_round", {})),
        ("drift_control_on", analysis.get("drift_control_on_ms_per_round", {})),
    ):
        for field, value in block.items():
            run.summary[f"stage1/paired/{key}/{field}"] = value

    for field, value in analysis.get(
        "attribution_m9_candidate_side", {}
    ).items():
        run.summary[f"stage1/attribution_m9/{field}"] = value

    run.summary["stage1/trajectories_identical"] = analysis[
        "trajectories_identical"
    ]
    run.summary["stage1/m9_round_share"] = analysis["m9_round_share"]
    run.summary["stage1/gpu_temp_entry_spread_c"] = analysis[
        "gpu_temp_entry_spread_c"
    ]
    run.summary["stage1/problems"] = analysis["problems"]
    for field, value in whole.items():
        if not isinstance(value, list):
            run.summary[f"stage1/whole_leg/{field}"] = value

    # Primary metric: round-weighted local milliseconds per round recovered.
    run.summary["primary/round_weighted_ms_per_round_recovered"] = allr.get(
        "mean"
    )
    run.summary["primary/direction"] = "maximize"

    # Desk ranked projection, EXTRAPOLATED, harness=ranked. Not a measurement.
    if whole.get("relative_gain") is not None:
        run.summary["ranked_extrapolated/local_relative_gain"] = whole[
            "relative_gain"
        ]
        run.summary["ranked_extrapolated/k_transfer"] = K_TRANSFER
        run.summary["ranked_extrapolated/published_points_on_receipt_a"] = (
            RECEIPT_A * whole["relative_gain"] * K_TRANSFER
        )
        run.summary["ranked_extrapolated/label"] = (
            "EXTRAPOLATED desk model, harness=ranked, not measured"
        )
        run.summary["ranked_extrapolated/receipt_a"] = RECEIPT_A
        run.summary["ranked_extrapolated/crown"] = CROWN

    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
