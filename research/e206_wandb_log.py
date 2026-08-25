#!/usr/bin/env python3
"""Publish the E206 cap-8 width-work map to W&B.

    usage: research/e206_wandb_log.py

E206 ran no GPU leg. Every census here is re-derived from evidence that already
existed: the cap-7 census from the E202 raw per-round traces (W&B lpwbno36) and
the cap-8 census from the E199 per-round tables (W&B 7y6ap5l8, replicated by
exdb3vyt). The dispatch enumeration is a source proof and the exposure map is a
desk decomposition of existing traces.

Every table is `harness=local`. Every ranked projection is a desk model marked
EXTRAPOLATED; none is an official or ranked score.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import wandb

ROOT = pathlib.Path(__file__).resolve().parent
DOC = ROOT / "e206-width-work-map.json"

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e206-cap8-width-work-map"

CROWN = 3.7291100105909
RECEIPT_A = 3.70784519415395
MUE_MS = 0.567


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    doc = json.loads(DOC.read_text())
    exp = doc["g3_exposure_map"]
    cap8 = doc["cap8_census"]["runs"]["7y6ap5l8"]
    cap7 = doc["cap7_census"]["total_by_served_width"]
    cap7_rounds = sum(cap7.values())

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e206-cap8-width-work-map",
        job_type="desk",
        config={
            "experiment": "e206-cap8-width-work-map",
            "harness": "local",
            "campaign_base_sha": doc["campaign_base"],
            "branch_head_sha": git("rev-parse", "HEAD"),
            "worktree_clean": git("status", "--porcelain") == "",
            "gpu_legs_run": 0,
            "timing_contrasts_run": 0,
            "cap7_census_source_run": "lpwbno36",
            "cap7_census_surface_sha": "7f10e147",
            "cap8_census_source_runs": ["7y6ap5l8", "exdb3vyt"],
            "cap8_census_surface": "c47c7284 = b9ee228c + "
            "segmentedVerifyDepthCap=8; b9ee228c has a zero-byte scored diff "
            "against the campaign base",
            "token_window": 512,
            "chip": "Apple M4 Pro",
            "ranked_score_boundary_check": "PASS",
        },
        notes="E206 desk and census map. No GPU leg. Cap-8 census recovered "
        "from existing W&B per-round tables; cap-7 census re-emitted from the "
        "E202 raw traces. m=9 dispatch enumeration is a source proof.",
    )

    census_rows = []
    for w in range(2, 10):
        c7 = cap7.get(str(w), cap7.get(w, 0))
        c8 = cap8["census_by_served_width"].get(str(w), 0)
        census_rows.append(
            [
                w,
                c7,
                100.0 * c7 / cap7_rounds,
                c8,
                100.0 * c8 / cap8["rounds"],
                cap8["block_request_ms_mean_by_width"].get(str(w)),
            ]
        )
    run.log(
        {
            "census/by_served_width": wandb.Table(
                columns=[
                    "served_width_qL",
                    "cap7_rounds",
                    "cap7_pct",
                    "cap8_rounds",
                    "cap8_pct",
                    "cap8_block_request_ms_mean",
                ],
                data=census_rows,
            )
        }
    )

    m9 = doc["m9_dispatch_map"]
    run.log(
        {
            "m9/dispatch_by_cell": wandb.Table(
                columns=[
                    "cell",
                    "k",
                    "n",
                    "invocations_per_round",
                    "groups_at_m9",
                    "covered_by_e195_at_m9",
                    "third_pass_ms_per_m9_round",
                    "share_of_third_pass",
                    "weight_bytes_share_of_pass",
                ],
                data=[
                    [
                        name,
                        rec["cell_key_k_n"][0],
                        rec["cell_key_k_n"][1],
                        rec["invocations_per_round"],
                        rec["shipped_groups_at_m9"],
                        rec["covered_by_e195_selective_plan_at_m9"],
                        exp["cell_split_e186_isolated"][name][
                            "third_pass_ms_per_m9_round"
                        ],
                        exp["cell_split_e186_isolated"][name][
                            "share_of_isolated_cell_sum"
                        ],
                        doc["weight_byte_model"]["per_cell"][name]["share_of_pass"],
                    ]
                    for name, rec in m9["cells"].items()
                ],
            ),
            "m9/ipg_options": wandb.Table(
                columns=[
                    "ipg",
                    "groups",
                    "tail_rows",
                    "legal_under_static_assert",
                    "within_shipped_register_regime",
                ],
                data=[
                    [
                        o["ipg"],
                        o["groups"],
                        o["tail_rows"],
                        o["legal_under_static_assert"],
                        o["max_ipg_proven_in_shipped_plan"],
                    ]
                    for o in m9["ipg_options_at_m9"]
                ],
            ),
            "exposure/step_law_check": wandb.Table(
                columns=[
                    "served_width_qL",
                    "rounds",
                    "measured_block_request_ms_mean",
                    "step_law_ms",
                    "uniform_staged_G",
                ],
                data=[
                    [
                        r["served_width"],
                        r["rounds"],
                        r["measured_block_request_ms_mean"],
                        r["step_law_ms_uniform_staged_G"],
                        r["uniform_staged_G"],
                    ]
                    for r in doc["step_law_consistency_check_within_7y6ap5l8"]
                ],
            ),
            "exposure/g3_band_split": wandb.Table(
                columns=["band", "step_ms_m8_to_m9"],
                data=[[k, v] for k, v in exp["band_split_e182_in_path_ms"].items()],
            ),
        }
    )

    proj = exp["ranked_published_score_projection"]
    run.summary.update(
        {
            "cap7/rounds": cap7_rounds,
            "cap7/qL8_pct_all_rounds": 100.0 * cap7.get("8", 0) / cap7_rounds,
            "cap7/qL9_pct_all_rounds": 100.0 * cap7.get("9", 0) / cap7_rounds,
            "cap7/accepted_draft_rate": 0.8770161290,
            "cap7/mean_accepted_draft_len": 5.5769230769,
            "cap8/rounds": cap8["rounds"],
            "cap8/qL9_rounds": cap8["census_by_served_width"].get("9"),
            "cap8/qL9_pct_all_rounds": 100.0 * exp["cap8_m9_round_share"],
            "cap8/qL8_pct_all_rounds": 100.0
            * cap8["census_by_served_width"].get("8", 0)
            / cap8["rounds"],
            "cap8/accepted_draft_rate": 0.83685,
            "cap8/mean_accepted_draft_len": 5.7368,
            "m9/e195_covers_any_family": False,
            "m9/uncovered_cells": len(m9["cells"]),
            "m9/matvecs_per_round": sum(
                r["invocations_per_round"] for r in m9["cells"].values()
            ),
            "m9/shipped_groups": 3,
            "m9/best_legal_group_reduction_ipg": 5,
            "m9/best_legal_group_reduction_G": 2,
            "exposure/local_third_pass_ms_per_m9_round": exp[
                "local_third_pass_ms_per_m9_round"
            ],
            "exposure/local_round_weighted_ms": exp[
                "local_round_weighted_third_pass_ms"
            ],
            "exposure/weight_bytes_per_pass_gb": doc["weight_byte_model"][
                "weight_bytes_per_round_per_pass"
            ]
            / 1e9,
            "exposure/top3_cell_share": exp["top3_cell_share_of_isolated_sum"],
            "transfer/ratio_measured_f505": exp["transfer"]["ratio_measured_f505"],
            "transfer/realized_fraction_of_naive_scaled_step": exp["transfer"][
                "realized_fraction_of_naive_scaled_step"
            ],
            "ranked_extrapolated/round_weighted_ms_a": exp[
                "ranked_extrapolated_round_weighted_ms"
            ]["a_measured_f505_transfer"],
            "ranked_extrapolated/published_delta_a": proj[
                "a_measured_f505_transfer"
            ]["published_delta"],
            "ranked_extrapolated/published_delta_b_low": proj[
                "b_e197_refit_transfer_low"
            ]["published_delta"],
            "ranked_extrapolated/published_delta_c": proj[
                "c_smooth_step_continuation"
            ]["published_delta"],
            "ranked_extrapolated/cap8_without_third_pass": proj[
                "a_measured_f505_transfer"
            ]["cap8_without_third_pass"],
            "baseline/receipt_a": RECEIPT_A,
            "baseline/crown": CROWN,
            "baseline/mue_ms": MUE_MS,
            "official_score": False,
            "rankable": False,
        }
    )
    run.finish()
    print(f"logged {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
