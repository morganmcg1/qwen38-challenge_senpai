#!/usr/bin/env python3
"""Publish the E135 composed-archive evidence to W&B.

    YUKON_API_TOKEN=... WANDB_API_KEY=... \
        python3 research/e135_log_composition_wandb.py

Two metrics the assignment requires were not derivable until the `572b2cc4`
receipt landed, so `e135rung1` and `e135ship` could not carry them:

`e135_launch_cost_us_per_column`
    The AVERAGE launch cost of one deleted threadgroup column, taken as the
    per-drafting-round saving divided by the mean number of columns the tight
    grid deletes at the drafting prompts' mean width. It is an average, not a
    marginal. The measured MARGINAL slope in width is logged beside it and is
    consistent with zero, which is the whole content of the F14/F15 dispute.

`e135_onepass_gain_under_tight_pct`
    What the one-pass `{6:6, 7:7}` table added on ranked hardware once the
    launch grid was already tight. Both receipts in the measured pair carry
    `onepass67`, so this is not a direct contrast; it is the observed board
    move minus the move predicted from the launch grid alone.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys

import wandb

sys.path.insert(0, "research")

from e135_f14_regression import board, receipt, recover_rounds  # noqa: E402

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
BEFORE, AFTER = "623e77af", "572b2cc4"
COMPOSED_SUBMISSION = "e003a86d-e1f0-4818-baa7-953469303616"
# Columns the tight grid deletes at width m under the `onepass67` table that
# both receipts in the measured pair carry: m - ceil(m / ipg).
DELETED_UNDER_ONEPASS67 = {3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 6}
# Predicted and observed published-score move for the launch grid alone.
PREDICTED_GRID_MOVE_PCT = 4.0462
GATE_LEG = pathlib.Path("research/out/e135c512/pipelines.json")


def interpolate_deleted(width: float) -> float:
    low = max(3, min(9, int(math.floor(width))))
    high = max(3, min(9, int(math.ceil(width))))
    if low == high:
        return float(DELETED_UNDER_ONEPASS67[low])
    span = DELETED_UNDER_ONEPASS67[high] - DELETED_UNDER_ONEPASS67[low]
    return DELETED_UNDER_ONEPASS67[low] + span * (width - low)


def main() -> int:
    rows = board(os.environ["YUKON_API_TOKEN"])
    before, after = receipt(BEFORE, rows), receipt(AFTER, rows)
    tokens = before["officialMetrics"]["decode_tokens"]
    new_by_hash = {p["prompt_sha256"]: p
                   for p in after["officialMetrics"]["per_prompt"]}

    per_drafting_round, drafting_widths = [], []
    for old in before["officialMetrics"]["per_prompt"]:
        new = new_by_hash[old["prompt_sha256"]]
        rounds, _ = recover_rounds(new["effective_mean_draft_len"], tokens)
        if new["non_drafting_round_count"] != 0:
            continue
        delta = (old["mtp_seconds_per_token_mean"]
                 - new["mtp_seconds_per_token_mean"])
        per_drafting_round.append(delta * (tokens / rounds) * 1e6)
        drafting_widths.append(1.0 + new["effective_mean_draft_len"])

    n = len(per_drafting_round)
    saving = sum(per_drafting_round) / n
    saving_sd = math.sqrt(sum((v - saving) ** 2 for v in per_drafting_round)
                          / (n - 1))
    mean_width = sum(drafting_widths) / n
    deleted = interpolate_deleted(mean_width)

    observed_move_pct = 100.0 * (after["officialScore"] / before["officialScore"]
                                 - 1.0)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    leg = json.loads(GATE_LEG.read_text()) if GATE_LEG.exists() else {}

    summary = {
        # Required by the assignment.
        "e135_tight_grid_candidate_leg_pct": 1.8064,
        "e135_launch_cost_us_per_column": saving / deleted,
        "e135_onepass_gain_under_tight_pct":
            observed_move_pct - PREDICTED_GRID_MOVE_PCT,
        "e135_exact_token_divergences": 0,
        # The ranked pair the two derived metrics come from.
        "e135_ranked_before_score": before["officialScore"],
        "e135_ranked_after_score": after["officialScore"],
        "e135_ranked_observed_move_pct": observed_move_pct,
        "e135_ranked_predicted_grid_move_pct": PREDICTED_GRID_MOVE_PCT,
        "e135_per_drafting_round_saving_us": saving,
        "e135_per_drafting_round_saving_sd_us": saving_sd,
        "e135_drafting_prompt_count": n,
        "e135_drafting_mean_width": mean_width,
        "e135_deleted_columns_at_mean_width": deleted,
        # F14/F15: which shape the saving takes. The marginal slope is the
        # number that decides whether wider verification saves more.
        "e135_marginal_slope_us_per_width": -74.7,
        "e135_marginal_slope_se_us_per_width": 71.1,
        "e135_log_model_press": 2.336e6,
        "e135_flat_model_press": 6.038e6,
        "e135_drafting_fraction_model_press": 5.065e5,
        "e135_log_coef_from_synthetic_flat_law": 1365.6,
        "e135_log_coef_from_real_receipts": 1354.3,
        # The composed archive.
        "e135_composed_submission_id": COMPOSED_SUBMISSION,
        "e135_composed_commit": commit,
        "e135_composed_probe_fraction": 0.15,
        "e135_composed_pass_boundary_width": 6,
        "e135_composed_pass_boundary_tier_factor": 1.45,
        "e135_composed_route": leg.get("default_route"),
        "e135_composed_grid": leg.get("default_grid"),
        "e135_composed_plan": leg.get("plan"),
        "e135_local_submit_runnable_on_host": False,
    }

    run = wandb.init(
        entity=PROJECT.split("/")[0],
        project=PROJECT.split("/")[1],
        id="e135compose",
        name="e135-composed-archive-and-cost-model",
        resume="allow",
        config={
            "experiment": "e135",
            "base_sha": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
            "host": "apple_m4_pro_48gib_g16s",
            "ranked_host": "m5_g17s",
            "decode_tokens": 512,
            "mechanisms": ["tight_launch_grid", "pb6_tier_1_45",
                           "probe_fraction_0_15", "revert_onepass67"],
        },
    )
    run.summary.update(summary)
    for key, value in sorted(summary.items()):
        print("%-46s %s" % (key, value))
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
