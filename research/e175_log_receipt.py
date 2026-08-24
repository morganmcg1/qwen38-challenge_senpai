#!/usr/bin/env python3
"""Log the E175 official receipt and its per-prompt decomposition to W&B.

harness=ranked throughout. Input is the terminal receipt record written by the
watcher, plus a board snapshot that carries the comparison receipts.

Run: python3 research/e175_log_receipt.py <receipt.json> <board.json>
"""

from __future__ import annotations

import json
import sys

import wandb

import e175_board_q_decomposition as board


def main() -> None:
    with open(sys.argv[1]) as fh:
        receipt = json.load(fh)
    metrics = receipt["officialMetrics"]
    ranked = board.main(sys.argv[2])

    run = wandb.init(
        project="qwen38-mlx-challenge-senpai",
        entity="wandb-applied-ai-team",
        name="e175-receipt-15017ddf-organizer-pure-plus-q",
        job_type="official-receipt",
        config={
            "experiment": "E175",
            "harness": "ranked",
            "submission_id": receipt["id"],
            "frozen_candidate_sha": "9c4fefe847dc088817034c8644f59b3dda4da748",
            "yukon_snapshot_commit": receipt["submissionCommitSha"],
            "base_argument": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
            "organizer_sha": "0863b06ac16e26e48fc06e97444095b00feb66d4",
            "comparison_receipt_a": "5a9f130a organizer-pure",
            "status": receipt["status"],
            "rejection_reason": receipt["rejectionReason"],
            "created_at": receipt["createdAt"],
            "updated_at": receipt["updatedAt"],
        },
    )

    scalar = {
        f"receipt/{k}": v
        for k, v in metrics.items()
        if isinstance(v, (int, float, bool))
    }
    run.log({**scalar, **{f"ranked/{k}": v for k, v in ranked.items()}})

    for prompt in metrics["per_prompt"]:
        run.log(
            {
                f"per_prompt/{k}": v
                for k, v in prompt.items()
                if isinstance(v, (int, float, bool))
            }
        )

    run.summary["official_score"] = receipt["officialScore"]
    run.summary["baseline_receipt_a_score"] = 3.70784519415395
    run.summary["delta_vs_a"] = receipt["officialScore"] - 3.70784519415395
    run.finish()
    print(f"W&B: {run.url}")


if __name__ == "__main__":
    main()
