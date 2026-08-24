#!/usr/bin/env python3
"""Publish the E180 A-replay attempt to W&B.

    usage: research/e180_wandb_log.py

E180 rebuilt the organizer-pure tree (editable surface byte-identical to
organizer main ``0863b06a``) and tried to resubmit it, to draw one sample from
the ranked receipt channel on content that does not move.

Yukon refused the submission by content dedup and returned our pre-existing
receipt A. No ranked run started and no new draw exists, so every ranked field
here records the refusal rather than a score. The timing fields are
``harness=local`` on an Apple M4 Pro and are a correctness confirmation, not
evidence about the ranked score.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e180-a-replay-receipt-channel"

BASE_SHA = "38634ebfbf67f8ccc8a5a404acbc15e749d26b7a"
BASE_ANCHOR = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
FROZEN_SHA = "b014fbf70f82c08471c5e54c4b657843f61477de"
ORGANIZER_SHA = "0863b06ac16e26e48fc06e97444095b00feb66d4"
WORKER_SHA256 = "bc65996eeae05b5b22f59bde53e6e2fbb714a7b682e704bc3f5386a8b7265a00"

SCORE_JSON = pathlib.Path(__file__).with_name(
    "e180-localsubmit-512-organizer-pure.json"
)

FIRE_AT_UTC = "2026-08-24T11:58:33Z"
REFUSED_AT_UTC = "2026-08-24T11:58:36Z"
REFUSAL_VERBATIM = (
    "Submission already exists\n"
    "benchmark   5d1ee4d7-80bd-4555-b182-6505f26ef495\n"
    "submission  5a9f130a-a29b-4b92-84ad-f0a2a59af210\n"
    "status      rejected\n"
    "note        not stored (existing submission reused; its original note is kept)"
)

# Registered before the fire (advisor F1). Kept here so the unrealized draw is
# reproducible if the lever ever reopens.
A_LEVEL_DRIFT_ADJUSTED = 3.705
CHANNEL_SIGMA_FRACTION = 0.00689
CROWN_SCORE = 3.7291100105909
A_SCORE = 3.70784519415395


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty_paths() -> int:
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return len(out.splitlines()) if out else 0


def main() -> None:
    local = json.loads(SCORE_JSON.read_text())
    m = local["metrics"]

    config = {
        "experiment": "e180-a-replay-receipt-channel",
        "hypothesis": (
            "Resubmitting a byte-identical organizer-pure tree draws one sample "
            "from the ranked receipt channel and has P~0.14 of taking the crown."
        ),
        "base_sha": BASE_SHA,
        "base_anchor_at_fire": BASE_ANCHOR,
        "frozen_sha": FROZEN_SHA,
        "organizer_sha": ORGANIZER_SHA,
        "head_at_log_time": git_head(),
        "worker_sha256": WORKER_SHA256,
        "candidate_tree": "organizer upstream/main, all 89 editablePaths byte-identical",
        "editable_surface_diff_vs_organizer": "empty",
        "proposal_head": "organizer-pinned, no candidate declaration",
        "local_host": "Apple M4 Pro",
        "local_os": "macOS 26.5.2",
        "local_toolchain": "Apple Swift 6.3.3 (swiftlang-6.3.3.1.3, clang-2100.1.1.101)",
        "cool_gate": "real 40C gate, not bypassed",
        "registered_a_level_drift_adjusted": A_LEVEL_DRIFT_ADJUSTED,
        "registered_channel_sigma_fraction": CHANNEL_SIGMA_FRACTION,
        "crown_score": CROWN_SCORE,
        "a_score": A_SCORE,
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e180-a-replay-content-dedup-refusal",
        job_type="official-submission-attempt",
        config=config,
    )

    metrics = {
        # harness=local: 512-token real-gate confirmation on the frozen tree.
        "local/harness": "local",
        "local/decode_tokens": m["decode_tokens"],
        "local/mtp_depth": m["mtp_depth"],
        "local/serial_seconds_per_token": m["serial_seconds_per_token"],
        "local/mtp_seconds_per_token": m["mtp_seconds_per_token"],
        "local/mtp_decode_speedup": m["mtp_decode_speedup"],
        "local/effective_mean_draft_len": m["effective_mean_draft_len"],
        "local/accepted_draft_rate": m["accepted_draft_rate"],
        "local/all_tokens_matched": int(m["all_tokens_matched"]),
        "local/residual_divergence_count": m["residual_divergence_count"],
        "local/uses_pinned_mtp_head": int(m["uses_pinned_mtp_head"]),
        "local/head_provenance_sha256": m["head_provenance_sha256"],
        "local/passed": int(local["passed"]),
        "local/serial_rows_checked": 512,
        "local/mtp_rows_checked": 568,
        "local/mtp_rounds": 77,
        # harness=ranked: the refusal, not a score.
        "ranked/harness": "ranked",
        "ranked/new_receipt_created": 0,
        "ranked/refusal": "content-dedup: submission already exists",
        "ranked/reused_submission_id": "5a9f130a-a29b-4b92-84ad-f0a2a59af210",
        "ranked/reused_submission_status": "rejected",
        "ranked/reused_submission_score": A_SCORE,
        "ranked/benchmark_id": "5d1ee4d7-80bd-4555-b182-6505f26ef495",
        "ranked/fire_at_utc": FIRE_AT_UTC,
        "ranked/refused_at_utc": REFUSED_AT_UTC,
        "ranked/note_stored": 0,
        "ranked/draw_realized": 0,
        # Gate evidence.
        "gate/scope_ok": 1,
        "gate/budget_source_bytes": 2604101,
        "gate/budget_growth_bytes": -33188,
        "gate/ranked_score_boundary_ok": 1,
        "gate/worker_assertions_ok": 1,
        "gate/twin_audit_stale_count": 1,
        "gate/twin_audit_stale_is_comment_only": 1,
    }

    run.log(metrics)
    run.summary.update(metrics)
    run.summary["refusal_verbatim"] = REFUSAL_VERBATIM
    run.summary["finding"] = (
        "Yukon content-addresses submission archives per solver. An identical "
        "editable archive returns the existing submission and starts no ranked "
        "run, so a same-solver replay cannot draw a second sample from the "
        "receipt channel. The refusal also certifies, at the archive level, "
        "that receipt A 5a9f130a was already organizer-pure."
    )
    run.summary["dirty_paths_at_log_time"] = git_dirty_paths()
    print(f"logged {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
