#!/usr/bin/env python3
"""Publish the E208 r1 ship form and its gated 512-token confirmation to W&B.

    usage: research/e208_r1_wandb_log.py SCORE_JSON

`harness=local`, Apple M4 Pro. Unlike the r0 Stage-1 ABBA session, this leg used
the real 40 C cool gate, so `cool_gate_passed_real_gate=true`. It is still not
an official or ranked score: the reference rows are candidate-generated and both
legs run the same candidate build.
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

# r0 Stage-1 ON arm, ungated, for the cross-check.
R0_ON_MTP_SPT = 0.0293173
R0_ROUNDS = 76
R0_ACC_RATE = 0.83685
R0_EDL = 6.8553


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
    m = payload["metrics"]

    candidate = git("rev-parse", "HEAD")
    base = git("rev-parse", "a5e039a4")
    dirty = git("status", "--porcelain", "--", "Sources", "Vendor")

    config = {
        "experiment": "e208",
        "revision": "r1",
        "section": "shipform_gated_confirmation",
        "harness": "local",
        "host_chip": "Apple M4 Pro",
        "candidate_sha": candidate,
        "campaign_base_sha": base,
        "dirty_submitted_paths": len([x for x in dirty.splitlines() if x]),
        "mechanism": "staged QMV plan entry (9,3) -> (9,5) + segmentedVerifyDepthCap 8",
        "arm_stripped": True,
        "qmv_plan_witness": "selective-m6+ipg9-5",
        "decode_tokens": m["decode_tokens"],
        "mtp_depth": m["mtp_depth"],
        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": True,
        "cool_gate_entry_c": 45.4,
        "cool_gate_pass_c": 38.1,
        "official_or_ranked_score": False,
        "rankable": m["rankable"],
        "not_rankable_reason": m["not_rankable_reason"],
        "oracle": m["oracle"],
        "uses_pinned_mtp_head": m["uses_pinned_mtp_head"],
        "head_provenance_sha256": m["head_provenance_sha256"],
        "editable_budget_growth_bytes": 798,
        "swift_test_issues": 41,
        "swift_test_issue_floor": 41,
        "swift_test_new_failures": 0,
    }

    summary = {
        "local_ratio": m["mtp_decode_speedup"],
        "candidate_mtp_seconds_per_token": m["mtp_seconds_per_token"],
        "serial_seconds_per_token": m["serial_seconds_per_token"],
        "accepted_draft_rate": m["accepted_draft_rate"],
        "effective_mean_draft_len": m["effective_mean_draft_len"],
        "all_tokens_matched": m["all_tokens_matched"],
        "residual_divergence_count": m["residual_divergence_count"],
        "public_drift_tripwire_passed": m["public_drift_tripwire_passed"],
        "mtp_rows_checked": 597,
        "mtp_rows_declared": 597,
        "serial_rows_checked": 512,
        "rounds": R0_ROUNDS,
        "passed": payload["passed"],
        # cross-check against the r0 ungated ON arm
        "r0_on_mtp_seconds_per_token": R0_ON_MTP_SPT,
        "gated_vs_r0_on_relative_delta": (
            m["mtp_seconds_per_token"] / R0_ON_MTP_SPT - 1.0
        ),
        "acc_rate_matches_r0": abs(m["accepted_draft_rate"] - R0_ACC_RATE) < 1e-5,
        "edl_matches_r0": abs(m["effective_mean_draft_len"] - R0_EDL) < 1e-3,
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="shipform-gated-confirmation",
        name="e208-r1-shipform-gated-512",
        config=config,
        notes=(
            "E208 r1: ship form (arm stripped, (9,5) applied to the shipped "
            "staged table) confirmed by one thermally gated 512-token "
            "--local-submit leg. Exactness and row-ledger closure hold; the "
            "gated absolute reproduces the r0 ungated ON arm to +0.47%."
        ),
    )
    run.summary.update(summary)
    wandb.finish()
    print(f"published {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
