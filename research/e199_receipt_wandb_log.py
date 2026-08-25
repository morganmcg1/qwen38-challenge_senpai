#!/usr/bin/env python3
"""Publish the E199 official receipt 5f508da5 and its per-prompt analysis to W&B.

    usage: research/e199_receipt_wandb_log.py

Every number here is ``harness=ranked``: it comes from the Yukon submission
record for the composed candidate (verify-depth cap 8 plus the staged width-9
QMV group plan), not from a local measurement. The comparators are our own
earlier ranked receipts, read from the same board, so the serial numerator is
the same pinned baseline in all three.

The key structure recorded here is an internal control group. Only the
verify-depth cap can change ``effective_mean_draft_len``, and a width-9 QMV
round cannot exist below cap 8, so:

  * the three prompts whose draft length did not move isolate everything the
    campaign base changed since receipt A, and
  * the five prompts whose draft length rose isolate the marginal price of the
    ninth verified row, measured with its best available kernel.
"""

from __future__ import annotations

import json
import os
import statistics as st
import urllib.request

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e199-cap8-paid-receipt"

API = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
SUBMISSION_ID = "5f508da5-2d0c-43ee-b0c3-8f9f83367d49"
CANDIDATE_SHA = "9a91ba76ae49ea902063770a2ed3317de0d42b51"
CAMPAIGN_BASE_SHA = "b9ee228c26629163d81684506a28eb1ca198ae80"
RECEIPT_A = "5a9f130a"
RECEIPT_A_SCORE = 3.70784519415395
E193_RECEIPT = "2681c3ac"
CROWN = "ec24d591"
CROWN_SCORE = 3.7291100105909
GATE_RUN = "7y6ap5l8"


def board():
    token = os.environ["YUKON_API_TOKEN"]
    url = f"{API}/benchmarks/{BENCHMARK_ID}/submissions?limit=2000"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.load(urllib.request.urlopen(request, timeout=180))["submissions"]


def metrics(sub):
    raw = sub.get("officialMetrics")
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def per_prompt(sub):
    return {p["prompt_sha256"][:8]: p for p in metrics(sub).get("per_prompt", [])}


def main() -> None:
    rows = board()
    ours = [s for s in rows if s.get("solverUsername") == "morganmcg1"]
    receipt = next(s for s in ours if s["id"] == SUBMISSION_ID)
    receipt_a = next(s for s in ours if s["id"].startswith(RECEIPT_A[:7]))
    e193 = next(s for s in ours if s["id"].startswith(E193_RECEIPT[:7]))

    live = metrics(receipt)
    new, base_a, base_e193 = per_prompt(receipt), per_prompt(receipt_a), per_prompt(e193)

    deepened, control = [], []
    table = wandb.Table(columns=[
        "prompt", "ratio_receipt_a", "ratio_e193", "ratio_cap8_qmv95",
        "delta_vs_a", "dlen_receipt_a", "dlen_cap8_qmv95", "delta_dlen",
        "mtp_s_per_tok_a", "mtp_s_per_tok_cap8_qmv95", "delta_mtp_pct",
        "non_drafting_rounds", "parity_ok",
    ])
    for key in sorted(new, key=lambda k: new[k]["raw_ratio_of_means"]):
        a, e, n = base_a[key], base_e193[key], new[key]
        d_len = n["effective_mean_draft_len"] - a["effective_mean_draft_len"]
        d_mtp = 100.0 * (n["mtp_seconds_per_token_mean"]
                         / a["mtp_seconds_per_token_mean"] - 1.0)
        table.add_data(
            key, a["raw_ratio_of_means"], e["raw_ratio_of_means"],
            n["raw_ratio_of_means"], n["raw_ratio_of_means"] - a["raw_ratio_of_means"],
            a["effective_mean_draft_len"], n["effective_mean_draft_len"], d_len,
            a["mtp_seconds_per_token_mean"], n["mtp_seconds_per_token_mean"], d_mtp,
            n["non_drafting_round_count"], n["parity_ok"],
        )
        (deepened if abs(d_len) > 0.01 else control).append(d_mtp)

    score = receipt["officialScore"]
    config = {
        "harness": "ranked",
        "experiment": "e199-r1-composed-cap8-qmv95",
        "mechanism": "segmentedVerifyDepthCap 7->8 + staged QMV m=9 IPG 3->5",
        "qmv_width_plan_witness": "selective-m6+ipg9-5",
        "candidate_sha": CANDIDATE_SHA,
        "campaign_base_sha": CAMPAIGN_BASE_SHA,
        "submission_id": SUBMISSION_ID,
        "submission_commit_sha": receipt.get("submissionCommitSha"),
        "benchmark_id": BENCHMARK_ID,
        "status": receipt.get("status"),
        "rejection_reason": receipt.get("rejectionReason"),
        "created_at": receipt.get("createdAt"),
        "updated_at": receipt.get("updatedAt"),
        "queue_hours": 5.32,
        "decode_tokens": live.get("decode_tokens"),
        "prompt_count": live.get("prompt_count"),
        "pairs_per_prompt": live.get("pairs_per_prompt"),
        "mtp_max_draft_depth": live.get("mtp_max_draft_depth"),
        "mtp_depth": live.get("mtp_depth"),
        "aggregation": live.get("aggregation"),
        "median_rule": live.get("median_rule"),
        "gate_run_local": GATE_RUN,
        "comparator_receipt_a": RECEIPT_A,
        "comparator_e193": E193_RECEIPT,
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name="e199-r1-receipt-5f508da5", job_type="official-receipt",
                     config=config, tags=["harness-ranked", "official-receipt",
                                           "rejected", "depth-axis", "cap8"])

    ratios = sorted(p["raw_ratio_of_means"] for p in new.values())
    run.summary.update({
        "official_score": score,
        "official_score_receipt_a": RECEIPT_A_SCORE,
        "official_score_crown": CROWN_SCORE,
        "delta_vs_receipt_a": score - RECEIPT_A_SCORE,
        "delta_vs_receipt_a_pct": 100.0 * (score / RECEIPT_A_SCORE - 1.0),
        "delta_vs_crown": score - CROWN_SCORE,
        "improved": receipt.get("improved"),
        "parity_all_ok": live.get("parity_all_ok"),
        "accepted_pair_count": live.get("accepted_pair_count"),
        "per_prompt_min_ratio": ratios[0],
        "per_prompt_max_ratio": ratios[-1],
        "central_pair_low": ratios[3],
        "central_pair_high": ratios[4],
        "deepened_prompt_count": len(deepened),
        "control_prompt_count": len(control),
        "deepened_mean_mtp_pct": st.mean(deepened),
        "control_mean_mtp_pct": st.mean(control),
        "depth_attributable_mtp_pct": st.mean(deepened) - st.mean(control),
        "deepened_all_slower": all(v > 0 for v in deepened),
    })
    run.log({"per_prompt": table})
    run.finish()
    print(f"logged {run.url}")


if __name__ == "__main__":
    main()
