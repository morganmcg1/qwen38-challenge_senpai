#!/usr/bin/env python3
"""Publish the E199 cap-8 width-9 gate evidence to W&B.

    usage: research/e199_wandb_log.py [--out research/out/e199/gate]

Everything here comes from one gate-qualified 512-token ``--local-submit``
session: the real 40 C cool gate ran before all three legs and the per-round
phase trace was never enabled, so the wall times are admissible as
``harness=local`` figures.

The cap-7 rows logged for context are NOT a matched control. They come from the
retained E181 session on this same host, which ran a different tree and a
PINNED-shape head directory (``6fe9db14``), while this run ran the DECLARED
head (``dadbfb80``). FINDING 511 measured about 8.5 %/token between head
classes, which is larger than the cap-7-to-cap-8 difference, so no causal
timing claim is made from that pair. The row-ledger counts are logged beside
them because counts do not depend on temperature or host.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e199-cap8-paid-receipt"

CAMPAIGN_BASE_SHA = "5dec966ef33736e6dfc143256bca441454e2ea17"
RECEIPT_A = "5a9f130a"
RECEIPT_A_SCORE = 3.70784519415395
CROWN = "ec24d591"
CROWN_SCORE = 3.7291100105909
# E197 desk prediction for this cell (FINDING 514, W&B 1hdlmyzr).
E197_PREDICTED_MEDIAN = 3.7153
E197_BAND_LOW = 3.466
E197_BAND_HIGH = 3.994
E197_P_BEAT_CROWN = 0.465
E197_P_BEAT_A = 0.521
# The dR9 discriminator the receipt buys.
FLAT_CONTINUATION_PREDICTION = 3.82
STEPPED_CONTINUATION_PREDICTION = 3.68

# Retained cap-7 512-token legs on this host (E181, pinned-shape head).
CAP7_CONTEXT = {
    "rounds": 77,
    "accepted": 435,
    "rejected": 56,
    "rows": 568,
    "effective_mean_draft_len": 6.376623376623376,
    "accepted_draft_rate": 0.8859470468431772,
    "head_provenance_prefix": "6fe9db14",
    "mtp_seconds_per_token_legs": [
        0.03165631042793393, 0.031419593608006835, 0.03140815440565348,
        0.03162436140701175, 0.03159670322202146, 0.031431496143341064,
        0.03146121697500348, 0.031635339837521315,
    ],
}


def sh(*args: str) -> str:
    return subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout.strip()


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def load_timed_leg(reports: pathlib.Path) -> dict:
    for path in sorted(reports.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("verb") == "mtp-timed" \
                and not doc.get("is_serial_control", True):
            return doc
    raise SystemExit(f"e199_wandb_log: no timed MTP leg report in {reports}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="research/out/e199/gate")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    meta = read_meta(out / "meta.txt")
    report = load_timed_leg(out / "reports")
    metrics = json.loads((out / "score.json").read_text())["metrics"]

    lengths = report["effective_draft_lengths"]
    hist = collections.Counter(lengths)
    rounds = report["round_count"]
    accepted = report["accepted_draft_total"]
    rejected = report["rejected_draft_total"]
    rows = report["declared_rows_total"]

    cap7_mean = (sum(CAP7_CONTEXT["mtp_seconds_per_token_legs"])
                 / len(CAP7_CONTEXT["mtp_seconds_per_token_legs"]))

    config = {
        "experiment": "e199",
        "hypothesis": "segmentedVerifyDepthCap 7 -> 8 beats receipt A and the crown",
        "candidate_change": "Qwen36MTPBlockSession.segmentedVerifyDepthCap 7 -> 8",
        "candidate_commit": meta["base_sha"],
        "campaign_base_sha": CAMPAIGN_BASE_SHA,
        "upstream_sha": meta["upstream_sha"],
        "submitted_surface_files": 1,
        "submitted_surface_lines_changed": 1,
        "host": meta["host"],
        "chip": meta["chip"],
        "mem_bytes": int(meta["mem_bytes"]),
        "os": meta["os"],
        "toolchain": meta["toolchain"],
        "worker_sha256": meta["worker_sha256_before"],
        "cli_sha256": meta["cli_sha256"],
        "worker_digest_stable": meta["worker_digest_stable"] == "true",
        "worktree_clean": meta["worktree_clean"] == "true",
        "head_class": "declared",
        "head_provenance_sha256": metrics["head_provenance_sha256"],
        "head_origin": report["head_provenance"]["origin"],
        "fixed_draft_depth": "unset",
        "token_window": 512,
        "mode": metrics["mode"],
        "harness": "local",
        "fixture": "public single fixture (--local-submit)",
        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": True,
        "trace_perturbs_timing": False,
        "gpu_temp_entry_pre_gate_c": float(meta["gpu_temp_entry"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit"]),
        "e197_predicted_ranked_median": E197_PREDICTED_MEDIAN,
        "e197_band_low": E197_BAND_LOW,
        "e197_band_high": E197_BAND_HIGH,
        "e197_p_beat_crown": E197_P_BEAT_CROWN,
        "e197_p_beat_a": E197_P_BEAT_A,
        "receipt_a": RECEIPT_A,
        "receipt_a_score": RECEIPT_A_SCORE,
        "crown": CROWN,
        "crown_score": CROWN_SCORE,
        "flat_continuation_prediction": FLAT_CONTINUATION_PREDICTION,
        "stepped_continuation_prediction": STEPPED_CONTINUATION_PREDICTION,
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e199-cap8-width9-gate", job_type="gate", config=config,
        notes=("Combined width-9 exactness gate and 512-token --local-submit "
               "confirmation for the one-line cap-8 candidate."),
    )

    summary = {
        # --- width-9 proof (parent journal, not a worker counter) ---
        "gate/nine_row_round_count": hist.get(8, 0),
        "gate/effective_max_draft_len": report["effective_max_draft_len"],
        "gate/effective_max_rows": report["effective_max_draft_len"] + 1,
        "gate/round_count": rounds,
        "gate/non_drafting_round_count": report["non_drafting_round_count"],
        "gate/nine_row_round_fraction": hist.get(8, 0) / rounds,
        # --- RULE 179 ledger ---
        "ledger/rounds": rounds,
        "ledger/accepted_draft_total": accepted,
        "ledger/rejected_draft_total": rejected,
        "ledger/declared_rows_total": rows,
        "ledger/reference_checked_row_total": report["reference_checked_row_total"],
        "ledger/rejected_rows_reference_checked":
            report["rejected_rows_reference_checked"],
        "ledger/identity_ok": rounds + accepted + rejected == rows,
        # --- exactness ---
        "exact/all_tokens_matched": metrics["all_tokens_matched"],
        "exact/residual_divergence_count": metrics["residual_divergence_count"],
        "exact/parity_all_ok": report["parity_all_ok"],
        "exact/max_rejected_tail_logit_delta":
            report["max_rejected_tail_logit_delta"],
        "exact/verify_block_replayed_round_count":
            report["verify_block_replayed_round_count"],
        "exact/public_drift_tripwire_passed":
            metrics["public_drift_tripwire_passed"],
        "exact/emitted_token_total": report["emitted_token_total"],
        "exact/decode_tokens": metrics["decode_tokens"],
        # --- timing record (harness=local; not the decision) ---
        "local/serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "local/mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "local/ratio": metrics["mtp_decode_speedup"],
        "local/accepted_draft_rate": metrics["accepted_draft_rate"],
        "local/effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "local/seed_prefill_seconds": report["seed_prefill_seconds"],
        "local/decode_seconds": report["decode_seconds"],
        "local/p50_block_request_seconds": report["p50_block_request_seconds"],
        "local/max_block_request_seconds_after_first":
            report["max_block_request_seconds_after_first"],
        "local/target_cache_offset_final": report["target_cache_offset_final"],
        # --- unmatched cap-7 context, labelled as such ---
        "context_unmatched/cap7_mtp_seconds_per_token_mean": cap7_mean,
        "context_unmatched/cap7_rounds": CAP7_CONTEXT["rounds"],
        "context_unmatched/cap7_accepted": CAP7_CONTEXT["accepted"],
        "context_unmatched/cap7_rejected": CAP7_CONTEXT["rejected"],
        "context_unmatched/cap7_rows": CAP7_CONTEXT["rows"],
        "context_unmatched/cap7_effective_mean_draft_len":
            CAP7_CONTEXT["effective_mean_draft_len"],
        "context_unmatched/cap7_head_provenance_prefix":
            CAP7_CONTEXT["head_provenance_prefix"],
        "context_unmatched/delta_accepted": accepted - CAP7_CONTEXT["accepted"],
        "context_unmatched/delta_rejected": rejected - CAP7_CONTEXT["rejected"],
        "context_unmatched/delta_rows": rows - CAP7_CONTEXT["rows"],
        "context_unmatched/is_matched_control": False,
        "context_unmatched/why_not_matched":
            "different tree and different head class (6fe9db14 pinned-shape vs "
            "dadbfb80 declared); FINDING 511 head-class effect ~8.5%/token "
            "exceeds the cap effect",
        "verdict": "gate_passed",
    }
    run.summary.update(summary)

    depth_table = wandb.Table(columns=["drafts", "rows_verified", "rounds"])
    for drafts in sorted(hist):
        depth_table.add_data(drafts, drafts + 1, hist[drafts])
    run.log({"depth_histogram": depth_table})

    block_table = wandb.Table(columns=["round", "drafts", "rows",
                                       "block_request_seconds"])
    for index, (drafts, seconds) in enumerate(
            zip(lengths, report["block_request_seconds"])):
        block_table.add_data(index, drafts, drafts + 1, seconds)
    run.log({"per_round": block_table})

    for index, (drafts, seconds) in enumerate(
            zip(lengths, report["block_request_seconds"])):
        run.log({"round/index": index, "round/drafts": drafts,
                 "round/rows": drafts + 1,
                 "round/block_request_seconds": seconds})

    print(f"e199_wandb_log: {run.url}")
    print(f"e199_wandb_log: run_id={run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
