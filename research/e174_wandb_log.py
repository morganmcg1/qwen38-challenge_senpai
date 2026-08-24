#!/usr/bin/env python3
"""Publish the E168 r1 single-leg confirmation of the organizer-pure cap-4 tree.

    usage: research/e174_wandb_log.py [--root research/out/e174]

One 512-token `--local-submit` leg with the real 40 C gate, run on the exact
tree that is submitted. It is `harness=local`: both legs of a local run use the
candidate build, so the serial-to-MTP ratio here is not the ranked numerator.
The leg's purpose is exactness, row-ledger closure and a width census on the
ship tree, plus a sanity check that absolute candidate MTP time on the
organizer-pure parent sits where the r0 cap arm sat.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e168-depth-cap-confirmation"

# E168 r0, same host, two-build palindrome session, real gate, 512 tokens.
R0_BASE_MS_PER_TOKEN = 31.535349669866264
R0_CAP_MS_PER_TOKEN = 30.081754783168435

CROWN = "ec24d591"
CROWN_SCORE = 3.7291100105909
RECEIPT_A = "5a9f130a"
RECEIPT_A_SCORE = 3.70784519415395

# A round of verified width M reads the backbone weight stream twice at M >= 6.
TWO_PASS_WIDTH = 6


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/out/e174")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    meta = read_meta(root / "meta.txt")
    metrics = json.loads((root / "score.json").read_text())["metrics"]

    timed_path = root / "reports" / "04-mtp-timed.json"
    report = json.loads(timed_path.read_text()) if timed_path.exists() else None
    widths = [1 + w for w in report["effective_draft_lengths"]] if report else []
    histogram: dict[int, int] = {}
    for width in widths:
        histogram[width] = histogram.get(width, 0) + 1
    two_pass_rounds = sum(n for m, n in histogram.items() if m >= TWO_PASS_WIDTH)

    ledger_expected = (
        len(widths) + sum((m - 1) * n for m, n in histogram.items()) if widths else None
    )
    ledger_closed = (
        ledger_expected == report["declared_rows_total"] if report else None
    )

    mtp_ms = 1000.0 * metrics["mtp_seconds_per_token"]

    config = {
        "experiment": "e168-r1-depth-cap-4-organizer-pure",
        "mechanism": "segmentedVerifyDepthCap 7 -> 4",
        "parent": "organizer main",
        "organizer_sha": meta.get("organizer_sha", ""),
        "candidate_sha": meta.get("candidate_sha", git("rev-parse", "HEAD")),
        "submitted_paths_changed": 1,
        "harness": "local",
        "leg_kind": "single-leg-gated-ship-tree-confirmation",
        "mode": metrics["mode"],
        "tokens": metrics["decode_tokens"],
        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": True,
        "gpu_temp_entry_is_pre_gate": True,
        "trace_perturbs_timing": False,
        "official_score": False,
        "primary_metric": "mtp_seconds_per_token",
        "host": meta.get("host", ""),
        "chip": meta.get("chip", ""),
        "worker_sha256": meta.get("worker_sha256", ""),
        "cli_sha256": meta.get("cli_sha256", ""),
        "metallib_sha256": meta.get("metallib_sha256", ""),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint", ""),
        "head_provenance_sha256": metrics.get("head_provenance_sha256", ""),
        "uses_pinned_mtp_head": metrics.get("uses_pinned_mtp_head"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "crown_submission": CROWN,
        "crown_score": CROWN_SCORE,
        "receipt_a_submission": RECEIPT_A,
        "receipt_a_score": RECEIPT_A_SCORE,
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e168-r1-cap4-organizer-pure-confirm-512",
        job_type="local-submit-confirmation",
        config=config,
    )

    summary = {
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "mtp_milliseconds_per_token": mtp_ms,
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "mtp_decode_speedup_local_ratio": metrics["mtp_decode_speedup"],
        "accepted_draft_rate": metrics["accepted_draft_rate"],
        "effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "decode_tokens": metrics["decode_tokens"],
        "all_tokens_matched": metrics["all_tokens_matched"],
        "residual_divergence_count": metrics["residual_divergence_count"],
        "public_drift_tripwire_passed": metrics["public_drift_tripwire_passed"],
        "gpu_temp_entry_c_pre_gate": meta.get("gpu_temp_entry_c", ""),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c", ""),
        "worker_digest_stable": meta.get("worker_sha256")
        == meta.get("post_run_worker_sha256"),
        "candidate_sha_stable": meta.get("candidate_sha")
        == meta.get("post_run_candidate_sha"),
        # Same host, same token window, same gate, r0 two-build session.
        "r0_base_milliseconds_per_token": R0_BASE_MS_PER_TOKEN,
        "r0_cap_milliseconds_per_token": R0_CAP_MS_PER_TOKEN,
        "delta_vs_r0_base_pct": 100.0 * (mtp_ms - R0_BASE_MS_PER_TOKEN)
        / R0_BASE_MS_PER_TOKEN,
        "delta_vs_r0_cap_pct": 100.0 * (mtp_ms - R0_CAP_MS_PER_TOKEN)
        / R0_CAP_MS_PER_TOKEN,
    }
    if report is not None:
        summary.update(
            {
                "rounds": len(widths),
                "mean_m": statistics.fmean(widths),
                "max_m": max(widths),
                "two_pass_rounds": two_pass_rounds,
                "two_pass_fraction": two_pass_rounds / len(widths),
                "declared_rows_total": report["declared_rows_total"],
                "accepted_draft_total": report["accepted_draft_total"],
                "emitted_token_total": report["emitted_token_total"],
                "row_ledger_expected": ledger_expected,
                "row_ledger_closed": ledger_closed,
            }
        )
    run.summary.update(summary)

    if histogram:
        table = wandb.Table(columns=["verified_width_m", "rounds", "two_pass"])
        for width in sorted(histogram):
            table.add_data(width, histogram[width], width >= TWO_PASS_WIDTH)
        run.log({"width_census": table})

    for key in sorted(summary):
        print(f"{key}={summary[key]}")
    print(f"width_histogram={dict(sorted(histogram.items()))}")
    print(f"wandb_url={run.url}")
    print(f"wandb_run_id={run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
