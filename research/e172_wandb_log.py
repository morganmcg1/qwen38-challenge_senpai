#!/usr/bin/env python3
"""Publish the E168 depth-cap two-build confirmation to W&B.

    usage: research/e172_wandb_log.py --report research/out/e172/report.json

Unlike the E168 census, every number here IS a timing measurement. The legs ran
the real 40 C cool gate, never enabled the per-round phase trace, and asserted
the worker digest before and after each leg, so the gate flags below are true
rather than disclaimers. They are still `harness=local`: both legs of a local
run use the candidate build, so this measures the candidate MTP leg against a
different build of the same leg, not the ranked numerator.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e168-depth-cap-confirmation"

CROWN = "ec24d591"
CROWN_SCORE = 3.7291100105909


def read_stage(root: pathlib.Path, arm: str) -> dict[str, str]:
    out: dict[str, str] = {}
    path = root / "bin" / arm / "stage.txt"
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[f"{arm}_{key}"] = value
    return out


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--root", default="research/out/e172")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    root = pathlib.Path(args.root)

    config = {
        "experiment": "e168-depth-cap",
        "mechanism": "segmentedVerifyDepthCap 7 -> 4",
        "harness": "local",
        "leg_kind": "two-build-gated-confirmation",
        "mode": "qwen-mtp-local-submit",
        "tokens": 512,
        "session_order": "base cap cap base",
        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": True,
        "trace_perturbs_timing": False,
        "official_score": False,
        "candidate_head": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "crown_submission": CROWN,
        "crown_score": CROWN_SCORE,
        # A local ratio cannot price the ranked numerator, so the primary
        # quantity is absolute candidate MTP time, not the serial-to-MTP ratio.
        "primary_metric": "mtp_seconds_per_token",
    }
    config.update(read_stage(root, "base"))
    config.update(read_stage(root, "cap"))

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e168-depth-cap-confirm-512", job_type="confirmation",
        config=config,
    )

    leg_columns = [
        "leg", "arm", "ms_per_token", "serial_ms_per_token",
        "mtp_decode_speedup", "mean_m", "multi_pass_fraction", "rounds",
        "accepted_draft_rate", "declared_rows_total", "emitted_token_total",
        "decode_tokens", "all_tokens_matched", "residual_divergence_count",
        "public_drift_tripwire_passed", "gpu_temp_entry", "gpu_temp_exit",
        "worker_digest_stable", "histogram",
    ]
    leg_table = wandb.Table(columns=leg_columns)
    for leg in report["legs"]:
        leg_table.add_data(
            leg["leg"], leg["arm"], leg["mtp_seconds_per_token"] * 1000.0,
            leg["serial_seconds_per_token"] * 1000.0, leg["mtp_decode_speedup"],
            leg["mean_m"], leg["multi_pass_fraction"], leg["rounds"],
            leg["accepted_draft_rate"], leg["declared_rows_total"],
            leg["emitted_token_total"], leg["decode_tokens"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["public_drift_tripwire_passed"], leg["gpu_temp_entry"],
            leg["gpu_temp_exit"], leg["worker_digest_stable"],
            json.dumps(leg["histogram"]),
        )

    arm_columns = [
        "arm", "legs", "median_ms_per_token", "half_range_pct", "sd_pct",
        "mean_m", "multi_pass_fraction", "accepted_draft_rate", "rounds",
    ]
    arm_table = wandb.Table(columns=arm_columns)
    summary: dict[str, float] = {}
    for arm, block in sorted(report["arms"].items()):
        arm_table.add_data(
            arm, block["legs"], block["median_ms_per_token"],
            block["half_range_pct"], block["sd_pct"], block["mean_m"],
            block["multi_pass_fraction"], block["accepted_draft_rate"],
            block["rounds"],
        )
        for field in (
            "median_ms_per_token", "half_range_pct", "sd_pct", "mean_m",
            "multi_pass_fraction", "accepted_draft_rate",
        ):
            summary[f"{arm}/{field}"] = block[field]

    for arm, contrast in report["contrasts"].items():
        summary[f"{arm}/change_pct_vs_base"] = contrast["change_pct"]
        summary[f"{arm}/combined_half_range_pct"] = contrast[
            "combined_half_range_pct"
        ]
        summary[f"{arm}/resolved"] = contrast["resolved"]

    summary["gate_failures"] = len(report["gate_failures"])
    summary["all_legs_gate_clean"] = not report["gate_failures"]

    run.log({"legs": leg_table, "arms": arm_table})
    run.summary.update(summary)
    print(f"logged {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
