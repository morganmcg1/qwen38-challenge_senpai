#!/usr/bin/env python3
"""E184: publish the prefill decomposition evidence as one W&B run.

Reads the reduction produced by `research/e184_analyse.py` on stdin, or the
session artifacts directly, and logs:
  - the isolated gated-delta scan probe table (T vs microseconds);
  - the in-path phase table for each granularity arm;
  - the trusted `seed_prefill_seconds` anchor for every arm and leg;
  - the derived mechanism pricing.

Every number is `harness=local` on a non-ranked host. `cool_gate_passed_real_gate`
and `gate_qualified_for_timing` are logged as False and never omitted.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# Ranked transfer constants recorded in the campaign ledger (harness=ranked).
RANKED_PREFILL_SHARE_OF_CANDIDATE_LEG = 0.1004  # prompt-weighted, n=32
RANKED_PREFILL_SPT_SD_FRACTION = 0.00200  # sd of prefill_seconds_per_token
PUBLISHED_SCORE_BASE = 3.70784519415395  # campaign best A


def git(*args):
    return subprocess.run(
        ["git", "-C", ROOT, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduction", required=True)
    parser.add_argument("--run-name", default="e184-prefill-cost-decomposition")
    args = parser.parse_args()

    with open(args.reduction) as handle:
        reduction = json.load(handle)

    probe = reduction.get("probe") or {}
    cells = {c["arm"]: c for c in probe.get("cells", [])}

    config = {
        "experiment": "e184-prefill-cost-decomposition",
        "harness": "local",
        "host_chip": "Apple M4 Pro",
        "host_gpu_cores": 20,
        "host_memory_gib": 48,
        "ranked_host": False,
        "base_sha": "cdb848a2fa02349a7e5ca1a8df0d1191ea347402",
        "candidate_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "token_window_decode": 8,
        "seed_tokens": 512,
        "mode": "qwen-mtp-local-iterate",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "cool_gate_env": "MLXFAST_LOCAL_COOL_GATE=0",
        "abba_counterbalanced": True,
        "instrument": "E184Prefill in-path phase brackets, default-off",
        "instrument_env_prefix": "DARKBLOOM_",
        "linear_layers": 48,
        "full_attention_layers": 16,
        "ranked_prefill_share_of_candidate_leg": RANKED_PREFILL_SHARE_OF_CANDIDATE_LEG,
        "ranked_prefill_spt_sd_fraction": RANKED_PREFILL_SPT_SD_FRACTION,
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.run_name,
        job_type="profiling", config=config,
        tags=["e184", "prefill", "decomposition", "harness-local", "ungated"],
    )

    # --- isolated scan probe -------------------------------------------------
    probe_rows = []
    for cell in probe.get("cells", []):
        probe_rows.append([
            cell["arm"], cell["timesteps"], cell["microseconds"],
            cell["forward_microseconds"], cell["reverse_microseconds"],
            cell["replicates"],
        ])
    run.log({"scan_probe": wandb.Table(
        columns=["arm", "timesteps", "microseconds", "forward_us",
                 "reverse_us", "replicates"],
        data=probe_rows)})

    scan_512 = cells.get("scan_T512", {}).get("microseconds")
    scan_512_l1 = cells.get("scan_T512_L1", {}).get("microseconds")
    chunk128 = cells.get("chunk128_proxy", {}).get("microseconds")
    scan_metrics = {}
    if scan_512 and scan_512_l1:
        scan_metrics.update({
            "scan_us_per_launch_T512": scan_512,
            "scan_us_fixed_term": scan_512_l1,
            "scan_us_loop_term": scan_512 - scan_512_l1,
            "scan_loop_fraction_of_launch": (scan_512 - scan_512_l1) / scan_512,
            "scan_seconds_whole_model": scan_512 * 48 / 1e6,
        })
    if scan_512 and chunk128:
        scan_metrics["chunk_parallel_ceiling_seconds"] = (
            (scan_512 - chunk128) * 48 / 1e6)

    # --- in-path arms -------------------------------------------------------
    anchor_rows = []
    phase_rows = []
    off_prefill = []
    for arm, payload in reduction["arms"].items():
        for record in payload["trusted"]:
            anchor_rows.append([
                arm, record["file"], record["seed_prefill_seconds"],
                record["decode_seconds"], record["prefill_seconds_per_token"],
            ])
            if arm.endswith("-off") and record["seed_prefill_seconds"]:
                off_prefill.append(record["seed_prefill_seconds"])
        for row in payload["phases"]:
            phase_rows.append([
                arm, row["phase"], row["mean_seconds"],
                row["share_of_bracketed"], row["n"], row["sd_rel"],
            ])
    run.log({
        "trusted_anchor": wandb.Table(
            columns=["arm", "report", "seed_prefill_seconds", "decode_seconds",
                     "prefill_seconds_per_token"],
            data=anchor_rows),
        "phase_table": wandb.Table(
            columns=["arm", "phase", "mean_seconds", "share_of_bracketed",
                     "n", "sd_rel"],
            data=phase_rows),
    })

    summary = dict(scan_metrics)
    if off_prefill:
        summary["off_seed_prefill_seconds_mean"] = statistics.mean(off_prefill)
        if len(off_prefill) > 1:
            summary["off_seed_prefill_seconds_sd"] = statistics.stdev(off_prefill)
    ceiling = summary.get("chunk_parallel_ceiling_seconds")
    anchor = summary.get("off_seed_prefill_seconds_mean")
    if ceiling and anchor:
        prefill_fraction = ceiling / anchor
        summary["chunk_parallel_prefill_fraction"] = prefill_fraction
        summary["chunk_parallel_published_gain_fraction"] = (
            prefill_fraction * RANKED_PREFILL_SHARE_OF_CANDIDATE_LEG)
        summary["chunk_parallel_published_delta"] = (
            prefill_fraction * RANKED_PREFILL_SHARE_OF_CANDIDATE_LEG
            * PUBLISHED_SCORE_BASE)

    for key, value in summary.items():
        run.summary[key] = value
    run.summary["mue_published_fraction_required"] = 0.0030

    print(json.dumps({"run_id": run.id, "url": run.url, "summary": summary},
                     indent=1))
    run.finish()


if __name__ == "__main__":
    main()
