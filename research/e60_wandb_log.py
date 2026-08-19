#!/usr/bin/env python3
"""Log one E60 leg to W&B, immediately after that leg closes.

The advisor's standing instruction is to log while measuring, never once at
session end, so this script resumes a single run id and is called by
research/e60_session.sh after every leg.

  research/e60_wandb_log.py --leg research/out/e60-t300-1A
  research/e60_wandb_log.py --summary research/e60-artifacts/e60-rung0.json

Round counts are derived from the score payload, which is exact:

  rounds        = tokens / (1 + accepted_rate * mean_draft_len)
  drafted rows  = rounds * mean_draft_len
  accepted rows = tokens - rounds
  rejected rows = drafted - accepted
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e60-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

LOCAL_NULL_FLOOR_PERCENT = 0.0629
RANKED_MDE_PERCENT = 0.283

ARM_SURFACE = {
    "A": "upstream/main 9e1ff9ec (promoted frontier 59b321ee: organizer main "
         "0c90733d plus warmTargetLaterWindowSDPA)",
    "B": "campaign base 38e43f07",
    "C": "campaign base 38e43f07 plus hand-applied warmTargetLaterWindowSDPA",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e60-campaign-composite-vs-organizer-main",
        "revision_id": "r1",
        "pr_number": 63,
        "base_sha": "38e43f07a0dde9f4abad96d69c1497e2e83db403",
        "upstream_sha": "0c90733d383f6b987a29682bf9eb9458a6172bfa",
        "frontier_sha": "9e1ff9ec7152a04b753f2efb91c3e559909ea4b9",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_model": "Mac16,11",
        "host_physical_memory_gib": 48,
        "harness": "local",
        "local_mode": "--local-iterate",
        "darkbloom_startup_memory_profile": "full",
        "mlx_max_mb_per_buffer": 512,
        "mlx_max_ops_per_buffer": 50,
        "command_buffer_geometry": "ranked",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "local_null_floor_percent": LOCAL_NULL_FLOOR_PERCENT,
        "ranked_mde_percent_2sd": RANKED_MDE_PERCENT,
        "live_promoted_frontier_score": 3.24985583421771,
        "organizer_main_score": 3.24929399,
        "our_best_official_score": 3.23250848263467,
        "arm_surfaces": ARM_SURFACE,
    }


def resume_run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id or None,
        resume="allow",
        name="e60-campaign-composite-vs-frontier",
        job_type="arm-comparison",
        config=identity(),
        tags=["e60", "arm-comparison", "qwen-alphonse", "local", "ranked-geometry"],
    )
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


def read_meta(leg: pathlib.Path) -> dict:
    meta: dict[str, str] = {}
    path = leg / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def leg_record(leg: pathlib.Path) -> dict:
    meta = read_meta(leg)
    score_path = leg / "score.json"
    record: dict = {
        "tag": leg.name,
        "arm": meta.get("arm", "?"),
        "tokens": int(meta.get("tokens", 0) or 0),
        "exit": int(meta.get("exit", -1) or -1),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "staged_worker_sha256": meta.get("staged_worker_sha256"),
        "mlx_max_mb_per_buffer": int(meta.get("mlx_max_mb_per_buffer", 0) or 0),
        "mlx_max_ops_per_buffer": int(meta.get("mlx_max_ops_per_buffer", 0) or 0),
        "darkbloom_startup_memory_profile": meta.get(
            "darkbloom_startup_memory_profile"
        ),
        "low_memory_notice_count": int(meta.get("low_memory_notice_count", -1) or -1),
        "low_memory_notice_count_under_auto": int(
            meta.get("low_memory_notice_count_under_auto", -1) or -1
        ),
        "peak_ram_gb_max": float(meta["peak_ram_gb_max"])
        if meta.get("peak_ram_gb_max") else None,
    }
    worker_env = leg / "worker-env.txt"
    record["worker_env_proof"] = (
        worker_env.read_text().split() if worker_env.exists() else []
    )
    if not score_path.exists():
        record["failed"] = True
        err = leg / "wrapper.err"
        if err.exists():
            record["error_tail"] = err.read_text().splitlines()[-25:]
        return record
    metrics = json.loads(score_path.read_text())["metrics"]
    record["failed"] = False
    record.update(
        {
            "decode_tokens": metrics["decode_tokens"],
            "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
            "serial_seconds_per_token": metrics["serial_seconds_per_token"],
            "mtp_decode_speedup": metrics["mtp_decode_speedup"],
            "effective_mean_draft_len": metrics["effective_mean_draft_len"],
            "accepted_draft_rate": metrics["accepted_draft_rate"],
            "all_tokens_matched": metrics["all_tokens_matched"],
            "residual_divergence_count": metrics["residual_divergence_count"],
            "mtp_depth": metrics["mtp_depth"],
            "public_drift_tripwire_passed": metrics["public_drift_tripwire_passed"],
            "head_provenance_sha256": metrics["head_provenance_sha256"],
        }
    )
    tokens = metrics["decode_tokens"]
    mdl = metrics["effective_mean_draft_len"]
    rate = metrics["accepted_draft_rate"]
    rounds = tokens / (1.0 + rate * mdl)
    drafted = rounds * mdl
    accepted = tokens - rounds
    record.update(
        {
            "rounds": rounds,
            "drafted_rows": drafted,
            "accepted_rows": accepted,
            "rejected_rows": drafted - accepted,
            "declared_rows": rounds + drafted,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg")
    parser.add_argument("--summary")
    args = parser.parse_args()

    run = resume_run()
    if args.leg:
        leg = pathlib.Path(args.leg)
        record = leg_record(leg)
        payload = {
            f"leg/{key}": value
            for key, value in record.items()
            if isinstance(value, (int, float, bool)) and not isinstance(value, bool)
        }
        payload["leg/tag"] = record["tag"]
        payload["leg/arm"] = record["arm"]
        run.log(payload)
        run.summary[f"legs/{record['tag']}"] = record
        print(json.dumps(record, indent=2, default=str))
    if args.summary:
        summary = json.loads(pathlib.Path(args.summary).read_text())
        run.summary.update(summary)
        print(json.dumps(summary, indent=2, default=str)[:4000])
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
