#!/usr/bin/env python3
"""Log one E62 leg to W&B, immediately after that leg closes.

The advisor's standing instruction is to log while measuring, never once at
session end, so this script resumes a single run id and is called by
research/e62_session.sh after every leg.

  research/e62_wandb_log.py --leg research/out/e62-r1-ops-01-ops50 \
      --position 1 --session r1-ops
  research/e62_wandb_log.py --summary research/e62-artifacts/e62-rung0.json \
      --summary-key rung0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e62-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

# The campaign's adjacent-only 0.0629 % floor is retracted. The null floor
# scales with leg separation; these are the E60 within-session same-arm spreads.
NULL_FLOOR_BY_SEPARATION_PERCENT = {1: 0.0032, 3: 0.1147, 5: 0.1634}
MINIMUM_USEFUL_EFFECT_PERCENT = -0.15
RANKED_SINGLE_PAIR_THRESHOLD_PERCENT = 2.10


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e62-ranked-allocator-command-buffer-geometry",
        "revision_id": "r1",
        "pr_number": 65,
        "base_sha": "ea683aae5e41d7c84518e19d5c9cbae8434df774",
        "anchor_base_sha": "7040406",
        "anchor_scored_surface": "d2139c92",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_model": "Mac16,11",
        "host_physical_memory_gib": 48,
        "harness": "local",
        "local_mode": "--local-iterate",
        "darkbloom_startup_memory_profile": "full",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "null_floor_by_separation_percent": NULL_FLOOR_BY_SEPARATION_PERCENT,
        "minimum_useful_effect_percent": MINIMUM_USEFUL_EFFECT_PERCENT,
        "ranked_single_pair_threshold_percent": RANKED_SINGLE_PAIR_THRESHOLD_PERCENT,
        "anchor_mtp_seconds_per_token_512": 0.0340649975114502,
        "our_best_official_score": 3.23250848263467,
        "live_promoted_frontier_score": 3.24985583421771,
    }


def resume_run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id or None,
        resume="allow",
        name="e62-ranked-allocator-command-buffer-geometry",
        job_type="knob-sweep",
        config=identity(),
        tags=["e62", "qwen-alphonse", "local", "command-buffer", "allocator"],
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


def _float(meta: dict, key: str):
    value = meta.get(key)
    return float(value) if value else None


def leg_record(leg: pathlib.Path) -> dict:
    meta = read_meta(leg)
    score_path = leg / "score.json"
    record: dict = {
        "tag": leg.name,
        "arm": meta.get("label") or meta.get("arm", "?"),
        "binary_arm": meta.get("arm", "?"),
        "tokens": int(meta.get("tokens", 0) or 0),
        "exit": int(meta.get("exit", -1) or -1),
        "mlx_max_mb_per_buffer": int(meta.get("mlx_max_mb_per_buffer", 0) or 0),
        "mlx_max_ops_per_buffer": int(meta.get("mlx_max_ops_per_buffer", 0) or 0),
        "wired_residency_requested": meta.get("wired_residency_requested"),
        "wired_residency_active": meta.get("wired_residency_active") == "true",
        "wired_outcome_line": meta.get("wired_outcome_line"),
        "gpu_temp_entry_c": _float(meta, "gpu_temp_entry_c"),
        "gpu_temp_exit_c": _float(meta, "gpu_temp_exit_c"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "staged_worker_sha256": meta.get("staged_worker_sha256"),
        "worker_text_sha256": meta.get("worker_text_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "darkbloom_startup_memory_profile": meta.get(
            "darkbloom_startup_memory_profile"
        ),
        "low_memory_notice_count": int(meta.get("low_memory_notice_count", -1) or -1),
        "low_memory_notice_count_under_auto": int(
            meta.get("low_memory_notice_count_under_auto", -1) or -1
        ),
        "worker_peak_rss_gb": _float(meta, "worker_peak_rss_gb"),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
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
    for key in (
        "decode_tokens",
        "mtp_seconds_per_token",
        "serial_seconds_per_token",
        "mtp_decode_speedup",
        "effective_mean_draft_len",
        "accepted_draft_rate",
        "all_tokens_matched",
        "residual_divergence_count",
        "mtp_depth",
        "public_drift_tripwire_passed",
        "head_provenance_sha256",
    ):
        if key in metrics:
            record[key] = metrics[key]
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
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--summary")
    parser.add_argument("--summary-key", default="")
    args = parser.parse_args()

    run = resume_run()
    if args.leg:
        leg = pathlib.Path(args.leg)
        record = leg_record(leg)
        record["leg_position"] = args.position
        record["session"] = args.session
        payload = {
            f"leg/{key}": value
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        payload["leg/tag"] = record["tag"]
        payload["leg/arm"] = record["arm"]
        payload["leg/session"] = args.session
        run.log(payload)
        run.summary[f"legs/{record['tag']}"] = record
        print(json.dumps(record, indent=2, default=str))
    if args.summary:
        summary = json.loads(pathlib.Path(args.summary).read_text())
        if args.summary_key:
            summary = {f"{args.summary_key}/{k}": v for k, v in summary.items()}
        run.summary.update(summary)
        print(json.dumps(summary, indent=2, default=str)[:4000])
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
