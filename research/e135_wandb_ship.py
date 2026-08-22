#!/usr/bin/env python3
"""Publish the E135 ship-commit evidence to W&B.

    usage: research/e135_wandb_ship.py [--run-id e135ship] [--dry]

This run is the archive that was submitted officially, not a timing session.
It carries the bare 512-token exactness leg, the launched-column witness read
off that leg's own dispatch argument, the worker digest recorded before and
after the leg, and the record of the `--local-submit` harness failure that the
leg replaced. Nothing here is a ranked or official score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e135-tight-qmv-launch-grid"

LEG_DIR = pathlib.Path("research/out/e135x512")

SUBMISSION = {
    "yukon_submission_id": "572b2cc4-0299-4871-9d5b-73eadbe95d3b",
    "yukon_benchmark_id": "5d1ee4d7-80bd-4555-b182-6505f26ef495",
    "yukon_base_sha": "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf",
    "yukon_model_attribution": "senpai",
    "submitted_at_utc": "2026-08-22T20:29Z",
    "beaten_baseline_receipt": "623e77af",
    "beaten_baseline_official_score": 3.52085227003175,
    "live_crown_official_score_at_call": 3.68172016051458,
}

# The `--local-submit` gate could not run on this host. Both attempts died in
# the MTP reference pass, which stands up a second depth-8 session, and the
# window length made no difference.
LOCAL_SUBMIT_DEFECT = {
    "local_submit_available": False,
    "local_submit_failure_stage": "mtp reference row generation, depth=8",
    "local_submit_failure_status": 15,
    "local_submit_failed_at_tokens": [512, 128],
    "local_submit_window_dependent": False,
    "host_physical_gib": 48,
    "host_startup_profile": "low-memory",
}


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def width_histogram(path: pathlib.Path) -> dict:
    """Count realised verify widths from the round trace: M = d + 1."""
    if not path.exists():
        return {}
    hist: dict[int, int] = {}
    for match in re.finditer(r"\bd=(\d+)", path.read_text()):
        width = int(match.group(1)) + 1
        hist[width] = hist.get(width, 0) + 1
    return {str(k): hist[k] for k in sorted(hist)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="e135ship")
    parser.add_argument("--name", default="e135-ship-commit-exactness")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    score = json.loads((LEG_DIR / "score.json").read_text())
    metrics = score["metrics"]
    meta = read_meta(LEG_DIR / "meta.txt")
    pipelines = json.loads((LEG_DIR / "pipelines.json").read_text())

    columns = {str(k): v for k, v in sorted(
        ((int(k), v) for k, v in pipelines["columns_by_width"].items()))}

    config = {
        "experiment": "e135",
        "rung": "ship",
        "harness": "local",
        "question": (
            "does the submitted tight QMV launch grid reproduce the serial "
            "token stream exactly over a full 512-token window in its shipped "
            "default configuration"
        ),
        "ship_sources_commit": "1efb1916",
        "leg_commit": meta.get("base_sha"),
        "campaign_base_sha": "ea546d1f8ab0d1de2169e5fbf19bdef2aa79003d",
        "worker_sha256": meta.get("worker_sha256"),
        "post_run_worker_sha256": meta.get("post_run_worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_bytes": int(meta.get("memory_bytes", 0)),
        "decode_tokens": metrics["decode_tokens"],
        "local_mode": meta.get("local_mode"),
        "grid_env_override": None,
        "table_env_override": None,
        "parsed_grid_enum": pipelines.get("grid"),
        "default_grid_witness": pipelines.get("default_grid"),
        "width_plan": pipelines.get("plan"),
        "gpu_temp_entry_c": float(meta.get("gpu_temp_entry_c", "nan")),
        "gpu_temp_exit_c": float(meta.get("gpu_temp_exit_c", "nan")),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        **SUBMISSION,
        **LOCAL_SUBMIT_DEFECT,
    }

    summary = {
        "e135_exact_token_divergences": metrics["residual_divergence_count"],
        "e135_all_tokens_matched": metrics["all_tokens_matched"],
        "e135_public_drift_tripwire_passed":
            metrics["public_drift_tripwire_passed"],
        "e135_leg_passed": score["passed"],
        "e135_ship_mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "e135_ship_serial_seconds_per_token":
            metrics["serial_seconds_per_token"],
        "e135_ship_local_ratio": metrics["mtp_decode_speedup"],
        "e135_effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "e135_accepted_draft_rate": metrics["accepted_draft_rate"],
        "e135_trace_rounds": int(meta.get("trace_rounds", 0)),
        "e135_columns_by_width": columns,
        "e135_realised_width_histogram": width_histogram(LEG_DIR / "trace.txt"),
        "e135_worker_digest_stable":
            meta.get("worker_sha256") == meta.get("post_run_worker_sha256"),
    }

    if args.dry:
        print(json.dumps({"config": config, "summary": summary}, indent=2,
                         default=str))
        return 0

    import wandb

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     id=args.run_id, name=args.name, resume="allow",
                     config=config,
                     tags=["e135", "ship", "exactness", "submitted"])
    run.summary.update(summary)
    run.finish()
    print(f"published {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
