#!/usr/bin/env python3
"""Publish the E185 decode-round GPU idle measurement to W&B.

    usage: research/e185_wandb.py --residency research/e185-residency.json \
                                  [--intervals research/e185-intervals.json] \
                                  [--name NAME] [--notes TEXT]

One run per session. The residency document is the stage 1 hardware-counter
screen and the intervals document is the stage 2 phase attribution from the
E90 command-buffer ledger. Both are `harness=local` observation legs and
neither is a timed contrast, so nothing here is a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e185-decode-gpu-idle-gaps"


def table(rows: list[dict]) -> wandb.Table:
    columns = sorted({k for row in rows for k in row})
    return wandb.Table(
        columns=columns,
        data=[[row.get(c) for c in columns] for row in rows],
    )


PHASES = [
    "d_pre", "d_flush", "d_head1", "d_submit1", "d_chain", "d_submit2",
    "snapshot", "verify_graph", "eval_wall", "readout", "commit", "upkeep",
    "inter_round_gap",
]


def phase_rows(agg: dict) -> list[dict]:
    rows = []
    for phase in PHASES:
        span = agg.get(f"{phase}_us", {}).get("median")
        busy = agg.get(f"{phase}_gpu_busy_us", {}).get("median")
        idle = agg.get(f"{phase}_gpu_idle_us", {}).get("median")
        if span is None:
            continue
        rows.append({
            "phase": phase,
            "median_us": span,
            "gpu_busy_us": busy,
            "gpu_idle_us": idle,
            "idle_pct": (100.0 * idle / span) if span and idle is not None else None,
        })
    round_us = agg.get("round_us", {}).get("median")
    round_idle = agg.get("gpu_idle_us_total", {}).get("median")
    if round_us:
        rows.append({
            "phase": "ROUND",
            "median_us": round_us,
            "gpu_busy_us": agg.get("gpu_busy_us_total", {}).get("median"),
            "gpu_idle_us": round_idle,
            "idle_pct": 100.0 * round_idle / round_us if round_idle is not None else None,
        })
    return rows


def log_scalars(run, prefix: str, doc: dict) -> None:
    for key, value in doc.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        run.summary[f"{prefix}/{key}"] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--residency", required=True)
    ap.add_argument("--intervals")
    ap.add_argument("--name", default="e185-decode-gpu-idle-gaps")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    residency = json.loads(Path(args.residency).read_text())
    intervals = json.loads(Path(args.intervals).read_text()) if args.intervals else None

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        job_type="diagnostic-session",
        notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "harness": "local",
            "local_mode": "--local-iterate",
            "timed_contrast": False,
            "official_or_ranked_score": False,
            "gate_qualified_for_timing": False,
            "cool_gate_passed_real_gate": False,
            "ranked_host": False,
            "decode_tokens": residency.get("tokens"),
            "base_sha": residency.get("base_sha"),
            "worker_sha256": residency.get("worker_sha256"),
            "host": residency.get("host"),
            "chip": residency.get("chip"),
            "stage1_instrument": "powermetrics --samplers gpu_power -f plist; "
            "per-sample idle_ns/elapsed_ns hardware counters",
            "stage1_sampler_interval_ms": residency.get("sampler_interval_ms"),
            "stage2_instrument": "E90GPUIntervalLedger: MTLCommandBuffer commit "
            "hook, gpuStartTime/gpuEndTime per completed buffer, union "
            "intersected with the mtp-anchor round windows",
        },
    )

    log_scalars(run, "stage1", residency)
    run.log({"stage1/phase_profile": table(residency["phase_profile"])})
    run.log({"stage1/alignment_sensitivity": table(residency["alignment_sensitivity"])})

    if intervals:
        docs = intervals if isinstance(intervals, list) else [intervals]
        for doc in docs:
            tag = doc.get("tag", "leg")
            for stratum in ("aggregate", "aggregate_clean"):
                agg = doc.get(stratum) or {}
                if not agg:
                    continue
                run.log({f"stage2/{tag}/{stratum}": table(phase_rows(agg))})
                for key, stats in agg.items():
                    if isinstance(stats, dict) and "median" in stats:
                        run.summary[f"stage2/{tag}/{stratum}/{key}_median"] = (
                            stats["median"]
                        )
            log_scalars(run, f"stage2/{tag}", {
                k: v for k, v in doc.items() if not isinstance(v, (list, dict))
            })

    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
