#!/usr/bin/env python3
"""Publish the E185 decode-round GPU idle measurement to W&B.

    usage: research/e185_wandb.py --residency RESIDENCY.json [MORE.json ...] \
                                  [--gaps GAPS.json ...] \
                                  [--reconcile RECONCILE.json] \
                                  [--name NAME] [--notes TEXT]

One run per session. A residency document is the `powermetrics` hardware-idle
counter view of one leg, a gaps document is the E90 command-buffer ledger view
of one leg, and the reconcile document pairs the two instruments on the one
leg that carried both. All are `harness=local` observation legs and none is a
timed contrast, so nothing here is a score.
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


def phase_rows(doc: dict) -> list[dict]:
    rows = [
        {
            "phase": p["phase"],
            "median_us": p["span_us_median"],
            "gpu_busy_us": p["gpu_busy_us_median"],
            "gpu_idle_us": p["gpu_idle_us_median"],
            "idle_pct": p["idle_pct"],
            "instrument_only": p["instrument_only"],
        }
        for p in doc.get("phases", [])
    ]
    round_us = doc.get("round_us", {}).get("median")
    if round_us:
        rows.append({
            "phase": "ROUND",
            "median_us": round_us,
            "gpu_busy_us": doc["round_gpu_busy_us"]["median"],
            "gpu_idle_us": doc["round_gpu_idle_us"]["median"],
            "idle_pct": 100.0 * doc["round_gpu_idle_us"]["median"] / round_us,
            "instrument_only": False,
        })
    return rows


def log_scalars(run, prefix: str, doc: dict) -> None:
    for key, value in doc.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        run.summary[f"{prefix}/{key}"] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--residency", required=True, nargs="+")
    ap.add_argument("--gaps", nargs="*", default=[])
    ap.add_argument("--reconcile")
    ap.add_argument("--biggaps")
    ap.add_argument("--name", default="e185-decode-gpu-idle-gaps")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    residencies = [json.loads(Path(p).read_text()) for p in args.residency]
    residency = residencies[0]
    gaps = [json.loads(Path(p).read_text()) for p in args.gaps]
    reconcile = json.loads(Path(args.reconcile).read_text()) if args.reconcile else None
    biggaps = json.loads(Path(args.biggaps).read_text()) if args.biggaps else None

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

    for doc in residencies:
        tag = doc.get("tag", "leg")
        log_scalars(run, f"hardware/{tag}", doc)
        run.log({f"hardware/{tag}/phase_profile": table(doc["phase_profile"])})
        run.log(
            {f"hardware/{tag}/alignment_sensitivity": table(doc["alignment_sensitivity"])}
        )

    for doc in gaps:
        tag = doc.get("tag", "leg")
        run.log({f"ledger/{tag}/phases": table(phase_rows(doc))})
        run.log({f"ledger/{tag}/gap_size_by_start_phase": table(
            [{"phase": k, **v} for k, v in doc.get("gap_size_by_start_phase", {}).items()]
        )})
        for key, stats in doc.items():
            if isinstance(stats, dict) and "median" in stats:
                run.summary[f"ledger/{tag}/{key}_median"] = stats["median"]
        log_scalars(run, f"ledger/{tag}", {
            k: v for k, v in doc.items() if not isinstance(v, (list, dict))
        })

    if biggaps:
        run.log({"ledger/gaps_at_least_1ms_by_start_phase": table(biggaps)})

    if reconcile:
        log_scalars(run, "reconcile", reconcile)
        for key, stats in reconcile.get("per_sample_regression", {}).items():
            run.summary[f"reconcile/per_sample_regression/{key}"] = stats
        run.log({"reconcile/gap_size_buckets": table(reconcile["gap_size_buckets"])})

    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
