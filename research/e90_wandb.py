#!/usr/bin/env python3
"""E90: publish the GPU busy/idle accounting of a drafting round to W&B.

usage:
  research/e90_wandb.py --rung0a PATH --intervals PATH [--name NAME] [--notes TEXT]

`--intervals` takes the JSON list written by `research/e90_intervals.py
--output`. One run carries the composed-tree exactness proof (rung 0a), the
per-leg interval table (rung 0b) in all three host-state strata, and the
per-round rows another agent needs to re-derive every median.

Every leg is UNGATED, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` travel with the run verbatim. The legs are
diagnostic, not a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from e90_intervals import ANCHORS, HOST_PHASES

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e90-close-the-gpu-idle-window-in-the-drafting-round"

STRATA = [("aggregate", "all"), ("aggregate_clean", "clean"),
          ("aggregate_stuck", "stuck")]

INTERVAL_COLS = ["tag", "stratum", "rounds", "interval", "median_us", "mean_us",
                 "min_us", "max_us", "gpu_busy_us", "gpu_idle_us", "idle_pct"]


def interval_rows(doc: dict) -> list[dict]:
    rows = []
    counts = {"all": doc["rounds_analysed"], "clean": doc["rounds_clean"],
              "stuck": doc["rounds_stuck"]}
    labels = [label for _, _, label in ANCHORS] + [
        "inter_round_gap", "host_phase_sum", "round"]
    for key, stratum in STRATA:
        agg = doc.get(key) or {}
        if not agg:
            continue
        for label in labels:
            span = agg.get("%s_us" % label)
            if not span or not span.get("n"):
                continue
            if label == "round":
                busy = agg["gpu_busy_us_total"]["median"]
                idle = agg["gpu_idle_us_total"]["median"]
            elif label == "host_phase_sum":
                idle = agg["host_phase_idle_us"]["median"]
                busy = span["median"] - idle
            else:
                busy = agg["%s_gpu_busy_us" % label]["median"]
                idle = agg["%s_gpu_idle_us" % label]["median"]
            rows.append({
                "tag": doc["tag"], "stratum": stratum,
                "rounds": counts[stratum], "interval": label,
                "median_us": span["median"], "mean_us": span["mean"],
                "min_us": span["min"], "max_us": span["max"],
                "gpu_busy_us": busy, "gpu_idle_us": idle,
                "idle_pct": 100.0 * idle / span["median"] if span["median"] else 0.0,
            })
    return rows


def table(columns, rows) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def log_leg(run, doc: dict) -> None:
    tag = doc["tag"]
    run.log({f"rung0b/{tag}/intervals": table(INTERVAL_COLS, interval_rows(doc))})

    per_round = doc["per_round"]
    columns = sorted(per_round[0])
    run.log({f"rung0b/{tag}/per_round": table(columns, per_round)})

    agg = doc["aggregate"]
    scalars = {
        f"rung0b/{tag}/round_us_median": agg["round_us"]["median"],
        f"rung0b/{tag}/gpu_busy_us_median": agg["gpu_busy_us_total"]["median"],
        f"rung0b/{tag}/gpu_idle_us_median": agg["gpu_idle_us_total"]["median"],
        f"rung0b/{tag}/gpu_idle_pct": 100.0 * agg["gpu_idle_us_total"]["median"]
        / agg["round_us"]["median"],
        f"rung0b/{tag}/gpu_idle_us_within_d_submit2":
            agg["gpu_idle_us_within_d_submit2"]["median"],
        f"rung0b/{tag}/d_submit2_gpu_busy_pct":
            100.0 * agg["d_submit2_gpu_busy_us"]["median"]
            / agg["d_submit2_us"]["median"],
        f"rung0b/{tag}/tiling_error_us_median": agg["tiling_error_us"]["median"],
        f"rung0b/{tag}/frac_rounds_host_stuck": doc["frac_rounds_host_stuck"],
        f"rung0b/{tag}/rounds_analysed": doc["rounds_analysed"],
        f"rung0b/{tag}/rounds_without_gpu_busy": doc["rounds_without_gpu_busy"],
        f"rung0b/{tag}/largest_idle_interval_us": max(
            agg["%s_gpu_idle_us" % label]["median"] for _, _, label in ANCHORS),
    }
    if agg.get("host_thread_cpu_ns", {}).get("n"):
        scalars[f"rung0b/{tag}/host_thread_cpu_ns_median"] = \
            agg["host_thread_cpu_ns"]["median"]
    for key, sub in (("clean", "aggregate_clean"), ("stuck", "aggregate_stuck")):
        block = doc.get(sub) or {}
        if not block:
            continue
        scalars[f"rung0b/{tag}/{key}/round_us_median"] = block["round_us"]["median"]
        scalars[f"rung0b/{tag}/{key}/gpu_idle_us_median"] = \
            block["gpu_idle_us_total"]["median"]
        scalars[f"rung0b/{tag}/{key}/host_phase_sum_us_median"] = \
            block["host_phase_sum_us"]["median"]
        if block.get("host_thread_cpu_ns", {}).get("n"):
            scalars[f"rung0b/{tag}/{key}/host_thread_cpu_ns_median"] = \
                block["host_thread_cpu_ns"]["median"]
    run.log(scalars)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung0a", required=True)
    ap.add_argument("--intervals", required=True)
    ap.add_argument("--name", default="e90-gpu-idle-accounting")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    rung0a = json.loads(Path(args.rung0a).read_text())
    docs = json.loads(Path(args.intervals).read_text())
    meta = docs[0]["meta"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name,
        job_type="diagnostic-session", notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "harness": "local",
            "local_mode": "--local-iterate",
            "timed": False,
            "decode_tokens": 512,
            "offered_depth": 8,
            "legs": [d["tag"] for d in docs],
            "host_phases": HOST_PHASES,
            "host_stuck_threshold_us": docs[0]["host_stuck_threshold_us"],
            "cool_gate": 0,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "host": "ip-10-231-2-12.ec2.internal",
            "chip": "Apple M4 Pro",
            "memory_bytes": 51539607552,
            "ranked_host": False,
            "base_sha": meta.get("base_sha"),
            "worker_sha256": meta.get("worker_sha256"),
            "head_dir": meta.get("head_dir"),
            "instrument": "E90GPUIntervalLedger: MTLCommandBuffer commit hook, "
                          "gpuStartTime/gpuEndTime per completed buffer, union "
                          "intersected with the round's inter-anchor windows",
        },
    )

    run.log({"rung0a/exactness": table(sorted(rung0a), [rung0a])})
    for key, value in rung0a.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            run.log({f"rung0a/{key}": value})

    for doc in docs:
        log_leg(run, doc)

    run.log({"raw/intervals": wandb.Table(
        columns=["json"], data=[[json.dumps(docs, indent=2)]])})

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
