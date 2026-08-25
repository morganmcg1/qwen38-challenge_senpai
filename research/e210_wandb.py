#!/usr/bin/env python3
"""Publish E210 GPU-idle census documents to W&B.

    usage: research/e210_wandb.py DOC.json [DOC.json ...] [--name NAME]
                                  [--notes TEXT]

Every document is a `harness=local`, ungated observation produced by
`research/e210_census.py`. None of them is an official or ranked score, and
none is a timing contrast: the ledger costs one lock and two clock reads per
command buffer, so absolute leg times from these legs are attribution figures
only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e210-gpu-idle-census"
GROUP = "qwen38-r1-e210-gpu-idle-census"

IDENTITY_KEYS = (
    "tag", "experiment", "harness", "official_or_ranked_score", "leg_role",
    "gate_qualified_for_timing", "cool_gate_passed_real_gate", "base_sha",
    "worker_sha256", "metallib_source_fingerprint", "host", "chip",
    "memory_bytes", "sandbox", "tokens", "stall_us",
    "head_provenance_sha256", "all_tokens_matched", "effective_mean_draft_len",
    "accepted_draft_rate", "mtp_seconds_per_token", "serial_seconds_per_token",
    "pid", "union_intervals",
)


def flatten(doc: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in doc.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}/"))
        elif isinstance(value, (bool, int, float, str)):
            flat[name] = value
    return flat


def segment_table(docs: list[dict]) -> wandb.Table:
    columns = ["tag", "leg", "segment", "span_us", "gpu_busy_us", "gpu_idle_us",
               "gpu_idle_pct_of_segment", "share_of_leg_idle_pct"]
    rows = []
    for doc in docs:
        for leg_name, leg in doc["legs"].items():
            for segment, block in leg["segments"].items():
                rows.append([
                    doc["tag"], leg_name, segment, block["span_us"],
                    block["gpu_busy_us"], block["gpu_idle_us"],
                    block["gpu_idle_pct_of_segment"],
                    leg["idle_share_of_leg_idle_pct"].get(segment),
                ])
            rows.append([
                doc["tag"], leg_name, "LEG_TOTAL", leg["leg_span_us"],
                leg["leg_gpu_busy_us"], leg["leg_gpu_idle_us"],
                leg["leg_gpu_idle_pct"], 100.0,
            ])
    return wandb.Table(columns=columns, data=rows)


def phase_table(docs: list[dict]) -> wandb.Table:
    columns = ["tag", "leg", "phase", "span_us", "gpu_busy_us", "gpu_idle_us",
               "gpu_idle_pct_of_phase", "idle_us_per_round"]
    rows = []
    for doc in docs:
        for leg_name, leg in doc["legs"].items():
            rounds = leg["rounds_analysed"]
            for phase, block in leg["phases"].items():
                rows.append([
                    doc["tag"], leg_name, phase, block["span_us"],
                    block["gpu_busy_us"], block["gpu_idle_us"],
                    block["gpu_idle_pct_of_segment"],
                    block["gpu_idle_us"] / rounds,
                ])
    return wandb.Table(columns=columns, data=rows)


def slice_table(docs: list[dict]) -> wandb.Table:
    """Coherent idle slices, keyed by the phase each slice starts in."""
    columns = ["tag", "leg", "start_phase", "count", "total_us",
               "us_per_round", "max_us"]
    rows = []
    for doc in docs:
        for leg_name, leg in doc["legs"].items():
            for phase, block in leg["coherent_slices"]["by_phase"].items():
                rows.append([
                    doc["tag"], leg_name, phase, block["count"],
                    block["total_us"], block["us_per_round"], block["max_us"],
                ])
    return wandb.Table(columns=columns, data=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--name", default=EXPERIMENT)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    docs = [json.loads(Path(p).read_text()) for p in args.docs]
    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "official_or_ranked_score": False,
        "timed_contrast": False,
    }
    config.update({k: docs[0][k] for k in IDENTITY_KEYS if k in docs[0]})

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, name=args.name,
        notes=args.notes, config=config, job_type="analysis")
    for doc in docs:
        prefix = doc.get("tag", "doc")
        for key, value in flatten(doc).items():
            run.summary[f"{prefix}/{key}"] = value
    run.log({"segments": segment_table(docs), "phases": phase_table(docs),
             "coherent_slices": slice_table(docs)})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
