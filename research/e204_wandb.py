#!/usr/bin/env python3
"""Publish E204 round-end seam documents to W&B.

    usage: research/e204_wandb.py DOC.json [DOC.json ...] [--name NAME]
                                  [--notes TEXT]

Every document is a `harness=local` observation or contrast report produced by
`research/e204_seam.py` (stage 0) or `research/e204_arms.py` (stage 2). None of
them is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e204-round-end-seam-overlap"

IDENTITY_KEYS = (
    "tag", "stage", "harness", "official_or_ranked_score", "leg_role",
    "timed_contrast", "gate_qualified_for_timing", "cool_gate_passed_real_gate",
    "base_sha", "worker_sha256", "host", "chip", "tokens", "head_dir",
    "gpu_temp_entry_c", "gpu_temp_exit_c", "rounds_analysed", "pid",
    "all_tokens_matched", "mtp_seconds_per_token", "serial_seconds_per_token",
    "effective_mean_draft_len", "accepted_draft_rate",
)


def flatten(doc: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in doc.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}/"))
        elif isinstance(value, bool) or isinstance(value, (int, float, str)):
            flat[name] = value
    return flat


def seam_table(doc: dict) -> wandb.Table:
    rows = []
    for window in ("protocol_seam", "overlappable"):
        for key, block in doc.get(window, {}).items():
            if not block.get("rounds"):
                continue
            for phase, span in block["phase_span_us_median"].items():
                rows.append({
                    "window": window, "rounds_group": key, "phase": phase,
                    "span_us_median": span,
                    "idle_us_median": block["phase_idle_us_median"][phase],
                })
            rows.append({
                "window": window, "rounds_group": key, "phase": "WINDOW_TOTAL",
                "span_us_median": block["seam_span_us"]["median"],
                "idle_us_median": block["seam_idle_us_production"]["median"],
            })
            rows.append({
                "window": window, "rounds_group": key,
                "phase": "LARGEST_COHERENT_SLICE",
                "span_us_median": None,
                "idle_us_median": block["largest_coherent_slice_us"]["median"],
            })
    columns = ["window", "rounds_group", "phase", "span_us_median", "idle_us_median"]
    return wandb.Table(
        columns=columns, data=[[row.get(c) for c in columns] for row in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--name", default=EXPERIMENT)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    docs = [json.loads(Path(p).read_text()) for p in args.docs]
    first = docs[0]
    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "official_or_ranked_score": False,
    }
    config.update({k: first[k] for k in IDENTITY_KEYS if k in first})

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name, notes=args.notes,
        config=config, job_type="analysis")
    for index, doc in enumerate(docs):
        prefix = doc.get("tag", "doc%d" % index)
        for key, value in flatten(doc).items():
            run.summary[f"{prefix}/{key}"] = value
        if "protocol_seam" in doc:
            run.log({f"{prefix}/seam": seam_table(doc)})
        if "arms" in doc:
            run.log({f"{prefix}/arms": wandb.Table(
                columns=sorted({k for row in doc["arms"] for k in row}),
                data=[[row.get(c) for c in sorted({k for r in doc["arms"] for k in r})]
                      for row in doc["arms"]])})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
