#!/usr/bin/env python3
"""E99: publish one W&B run per timed leg.

The advisor asked for a run URL per leg, including the threshold-sweep legs,
so that the campaign report can cite each measurement on its own. Every row
already carries its full identity tuple from `research/e99_rung5_price.py`, so
this script only splits those rows into runs and prints the URLs.

usage:
  research/e99_wandb_legs.py research/e99-artifacts/rung5-*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e99-oracle-allocation-bound"

IDENTITY = ("leg", "gate", "threshold", "offered_cap", "worker_sha256",
            "base_sha", "head_provenance_sha256", "entry_c", "exit_c",
            "cool_gate_passed_real_gate", "gate_qualified_for_timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", nargs="+")
    parser.add_argument("--branch-commit", default="")
    args = parser.parse_args()

    urls = []
    for path in args.bundles:
        bundle = json.loads(Path(path).read_text())
        arms = (("off", bundle["off"]), ("on", bundle["on"]),
                ("sweep", bundle["sweep"]))
        for role, rows in arms:
            for row in rows:
                config = {key: row[key] for key in IDENTITY}
                config.update(experiment=EXPERIMENT, assignment_pr=101,
                              student="qwen-edward", role=role,
                              bundle=Path(path).stem, harness="local",
                              tokens=512, local_mode="--local-iterate",
                              official_or_ranked_score=False,
                              branch_commit=args.branch_commit)
                run = wandb.init(entity=ENTITY, project=PROJECT,
                                 name=row["leg"], group=EXPERIMENT,
                                 job_type=f"timed-leg-{role}", config=config,
                                 reinit=True)
                for key, value in row.items():
                    if key not in IDENTITY:
                        run.summary[key] = value
                urls.append((row["leg"], role, row["offered_cap"], run.url))
                run.finish()

    for leg, role, cap, url in urls:
        print(f"{leg}\t{role}\tcap{cap}\t{url}")


if __name__ == "__main__":
    main()
