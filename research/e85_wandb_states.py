#!/usr/bin/env python3
"""Report the terminal state of every W&B run quoted in the E85 result."""
from __future__ import annotations

import json
import pathlib

import wandb

from e85_wandb_log import ENTITY, PROJECT

SESSIONS = [
    "research/out/e85-abba-512-base-vs-ab",
    "research/out/e85-tax-512",
    "research/out/e85-abba-512-traced",
]

api = wandb.Api()
out = []
for session in SESSIONS:
    lines = pathlib.Path(session, "wandb-runs.tsv").read_text().splitlines()[1:]
    for index, line in enumerate(lines):
        run_id, url = line.split("\t")
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        out.append({"session": pathlib.Path(session).name, "run_id": run_id,
                    "url": url, "state": run.state, "name": run.name,
                    "summary": index == len(lines) - 1})

for row in out:
    flag = "SUMMARY" if row["summary"] else ""
    print(f"{row['state']:<10s} {row['run_id']:<10s} {row['name']:<44s} {flag}")

pathlib.Path("research/e85-artifacts/wandb-states.json").write_text(
    json.dumps(out, indent=2) + "\n")
print(f"\n{len(out)} runs, states: "
      f"{sorted({r['state'] for r in out})}")
