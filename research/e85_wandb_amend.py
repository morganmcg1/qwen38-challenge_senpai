#!/usr/bin/env python3
"""Add analysis to an existing E85 summary run without creating a duplicate.

    usage: research/e85_wandb_amend.py RUN_ID PREFIX=FILE.json [PREFIX=FILE.json ...]

Re-running the leg publisher would create a second copy of every run. When a
correction lands after publication, resume the summary run and flatten the new
report into it instead.
"""
from __future__ import annotations

import json
import pathlib
import sys

import wandb

from e85_wandb_log import ENTITY, PROJECT, flatten


def main() -> None:
    run_id = sys.argv[1]
    payload: dict = {}
    for spec in sys.argv[2:]:
        prefix, _, path = spec.partition("=")
        flatten(prefix, json.loads(pathlib.Path(path).read_text()), payload)

    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="must")
    run.summary.update(payload)
    print(f"amended {run.url} with {len(payload)} keys")
    run.finish()


if __name__ == "__main__":
    main()
