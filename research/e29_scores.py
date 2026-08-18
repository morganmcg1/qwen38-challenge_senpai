#!/usr/bin/env python3
"""Print the score/correctness fields of each E29 arm's score.json."""
import json
import sys
from pathlib import Path

WANTED = (
    "serial_seconds_per_token",
    "mtp_seconds_per_token",
    "speedup",
    "all_tokens_matched",
    "effective_mean_draft_len",
    "accepted_draft_rate",
    "reference_checked_rows",
    "residual_divergence_count",
)


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix, obj


for arm in sys.argv[1:]:
    path = Path(arm)
    if path.is_dir():
        path = path / "score.json"
    print(f"--- {path.parent.name}")
    if not path.exists():
        print("  (missing)")
        continue
    flat = dict(walk(json.load(path.open())))
    for key, value in flat.items():
        leaf = key.split(".")[-1]
        if leaf in WANTED:
            print(f"  {key}: {value}")
