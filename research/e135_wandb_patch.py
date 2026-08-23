#!/usr/bin/env python3
"""Correct the column-pricing summary keys on an already-published run.

    python3 research/e135_wandb_patch.py --run 8c5y1abg --label c1

The first publication of T29-A priced a threadgroup column against the
pipeline log's `by_width`, which is a warm census and not decode traffic. The
corrected values come from `e135_width_histogram.py`, which weights each width
by how often a drafting round actually verifies at it.
"""
from __future__ import annotations

import argparse
import json
import pathlib

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

KEYS = (
    "e135_launch_cost_us_per_column",
    "e135_launch_cost_us_per_column_se",
    "e135_working_column_us",
    "e135_working_column_us_se",
    "e135_verify_width_mean",
    "e135_width_6_7_round_share_pct",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", default="c1")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(
        f"research/e135-artifacts/{args.label}-per-width.json").read_text())

    import wandb

    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{args.run}")
    for key in KEYS:
        run.summary[key] = data[key]
    run.summary["e135_columns_per_round"] = data["columns_per_round"]
    run.summary["e135_verify_width_histogram"] = data["verify_width_histogram"]
    run.summary["e135_column_pricing_corrected"] = True
    run.summary.update()
    print(f"patched {run.url}")
    for key in KEYS:
        print(f"  {key} = {data[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
