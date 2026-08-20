#!/usr/bin/env python3
"""Stream E71 census blocks to W&B while the session is still timing.

Reads the test process' stdout on stdin, tees it to this process' stdout and to
a log file, and logs every `E71_BLOCK {json}` line to W&B the moment it appears.
The run is created before the first block and finished when stdin closes, so no
measurement is ever logged retroactively at session end.

    usage: research/e71_wandb_stream.py --tag TAG --meta META.txt [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import sys

import wandb

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
PREFIX = "E71_BLOCK "


def read_meta(path: str) -> dict:
    meta = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key.strip()] = value.strip()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--log")
    parser.add_argument("--experiment", default="e71-in-situ-width-tax-census")
    parser.add_argument("--group", default="e71-width-tax-census")
    args = parser.parse_args()

    meta = read_meta(args.meta)
    entity, project = PROJECT.split("/", 1)
    run = wandb.init(
        entity=entity,
        project=project,
        name=args.tag,
        group=args.group,
        job_type="census",
        config={
            "experiment": args.experiment,
            "harness": "local",
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            **meta,
        },
    )
    print(f"E71_WANDB_URL {run.url}", flush=True)
    print(f"E71_WANDB_ID {run.id}", flush=True)

    log = open(args.log, "a") if args.log else None
    blocks = []
    try:
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log:
                log.write(line)
                log.flush()
            if not line.startswith(PREFIX):
                continue
            try:
                block = json.loads(line[len(PREFIX):])
            except json.JSONDecodeError:
                continue
            blocks.append(block)
            row = {
                "block/order": block.get("order"),
                "block/width": block.get("width"),
                "block/pin_rows": block.get("pin_rows"),
                "block/median_ms": 1e3 * block["seconds_median"],
                "block/min_ms": 1e3 * block["seconds_min"],
                "block/max_ms": 1e3 * block["seconds_max"],
                "block/mean_ms": 1e3 * block["seconds_mean"],
                "block/kl_start": block.get("kl_start"),
                "block/kl_end": block.get("kl_end"),
            }
            for key in ("gpu_temp_entry_c", "gpu_temp_exit_c"):
                if block.get(key) is not None:
                    row[f"block/{key}"] = block[key]
            arm = block.get("arm", "unknown")
            width = block.get("width")
            row[f"arm/{arm}/w{width}/median_ms"] = row["block/median_ms"]
            run.log(row, step=block.get("order", len(blocks) - 1))
    finally:
        if blocks:
            table = wandb.Table(
                columns=[
                    "order", "arm", "width", "pin_rows", "reps",
                    "median_ms", "min_ms", "max_ms", "mean_ms",
                    "kl_start", "kl_end", "gpu_temp_entry_c", "gpu_temp_exit_c",
                ],
                data=[
                    [
                        b.get("order"), b.get("arm"), b.get("width"),
                        b.get("pin_rows"), b.get("reps"),
                        1e3 * b["seconds_median"], 1e3 * b["seconds_min"],
                        1e3 * b["seconds_max"], 1e3 * b["seconds_mean"],
                        b.get("kl_start"), b.get("kl_end"),
                        b.get("gpu_temp_entry_c"), b.get("gpu_temp_exit_c"),
                    ]
                    for b in blocks
                ],
            )
            run.log({"blocks": table})
            temps = [
                b["gpu_temp_entry_c"] for b in blocks
                if b.get("gpu_temp_entry_c") is not None
            ]
            if temps:
                run.summary["entry_temp_min_c"] = min(temps)
                run.summary["entry_temp_max_c"] = max(temps)
                run.summary["entry_temp_spread_c"] = max(temps) - min(temps)
            run.summary["blocks_logged"] = len(blocks)
        run.finish()
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
