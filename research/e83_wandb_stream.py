#!/usr/bin/env python3
"""Stream E83 prefill-decomposition blocks to W&B while the session is timing.

Reads the test process' stdout on stdin, tees it to this process' stdout and to
a log file, and logs every `E83_BLOCK {json}` line to W&B the moment it appears.
The run is created before the first block and finished when stdin closes, so no
measurement is logged retroactively at session end.

Three block kinds arrive on the same stream and each gets its own row shape:

    begin                      one `begin()` replay, whole or phased
    boundary_census            dispatch and command-buffer counts per phase
    isolated_quantized_matmul  one prefill GEMM shape in isolation
    isolated_sdpa              the 512-row causal attention shape in isolation

    usage: research/e83_wandb_stream.py --tag TAG --meta META.txt [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import sys

import wandb

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
PREFIX = "E83_BLOCK "


def read_meta(path: str) -> dict:
    meta = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key.strip()] = value.strip()
    return meta


def begin_row(block: dict) -> dict:
    arm = block.get("arm", "unknown")
    form = "phased" if block.get("phased") else "whole"
    row = {
        "begin/order": block.get("order"),
        "begin/arm": arm,
        "begin/phased": bool(block.get("phased")),
        "begin/pin_rows": block.get("pin_rows"),
        "begin/begin_ms": 1e3 * block["begin_seconds"],
        "begin/build_ms": 1e3 * block["build_seconds"],
        "begin/final_eval_ms": 1e3 * block["final_eval_seconds"],
        "begin/readback_ms": 1e3 * block["readback_seconds"],
        f"arm/{arm}/{form}/begin_ms": 1e3 * block["begin_seconds"],
    }
    for name, seconds in (block.get("phases") or {}).items():
        row[f"phase/{name}_ms"] = 1e3 * seconds
    if block.get("phase_sum_seconds") is not None:
        row["begin/phase_sum_ms"] = 1e3 * block["phase_sum_seconds"]
        row["begin/phase_closure_ms"] = 1e3 * (
            block["begin_seconds"] - block["phase_sum_seconds"]
        )
    if block.get("stall_phase"):
        row["control/stall_phase"] = block["stall_phase"]
        row["control/stall_ms"] = block.get("stall_millis")
    return row


def isolated_row(block: dict) -> dict:
    family = block.get("family", "unknown")
    row = {
        "isolated/order": block.get("order"),
        "isolated/family": family,
        "isolated/median_ms": 1e3 * block["seconds_median"],
        "isolated/tflop_per_second": block.get("tflop_per_second"),
        "isolated/modelled_prefill_ms": 1e3 * block["modelled_prefill_seconds"],
        f"shape/{family}/median_ms": 1e3 * block["seconds_median"],
        f"shape/{family}/tflop_per_second": block.get("tflop_per_second"),
    }
    if block.get("gb_per_second") is not None:
        row["isolated/gb_per_second"] = block["gb_per_second"]
    return row


def census_row(block: dict) -> dict:
    row = {"census/order": block.get("order")}
    totals = block.get("totals") or {}
    row["census/total_dispatches"] = totals.get("dispatches")
    row["census/total_commits"] = totals.get("command_buffer_commits")
    for phase, snap in (block.get("per_phase") or {}).items():
        row[f"census/{phase}/dispatches"] = snap.get("dispatches")
        row[f"census/{phase}/commits"] = snap.get("command_buffer_commits")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--log")
    parser.add_argument("--experiment", default="e83-prefill-decomposition")
    parser.add_argument("--group", default="e83-prefill-decomposition")
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
    print(f"E83_WANDB_URL {run.url}", flush=True)
    print(f"E83_WANDB_ID {run.id}", flush=True)

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
            kind = block.get("kind")
            if kind == "begin":
                row = begin_row(block)
            elif kind == "boundary_census":
                row = census_row(block)
            elif kind in ("isolated_quantized_matmul", "isolated_sdpa"):
                row = isolated_row(block)
            else:
                row = {"unknown/order": block.get("order")}
            for key in ("gpu_temp_entry_c", "gpu_temp_exit_c"):
                if block.get(key) is not None:
                    row[f"block/{key}"] = block[key]
            run.log(row, step=block.get("order", len(blocks) - 1))
    finally:
        begins = [b for b in blocks if b.get("kind") == "begin"]
        if begins:
            run.log(
                {
                    "begin_blocks": wandb.Table(
                        columns=[
                            "order", "arm", "phased", "pin_rows", "begin_ms",
                            "build_ms", "final_eval_ms", "readback_ms",
                            "phase_sum_ms", "stall_phase",
                            "gpu_temp_entry_c", "gpu_temp_exit_c",
                        ],
                        data=[
                            [
                                b.get("order"), b.get("arm"), b.get("phased"),
                                b.get("pin_rows"), 1e3 * b["begin_seconds"],
                                1e3 * b["build_seconds"],
                                1e3 * b["final_eval_seconds"],
                                1e3 * b["readback_seconds"],
                                1e3 * b["phase_sum_seconds"]
                                if b.get("phase_sum_seconds") is not None
                                else None,
                                b.get("stall_phase"),
                                b.get("gpu_temp_entry_c"), b.get("gpu_temp_exit_c"),
                            ]
                            for b in begins
                        ],
                    )
                }
            )
        isolated = [
            b for b in blocks
            if b.get("kind") in ("isolated_quantized_matmul", "isolated_sdpa")
        ]
        if isolated:
            run.log(
                {
                    "isolated_shapes": wandb.Table(
                        columns=[
                            "family", "m", "k", "n", "layers", "median_ms",
                            "tflop_per_second", "gb_per_second",
                            "modelled_prefill_ms",
                        ],
                        data=[
                            [
                                b.get("family"), b.get("m"), b.get("k"),
                                b.get("n"), b.get("layers"),
                                1e3 * b["seconds_median"],
                                b.get("tflop_per_second"),
                                b.get("gb_per_second"),
                                1e3 * b["modelled_prefill_seconds"],
                            ]
                            for b in isolated
                        ],
                    )
                }
            )
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
