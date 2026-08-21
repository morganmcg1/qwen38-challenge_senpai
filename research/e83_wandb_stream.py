#!/usr/bin/env python3
"""Stream E83 prefill-decomposition blocks to W&B while the session is timing.

Reads the test process' stdout on stdin, tees it to this process' stdout and to
a log file, and logs every `E83_BLOCK {json}` line to W&B the moment it appears.
The run is created before the first block and finished when stdin closes, so no
measurement is logged retroactively at session end.

Several block kinds arrive on the same stream and each gets its own row shape:

    begin                      one `begin()` replay, whole or phased
    boundary_census            dispatch and command-buffer counts per phase
    ladder_step                one seed width across the prefill-ladder edge
    gate_arm                   one prefill-width fusion-bound arm
    isolated_quantized_matmul  one prefill GEMM shape in isolation
    isolated_sdpa              the 512-row causal attention shape in isolation

    usage: research/e83_wandb_stream.py --tag TAG --meta META.txt [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
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
        row["control/stall_nominal_ms"] = block.get("stall_millis")
        if block.get("stall_actual_seconds") is not None:
            row["control/stall_actual_ms"] = 1e3 * block["stall_actual_seconds"]
    return row


def head_row(block: dict) -> dict:
    rows = block.get("rows")
    return {
        "head/order": block.get("order"),
        "head/rows": rows,
        "head/step_ms": 1e3 * block["seconds_median"],
        "head/total_bytes_per_ms": block.get("total_bytes_per_ms"),
        "head/gemm_bytes_per_ms": block.get("gemm_bytes_per_ms"),
        f"head/m{rows}/step_ms": 1e3 * block["seconds_median"],
    }


def gate_row(block: dict) -> dict:
    arm = block.get("gate_arm")
    return {
        "gate/order": block.get("order"),
        "gate/rep": block.get("rep"),
        "gate/begin_ms": 1e3 * block["begin_seconds"],
        "gate/fused_in_proj_max_rows": block.get("fused_in_proj_max_rows"),
        "gate/fused_gate_up_max_rows": block.get("fused_gate_up_max_rows"),
        "gate/pack_builds_after": block.get("pack_builds_after"),
        f"gate/{arm}/begin_ms": 1e3 * block["begin_seconds"],
    }


def ladder_row(block: dict) -> dict:
    width = block.get("seed_length")
    return {
        "ladder/order": block.get("order"),
        "ladder/width": width,
        "ladder/forced_eval_points": block.get("forced_eval_points"),
        "ladder/begin_ms": 1e3 * block["begin_seconds"],
        "ladder/final_eval_ms": 1e3 * block["final_eval_seconds"],
        "ladder/ms_per_token": 1e3 * block["begin_seconds"] / width,
        f"ladder/w{width}/begin_ms": 1e3 * block["begin_seconds"],
    }


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


def generic_row(block: dict) -> dict:
    """Flatten any block kind this streamer has no bespoke row shape for.

    E91 adds `e91_ceiling_cell`, `e91_ceiling_summary` and `e91_machine_peaks`
    to the same stream. A named row shape per kind buys nothing here: every
    field is already a scalar or a short list of scalars.
    """
    kind = block.get("kind", "unknown")
    family = block.get("family")
    stem = f"{kind}/{family}" if family else kind
    row = {}
    for key, value in block.items():
        if key in ("kind", "family", "seconds"):
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            row[f"{stem}/{key}"] = value
    row[f"{kind}/order"] = block.get("order")
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
            elif kind == "ladder_step":
                row = ladder_row(block)
            elif kind == "gate_arm":
                row = gate_row(block)
            elif kind == "isolated_swiglu":
                row = {
                    "swiglu/order": block.get("order"),
                    "swiglu/unfused_ms": 1e3 * block["unfused_seconds_median"],
                    "swiglu/fused_ms": 1e3 * block["fused_seconds_median"],
                    "swiglu/saving_prefill_ms":
                        1e3 * block["saving_modelled_prefill_seconds"],
                }
            elif kind == "pinned_head_step":
                row = head_row(block)
            elif kind == "stall_calibration":
                row = {
                    "control/calibration_nominal_ms": block["nominal_millis"],
                    "control/calibration_median_ms": 1e3 * block["seconds_median"],
                    "control/calibration_min_ms": 1e3 * block["seconds_min"],
                    "control/calibration_max_ms": 1e3 * block["seconds_max"],
                }
            elif kind in ("isolated_quantized_matmul", "isolated_sdpa"):
                row = isolated_row(block)
            else:
                row = generic_row(block)
            for key in ("gpu_temp_entry_c", "gpu_temp_exit_c"):
                if block.get(key) is not None:
                    row[f"block/{key}"] = block[key]
            # Step is the stream position, not the block's `order`: a session
            # may run several tests and each restarts its own ordering.
            run.log(row, step=len(blocks) - 1)
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
        gates = [b for b in blocks if b.get("kind") == "gate_arm"]
        if gates:
            run.log(
                {
                    "gate_arms": wandb.Table(
                        columns=[
                            "order", "rep", "arm", "fused_in_proj_max_rows",
                            "fused_gate_up_max_rows", "begin_ms",
                            "pack_builds_before", "pack_builds_after",
                            "first_primary", "gpu_temp_entry_c",
                            "gpu_temp_exit_c",
                        ],
                        data=[
                            [
                                b.get("order"), b.get("rep"), b.get("gate_arm"),
                                b.get("fused_in_proj_max_rows"),
                                b.get("fused_gate_up_max_rows"),
                                1e3 * b["begin_seconds"],
                                b.get("pack_builds_before"),
                                b.get("pack_builds_after"),
                                b.get("first_primary"),
                                b.get("gpu_temp_entry_c"),
                                b.get("gpu_temp_exit_c"),
                            ]
                            for b in gates
                        ],
                    )
                }
            )
            # Each rep runs every arm, so a paired within-rep delta cancels the
            # monotone thermal trend that an unpaired arm median would absorb.
            by_rep = {b["rep"]: b["begin_seconds"]
                      for b in gates if b.get("gate_arm") == "gate_baseline"}
            summary = {}
            for arm in sorted({b.get("gate_arm") for b in gates}):
                xs = [1e3 * b["begin_seconds"]
                      for b in gates if b.get("gate_arm") == arm]
                summary[f"gate/{arm}/median_begin_ms"] = statistics.median(xs)
                summary[f"gate/{arm}/n"] = len(xs)
                if arm == "gate_baseline":
                    continue
                deltas = [1e3 * (by_rep[b["rep"]] - b["begin_seconds"])
                          for b in gates
                          if b.get("gate_arm") == arm and b["rep"] in by_rep]
                if deltas:
                    summary[f"gate/{arm}/paired_median_saving_ms"] = (
                        statistics.median(deltas))
                    summary[f"gate/{arm}/paired_min_saving_ms"] = min(deltas)
                    summary[f"gate/{arm}/paired_max_saving_ms"] = max(deltas)
            run.summary.update(summary)

        ladder = [b for b in blocks if b.get("kind") == "ladder_step"]
        if ladder:
            by_width: dict[int, list[float]] = {}
            for b in ladder:
                by_width.setdefault(b["seed_length"], []).append(
                    1e3 * b["begin_seconds"])
            widths = sorted(by_width)
            med = {w: statistics.median(by_width[w]) for w in widths}
            run.log(
                {
                    "ladder_steps": wandb.Table(
                        columns=[
                            "width", "forced_eval_points", "n",
                            "median_begin_ms", "ms_per_token",
                        ],
                        data=[
                            [
                                w, 22 if w >= 512 else 0, len(by_width[w]),
                                med[w], med[w] / w,
                            ]
                            for w in widths
                        ],
                    )
                }
            )
            # Fit median begin_ms against width on the ladder-off side only,
            # then read the residual at each ladder-on width. That residual is
            # the net cost of arming 22 forced evaluation points, with the
            # arithmetic difference already removed by the fit.
            off = [w for w in widths if w < 512]
            on = [w for w in widths if w >= 512]
            if len(off) >= 2 and on:
                sx = sum(off)
                sy = sum(med[w] for w in off)
                sxx = sum(w * w for w in off)
                sxy = sum(w * med[w] for w in off)
                n = len(off)
                denom = n * sxx - sx * sx
                if denom:
                    slope = (n * sxy - sx * sy) / denom
                    intercept = (sy - slope * sx) / n
                    summary = {}
                    for w in on:
                        resid = med[w] - (slope * w + intercept)
                        summary[f"ladder/residual_ms_w{w}"] = resid
                        summary[f"ladder/residual_per_boundary_ms_w{w}"] = resid / 22
                    summary["ladder/fit_slope_ms_per_token"] = slope
                    summary["ladder/fit_intercept_ms"] = intercept
                    run.summary.update(summary)
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
        head = [b for b in blocks if b.get("kind") == "pinned_head_step"]
        if head:
            run.log(
                {
                    "pinned_head_steps": wandb.Table(
                        columns=[
                            "rows", "cache_length", "step_ms",
                            "total_bytes", "total_bytes_per_ms",
                            "gemm_bytes", "gemm_bytes_per_ms",
                        ],
                        data=[
                            [
                                b.get("rows"), b.get("cache_length"),
                                1e3 * b["seconds_median"],
                                b.get("total_bytes"), b.get("total_bytes_per_ms"),
                                b.get("gemm_bytes"), b.get("gemm_bytes_per_ms"),
                            ]
                            for b in head
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
