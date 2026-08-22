#!/usr/bin/env python3
"""Publish the E117 shipped-width-frame evidence to W&B.

    usage: research/e117_wandb_log.py research/out/TAG [--name NAME] [--rung R]

The run carries one isolated probe session: one process, one MLX stream,
palindrome ordering inside every block, a discarded fixed-duration ramp burst
after every temperature sample, and three arms plus a host-cost control arm
across the shipped M partition table.

This is a within-session relative measurement on a research instrument. It holds
no model, runs no benchmark wrapper and passes no thermal gate, so it logs
`cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
verbatim and is never a score.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e117-gate-up-na4-rate-dip-and-the-serialised-n-split"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
BASE_SHA = "1d2320bece29cddc94b95e5f99f00331b05a5025"
FRONTIER_PROMOTED = "51b9bf85"
OUR_BEST_SERIAL_FREE = "b8b8b860"

ARM_ROLE = {
    "a_one": "one dispatch, M rows, full N. The reference.",
    "c_nsplit": (
        "two concurrent dispatches over disjoint N halves, half views hoisted "
        "out of the timed body"
    ),
    "e_nsplit_serial": (
        "the same two N-split dispatches in separate blocking evals. Removes "
        "concurrency and holds everything else. The payoff arm."
    ),
    "control.small": (
        "same arm structures on a tensor with about 1.5 us of GPU work: the "
        "host cost of each structure"
    ),
}

PARTITION_NOTE = (
    "quantized.cpp:251-254 dispatches grid_dims(M, N/8, B), so ntg.x == M. "
    "qmv_fast_crossrow_affine4_g64_m at quantized.h:1157-1186 early-returns "
    "every threadgroup whose first_m = tid.x * IPG is at or past M, so the "
    "working-group count is fixed by M alone: "
    "M=1..5 one group, M=6 [3+3], M=7 [4+3], M=8 [4+4], M=9 [3+3+3]."
)


def gate_flags(instrument: str, gpu_seconds: float) -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "gpu_seconds_used": gpu_seconds,
        "instrument": instrument,
    }


def session_seconds(meta: dict[str, str]) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start, end = meta.get("started_utc"), meta.get("finished_utc")
    if not start or not end:
        return 0.0
    return (datetime.datetime.strptime(end, fmt)
            - datetime.datetime.strptime(start, fmt)).total_seconds()


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--name", default="e117-rung0-m-frame")
    parser.add_argument("--rung", default="0")
    parser.add_argument(
        "--question",
        default=(
            "does the mlp.gate_up rate dip exist in the shipped [4+4] "
            "configuration at ntg.x = 8, where two groups of four already "
            "share one dispatch and one weight stream"
        ))
    args = parser.parse_args()

    summary = json.loads((args.out_dir / "summary.json").read_text())
    meta = read_meta(args.out_dir / "meta.txt")
    cells = json.loads((args.out_dir / "cells.json").read_text())

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="isolated-probe",
        name=args.name,
        config={
            "experiment": GROUP,
            "rung": args.rung,
            "pr": 119,
            "question": args.question,
            "base_sha": BASE_SHA,
            "git_head": meta.get("git_head"),
            "git_dirty": meta.get("git_dirty"),
            "frontier_promoted": FRONTIER_PROMOTED,
            "our_best_serial_free": OUR_BEST_SERIAL_FREE,
            "host": HOST,
            "instance": meta.get("host"),
            "chip": meta.get("chip"),
            "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "swift_config": meta.get("swift_config"),
            "shapes": meta.get("shapes"),
            "widths": meta.get("widths"),
            "blocks": meta.get("blocks"),
            "estimator_pass": summary["estimator"],
            "drop_first_block": summary["drop_first_block"],
            "ramp_seconds": summary.get("ramp_seconds"),
            "defect16_fix": (
                "Temperature is sampled only at block boundaries and every "
                "sample is followed by a discarded ramp burst of fixed "
                "WALL-CLOCK duration. E115 used a fixed replicate count, which "
                "on mlp.gate_up was only about 43 ms and left part of the DVFS "
                "ramp on the first timed arm."
            ),
            "partition_table": PARTITION_NOTE,
            "probe_file": "Tests/MLXFastTests/E117WidthFrameProbeTests.swift",
            "runner": "research/e117_probe.sh",
            "analysis": "research/e117_analysis.py",
            "group_size": cells["group_size"],
            "bits": cells["bits"],
            "eval_overhead_us": cells["eval_overhead_us"],
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "arm_roles": ARM_ROLE,
            **gate_flags(
                "E117 MLX-level quantizedMM shipped-width probe, GPU",
                session_seconds(meta)),
        },
        reinit=True,
    )

    widths = wandb.Table(
        columns=[
            "shape", "outputs", "hidden", "m", "partition", "groups",
            "a_one_raw_us", "a_one_net_us", "a_one_net_gbs", "control_us",
        ])
    contrasts = wandb.Table(
        columns=[
            "shape", "outputs", "m", "partition", "arm", "arm_raw_us",
            "arm_net_us", "net_pct_faster_mean", "net_pct_faster_sem",
            "raw_pct_faster_mean", "raw_pct_faster_sem", "n_blocks", "role",
        ])
    for shape, entry in summary["shapes"].items():
        for m, w in sorted(entry["widths"].items(), key=lambda kv: int(kv[0])):
            widths.add_data(
                shape, entry["outputs"], entry["hidden"], int(m),
                w["partition"], w["groups"], w["a_one_raw_us"],
                w["a_one_net_us"], w["a_one_net_gbs"], w["control_us"])
            for arm, a in w["arms"].items():
                contrasts.add_data(
                    shape, entry["outputs"], int(m), w["partition"], arm,
                    a["arm_raw_us"], a["arm_net_us"],
                    a["net_pct_faster_mean"], a["net_pct_faster_sem"],
                    a["raw_pct_faster_mean"], a["raw_pct_faster_sem"],
                    a["n_blocks"], ARM_ROLE.get(arm, ""))
    run.log({f"rung{args.rung}/widths": widths})
    run.log({f"rung{args.rung}/contrasts": contrasts})

    blocks = wandb.Table(
        columns=[
            "shape", "outputs", "m", "arm", "block", "forward_us",
            "reverse_us", "mean_us", "dispatches", "evals_per_replicate",
            "replicates",
        ])
    for cell in cells["cells"]:
        blocks.add_data(
            cell["shape"], cell["outputs"], cell["width"], cell["arm"],
            cell["block"], cell["forward_us"], cell["reverse_us"],
            (cell["forward_us"] + cell["reverse_us"]) / 2, cell["dispatches"],
            cell["evals_per_replicate"], cell["replicates"])
    run.log({f"rung{args.rung}/blocks": blocks})

    fidelity = wandb.Table(
        columns=[
            "shape", "outputs", "m", "nsplit_bit_exact",
            "positive_control_differs", "one_digest", "nsplit_digest",
            "wrong_split_digest",
        ])
    for record in cells["exactness"]:
        fidelity.add_data(
            record["shape"], record["outputs"], record["width"],
            record["nsplit_bit_exact"], record["positive_control_differs"],
            record["one_digest"], record["nsplit_digest"],
            record["wrong_split_digest"])
    run.log({f"rung{args.rung}/fidelity": fidelity})

    residual = wandb.Table(
        columns=["arm", "mean_pct", "sem_pct", "min_pct", "max_pct", "n"])
    for arm, value in summary["defect16_residual"].items():
        residual.add_data(
            arm, value["mean_pct"], value["sem_pct"], value["min_pct"],
            value["max_pct"], value["n"])
    run.log({f"rung{args.rung}/defect16_residual": residual})

    thermal = wandb.Table(
        columns=["shape", "m", "block", "gpu_temp_entry_c", "gpu_temp_exit_c"])
    for line in (args.out_dir / "probe.log").read_text().splitlines():
        if not line.startswith("E117_BLOCK "):
            continue
        record = json.loads(line[len("E117_BLOCK "):])
        thermal.add_data(
            record["shape"], record["width"], record["block"],
            record.get("gpu_temp_entry_c"), record.get("gpu_temp_exit_c"))
    run.log({f"rung{args.rung}/thermal": thermal})

    for shape, entry in summary["shapes"].items():
        disc = entry.get("discriminator")
        if not disc:
            continue
        tag = shape.replace(".", "_")
        run.summary[f"{tag}/m8_over_two_m4"] = disc["m8_over_two_m4"]
        run.summary[f"{tag}/m8_saving_vs_two_m4_pct"] = (
            disc["m8_saving_vs_two_m4_pct"])
        m8 = entry["widths"].get("8") or entry["widths"].get(8)
        if m8 and "e_nsplit_serial" in m8["arms"]:
            run.summary[f"{tag}/e_nsplit_serial_net_pct_m8"] = (
                m8["arms"]["e_nsplit_serial"]["net_pct_faster_mean"])
            run.summary[f"{tag}/e_nsplit_serial_net_pct_m8_sem"] = (
                m8["arms"]["e_nsplit_serial"]["net_pct_faster_sem"])

    gate_up = summary["shapes"].get("mlp.gate_up")
    if gate_up:
        m8 = gate_up["widths"].get("8") or gate_up["widths"].get(8)
        if m8 and "e_nsplit_serial" in m8["arms"]:
            run.summary["e117_serial_nsplit_pct_faster_vs_one_dispatch_gate_up_m8"] = (
                m8["arms"]["e_nsplit_serial"]["net_pct_faster_mean"])

    run.summary["exactness_cells"] = summary["exactness"]["cells"]
    run.summary["exactness_not_bit_exact"] = summary["exactness"]["not_bit_exact"]
    run.summary["exactness_positive_control_failed"] = (
        summary["exactness"]["positive_control_failed"])
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
