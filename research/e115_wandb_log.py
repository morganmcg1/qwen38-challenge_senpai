#!/usr/bin/env python3
"""Publish the E115 concurrency-discriminator evidence to W&B.

    usage: research/e115_wandb_log.py research/out/TAG

The run carries the isolated rung 1 probe: one process, one MLX stream,
palindrome ordering, first block discarded, six arms plus a host-cost control
arm at four widths on four scored shapes.

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
GROUP = "e115-concurrency-discriminator-and-the-n-split"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
BASE_SHA = "91b51ec3c5c3eb86b917de1efb3de7219dc3eecb"
FRONTIER_PROMOTED = "51b9bf85"
OUR_BEST_PROMOTED = "f04b102e"

ARM_ROLE = {
    "a_one": "one dispatch, NA rows, full N. The reference.",
    "b_msplit": (
        "two concurrent dispatches over disjoint M rows, full N each, so both "
        "read the SAME weights and the pass count doubles. The shipped G=2 "
        "shape."
    ),
    "c_nsplit": (
        "two concurrent dispatches over disjoint N halves, slices built inside "
        "the timed body. The payoff arm."
    ),
    "c_nsplit_pre": "c_nsplit with the half views hoisted out of the timed body",
    "d_indep": (
        "two concurrent dispatches over two SEPARATE weight buffers of N/2 "
        "rows. Removes weight sharing and holds everything else."
    ),
    "e_nsplit_serial": (
        "the two N-split dispatches in separate blocking evals. Removes "
        "concurrency and holds everything else."
    ),
    "f_nsplit4": "four concurrent dispatches over disjoint N quarters",
    "control.small": (
        "same arm structures on a tensor with about 1.5 us of GPU work: the "
        "host cost of each structure"
    ),
}


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
    """Wall-clock seconds of the probe session, build included."""
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
    parser.add_argument("--name", default="e115-rung1-concurrency-discriminator")
    args = parser.parse_args()

    summary_path = args.out_dir / "summary.json"
    payload = json.loads(summary_path.read_text())
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
            "rung": "1",
            "pr": 117,
            "question": (
                "does a second concurrent QMV dispatch buy request-level "
                "overlap (H1), shared-weight caching (H2) or slicing (H3)"
            ),
            "base_sha": BASE_SHA,
            "git_head": meta.get("git_head"),
            "git_dirty": meta.get("git_dirty"),
            "frontier_promoted": FRONTIER_PROMOTED,
            "our_best_promoted": OUR_BEST_PROMOTED,
            "host": HOST,
            "instance": meta.get("host"),
            "chip": meta.get("chip"),
            "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "swift_config": meta.get("swift_config"),
            "shapes": meta.get("shapes"),
            "widths": meta.get("widths"),
            "blocks": meta.get("blocks"),
            "blocks_dropped": payload["blocks_dropped"],
            "probe_file": (
                "Tests/MLXFastTests/E115ConcurrentDispatchProbeTests.swift"
            ),
            "runner": "research/e115_probe.sh",
            "analysis": "research/e115_analysis.py",
            "group_size": cells["group_size"],
            "bits": cells["bits"],
            "eval_overhead_us": payload["eval_overhead_us"],
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "arm_roles": ARM_ROLE,
            **gate_flags(
                "E115 MLX-level quantizedMM concurrency probe, GPU",
                session_seconds(meta),
            ),
        },
        reinit=True,
    )

    timing = wandb.Table(
        columns=[
            "shape", "na", "arm", "raw_us", "net_us",
            "raw_pct_faster_vs_a_one", "net_pct_faster_vs_a_one",
            "net_pct_faster_min", "net_pct_faster_max", "blocks", "role",
        ])
    for entry in payload["cells"].values():
        timing.add_data(
            entry["shape"], entry["width"], entry["arm"], entry["raw_us"],
            entry["net_us"], entry["raw_pct_faster_vs_a_one"],
            entry["net_pct_faster_vs_a_one"], entry["net_pct_faster_min"],
            entry["net_pct_faster_max"], entry["blocks"],
            ARM_ROLE.get(entry["arm"], ""))
    run.log({"rung1/timing": timing})

    blocks = wandb.Table(
        columns=[
            "shape", "na", "arm", "block", "forward_us", "reverse_us",
            "mean_us", "dispatches", "weight_passes", "evals_per_replicate",
            "replicates",
        ])
    for cell in cells["cells"]:
        blocks.add_data(
            cell["shape"], cell["width"], cell["arm"], cell["block"],
            cell["forward_us"], cell["reverse_us"],
            (cell["forward_us"] + cell["reverse_us"]) / 2, cell["dispatches"],
            cell["weight_passes"], cell["evals_per_replicate"],
            cell["replicates"])
    run.log({"rung1/blocks": blocks})

    fidelity = wandb.Table(
        columns=[
            "shape", "na", "nsplit_bit_exact", "positive_control_differs",
            "nsplit4_bit_exact", "one_digest", "nsplit_digest",
            "wrong_split_digest",
        ])
    for record in cells["exactness"]:
        fidelity.add_data(
            record["shape"], record["width"], record["nsplit_bit_exact"],
            record["positive_control_differs"],
            record.get("nsplit4_bit_exact"), record["one_digest"],
            record["nsplit_digest"], record["wrong_split_digest"])
    run.log({"rung1/fidelity": fidelity})

    aliasing = wandb.Table(
        columns=["shape", "delta_bytes", "full_tensor_bytes", "aliases"])
    for record in cells["slice_aliasing"]:
        aliasing.add_data(
            record["shape"], record["delta_bytes"],
            record["full_tensor_bytes"],
            abs(record["delta_bytes"]) < record["full_tensor_bytes"] // 8)
    run.log({"rung1/slice_aliasing": aliasing})

    flat: dict[str, object] = {}
    for key, entry in payload["cells"].items():
        prefix = f"cell/{entry['shape']}/na{entry['width']}/{entry['arm']}"
        flat[f"{prefix}/net_us"] = entry["net_us"]
        flat[f"{prefix}/net_pct_faster"] = entry["net_pct_faster_vs_a_one"]
    for key, value in payload["round_weighted_net_pct"].items():
        flat[f"round_weighted/{key.replace('|', '/')}"] = value

    gate_up = payload["kill_rule_inputs"].get("mlp.gate_up|c_nsplit_pre")
    lm_head = payload["kill_rule_inputs"].get("lm_head|c_nsplit_pre")
    flat.update({
        "e115_nsplit_isolated_pct_faster_vs_one_dispatch_na4_gate_up": gate_up,
        "e115_nsplit_isolated_pct_faster_vs_one_dispatch_na4_lm_head": lm_head,
        "kill_rule_pct": payload["kill_rule_pct"],
        "kill_rule_passed": payload["kill_rule_passed"],
        "all_nsplit_bit_exact": all(
            r["nsplit_bit_exact"] for r in cells["exactness"]),
        "all_positive_controls_differ": all(
            r["positive_control_differs"] for r in cells["exactness"]),
    })
    run.summary.update(flat)
    run.finish()
    print(f"logged {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
