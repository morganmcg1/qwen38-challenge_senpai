#!/usr/bin/env python3
"""Publish one E120 candidate-owned QMV dispatch session to W&B.

    usage: research/e120_wandb_log.py research/out/TAG --name NAME --rung R

The session directory may hold any of `cells.json` (rung-1 matched timing),
`fill.json` (rung-2 in-stream fill cost), `exact.json` (bit exactness with the
three positive controls), `census.json` (register, spill and text on both
architectures) and `probe.log`. Whatever is present is logged.

This is a within-session relative measurement on a research instrument. It
holds no model, runs no benchmark wrapper and passes no thermal gate, so it
records `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
verbatim and is never a score.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e120-own-the-qmv-dispatch"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
BUDGET_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"

# Advisor addendum, PR #121 comment 5377595804. A standalone fill dispatch pays
# for itself while it costs less than this.
FILL_BREAK_EVEN_US = 22.7

ARM_ROLE = {
    "a_mlx": "MLX's own quantizedMM launcher. The incumbent.",
    "b_replica": (
        "candidate-dispatched replica of affine_qmv_fast: same arithmetic, "
        "same group indexing, same geometry, our dispatch"
    ),
    "a_replica": "replica with no chunk-sum table at all. The fill-cost reference.",
    "b_fill_noconsume": (
        "replica plus a live chunk-sum table that the kernel never reads. The "
        "fill dispatch really runs in the stream, so b minus a is the fill cost "
        "and nothing else. The census confirms this arm compiles to text "
        "identical to a_replica."
    ),
    "c_sumtable": (
        "replica reading the chunk sums from the table. b minus c is the "
        "consumer gain, a minus c is the net."
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
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start, end = meta.get("started_utc"), meta.get("finished_utc")
    if not start or not end:
        return 0.0
    return (
        datetime.datetime.strptime(end, fmt) - datetime.datetime.strptime(start, fmt)
    ).total_seconds()


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def arm_samples(cells: list[dict]) -> dict[tuple, dict[str, list[float]]]:
    """Every per-arm sample, keyed by cell. Forward and reverse are separate
    samples: the palindrome cancels drift in the mean, and keeping both makes
    the dispersion visible."""
    per: dict[tuple, dict[str, list[float]]] = {}
    for cell in cells:
        key = (cell["shape"], cell["width"])
        bucket = per.setdefault(key, {})
        for arm in cell["arms"]:
            bucket.setdefault(arm["arm"], []).extend(
                [arm["forward_us"], arm["reverse_us"]]
            )
    return per


def host_cost(log: pathlib.Path) -> dict | None:
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        if line.startswith("E120_HOSTCOST "):
            return json.loads(line[len("E120_HOSTCOST ") :])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--census", type=pathlib.Path)
    args = parser.parse_args()

    meta = read_meta(args.out_dir / "meta.txt")
    cells = load(args.out_dir / "cells.json")
    fill = load(args.out_dir / "fill.json")
    exact = load(args.out_dir / "exact.json")
    census = load(args.census) if args.census else load(args.out_dir / "census.json")
    hosts = host_cost(args.out_dir / "probe.log")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="isolated-probe",
        name=args.name,
        config={
            "experiment": GROUP,
            "rung": args.rung,
            "pr": 121,
            "question": args.question,
            "base_sha": BASE_SHA,
            "budget_base_sha": BUDGET_BASE_SHA,
            "git_head": meta.get("git_head"),
            "git_dirty": meta.get("git_dirty"),
            "host": HOST,
            "instance": meta.get("host"),
            "chip": meta.get("chip"),
            "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "swift_config": meta.get("swift_config"),
            "shapes": meta.get("shapes"),
            "widths": meta.get("widths"),
            "blocks": meta.get("blocks"),
            "layers": meta.get("layers"),
            "ramp_seconds": meta.get("ramp_s"),
            "target_us": meta.get("target_us"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "fill_break_even_us": FILL_BREAK_EVEN_US,
            "arm_roles": ARM_ROLE,
            "probe_file": "Tests/MLXFastTests/E120CustomQMVProbeTests.swift",
            "runner": "research/e120_probe.sh",
            "census_tool": "research/e120_census.py",
            **gate_flags(
                "E120 candidate-owned QMV dispatch probe, GPU",
                session_seconds(meta),
            ),
        },
        reinit=True,
    )

    summary: dict[str, object] = {}

    if cells:
        table = wandb.Table(
            columns=[
                "shape", "outputs", "hidden", "m", "mlx_us", "replica_us",
                "delta_us", "delta_pct", "samples",
            ])
        worst = None
        for (shape, m), arms in sorted(arm_samples(cells["cells"]).items()):
            if "a_mlx" not in arms or "b_replica" not in arms:
                continue
            a = statistics.median(arms["a_mlx"])
            b = statistics.median(arms["b_replica"])
            pct = 100.0 * (b - a) / a
            worst = pct if worst is None else max(worst, pct)
            cell = next(c for c in cells["cells"] if c["shape"] == shape and c["width"] == m)
            table.add_data(
                shape, cell["outputs"], cell["hidden"], m, a, b, b - a, pct,
                len(arms["a_mlx"]))
        run.log({f"rung{args.rung}/matched_timing": table})
        if worst is not None:
            summary["replica_worst_deficit_pct"] = worst

    if fill:
        table = wandb.Table(
            columns=[
                "shape", "outputs", "hidden", "m", "k_blocks", "table_bytes",
                "layers", "a_replica_us", "b_fill_noconsume_us",
                "c_sumtable_us", "fill_us_per_dispatch",
                "consumer_gain_us_per_matvec", "net_us_per_matvec",
                "break_even_us", "pays_for_itself", "samples",
            ])
        for (shape, m), arms in sorted(arm_samples(fill["cells"]).items()):
            a = statistics.median(arms["a_replica"])
            b = statistics.median(arms["b_fill_noconsume"])
            c = statistics.median(arms["c_sumtable"])
            layers = fill["layers"]
            cell = next(x for x in fill["cells"] if x["shape"] == shape and x["width"] == m)
            fill_us = (b - a) / layers
            gain_us = (b - c) / layers
            net_us = (a - c) / layers
            table.add_data(
                shape, cell["outputs"], cell["hidden"], m, cell["k_blocks"],
                cell["table_bytes"], layers, a, b, c, fill_us, gain_us, net_us,
                FILL_BREAK_EVEN_US, bool(fill_us < FILL_BREAK_EVEN_US),
                len(arms["a_replica"]))
            summary[f"fill_us/{shape}/m{m}"] = fill_us
            summary[f"consumer_gain_us/{shape}/m{m}"] = gain_us
            summary[f"net_us/{shape}/m{m}"] = net_us
        run.log({f"rung{args.rung}/fill_cost": table})

    for source, name in ((cells, "matched_timing"), (fill, "fill_cost")):
        if not source:
            continue
        blocks = wandb.Table(
            columns=[
                "shape", "outputs", "m", "block", "arm", "forward_us",
                "reverse_us", "mean_us", "replicates", "gpu_temp_entry_c",
                "gpu_temp_exit_c", "role",
            ])
        for cell in source["cells"]:
            for arm in cell["arms"]:
                blocks.add_data(
                    cell["shape"], cell["outputs"], cell["width"], cell["block"],
                    arm["arm"], arm["forward_us"], arm["reverse_us"],
                    0.5 * (arm["forward_us"] + arm["reverse_us"]),
                    cell["replicates"], cell.get("gpu_temp_entry_c"),
                    cell.get("gpu_temp_exit_c"), ARM_ROLE.get(arm["arm"], ""))
        run.log({f"rung{args.rung}/{name}_blocks": blocks})

    if exact:
        table = wandb.Table(
            columns=[
                "shape", "outputs", "hidden", "m", "arm", "elements",
                "differing_elements", "max_abs_diff", "bit_exact", "x_hit",
                "meta_hit", "table_hit", "restored_diff",
                "positive_control_can_fail",
            ])
        exact_all = True
        for record in exact["records"]:
            exact_all = exact_all and record["differing_elements"] == 0
            table.add_data(
                record["shape"], record["outputs"], record["hidden"],
                record["width"], record["arm"], record["elements"],
                record["differing_elements"], record["max_abs_diff"],
                record["bit_exact"], record["x_hit"], record["meta_hit"],
                record.get("table_hit"), record.get("restored_diff"),
                record["positive_control_can_fail"])
        run.log({f"rung{args.rung}/exactness": table})
        summary["bit_exact_all_cells"] = exact_all
        summary["exactness_cells"] = len(exact["records"])

    if census:
        archs = sorted(census)
        table = wandb.Table(columns=["kernel", "arch", "registers", "spill_bytes", "text_bytes"])
        for arch in archs:
            for kernel, record in sorted(census[arch].items()):
                table.add_data(
                    kernel, arch, record["registers"], record["spill_bytes"],
                    record["text_bytes"])
        run.log({f"rung{args.rung}/census": table})
        summary["max_spill_bytes"] = max(
            record["spill_bytes"] for arch in archs for record in census[arch].values())

    if hosts:
        run.log({f"rung{args.rung}/host_cost": wandb.Table(
            columns=list(hosts), data=[[hosts[k] for k in hosts]])})
        summary["custom_dispatch_host_us"] = hosts["custom_minus_mlx_us"]

    run.summary.update(summary)
    print(f"run_id={run.id}")
    print(f"url={run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
