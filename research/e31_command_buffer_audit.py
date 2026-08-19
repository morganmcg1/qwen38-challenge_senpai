#!/usr/bin/env python3
"""qwen38-r1-e31: source audit of MLX command-buffer geometry, and the bound
E29's own ladder sweep already places on the commit-boundary axis.

No GPU work. Every number is derived from vendored MLX source, the shipped
Swift startup policy, the on-disk checkpoint headers, and E29's in-tree
analysis file.

usage:
  research/e31_command_buffer_audit.py [--weights weights] [--wandb]

Two independent derivations:

  geometry  - the effective (max_ops, max_mb) pair actually in force in the
              scored worker on each host profile, and the resulting automatic
              command-buffer commit count for one 969-dispatch verify forward.
              `buffer_sizes_` accumulates array::data_size(), documented in
              units of item_size and NOT bytes, so the "MB" cap is a
              mebi-ELEMENT cap.
  bound     - a least-squares fit of E29's four-arm ladder sweep (0/2/8/17
              forced commit boundaries) giving the per-boundary cost with a
              95% interval, and the implied ceiling for removing every
              automatic commit.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

# --- constants read out of the vendored source (file:line in the report) ----

# Vendor/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp:25
MAX_ACTIVE_TASKS = 10

# Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp:573-595,
# keyed by the last character of the GPU architecture name.
ARCH_DEFAULTS = {
    "p": (20, 40),  # phone
    "g": (40, 40),  # base, pro
    "s": (50, 50),  # max
    "d": (50, 50),  # ultra
    "": (40, 40),  # default
}

# Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift:66-73 (>=96 GiB
# no-overwrite install) and :112-113 / :145-146 (profile scalars). The MTP
# worker (Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:133,480-490)
# force-applies the profile scalars ONLY on the low-memory branch.
PROFILES = {
    "ranked_m5_max_128gb": {
        "physical_gib": 128,
        "low_memory": False,
        "max_ops": 50,
        "max_mb": 512,
        "source": "installQwenMTPFullProfileCommandBufferDefaults (setenv, overwrite=0)",
    },
    "local_m4_pro_48gib": {
        "physical_gib": 48,
        "low_memory": True,
        "max_ops": 64,
        "max_mb": 128,
        "source": "worker low-memory branch (setenv, overwrite=1)",
    },
}

# Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:2118-2132, fired at
# :2218-2226. Each rung is one asyncEval => one forced commit boundary.
LADDER_RUNGS = {"off": 0, "front": 2, "default": 8, "dense": 17}

# E29 (research/results/e29-round-overhead-host-graph.md): the verify-forward
# dispatch inventory is constant at 969 for M=6..9.
VERIFY_DISPATCHES = 969
LAYERS = 64

# Two-sided 95% t quantile at 2 degrees of freedom (4 arms, 2 fit parameters).
T_95_DF2 = 4.302652729911275


def element_inventory(weights_dir: Path) -> dict:
    """Elements (array::data_size units) referenced by one full forward."""
    total = 0
    layer_total = 0
    tensors = 0
    layer_ids = set()
    for path in sorted(weights_dir.glob("*.safetensors")):
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in spec["shape"]:
                count *= dim
            total += count
            tensors += 1
            parts = name.split(".")
            if "layers" in parts:
                layer_ids.add(int(parts[parts.index("layers") + 1]))
                layer_total += count
    return {
        "tensors": tensors,
        "total_elements": total,
        "layer_elements": layer_total,
        "non_layer_elements": total - layer_total,
        "layers_seen": len(layer_ids),
        "elements_per_layer": layer_total / max(len(layer_ids), 1),
    }


def commits_per_forward(max_ops: int, max_mb: int, inventory: dict) -> dict:
    """Automatic commits for one verify forward.

    `CommandEncoder::needs_commit()` (device.cpp:484-487) fires when either
    counter passes its cap; a commit resets both (device.cpp:528-529). Both
    axes are reported, plus the joint count from walking the layer stack.
    """
    elements = inventory["total_elements"]
    threshold_elements = max_mb * (1 << 20)
    ops_axis = VERIFY_DISPATCHES / (max_ops + 1)
    element_axis = elements / threshold_elements

    dispatches_per_layer = VERIFY_DISPATCHES / LAYERS
    elements_per_layer = inventory["elements_per_layer"]
    ops_fraction = dispatches_per_layer / (max_ops + 1)
    element_fraction = elements_per_layer / threshold_elements
    joint_layer_stack = LAYERS * max(ops_fraction, element_fraction)
    non_layer = inventory["non_layer_elements"] / threshold_elements
    return {
        "max_ops": max_ops,
        "max_mb": max_mb,
        "threshold_elements": threshold_elements,
        "ops_axis_commits": ops_axis,
        "element_axis_commits": element_axis,
        "binding_axis": "ops" if ops_fraction >= element_fraction else "elements",
        "automatic_commits": joint_layer_stack + non_layer,
    }


def ladder_fit(analysis_path: Path, decode_tokens: int = 256) -> dict:
    data = json.loads(analysis_path.read_text())
    arms = {"L0": "off", "L1": "front", "D0": "default", "L2": "dense"}
    table = []
    for arm, ladder in arms.items():
        entry = data[arm]
        serial = entry["parent_block_seconds"]["03-mtp-timed.json"]
        mtp = entry["parent_block_seconds"]["04-mtp-timed.json"]
        table.append(
            {
                "arm": arm,
                "ladder": ladder,
                "rungs": LADDER_RUNGS[ladder],
                "serial_ms_per_token": 1000.0 * serial["sum_s"] / decode_tokens,
                "mtp_ms_per_token": 1000.0 * mtp["sum_s"] / decode_tokens,
                "mtp_ms_per_round": mtp["mean_ms"],
                "local_ratio": serial["sum_s"] / mtp["sum_s"],
                "accepted_draft_total": entry["accepted_draft_total"],
            }
        )
    xs = [row["rungs"] for row in table]
    ys = [row["mtp_ms_per_round"] for row in table]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sigma2 = sum(r * r for r in residuals) / (n - 2)
    slope_se = (sigma2 / sxx) ** 0.5
    return {
        "table": table,
        "round_mean_ms": my,
        "slope_us_per_boundary": 1000.0 * slope,
        "slope_se_us": 1000.0 * slope_se,
        "slope_ci95_us": (
            1000.0 * (slope - T_95_DF2 * slope_se),
            1000.0 * (slope + T_95_DF2 * slope_se),
        ),
        "mtp_spread_pct": 100.0 * (max(ys) - min(ys)) / min(ys),
        "serial_spread_pct": 100.0
        * (
            max(r["serial_ms_per_token"] for r in table)
            - min(r["serial_ms_per_token"] for r in table)
        )
        / min(r["serial_ms_per_token"] for r in table),
    }


def removal_ceiling(fit: dict, commits: float) -> dict:
    """Effect of deleting `commits` boundaries, extrapolating E29's fit.

    Round-time change is signed so that positive means SLOWER. The fitted
    slope is negative (more boundaries measured marginally faster), so the
    central extrapolation of removing boundaries is a small slowdown, and the
    best case for the mechanism sits at the upper edge of the slope interval.
    """
    scale = 100.0 / fit["round_mean_ms"]
    central_ms = -fit["slope_us_per_boundary"] * commits / 1000.0
    worst_ms = -fit["slope_ci95_us"][0] * commits / 1000.0
    best_ms = -fit["slope_ci95_us"][1] * commits / 1000.0
    return {
        "commits_removed": commits,
        "round_time_change_ms": central_ms,
        "round_time_change_pct": central_ms * scale,
        "round_time_change_ci95_pct": (best_ms * scale, worst_ms * scale),
        "best_case_speedup_pct": -best_ms * scale,
    }


def build_report(weights_dir: Path, analysis_path: Path) -> dict:
    inventory = element_inventory(weights_dir)
    geometry = {
        name: commits_per_forward(spec["max_ops"], spec["max_mb"], inventory)
        for name, spec in PROFILES.items()
    }
    geometry["e29_claimed_50_50"] = commits_per_forward(50, 50, inventory)
    fit = ladder_fit(analysis_path)
    ranked = geometry["ranked_m5_max_128gb"]["automatic_commits"]
    local = geometry["local_m4_pro_48gib"]["automatic_commits"]
    return {
        "inventory": inventory,
        "geometry": geometry,
        "ladder_fit": fit,
        "ceilings": {
            "ranked": removal_ceiling(fit, ranked),
            "local": removal_ceiling(fit, local),
        },
        "max_active_tasks": MAX_ACTIVE_TASKS,
        "verify_dispatches": VERIFY_DISPATCHES,
    }


def print_report(report: dict) -> None:
    inv = report["inventory"]
    print("== checkpoint element inventory (array::data_size units) ==")
    print(f"  tensors                {inv['tensors']}")
    print(f"  total elements         {inv['total_elements']:.6e}")
    print(f"  layer elements         {inv['layer_elements']:.6e} "
          f"({inv['layers_seen']} layers)")
    print(f"  elements per layer     {inv['elements_per_layer']:.6e}")
    print()
    print("== automatic commits per 969-dispatch verify forward ==")
    print(f"  {'profile':22} {'ops':>4} {'mb':>5} {'by ops':>8} {'by elem':>8} "
          f"{'binding':>8} {'commits':>8}")
    for name, geo in report["geometry"].items():
        print(
            f"  {name:22} {geo['max_ops']:4d} {geo['max_mb']:5d} "
            f"{geo['ops_axis_commits']:8.1f} {geo['element_axis_commits']:8.1f} "
            f"{geo['binding_axis']:>8} {geo['automatic_commits']:8.1f}"
        )
    print()
    fit = report["ladder_fit"]
    print("== E29 ladder sweep, both legs (local M4 Pro, 256 decode tokens) ==")
    print(f"  {'arm':4} {'ladder':8} {'rungs':>5} {'serial ms/tok':>13} "
          f"{'mtp ms/tok':>11} {'mtp ms/round':>12} {'local ratio':>11} {'acc':>5}")
    for row in fit["table"]:
        print(
            f"  {row['arm']:4} {row['ladder']:8} {row['rungs']:5d} "
            f"{row['serial_ms_per_token']:13.3f} {row['mtp_ms_per_token']:11.3f} "
            f"{row['mtp_ms_per_round']:12.3f} {row['local_ratio']:11.4f} "
            f"{row['accepted_draft_total']:5d}"
        )
    print(f"  MTP-leg spread {fit['mtp_spread_pct']:.3f}% | "
          f"serial-leg spread {fit['serial_spread_pct']:.2f}%")
    print(
        f"  per-boundary cost {fit['slope_us_per_boundary']:.1f} us "
        f"(SE {fit['slope_se_us']:.1f}, 95% CI "
        f"[{fit['slope_ci95_us'][0]:.1f}, {fit['slope_ci95_us'][1]:.1f}])"
    )
    print()
    print("== deleting every automatic commit (positive = SLOWER round) ==")
    for name, ceiling in report["ceilings"].items():
        print(
            f"  {name:8} remove {ceiling['commits_removed']:.1f} boundaries -> "
            f"round time {ceiling['round_time_change_pct']:+.3f}% central, "
            f"95% CI [{ceiling['round_time_change_ci95_pct'][0]:+.3f}%, "
            f"{ceiling['round_time_change_ci95_pct'][1]:+.3f}%]; best-case "
            f"speedup {ceiling['best_case_speedup_pct']:.3f}%"
        )


def publish(report: dict, run_name: str) -> None:
    import wandb

    fit = report["ladder_fit"]
    ranked = report["geometry"]["ranked_m5_max_128gb"]
    local = report["geometry"]["local_m4_pro_48gib"]
    claimed = report["geometry"]["e29_claimed_50_50"]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        job_type="source-audit",
        tags=["qwen38-r1-e31", "command-buffer-geometry", "source-audit", "no-gpu"],
        config={
            "assignment_id": "qwen38-r1-e31-mlx-command-buffer-geometry",
            "max_active_tasks": report["max_active_tasks"],
            "max_active_tasks_settable": False,
            "max_active_tasks_site": "Vendor/mlx-swift/.../mlx/transforms.cpp:25",
            "max_ops_env": "MLX_MAX_OPS_PER_BUFFER",
            "max_mb_env": "MLX_MAX_MB_PER_BUFFER",
            "env_read_site": "Vendor/mlx-swift/.../mlx/utils.h:178-188",
            "needs_commit_site": "Vendor/mlx-swift/.../backend/metal/device.cpp:484-487",
            "buffer_sizes_unit": "elements (array.h:346), not bytes",
            "ladder_env": "MLX_QWEN_MTP_LADDER",
            "ladder_site": "Vendor/mlx-swift-lm/.../Qwen35.swift:2118-2132",
            "verify_dispatches": report["verify_dispatches"],
            "checkpoint_tensors": report["inventory"]["tensors"],
            "checkpoint_elements": report["inventory"]["total_elements"],
        },
    )
    summary = {
        "e31/verify_forward_command_buffer_commits": ranked["automatic_commits"],
        "e31/commits_ranked_automatic": ranked["automatic_commits"],
        "e31/commits_ranked_with_ladder": ranked["automatic_commits"]
        + LADDER_RUNGS["default"],
        "e31/commits_local_automatic": local["automatic_commits"],
        "e31/commits_local_with_ladder": local["automatic_commits"]
        + LADDER_RUNGS["default"],
        "e31/commits_if_e29_50_50_were_true": claimed["automatic_commits"],
        "e31/ranked_max_ops": ranked["max_ops"],
        "e31/ranked_max_mb": ranked["max_mb"],
        "e31/local_max_ops": local["max_ops"],
        "e31/local_max_mb": local["max_mb"],
        "e31/binding_axis_ranked": ranked["binding_axis"],
        "e31/binding_axis_local": local["binding_axis"],
        "e31/per_boundary_cost_us": fit["slope_us_per_boundary"],
        "e31/per_boundary_cost_se_us": fit["slope_se_us"],
        "e31/per_boundary_ci95_low_us": fit["slope_ci95_us"][0],
        "e31/per_boundary_ci95_high_us": fit["slope_ci95_us"][1],
        "e31/mtp_leg_spread_pct": fit["mtp_spread_pct"],
        "e31/serial_leg_spread_pct": fit["serial_spread_pct"],
        "e31/removal_best_case_speedup_ranked_pct": report["ceilings"]["ranked"][
            "best_case_speedup_pct"
        ],
        "e31/removal_round_time_change_ranked_pct": report["ceilings"]["ranked"][
            "round_time_change_pct"
        ],
        "e31/timed_arms_run": 0,
    }
    run.summary.update(summary)
    table = wandb.Table(
        columns=[
            "arm",
            "ladder",
            "rungs",
            "serial_ms_per_token",
            "mtp_ms_per_token",
            "mtp_ms_per_round",
            "local_ratio",
            "accepted_draft_total",
        ]
    )
    for row in fit["table"]:
        table.add_data(*[row[column] for column in table.columns])
    run.log({"e31/ladder_sweep_two_leg": table})
    print(f"wandb run: {run.url}")
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights")
    parser.add_argument(
        "--analysis", default="research/results/e29-analysis.json"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", default="e31-command-buffer-geometry-audit")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    report = build_report(Path(args.weights), Path(args.analysis))
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    if args.wandb:
        publish(report, args.name)


if __name__ == "__main__":
    main()
