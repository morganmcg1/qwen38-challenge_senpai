#!/usr/bin/env python3
"""Log the E14 IPG weight-pass arm comparison to W&B as one analysis run.

    research/ipg_wandb_log.py --breakdown TAG=PATH [...] --run-name NAME

Each breakdown is the JSON written by `research/ipg_shape_breakdown.py`. The
per-arm cost curves are logged separately by `qmv_cost_curve_summary.py
--wandb`; this run carries the comparison that the curves cannot express on
their own: drift-adjusted arm ratios against a same-session reference, the
derived h vector per arm, and the per-shape excess fingerprint that separates a
weight-traffic effect from a register-pressure effect.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

WIDTHS = [str(m) for m in range(1, 10)]


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def arm_state(tag: str) -> dict:
    p = pathlib.Path(f".mlxfast-private/ipg-arms/{tag}/arm-state.json")
    return json.loads(p.read_text()) if p.exists() else {}


def start_temp(tag: str) -> float | None:
    p = pathlib.Path(f".mlxfast-private/qmv-curve/{tag}/start-temps.txt")
    if not p.exists():
        return None
    for line in p.read_text().split("\n"):
        parts = line.split()
        if len(parts) == 2:
            return float(parts[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", action="append", required=True,
                    metavar="TAG=PATH")
    ap.add_argument("--run-name", default="e14-ipg-weight-passes")
    ap.add_argument("--group", default="qwen38-r1-e14-ipg-weight-passes")
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--host", default="unknown")
    args = ap.parse_args()

    import wandb

    arms = {}
    for spec in args.breakdown:
        tag, path = spec.split("=", 1)
        arms[tag] = json.loads(pathlib.Path(path).read_text())

    ref_tag = next(iter(arms.values()))["ref_tag"]
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=args.run_name,
        job_type="analysis",
        group=args.group,
        config={
            "experiment": "qwen38-r1-e14-ipg-weight-passes",
            "base_sha": args.base_sha,
            "candidate_sha": git_sha(),
            "host": args.host,
            "reference_tag": ref_tag,
            "estimator": next(iter(arms.values()))["estimator"],
            "arms": sorted(arms),
            "reference_start_temp_c": start_temp(ref_tag),
            "reference_quantized_h_sha256": arm_state(ref_tag).get(
                "quantized_h_sha256"
            ),
            "reference_quantized_cpp_sha256": arm_state(ref_tag).get(
                "quantized_cpp_sha256"
            ),
        },
    )

    arm_rows = wandb.Table(
        columns=[
            "arm", "arm_source", "changed_width", "ratio", "ratio_drift_adjusted",
            "excess_h_units", "median_control_drift", "noise_floor_pct",
            "control_ratio_min", "control_ratio_max", "start_temp_c",
            "quantized_h_sha256",
        ]
    )
    shape_rows = wandb.Table(
        columns=[
            "arm", "shape", "sweep_order", "n", "k", "weight_mib",
            "calls_per_verify", "excess_pct", "weighted_excess_seconds",
            "share_of_excess_pct", "arm_intra_run_spread",
        ]
    )
    h_rows = wandb.Table(columns=["arm", "d", "h", "passes_ref", "na_ref"])
    wvs_rows = wandb.Table(columns=["arm", "m", "weighted_verify_seconds"])

    # Shipped dispatch table, verified at quantized.h:1809.
    passes_ref = {3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}
    na_ref = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

    summary = {}
    for tag, b in sorted(arms.items()):
        st = arm_state(tag)
        arm_rows.add_data(
            tag, st.get("arm", "?"), b["changed_width"], b["changed_width_ratio"],
            b["changed_width_ratio_drift_adjusted"],
            b["changed_width_excess_in_h_units"], b["median_control_drift"],
            b["noise_floor_halfwidth_pct"], b["control_ratio_min"],
            b["control_ratio_max"], start_temp(tag), st.get("quantized_h_sha256"),
        )
        total_excess = sum(s["weighted_excess_seconds"] for s in b["shapes"]) or 1.0
        for s in b["shapes"]:
            shape_rows.add_data(
                tag, s["name"], s["sweep_order"], s["n"], s["k"], s["weight_mib"],
                s["calls_per_verify"], 100.0 * s["excess_fraction"],
                s["weighted_excess_seconds"],
                100.0 * s["weighted_excess_seconds"] / total_excess,
                s["arm_intra_run_spread"],
            )
        for i, h in enumerate(b["h_arm"], start=1):
            h_rows.add_data(tag, i, h, passes_ref.get(i + 1), na_ref.get(i + 1))
        for m in WIDTHS:
            wvs_rows.add_data(tag, int(m), b["weighted_verify_seconds_arm"][m])
        short = tag.split("-")[-1]
        summary[f"arm/{short}/ratio_drift_adjusted"] = b[
            "changed_width_ratio_drift_adjusted"
        ]
        summary[f"arm/{short}/excess_h_units"] = b["changed_width_excess_in_h_units"]
        summary[f"arm/{short}/noise_floor_pct"] = b["noise_floor_halfwidth_pct"]

    ref = next(iter(arms.values()))
    for i, h in enumerate(ref["h_ref"], start=1):
        h_rows.add_data(ref_tag, i, h, passes_ref.get(i + 1), na_ref.get(i + 1))
    for m in WIDTHS:
        wvs_rows.add_data(ref_tag, int(m), ref["weighted_verify_seconds_ref"][m])

    h_ref = ref["h_ref"]
    flat = [h_ref[i] for i in (4, 5, 6)] + [h_ref[2]]
    spike = (h_ref[3] + h_ref[7]) / 2.0
    summary.update(
        {
            "ref/h_vector": h_ref,
            "ref/h4_pass_spike": h_ref[3],
            "ref/h8_pass_spike": h_ref[7],
            "ref/flat_step_mean": sum(flat) / len(flat),
            "ref/structural_pass_cost_h_units": spike - sum(flat) / len(flat),
            "ref/row_tax_over_stream_tax": (sum(flat) / len(flat))
            / max(spike - sum(flat) / len(flat), 1e-9),
        }
    )
    run.summary.update(summary)
    run.log(
        {
            "e14/arms": arm_rows,
            "e14/per_shape_excess": shape_rows,
            "e14/h_by_depth": h_rows,
            "e14/weighted_verify_seconds": wvs_rows,
        }
    )
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
