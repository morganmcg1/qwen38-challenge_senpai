#!/usr/bin/env python3
"""Publish one E56 ABBA session as a single W&B summary run.

The four leg runs carry their own traces. This run carries the comparison the
decision is actually made on: the arm contrasts against the null-arm floor, the
correctness and provenance fields that must be identical across arms, the
verify-width histogram of each arm, and the cost model the sched arm ran, read
from the live schedule through e56_walk_probe.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import wandb

import e56_walk_probe as probe

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e56-stream-aware-draft-depth-schedule"


def cost_model() -> dict:
    ipg = probe.inputs_per_group()
    marginal, cumulative = probe.cost_table(ipg)
    out = {
        "cost_verify_stream_ratio": probe.stream_cost_ratio(),
        "cost_head_step_ratio": probe.HEAD_STEP_COST_RATIO,
    }
    for depth in range(len(marginal)):
        out[f"cost_marginal_d{depth}"] = marginal[depth]
        out[f"cost_reach_floor_d{depth}"] = (
            marginal[depth] * (depth + 1) / cumulative[depth])
    out["cost_closed_steps"] = str(
        [d for d in range(len(marginal))
         if out[f"cost_reach_floor_d{d}"] >= 1.0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="s2")
    parser.add_argument("--abba", default="research/e56-abba.json")
    args = parser.parse_args()

    data = json.loads((ROOT / args.abba).read_text(encoding="utf-8"))

    metrics: dict[str, float] = {}
    for name, contrast in data["contrasts"].items():
        metrics[f"{name}__base"] = contrast["base_mean"]
        metrics[f"{name}__sched"] = contrast["sched_mean"]
        metrics[f"{name}__delta_pct"] = contrast["delta_pct"]
        metrics[f"{name}__null_spread_pct"] = contrast["base_null_spread_pct"]
        metrics[f"{name}__sched_spread_pct"] = contrast["sched_spread_pct"]

    summary: dict[str, object] = dict(metrics)
    summary["null_floor_pct"] = data["null_floor_pct"]
    for field, per_leg in data["correctness"].items():
        values = set(per_leg.values())
        summary[f"correctness_{field}"] = str(next(iter(values)))
        summary[f"correctness_{field}_identical"] = len(values) == 1

    for leg in data["legs"]:
        tag = leg["tag"]
        widths = leg["widths"]
        summary[f"{tag}_arm"] = leg["arm"]
        summary[f"{tag}_rounds"] = widths["rounds"]
        summary[f"{tag}_mean_verify_width"] = widths["mean_verify_width"]
        for width in range(1, 10):
            summary[f"{tag}_width_share_{width}"] = (
                widths["share"].get(str(width), 0.0))
        for field in ("entry_gpu_temp_c", "exit_gpu_temp_c",
                      "cool_gate_passed_real_gate", "worker_sha256",
                      "metallib_sha256", "arm_schedule_blob"):
            if field in leg["meta"]:
                summary[f"{tag}_{field}"] = leg["meta"][field]

    summary.update(cost_model())

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=f"e56-{args.session}-abba-summary",
                     job_type="analysis",
                     config={"experiment": GROUP, "session": args.session,
                             "arms": "base,sched,sched,base"},
                     reinit=True)
    run.log(metrics)
    run.summary.update(summary)
    run.finish()
    print(f"e56_log_session: logged {args.session} as {run.id}")


if __name__ == "__main__":
    main()
