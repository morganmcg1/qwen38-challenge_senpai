#!/usr/bin/env python3
"""Publish an E136 rung to W&B.

    usage: research/e136_wandb_log.py --rung 0|0b [--dry]

Rung 0 times Metal kernels but it is NOT a decode leg, a gated measurement or
a score. It times the survivor-selection dispatches standing alone, off the
scored path, so every run logs `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e136-c1-sketch-readout-build"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "35d8cf586b8671dc3d01faf3cdbd724ec603801b"

ARM_COLUMNS = [
    "arm", "survivors", "tiles", "dispatches", "added_us_batched",
    "added_us_isolated", "added_us_parts_plus_ledger_f", "added_ranked_pct",
    "added_ranked_pct_e93_rate", "recall_of_true_top32",
    "us_per_call", "batched_us_per_selection",
]

PART_COLUMNS = ["arm", "part", "us_above_floor"]

RUNGS = {
    "0": {
        "run_name": "e136-rung0-widened-selection-microbenchmark",
        "file": "research/e136-selection-bench.json",
        "question":
            "does selecting the top N of 34,424 sketch-scored rows with a "
            "histogram threshold cost less added GPU time per draft step "
            "than the 37.483 MB the C1 readout removes is worth",
        "command":
            "research/e133_job.sh research/await-lock-then-run.sh 1800 "
            "python3 research/e136_selection_bench.py --replicates 3",
    },
}


def rung0_summary(payload: dict) -> tuple[dict, dict]:
    verdict = payload["verdict"]
    arms, parts = [], []
    for label, row in payload["arms"].items():
        if "added_us_per_step" not in row:
            continue
        plan = row.get("plan", {})
        arms.append({
            "arm": label,
            "survivors": plan.get("real_count"),
            "tiles": plan.get("tiles"),
            "dispatches": row.get("dispatches_per_step"),
            "added_us_batched": row["added_us_per_step"],
            "added_us_isolated": row["added_us_isolated"],
            "added_us_parts_plus_ledger_f":
                row["added_us_from_parts_plus_ledger_f"],
            "added_ranked_pct": row["added_ranked_pct"],
            "added_ranked_pct_e93_rate":
                row["added_us_per_step"] / 247.2,
            "recall_of_true_top32":
                payload["correctness"][label]["recall_of_true_top32"],
            "us_per_call": row["us_per_call"],
            "batched_us_per_selection": row["batch"]["us_per_selection"],
        })
        for part, value in row.get("parts_us_above_floor", {}).items():
            parts.append({"arm": label, "part": part, "us_above_floor": value})

    summary = {
        # The metric the assignment names. The whole-arm batched slope is the
        # estimator: it replicates to 0.4 percent where the part arms move by
        # a factor of three.
        "e136_widened_selection_us_per_draft_step":
            verdict["added_us_per_draft_step"],
        "e136_widened_selection_us_spread":
            verdict["added_us_per_draft_step_spread"],
        "e136_widened_selection_ranked_pct":
            verdict["added_ranked_pct_at_265gbs"],
        "e136_widened_selection_ranked_pct_e93_rate":
            verdict["added_ranked_pct_at_measured_186_7gbs"],
        "e136_rung0_stop_rule": verdict["stop_rule"],
        "e136_rung0_stop_rule_on_worst_estimate":
            verdict["stop_rule_on_worst_estimate"],
        "e136_rung0_net_pct_after_selection_cost":
            verdict["net_pct_after_selection_cost"],
        "e136_rung0_recall_min":
            min(c["recall_of_true_top32"]
                for c in payload["correctness"].values()),
        "e136_rung0_positive_control_fires":
            all(c["positive_control_detects_the_change"]
                for c in payload["correctness"].values()),
        "e136_rung0_device_threshold_matches_host":
            all(c["device_tau_matches_host"]
                for c in payload["correctness"].values()),
    }
    for key, value in verdict.items():
        if isinstance(value, (int, float, str, bool)):
            summary["verdict/%s" % key] = value
    for key, value in payload["anchor_check"].items():
        if isinstance(value, (int, float, str, bool)):
            summary["anchor/%s" % key] = value
    for i, rep in enumerate(payload["replicates"]):
        for label, value in rep.items():
            summary["replicate/%d/%s" % (i, label)] = value
    return summary, {
        "rung0_arms": table(ARM_COLUMNS, arms),
        "rung0_parts": table(PART_COLUMNS, parts),
    }


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


BUILDERS = {"0": rung0_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    payload = json.loads(pathlib.Path(spec["file"]).read_text())
    summary, tables = BUILDERS[args.rung](payload)
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "timing_valid": False,
        "host": HOST,
        "base_sha": BASE_SHA,
        "rung": args.rung,
        "question": spec["question"],
        "command": spec["command"],
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type="kernel-microbenchmark",
                     config={"experiment": "E136", "rung": args.rung,
                             "base_sha": BASE_SHA, "host": HOST,
                             "command": spec["command"],
                             "question": spec["question"]})
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e136-rung%s" % args.rung,
                              type="kernel-microbenchmark")
    artifact.add_file(spec["file"])
    run.log_artifact(artifact)
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
