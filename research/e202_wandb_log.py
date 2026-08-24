#!/usr/bin/env python3
"""Publish the E202 eval()-barrier settlement session to W&B.

    usage: research/e202_wandb_log.py \
        --report research/e202-session.json \
        --session research/out/e202/session-submit-<name>

Everything here is `harness=local`. The legs ran with
`MLXFAST_LOCAL_COOL_GATE=0` under the standing counterbalanced-session
allowance, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` are carried verbatim and no number here is an
official or ranked score. The arms rotate IN PROCESS at the round boundary, so
every leg carries the same arm composition and the leg-level spread is a null
control rather than an arm contrast.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e202-eval-barrier-dispatch-settlement"

CROWN = 3.7291100105909
RECEIPT_A = 3.70784519415395
MUE_MS = 0.567
ARTIFACT_FLOOR_MS = 0.2
TRUST_BAR_MS = 0.5


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_build(path: pathlib.Path) -> dict:
    record = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key and value and "<<EOF" not in line:
                record[key.strip()] = value.strip()
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=pathlib.Path, required=True)
    ap.add_argument("--session", type=pathlib.Path, required=True)
    ap.add_argument(
        "--build", type=pathlib.Path, default=pathlib.Path("research/e202-build.txt")
    )
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    build = read_build(args.build)
    legs = report["legs"]
    serving = report["contrasts_branch_serving_rounds"]
    idle = report["contrasts_null_control_non_serving_rounds"]

    config = {
        "experiment": "e202-eval-barrier-dispatch-settlement",
        "harness": "local",
        "host_chip": "M4 Pro",
        "arms": "SHIPPED | BARRIER-LAST | BARRIER-ALL",
        "arm_switch": "in-process, at every round boundary (RULE 388)",
        "arm_switch_env": "DARKBLOOM_E202_BARRIER_OFFSET",
        "decode_tokens": legs[0].get("decode_tokens") if legs else None,
        "legs": len(legs),
        "assignment_base_sha": "7f10e147f2e270a8893c7c3a4ebe0e05cdc2fdd8",
        "candidate_sha": git("rev-parse", "HEAD"),
        "organizer_sha": build.get("organizer_sha"),
        "worker_sha256": build.get("worker_sha256"),
        "metallib_sha256": build.get("metallib_sha256"),
        "head_provenance_sha256": legs[0].get("head_provenance_sha256") if legs else None,
        "head_class": "declared",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "comparable_with_sandboxed_history": False,
        "worker_sandbox": "disabled",
        "added_timing_instrument": "e202-per-round-trace-and-arm-witness-all-arms",
        "timing_source": "per-round-trace-round_us",
        "official_score": False,
        "rankable": False,
        "receipt_a": RECEIPT_A,
        "crown": CROWN,
        "mue_ms_per_round": MUE_MS,
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=f"e202-{args.session.name}",
        job_type="local-measurement",
        config=config,
        tags=["e202", "harness-local", "ungated", "in-process-arm-switch", "finding-507"],
    )

    def contrast_metrics(prefix: str, block: dict) -> dict:
        out = {}
        for key in (
            "sync_ms_per_round",
            "inner_ms_per_round",
            "net_interior_ms_per_round",
            "all_minus_shipped_ms_per_round",
        ):
            stat = block.get(key) or {}
            out[f"{prefix}/{key}/mean"] = stat.get("mean")
            out[f"{prefix}/{key}/two_sigma"] = stat.get("two_sigma")
            out[f"{prefix}/{key}/n"] = stat.get("n")
        out[f"{prefix}/mean_split_calls_per_round"] = block.get(
            "mean_split_calls_per_round"
        )
        out[f"{prefix}/sync_us_per_barrier"] = block.get("sync_us_per_barrier")
        out[f"{prefix}/inner_us_per_barrier"] = block.get("inner_us_per_barrier")
        return out

    summary = {
        "witnesses_valid": report["witnesses_valid"],
        "verdict": report["verdict"],
        "artifact_floor_ms_per_round": ARTIFACT_FLOOR_MS,
        "trust_bar_ms_per_round": TRUST_BAR_MS,
        "leg_level_null_spread_ms": report["leg_level_null_control"]["spread_ms"],
        "leg_level_null_stdev_ms": report["leg_level_null_control"]["stdev_ms"],
    }
    summary.update(contrast_metrics("serving", serving))
    summary.update(contrast_metrics("null_nonserving", idle))

    leg_table = wandb.Table(
        columns=[
            "leg",
            "offset",
            "gpu_temp_entry",
            "gpu_temp_exit",
            "all_tokens_matched",
            "residual_divergence_count",
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "mtp_decode_speedup",
            "effective_mean_draft_len",
            "accepted_draft_rate",
            "leg_mean_round_ms",
            "witness_valid",
            "rounds",
            "branch_serving_rounds",
            "armed_rounds",
            "serving_triples",
            "split_call_census_by_width",
            "worker_sha256",
        ]
    )
    for leg in legs:
        leg_table.add_data(
            leg["leg"],
            leg["offset"],
            leg["gpu_temp_entry"],
            leg["gpu_temp_exit"],
            leg["all_tokens_matched"],
            leg["residual_divergence_count"],
            leg["mtp_seconds_per_token"],
            leg["serial_seconds_per_token"],
            leg["mtp_decode_speedup"],
            leg["effective_mean_draft_len"],
            leg["accepted_draft_rate"],
            leg["leg_mean_round_ms"],
            leg["witness"]["valid"],
            leg["witness"]["rounds"],
            leg["witness"]["branch_serving_rounds"],
            leg["witness"]["armed_rounds"],
            leg["serving_triples"],
            json.dumps(leg["split_call_census_by_width"]),
            leg["worker_sha256"],
        )

    census_table = wandb.Table(columns=["served_width_qL", "split_cell_calls"])
    for width, count in report["session_split_call_census_by_width"].items():
        census_table.add_data(int(width), count)

    run.log({"legs": leg_table, "split_cell_width_census": census_table})
    run.summary.update(summary)
    run.summary["report"] = json.dumps(report)[:100000]
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
