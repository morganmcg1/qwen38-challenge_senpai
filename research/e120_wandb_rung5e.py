#!/usr/bin/env python3
"""Publish one E120 rung 5e in-situ ABBA session to W&B.

    usage: research/e120_wandb_rung5e.py OUT_DIR --name NAME

Reads the tree that `research/e120_rung5e_session.sh` wrote and the
`rung5e_report.json` that `research/e120_rung5e_report.py` derived from it, and
records the whole experiment identity tuple beside the numbers so another agent
can tell which binary, head, host, window and fixture produced them.

The headline logged here is ABSOLUTE candidate seconds per token on the
native-MTP leg. The local serial-to-MTP ratio is logged only as a control,
because a wide-QMV change speeds both local legs and partly cancels there.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e120-own-the-qmv-dispatch"
BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
BUDGET_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
FIXTURE = "correctness_prompts/public_longcopy_gate_english_512_256.json"

# The isolated per-cell grid this session's prediction comes from.
SOURCE_RUNS = {
    "qql6zari": "rung 5d, widths 3,4,8, all seven shapes",
    "iyzornb9": "rung 5d-na, widths 5,6,7,9, four shapes",
    "by2wpwg5": "rung 5d-na2, widths 5,6,7,9, remaining shapes",
    "e3m044ng": "rung 5d derived analysis, grid -> round price -> ranked %",
}


def read_json(path: pathlib.Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rung", default="5e")
    args = parser.parse_args()

    out_dir = args.out_dir
    report = read_json(out_dir / "rung5e_report.json")
    meta = report["meta"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="in-situ-abba",
        name=args.name,
        config={
            "experiment": GROUP,
            "rung": args.rung,
            "pr": 121,
            "question": (
                "In situ, over a full 512-token decode, does the hoisted "
                "activation-sum table lower ABSOLUTE candidate seconds per "
                "token against the same binary with the table off?"
            ),
            # --- experiment identity tuple -------------------------------
            "base_sha": BASE_SHA,
            "budget_base_sha": BUDGET_BASE_SHA,
            "git_head": meta.get("git_head"),
            "git_dirty": meta.get("git_dirty"),
            "worker_sha256": meta.get("worker_sha256_start"),
            "worker_sha256_end": meta.get("worker_sha256_end"),
            "mtp_head_dir": meta.get("head_dir"),
            "mtp_head_safetensors_sha256": meta.get("head_safetensors_sha256"),
            "host": meta.get("host"),
            "chip": meta.get("chip"),
            "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "fixture": FIXTURE,
            "decode_tokens": meta.get("tokens"),
            "offered_draft_depth": meta.get("offered_draft_depth"),
            "arm_order": meta.get("order"),
            "arm_env_var": "MLX_E120_QMV_ARM",
            "source_runs": SOURCE_RUNS,
            # --- honesty flags -------------------------------------------
            "instrument": meta.get("instrument"),
            "harness": "local",
            "cool_gate_command": meta.get("cool_gate_command"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate")
            == "true",
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing")
            == "true",
            "official_or_ranked_score": False,
            "timing_valid": True,
            "headline_metric": report["headline_metric"],
        },
        reinit=True,
    )

    summary: dict[str, object] = {
        "exactness_ok": report["exactness_ok"],
        "failed_legs": len(report["failed_legs"]),
    }

    # --- one row per timed leg -------------------------------------------
    legs = wandb.Table(
        columns=[
            "label",
            "arm",
            "seconds_per_token",
            "all_tokens_matched",
            "residual_divergence_count",
            "decode_token_count",
            "accepted_draft_rate",
            "effective_mean_draft_len",
        ]
    )
    for record in report["fidelity"]:
        legs.add_data(
            record["label"],
            record["arm"],
            record["seconds_per_token"],
            record["all_tokens_matched"],
            record["residual_divergence_count"],
            record["decode_token_count"],
            record["accepted_draft_rate"],
            record["effective_mean_draft_len"],
        )
    run.log({"legs": legs})

    for arm, arm_summary in report["arms"].items():
        for key in ("legs", "mean", "median", "sd", "min", "max"):
            summary[f"arm/{arm}/{key}"] = arm_summary[key]
        entry = report["gpu_temp_entry_c"].get(arm)
        if entry:
            summary[f"arm/{arm}/gpu_temp_entry_c_mean"] = entry["mean"]
            summary[f"arm/{arm}/gpu_temp_entry_c_spread"] = entry["max"] - entry["min"]

    # --- the headline ------------------------------------------------------
    effect = report.get("effect")
    if effect:
        summary["headline/candidate_seconds_per_token"] = effect[
            "candidate_seconds_per_token"
        ]
        summary["headline/baseline_seconds_per_token"] = effect[
            "baseline_seconds_per_token"
        ]
        summary["headline/delta_seconds_per_token"] = effect["delta_seconds_per_token"]
        summary["headline/measured_leg_pct"] = effect["measured_leg_pct"]
        summary["headline/ranked_pct"] = effect["ranked_pct"]
        summary["headline/entry_temp_spread_c"] = effect["entry_temp_spread_c"]
        ratio_control = effect.get("local_ratio_control", {})
        for arm in ("off", "sumtable"):
            if arm in ratio_control:
                summary[f"control/local_ratio/{arm}"] = ratio_control[arm]

    dispersion = report.get("dispersion")
    if dispersion:
        for key, value in dispersion.items():
            summary[f"dispersion/{key}"] = value

    serial = report.get("serial_control")
    if serial:
        for arm, value in serial["by_arm"].items():
            summary[f"control/serial/{arm}_seconds_per_token"] = value[
                "seconds_per_token"
            ]
        summary["control/serial/spread_pct"] = serial["spread_pct"]
        summary["control/serial/arm_independent"] = serial["arm_independent"]

    worker = report["worker_assertion"]
    summary["provenance/one_binary_for_every_leg"] = worker["one_binary_for_every_leg"]
    summary["provenance/per_leg_worker_unchanged"] = worker["per_leg_unchanged"]

    # --- realised width histogram, from these timed legs -------------------
    histogram = report["realised_width_histogram"]
    widths = wandb.Table(columns=["width_m", "rounds", "fraction"])
    for width, count in sorted(histogram["counts"].items(), key=lambda kv: int(kv[0])):
        widths.add_data(int(width), count, histogram["fraction"][width])
    run.log({"realised_width_histogram": widths})
    summary["width/mean"] = histogram["mean_width"]
    summary["width/total_rounds"] = histogram["total_rounds"]

    predicted = report.get("predicted_from_rung5d")
    if predicted:
        summary["predicted/wide_qmv_pct"] = predicted["wide_qmv_pct"]
        summary["predicted/leg_pct"] = predicted["leg_pct"]
        summary["predicted/leg_pct_low"] = predicted["leg_pct_interval"][0]
        summary["predicted/leg_pct_high"] = predicted["leg_pct_interval"][1]
        summary["predicted/ranked_pct"] = predicted["ranked_pct"]
        summary["predicted/rounds_modelled"] = predicted["rounds_modelled"]
        summary["predicted/rounds_unmodelled"] = predicted["rounds_unmodelled"]
    ratio = report.get("isolated_to_in_situ_transfer_ratio")
    if ratio is not None:
        summary["isolated_to_in_situ_transfer_ratio"] = ratio

    run.summary.update(summary)
    print(f"logged {run.id} {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
