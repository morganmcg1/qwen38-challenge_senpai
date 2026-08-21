#!/usr/bin/env python3
"""Publish the E95 `target_verify` width census and the E95 rider to W&B.

    usage:
      research/e95_wandb_log.py census LEG@M [LEG@M ...]
      research/e95_wandb_log.py model
      research/e95_wandb_log.py rider --exact DIR --base LEG --cand LEG

`census` writes one run per width leg: the identity tuple, the local score
metrics, the host thermal record and the per-phase and per-class dispatch
counters of that leg.

`model` writes the analysis run: the three-term width model, its residuals, the
per-class row-cost regression that prices the idle x-groups, and the ranked
cost weighting of the target weight stream.

`rider` writes the rider run: the 512-token exactness gate, its positive
control, the base-to-rider draft-head dispatch and byte diff, and the ranked
price of the removed dead rows.

Every leg in this experiment is a counting and GPU-clock attribution leg, never
a gate-qualified timed arm, so `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged false verbatim on every run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e95_verify_census as census  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e95-dispatch-level-dead-work"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

# Fitted on the six census widths below. Reproduce with
#   research/e95_verify_census.py model e95-verify-m3@3 ... e93-gpu-insitu-long@9
WIDTH_MODEL = {
    "fixed_us_per_round": 10_920.0,
    "us_per_weight_pass": 27_377.0,
    "us_per_verify_row": 10_268.0,
    "max_abs_residual_pct": 1.08,
    "out_of_sample_width": 6,
    "out_of_sample_residual_pct": -1.34,
    "weight_stream_gb_per_s": 526.4,
    "arithmetic_tmac_per_s": 2.50,
}

# Per-class row cost, from the M=5 to M=8 pair. `ns_per_group` is the cost of
# one threadgroup at fixed work; `us_per_mmac` is the cost of the work itself.
ROW_COST = [
    ("gate_up  O=17408", 16.01, 0.3909),
    ("in_proj  O=27648", 16.01, 0.3909),
    ("lm_head  O=248320", 15.92, 0.3888),
    ("qkv      O=7168", 16.46, 0.4019),
    ("out+down O=5120", 37.07, 0.3935),
]

# Ranked cost weighting of the target weight stream, over the two public
# prompts at their measured mean accepted-draft widths.
RANKED_WEIGHTING = {
    "mean_width_beagle": 5.38,
    "mean_width_essays": 6.09,
    "phase_share_pct": 90.8,
    "fixed_share_pct_beagle": 9.0,
    "fixed_share_pct_essays": 8.5,
    "weight_stream_share_pct_beagle": 45.3,
    "weight_stream_share_pct_essays": 42.7,
    "arithmetic_share_pct_beagle": 45.7,
    "arithmetic_share_pct_essays": 48.8,
    "class_share_pct_gate_up_beagle": 40.5,
    "class_share_pct_gate_up_essays": 40.7,
    "class_share_pct_out_down_beagle": 27.4,
    "class_share_pct_out_down_essays": 27.6,
    "class_share_pct_gdn_in_proj_beagle": 14.4,
    "class_share_pct_gdn_in_proj_essays": 14.5,
    "class_share_pct_lm_head": 4.5,
    "class_share_pct_attn": 4.2,
}

# Draft-head q projection, per marginal draft. The island overwrites 1024 of
# the 12288 q output rows, so the pack only has to produce the other 11264.
Q_ROWS_TOTAL = 12_288
Q_ROWS_ISLAND = 1_024
Q_ROWS_LIVE = Q_ROWS_TOTAL - Q_ROWS_ISLAND
Q_BYTES_PER_ROW = 5120 * 0.5625  # affine-4 group-64: 4 bits + scale + bias
HEAD_WEIGHT_BYTES_PER_CALL = 427_476_480  # class-1 head weight stream, e93


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def read_meta(leg: str) -> dict[str, str]:
    meta = {}
    path = OUT / leg / "meta.txt"
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def gate_flags() -> dict[str, bool]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def identity(meta: dict[str, str]) -> dict[str, object]:
    return {
        "host": HOST,
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "head_provenance_sha256": None,
        "tokens": int(meta.get("tokens", 0)) or None,
        "forced_drafts": int(meta.get("forced_drafts", -1)),
        "local_mode": meta.get("local_mode"),
        "census": meta.get("census") == "1",
        "gputime": meta.get("gputime") == "1",
        "ops_per_buffer": meta.get("ops_per_buffer"),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "started": meta.get("started"),
        "finished": meta.get("finished"),
    }


def steady_rounds(leg: str, width: int) -> list[dict]:
    """Rounds of the MTP leg that ran at the forced verify width."""
    records = census.load(OUT / leg / "census.jsonl")
    return [r for r in records
            if r.get("event") == "round" and int(r["width"]) == width]


def leg_phase_table(leg: str, width: int):
    """Per-round dispatch count and encode microseconds for every phase."""
    rounds = steady_rounds(leg, width)
    table: dict[str, dict[str, float]] = {}
    for record in rounds:
        for phase, bucket in record["phases"].items():
            entry = table.setdefault(
                phase, {"dispatches_per_round": 0.0, "commits_per_round": 0.0})
            entry["dispatches_per_round"] += bucket["dispatches"]
            entry["commits_per_round"] += bucket["commits"]
    for entry in table.values():
        entry["dispatches_per_round"] /= len(rounds)
        entry["commits_per_round"] /= len(rounds)
    return len(rounds), table


def phase_shapes(leg: str, width: int, phase: str) -> dict[str, float]:
    """Per-round count of every dispatch shape inside one phase."""
    rounds = steady_rounds(leg, width)
    totals: dict[str, float] = {}
    for record in rounds:
        bucket = record["phases"].get(phase)
        if bucket is None:
            continue
        for shape, count in bucket.get("shapes", {}).items():
            totals[shape] = totals.get(shape, 0.0) + count
    return {shape: total / len(rounds) for shape, total in totals.items()}


def log_census(specs: list[str]) -> None:
    for spec in specs:
        leg, _, width_text = spec.partition("@")
        width = int(width_text)
        meta = read_meta(leg)
        score = read_json(OUT / leg / "score.json")
        metrics = score["metrics"]
        rounds, phases = leg_phase_table(leg, width)

        config = identity(meta)
        config["head_provenance_sha256"] = metrics.get("head_provenance_sha256")
        config["verify_width_m"] = width
        config["leg"] = leg
        config.update(gate_flags())

        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            job_type="width-census-leg", name=f"e95-census-{leg}",
            config=config, reinit=True)
        summary = {
            "rounds": rounds,
            "verify_width_m": width,
            "decode_tokens": metrics["decode_tokens"],
            "all_tokens_matched": metrics["all_tokens_matched"],
            "residual_divergence_count": metrics["residual_divergence_count"],
            "effective_mean_draft_len": metrics["effective_mean_draft_len"],
            "accepted_draft_rate": metrics["accepted_draft_rate"],
            "mtp_decode_speedup_census_serialised": metrics["mtp_decode_speedup"],
        }
        for phase, values in phases.items():
            summary[f"phase/{phase}/dispatches_per_round"] = values[
                "dispatches_per_round"]
            summary[f"phase/{phase}/us_per_round"] = values["us_per_round"]
        run.summary.update(summary)
        run.finish()


def log_model() -> None:
    config = {
        "host": HOST,
        "experiment": "e95-target-verify-width-model",
        "widths_fitted": [3, 4, 5, 8, 9],
        "width_held_out": 6,
        "tokens_per_leg": 384,
    }
    config.update(gate_flags())
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="width-model", name="e95-width-model", config=config,
        reinit=True)

    run.summary.update({f"model/{k}": v for k, v in WIDTH_MODEL.items()})
    run.summary.update({f"ranked/{k}": v for k, v in RANKED_WEIGHTING.items()})

    rows = wandb.Table(
        columns=["class", "ns_per_threadgroup", "us_per_mmac"], data=ROW_COST)
    run.log({"row_cost_by_class": rows})
    run.summary.update({
        "dead_work/idle_x_groups_are_free": True,
        "dead_work/launch_only_fit_rms_pct": 29.4,
        "dead_work/work_only_fit_rms_pct_low": 0.42,
        "dead_work/work_only_fit_rms_pct_high": 0.52,
        "dead_work/joint_fit_ns_per_threadgroup_low": -0.20,
        "dead_work/joint_fit_ns_per_threadgroup_high": -0.17,
        "dead_work/quantized_cpp_in_editable_paths": False,
    })
    run.finish()


def log_rider(exact_dir: pathlib.Path, base_leg: str, cand_leg: str) -> None:
    summary_path = exact_dir / "rung0a-summary.json"
    exact = read_json(summary_path)
    meta = {}
    for line in (exact_dir / "meta.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value

    base_meta = read_meta(base_leg)
    cand_meta = read_meta(cand_leg)
    base_score = read_json(OUT / base_leg / "score.json")["metrics"]
    cand_score = read_json(OUT / cand_leg / "score.json")["metrics"]

    base_shapes = census.draft_head_shapes(OUT / base_leg / "census.jsonl")
    cand_shapes = census.draft_head_shapes(OUT / cand_leg / "census.jsonl")

    saved_bytes = int((Q_ROWS_TOTAL - Q_ROWS_LIVE) * Q_BYTES_PER_ROW)
    config = {
        "host": HOST,
        "experiment": "e95-rider-q-row-shrink",
        "candidate_commit": meta.get("commit"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "head_provenance_sha256": meta.get("head_provenance_sha256"),
        "fixture": meta.get("fixture"),
        "tokens": int(meta.get("tokens", 0)),
        "mtp_depth": int(meta.get("depth", 0)),
        "dirty_candidate_paths": int(meta.get("dirty_candidate_paths", -1)),
        "base_census_leg": base_leg,
        "candidate_census_leg": cand_leg,
        "base_census_worker_sha256": base_meta.get("worker_sha256"),
        "candidate_census_worker_sha256": cand_meta.get("worker_sha256"),
    }
    config.update(gate_flags())

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="rider-exactness", name="e95-rider-q-row-shrink",
        config=config, reinit=True)

    run.summary.update({
        "exact/gate_exit": exact.get("gate_exit"),
        "exact/control_exit": exact.get("control_exit"),
        "exact/tokens": exact.get("tokens"),
        "exact/all_tokens_matched": exact.get("all_tokens_matched"),
        "exact/row_ledger_closed": exact.get("row_ledger_closed"),
        "exact/rows_accounted": exact.get("rows_accounted"),
        "exact/post_eos_tokens_checked": exact.get("post_eos_tokens_checked"),
        "exact/control_detected_mutation": exact.get("control_detected_mutation"),
        "rider/q_rows_before": Q_ROWS_TOTAL,
        "rider/q_rows_after": Q_ROWS_LIVE,
        "rider/q_island_rows": Q_ROWS_ISLAND,
        "rider/q_bytes_before": int(Q_ROWS_TOTAL * Q_BYTES_PER_ROW),
        "rider/q_bytes_after": int(Q_ROWS_LIVE * Q_BYTES_PER_ROW),
        "rider/q_bytes_saved": saved_bytes,
        "rider/head_weight_stream_share_pct":
            100.0 * saved_bytes / HEAD_WEIGHT_BYTES_PER_CALL,
        "census/base_effective_mean_draft_len":
            base_score["effective_mean_draft_len"],
        "census/candidate_effective_mean_draft_len":
            cand_score["effective_mean_draft_len"],
        "census/base_accepted_draft_rate": base_score["accepted_draft_rate"],
        "census/candidate_accepted_draft_rate":
            cand_score["accepted_draft_rate"],
        "census/draft_proposals_identical":
            base_score["effective_mean_draft_len"]
            == cand_score["effective_mean_draft_len"]
            and base_score["accepted_draft_rate"]
            == cand_score["accepted_draft_rate"],
        "census/base_all_tokens_matched": base_score["all_tokens_matched"],
        "census/candidate_all_tokens_matched": cand_score["all_tokens_matched"],
    })

    shape_rows = []
    for shape in sorted(set(base_shapes) | set(cand_shapes)):
        shape_rows.append([
            shape,
            base_shapes.get(shape, 0.0),
            cand_shapes.get(shape, 0.0),
            cand_shapes.get(shape, 0.0) - base_shapes.get(shape, 0.0),
        ])
    run.log({"draft_head_shapes": wandb.Table(
        columns=["shape", "base_per_round", "candidate_per_round", "delta"],
        data=shape_rows)})
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_census = sub.add_parser("census")
    p_census.add_argument("specs", nargs="+")
    sub.add_parser("model")
    p_rider = sub.add_parser("rider")
    p_rider.add_argument("--exact", required=True, type=pathlib.Path)
    p_rider.add_argument("--base", required=True)
    p_rider.add_argument("--cand", required=True)
    args = parser.parse_args()

    if args.command == "census":
        log_census(args.specs)
    elif args.command == "model":
        log_model()
    else:
        log_rider(args.exact, args.base, args.cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
