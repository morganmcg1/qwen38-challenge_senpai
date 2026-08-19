#!/usr/bin/env python3
"""qwen38-r1-e30: publish the post-NA=5 M=9 re-measurement to W&B.

usage:
  research/e30_log_wandb.py research/results/e30-analysis.json [--name N]

Consumes `research/e30_adjudicate.py --json-out`. Everything published here is
paired: the E29 pre-NA=5 arm and the E30 post-NA=5 arms, per width, from BOTH
the six-way trace and the trusted parent's own `block_request_seconds`. Both
decode legs are logged separately on true decode; no bare ratio is published as
a headline, because both local legs run the candidate build and the ranked score
divides by the serial leg measured in the same session.

Pre-registered thresholds travel with the run as config, so the verdicts can be
re-checked later without reading the report prose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

SEGMENTS = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]


def m(entry: dict, key: str) -> float | None:
    v = entry.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def sd(entry: dict, key: str) -> float | None:
    v = entry.get(key)
    return v.get("sd") if isinstance(v, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--group", default="qwen38-r1-e30")
    ap.add_argument("--name", default="e30-m9-cliff-after-na5")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    doc = json.loads(args.analysis.read_text())
    prereg = doc["prereg"]
    base = doc["baseline"]
    cands = doc["candidates"]
    primary_name = next(iter(cands))
    primary = cands[primary_name]
    adj = doc["adjudication"]
    meta = primary.get("meta") or {}

    config = {
        "experiment": "qwen38-r1-e30-m9-cliff-after-na5",
        "assignment_pr": 35,
        "base_sha": "d08feb85bf65959d7eaa1455e36a0173b3edd8d9",
        "e29_baseline_tree": (base.get("meta") or {}).get("head_sha"),
        "e30_tree": meta.get("head_sha"),
        "dirty": meta.get("dirty"),
        "arms": sorted(cands),
        "primary_arm": primary_name,
        "tokens": meta.get("tokens"),
        "offered_depth": meta.get("offered_depth"),
        "ladder": meta.get("ladder"),
        "chip": meta.get("chip"),
        "host": meta.get("host"),
        "cli_sha256": meta.get("cli_sha256"),
        "worker_sha256": meta.get("worker_sha256"),
        # The single live delta on this host: E27 raised the wide crossrow QMV
        # accumulator ceiling to NA=5 and took M=9 from IPG 3 to IPG 5, i.e.
        # three weight streams to two. M=8 stayed <T,8,4>.
        "treatment": "E27 NA=5: <T,9,3> -> <T,9,5> (3 -> 2 weight streams)",
        "control_width": 8,
        "treated_width": 9,
        "m8_dispatch_unchanged": True,
        "inert_on_this_host": [
            "Qwen36MTPBlockSession residency wiring (>=96 GiB gate)",
            "RuntimeStartupMemoryPolicy command-buffer defaults (>=96 GiB gate)",
        ],
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "thermal_before": meta.get("thermal_before"),
        "thermal_after": meta.get("thermal_after"),
        "trace_sync_head_enabled": False,
        "notes": args.notes,
        **{f"prereg/{k}": v for k, v in prereg.items()},
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        name=args.name,
        job_type="re-measurement",
        config=config,
        notes=args.notes or None,
    )

    arms = [("E29-D0", base)] + [(k, v) for k, v in cands.items()]

    leg_rows, width_rows, parent_rows, hist_rows, delta_rows = [], [], [], [], []
    for label, a in arms:
        lg = a["legs"]
        leg_rows.append([
            label, (a.get("meta") or {}).get("head_sha"),
            lg["serial"]["true_decode_ms_per_token"],
            lg["mtp"]["true_decode_ms_per_token"],
            lg["mtp"]["ms_per_round"],
            lg["true_decode_serial_over_mtp"],
            lg["serial"]["decode_seconds_prefill_inclusive"],
            lg["mtp"]["decode_seconds_prefill_inclusive"],
            lg["mtp"]["seed_prefill_seconds"],
            lg["mtp"]["prefill_share_of_decode_seconds_pct"],
            lg["mtp"]["all_tokens_matched"],
            lg["mtp"]["residual_divergence_count"],
            lg["mtp"]["accepted_draft_total"],
            lg["mtp"]["accepted_draft_rate"],
            lg["mtp"]["effective_mean_draft_len"],
            (a.get("meta") or {}).get("thermal_before"),
            (a.get("meta") or {}).get("thermal_after"),
        ])

        tr = a["trace"]
        for width, e in tr["per_width"].items():
            width_rows.append([
                label, int(width), e["rounds"],
                m(e, "round_ms"), sd(e, "round_ms"),
                e["round_sequence_slope_ms_per_round"],
                e["full_accept_ms_per_token"],
                m(e, "host_tail_ms"),
                *[m(e, s) for s in SEGMENTS],
                *[sd(e, s) for s in SEGMENTS],
            ])
        for width, n in tr["histogram"].items():
            hist_rows.append([label, int(width), n,
                              100.0 * n / tr["steady_rounds"]])

        p = a.get("parent")
        if p:
            for width, e in p["per_width"].items():
                parent_rows.append([
                    label, int(width), e["n"], e["mean"], e["sd"],
                    e["min"], e["max"], e["full_accept_ms_per_token"],
                    e["sequence_slope_ms_per_round"],
                ])

    btr, ptr = base["trace"], primary["trace"]
    for width in sorted(set(btr["per_width"]) & set(ptr["per_width"]),
                        key=int):
        b, c = btr["per_width"][width], ptr["per_width"][width]
        row = [int(width), b["rounds"], c["rounds"]]
        for key in ["round_ms", "verify_build_us", "eval_wall_us",
                    "draft_build_us", "host_tail_ms"]:
            bm, cm = m(b, key), m(c, key)
            row += [bm, cm, cm - bm, 100.0 * (cm / bm - 1.0) if bm else None]
        delta_rows.append(row)

    delta_cols = ["M", "e29_rounds", "e30_rounds"]
    for key in ["round_ms", "verify_build", "eval_wall", "draft_build",
                "host_tail"]:
        delta_cols += [f"{key}_e29", f"{key}_e30", f"{key}_delta_ms",
                       f"{key}_delta_pct"]

    run.log({
        "legs": wandb.Table(
            columns=["arm", "tree", "serial_ms_per_token_true_decode",
                     "mtp_ms_per_token_true_decode", "mtp_ms_per_round",
                     "true_decode_serial_over_mtp",
                     "serial_decode_seconds_prefill_inclusive",
                     "mtp_decode_seconds_prefill_inclusive",
                     "seed_prefill_seconds", "prefill_share_pct",
                     "all_tokens_matched", "residual_divergence_count",
                     "accepted_draft_total", "accepted_draft_rate",
                     "effective_mean_draft_len",
                     "thermal_before", "thermal_after"],
            data=leg_rows),
        "per_width_trace": wandb.Table(
            columns=["arm", "M", "rounds", "round_mean_ms", "round_sd_ms",
                     "round_sequence_slope_ms", "full_accept_ms_per_token",
                     "host_tail_mean_ms",
                     *[f"{s}_mean_ms" for s in SEGMENTS],
                     *[f"{s}_sd_ms" for s in SEGMENTS]],
            data=width_rows),
        "per_width_parent": wandb.Table(
            columns=["arm", "M", "rounds", "block_mean_ms", "block_sd_ms",
                     "block_min_ms", "block_max_ms",
                     "full_accept_ms_per_token", "sequence_slope_ms"],
            data=parent_rows),
        "depth_histogram": wandb.Table(
            columns=["arm", "M", "rounds", "share_of_steady_rounds_pct"],
            data=hist_rows),
        "e29_to_e30_width_deltas": wandb.Table(
            columns=delta_cols, data=delta_rows),
    })

    p1, p2, ctl = adj["p1_bound_M8"], adj["p2_vbuild_m9"], adj["controls"]
    summary = {
        # Primary metric, always evaluated at M=8 even once M=8 stops being the
        # cheapest width. Negative means forcing every round to M=8 would cost.
        "e30/all_at_m8_upper_bound_pct": p1["measured"],
        "e30/all_at_m8_upper_bound_pct_baseline": p1["baseline_pct"],
        "e30/all_at_m8_upper_bound_pct_e29_measured": p1["e29_measured_pct"],
        "e30/all_at_m8_upper_bound_pct_parent_side": p1["parent_side_pct"],
        "e30/p1_threshold_met": p1["threshold_met"],
        "e30/p1_student_point_error_pct": p1["student_point_error"],
        "e30/best_width": p1["best_width_now"],
        "e30/best_width_headroom_pct": p1["best_width_headroom_pct"],
        "e30/m8_to_m9_round_step_e29": p1["m8_to_m9_round_step_e29"],
        "e30/m8_to_m9_round_step_e30": p1["m8_to_m9_round_step_e30"],
        "e30/m9_per_row_vs_m8_pct_e29": p1["m9_per_row_vs_m8_pct_e29"],
        "e30/m9_per_row_vs_m8_pct_e30": p1["m9_per_row_vs_m8_pct_e30"],

        "e30/vbuild_m9_ms_e29": p2["e29_ms"],
        "e30/vbuild_m9_ms_e30": p2["e30_ms"],
        "e30/vbuild_m9_delta_ms": p2["measured"],
        "e30/vbuild_m9_delta_ms_student_point": p2["student_point"],
        "e30/vbuild_m9_flat_band_2sd_ms": p2["flat_band_2sd_ms"],
        "e30/vbuild_m9_is_flat": p2["is_flat"],
        "e30/p2_threshold_met": p2["threshold_met"],
        "e30/eval_wall_m9_delta_ms": p2["delta_eval_wall_m9_ms"],
        "e30/round_m9_delta_ms": p2["delta_round_m9_ms"],
        "e30/share_of_round_delta_in_vbuild_pct":
            p2["share_of_round_delta_in_vbuild_pct"],

        "e30/control_m8_round_ms_e29": ctl["m8_round_ms_e29"],
        "e30/control_m8_round_ms_e30": ctl["m8_round_ms_e30"],
        "e30/control_m8_delta_pct": ctl["m8_delta_pct"],
        "e30/control_m8_within_tolerance": ctl["m8_within_tolerance"],
        "e30/serial_ms_per_token_e29": ctl["serial_ms_per_token_e29"],
        "e30/serial_ms_per_token_e30": ctl["serial_ms_per_token_e30"],
        "e30/serial_delta_pct": ctl["serial_delta_pct"],
        "e30/serial_within_tolerance": ctl["serial_within_tolerance"],
        "e30/mtp_ms_per_token_e29": ctl["mtp_ms_per_token_e29"],
        "e30/mtp_ms_per_token_e30": ctl["mtp_ms_per_token_e30"],
        "e30/mtp_delta_pct": ctl["mtp_delta_pct"],
        "e30/histogram_unchanged": ctl["histogram_unchanged"],
        "e30/accepted_draft_total_e29": ctl["accepted_draft_total_e29"],
        "e30/accepted_draft_total_e30": ctl["accepted_draft_total_e30"],
        "e30/gate_qualified_for_timing": False,
        "e30/cool_gate_passed_real_gate": False,
    }
    for k, v in ctl["exactness_e30"].items():
        summary[f"e30/correctness/{k}"] = v

    rep = doc.get("repeat_dispersion")
    if rep:
        summary["e30/repeat_serial_ms_per_token_sd"] = rep[
            "serial_ms_per_token"]["sd"]
        summary["e30/repeat_mtp_ms_per_token_sd"] = rep[
            "mtp_ms_per_token"]["sd"]
        summary["e30/repeat_bound_M8_pct_sd"] = rep["bound_M8_pct"]["sd"]
        for width, stats in rep["per_width_round_mean_ms"].items():
            if stats["sd"] is not None:
                summary[f"e30/repeat_round_sd_ms/M{width}"] = stats["sd"]
        for width, stats in rep["per_width_vbuild_mean_ms"].items():
            if stats["sd"] is not None:
                summary[f"e30/repeat_vbuild_sd_ms/M{width}"] = stats["sd"]

    run.summary.update(summary)
    print(f"logged {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
