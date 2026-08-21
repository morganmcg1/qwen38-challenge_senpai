#!/usr/bin/env python3
"""E92: publish the read-bandwidth roofline and the verify-width sweep to W&B.

usage:
  research/e92_wandb.py --rung1 PATH [--rung1-reversed PATH] [--rung2 PATH]
                        [--name NAME] [--notes TEXT]

`--rung1` takes the JSON written by `research/e92_bandwidth.py --output`, and
`--rung2` the JSON written by `research/e92_widths.py --output`.

Every leg is UNGATED, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` travel with the run verbatim. Nothing here is
a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e92-what-limits-the-target-verify-pass"

# Apple's published unified-memory bandwidth of the M4 Pro in this host.
SPEC_BANDWIDTH_GBS = 273.0

BW_COLS = ["source", "kind", "megabytes", "bytes", "gpu_busy_us_median",
           "gpu_busy_us_min", "gpu_busy_us_max", "wall_us_median",
           "achieved_bandwidth_gbs", "wall_bandwidth_gbs",
           "marginal_bytes", "marginal_gpu_busy_us", "marginal_bandwidth_gbs",
           "gpu_fraction_of_wall", "frac_of_spec_peak"]

LEG_COLS = ["tag", "M", "form", "order", "rounds_on_pin", "rounds_analysed",
            "pin_purity", "width_is_delta", "accepted_mean", "round_us",
            "round_gpu_busy_us", "round_gpu_idle_us", "verify_gpu_busy_us",
            "head_gpu_busy_us", "d_submit2_gpu_busy_us", "snapshot_gpu_busy_us",
            "verify_gpu_busy_us_clean", "verify_gpu_busy_us_stuck",
            "frac_rounds_host_stuck", "tiling_error_us", "host_thread_cpu_ns",
            "gpu_temp_entry_c", "gpu_temp_exit_c"]

WIDTH_COLS = ["M", "legs", "verify_gpu_busy_us", "head_gpu_busy_us",
              "round_gpu_busy_us", "round_gpu_idle_us", "round_us",
              "verify_share_of_round_busy", "head_share_of_round_busy",
              "closure_share", "G", "modelled_weight_bytes",
              "achieved_bandwidth_gbs", "ratio_to_peak", "implied_bytes",
              "implied_over_weight_stream", "implied_over_G_times_stream"]


def table(columns, rows) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def bandwidth_rows(doc: dict, source: str) -> list[dict]:
    rows = []
    for row in doc["rows"]:
        out = dict(row)
        out["source"] = source
        out["frac_of_spec_peak"] = row["achieved_bandwidth_gbs"] / SPEC_BANDWIDTH_GBS
        rows.append(out)
    return rows


def log_rung1(run, doc: dict, source: str, prefix: str) -> None:
    rows = bandwidth_rows(doc, source)
    run.log({f"{prefix}/bandwidth": table(BW_COLS, rows)})
    plateau = [r for r in rows if r["kind"] == "int32" and r["megabytes"] >= 64]
    scalars = {
        f"{prefix}/peak_achieved_bandwidth_gbs": doc["peak_achieved_bandwidth_gbs"],
        f"{prefix}/min_achieved_bandwidth_gbs": doc["min_achieved_bandwidth_gbs"],
        f"{prefix}/relative_spread": doc["relative_spread"],
        f"{prefix}/flat_within_10_percent": int(doc["flat_within_10_percent"]),
        f"{prefix}/frac_of_spec_peak": doc["peak_achieved_bandwidth_gbs"]
        / SPEC_BANDWIDTH_GBS,
        f"{prefix}/committed_buffers": doc["buffer_stats"]["committed_total"],
        f"{prefix}/invalid_buffers": doc["buffer_stats"]["invalid_total"],
    }
    if plateau:
        lo = min(r["achieved_bandwidth_gbs"] for r in plateau)
        hi = max(r["achieved_bandwidth_gbs"] for r in plateau)
        scalars[f"{prefix}/plateau_min_gbs"] = lo
        scalars[f"{prefix}/plateau_max_gbs"] = hi
        scalars[f"{prefix}/plateau_relative_spread"] = (hi - lo) / hi
        scalars[f"{prefix}/plateau_flat_within_10_percent"] = int(
            (hi - lo) / hi <= 0.10)
    bf16 = [r for r in rows if r["kind"] == "bfloat16"]
    if bf16:
        same = [r for r in rows
                if r["kind"] == "int32" and r["megabytes"] == bf16[0]["megabytes"]]
        scalars[f"{prefix}/bf16_gbs"] = bf16[0]["achieved_bandwidth_gbs"]
        if same:
            scalars[f"{prefix}/bf16_over_int32"] = (
                bf16[0]["achieved_bandwidth_gbs"]
                / same[0]["achieved_bandwidth_gbs"])
    run.log(scalars)
    for size in sorted({r["megabytes"] for r in rows if r["kind"] == "int32"}):
        row = next(r for r in rows
                   if r["kind"] == "int32" and r["megabytes"] == size)
        run.log({f"{prefix}/gbs_by_size/{size}mb": row["achieved_bandwidth_gbs"]})


def leg_rows(doc: dict) -> list[dict]:
    rows = []
    for leg in doc["legs"]:
        meta = leg["meta"]
        out = {k: leg.get(k) for k in LEG_COLS}
        out["tag"] = leg["tag"]
        out["M"] = leg["M"]
        out["form"] = meta.get("e92_form")
        out["order"] = leg["tag"][-1]
        out["gpu_temp_entry_c"] = float(meta["gpu_temp_entry_c"])
        out["gpu_temp_exit_c"] = float(meta["gpu_temp_exit_c"])
        out["width_is_delta"] = int(leg["width_is_delta"])
        rows.append(out)
    return rows


def log_rung2(run, doc: dict, prefix: str = "rung2") -> None:
    legs = leg_rows(doc)
    run.log({f"{prefix}/legs": table(LEG_COLS, legs)})
    run.log({f"{prefix}/width_histograms": table(
        ["tag", "M", "observed_width", "rounds"],
        [{"tag": leg["tag"], "M": leg["M"], "observed_width": int(width),
          "rounds": count}
         for leg in doc["legs"]
         for width, count in leg["width_histogram"].items()])})

    table_rows = []
    for row in doc["table"]:
        out = dict(row)
        out["legs"] = ",".join(row["legs"])
        verify = row.get("verify_share_of_round_busy") or 0.0
        head = row.get("head_share_of_round_busy") or 0.0
        out["closure_share"] = verify + head
        table_rows.append(out)
    run.log({f"{prefix}/by_width": table(WIDTH_COLS, table_rows)})

    scalars = {
        f"{prefix}/anchor_verify_gpu_busy_us": doc["anchor_verify_gpu_busy_us"],
        f"{prefix}/legs_with_delta_histogram": sum(
            1 for leg in doc["legs"] if leg["width_is_delta"]),
        f"{prefix}/legs_total": len(doc["legs"]),
        f"{prefix}/max_abs_tiling_error_us": max(
            abs(leg["tiling_error_us"]) for leg in doc["legs"]),
        f"{prefix}/min_pin_purity": min(leg["pin_purity"] for leg in doc["legs"]),
        f"{prefix}/entry_temp_spread_c": (
            max(leg["gpu_temp_entry_c"] for leg in legs)
            - min(leg["gpu_temp_entry_c"] for leg in legs)),
    }
    for row in table_rows:
        width = row["M"]
        scalars[f"{prefix}/verify_gpu_busy_us/M{width}"] = row["verify_gpu_busy_us"]
        scalars[f"{prefix}/head_gpu_busy_us/M{width}"] = row["head_gpu_busy_us"]
        scalars[f"{prefix}/closure_share/M{width}"] = row["closure_share"]
        for key in ("achieved_bandwidth_gbs", "ratio_to_peak", "implied_bytes",
                    "implied_over_weight_stream", "implied_over_G_times_stream"):
            if row.get(key) is not None:
                scalars[f"{prefix}/{key}/M{width}"] = row[key]
    run.log(scalars)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung1", required=True)
    ap.add_argument("--rung1-reversed")
    ap.add_argument("--rung2")
    ap.add_argument("--rung2-production")
    ap.add_argument("--peak-gbs", type=float)
    ap.add_argument("--name", default="e92-verify-pass-bandwidth")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    rung1 = json.loads(Path(args.rung1).read_text())
    meta = rung1["meta"]

    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "local_mode": "--local-iterate",
        "timed": False,
        "official_or_ranked_score": False,
        "cool_gate": 0,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "ranked_host": False,
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_bytes": int(meta["memory_bytes"]),
        "spec_bandwidth_gbs": SPEC_BANDWIDTH_GBS,
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "head_dir": meta.get("head_dir"),
        "rung1_sizes_mb": meta.get("e92_bandwidth_sizes_mb"),
        "rung1_reps": meta.get("e92_bandwidth_reps"),
        "rung1_probe": "sum(zeros([n]) + 1), read-only, GPU busy from the "
                       "E90 command-buffer ledger, model resident",
        "peak_gbs_used_for_rung2": args.peak_gbs,
    }
    if args.rung2:
        rung2 = json.loads(Path(args.rung2).read_text())
        first = rung2["legs"][0]["meta"]
        config.update({
            "decode_tokens": int(first["tokens"]),
            "offered_depth": 8,
            "pin_mechanism": "MLX_E92_PIN_DRAFTS, clamped to the offer and to "
                             "Qwen36MTPLimits.maxDepth, bypassing the cost model",
            "rung2_legs": [leg["tag"] for leg in rung2["legs"]],
            "rung2_order": "ascending width 1..9, then descending 9..1 (ABBA)",
        })
    else:
        rung2 = None

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name,
        job_type="diagnostic-session", notes=args.notes, config=config)

    log_rung1(run, rung1, "forward", "rung1")
    if args.rung1_reversed:
        reversed_doc = json.loads(Path(args.rung1_reversed).read_text())
        log_rung1(run, reversed_doc, "reversed", "rung1_reversed")
    if rung2:
        log_rung2(run, rung2)
    if args.rung2_production:
        log_rung2(run, json.loads(Path(args.rung2_production).read_text()),
                  "rung2_production")

    print(run.url)
    print(run.id)
    run.finish()


if __name__ == "__main__":
    main()
