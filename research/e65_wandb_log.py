#!/usr/bin/env python3
"""Log one E65 leg, or a summary object, to W&B.

Every leg is logged as soon as it closes, not once at session end. Alongside
the usual leg record this logs the E65 instrument itself: the per-round census
of the leg's own trace, the kL >= 1024 crossing probe, and the per-(M, repaired)
cell table, so the round-latency evidence is reproducible from W&B alone.

  research/e65_wandb_log.py --leg research/out/e65-r0-01-census512a \
      --position 1 --session r0
  research/e65_wandb_log.py --summary research/e65-artifacts/rung0.json \
      --summary-key rung0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

import e65_round_census as census

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e65-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

# Ledger 202: the adjacent-only 0.0629 % null floor is retracted. Take the
# LARGEST same-arm spread inside the session that measures the effect. These
# are this student's own E60 within-session same-arm spreads, kept only as the
# prior expectation for how the floor grows with leg separation.
NULL_FLOOR_BY_SEPARATION_PERCENT = {1: 0.0032, 3: 0.1147, 5: 0.1634}
LIVE_PROMOTED_FRONTIER_SCORE = 3.25187972017987
OUR_BEST_OFFICIAL_SCORE = 3.23250848263467
RANKED_RUN_PUBLISHED_SCORE_SD_PERCENT = 0.756
RANKED_CANDIDATE_LEG_SD_PERCENT = 1.165


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e65-cold-kernel-first-touch-census",
        "revision_id": "r1",
        "pr_number": 68,
        "assignment_base_sha": "3bf0e1f20fcdcc9b90d6e5ded52329bf74e4b52c",
        "merged_base_sha": "45b4f3a800f879e3579ca27ef0b1c0ef40e4473d",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_gpu_architecture": "applegpu_g16s",
        "host_mlx_devc": "s",
        "host_mlx_arch_gen": 16,
        "sdpa_two_pass_reachable_locally": True,
        "harness": "local",
        "local_mode": "--local-iterate",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "null_floor_by_separation_percent": NULL_FLOOR_BY_SEPARATION_PERCENT,
        "live_promoted_frontier_score": LIVE_PROMOTED_FRONTIER_SCORE,
        "our_best_official_score": OUR_BEST_OFFICIAL_SCORE,
        "ranked_run_published_score_sd_percent":
            RANKED_RUN_PUBLISHED_SCORE_SD_PERCENT,
        "ranked_candidate_leg_sd_percent": RANKED_CANDIDATE_LEG_SD_PERCENT,
    }


def resume_run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id or None,
        resume="allow",
        name="e65-cold-kernel-first-touch-census",
        job_type="round-latency-census",
        config=identity(),
        tags=["e65", "qwen-alphonse", "local", "warm-path", "sdpa", "census"],
    )
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


def read_meta(leg: pathlib.Path) -> dict:
    meta: dict[str, str] = {}
    path = leg / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def _float(meta: dict, key: str):
    value = meta.get(key)
    try:
        return float(value) if value else None
    except ValueError:
        return None


def census_record(leg: pathlib.Path) -> dict:
    """Run the E65 round census over this leg's own trace."""
    trace = leg / "trace.txt"
    if not trace.exists() or trace.stat().st_size == 0:
        return {}
    sessions = census.parse(str(trace))
    best = None
    for index, session in enumerate(sessions):
        rows = census.annotate(session)
        if len(rows) < 20 or all(r["d"] == 0 for r in rows):
            continue
        (stats, outliers, singular,
         small, pooled_iqr) = census.cell_outliers(rows, 3.0)
        leg_us = sum(r["round_us"] for r in rows)
        timed_leg_us = (leg_us + session["begin_build_us"]
                        + session["begin_eval_wall_us"])
        excess = sum(o["excess_us"] for o in outliers)
        small_excess = sum(o["excess_us"] for o in small)
        entry = {
            "census_session_index": index,
            "census_rounds": len(rows),
            "census_tokens": sum(1 + r["acc"] for r in rows),
            "census_round_series_us": leg_us,
            "census_timed_leg_us": timed_leg_us,
            "census_begin_build_us": session["begin_build_us"],
            "census_begin_eval_wall_us": session["begin_eval_wall_us"],
            "census_kL_max": max(r["kL_verify"] for r in rows),
            "census_rounds_kL_ge_1024": sum(
                1 for r in rows if r["crosses_two_pass"]),
            "census_outlier_count": len(outliers),
            "census_outlier_excess_us": excess,
            "census_outlier_excess_pct_of_leg": (
                100.0 * excess / timed_leg_us if timed_leg_us else 0.0),
            "census_pooled_iqr_us": pooled_iqr,
            "census_small_cell_outlier_count": len(small),
            "census_small_cell_excess_us": small_excess,
            "census_combined_excess_pct_of_leg": (
                100.0 * (excess + small_excess) / timed_leg_us
                if timed_leg_us else 0.0),
            "census_cells": stats,
            "census_singular_cells": sorted(singular),
            "census_structural_events": census.structural_events(rows),
            "census_crossing_probe": census.crossing_probe(rows),
            "census_segment_totals_us": {
                s: sum(r[s] for r in rows) for s in census.SEGMENTS},
        }
        probes = entry["census_crossing_probe"]
        if probes and probes[0].get("peers"):
            p = probes[0]
            entry["crossing_round"] = p["round"]
            entry["crossing_M"] = p["M"]
            entry["crossing_kL"] = p["kL_verify"]
            entry["crossing_is_last_round"] = p["is_last_round"]
            entry["crossing_round_us"] = p["round_us"]
            entry["crossing_peer_median_us"] = p["peer_median_us"]
            entry["crossing_peer_spread_us"] = p["peer_spread_us"]
            entry["crossing_excess_us"] = p["excess_us"]
            entry["crossing_excess_in_peer_spreads"] = (
                p["excess_in_peer_spreads"])
            entry["crossing_excess_pct_of_leg"] = (
                100.0 * p["excess_us"] / timed_leg_us if timed_leg_us else 0.0)
            for segment, value in p["segment_excess_us"].items():
                entry[f"crossing_excess_{segment}_us"] = value
        for o in small:
            if o["round"] != 1:
                continue
            entry["first_round_cell"] = o["cell"]
            entry["first_round_us"] = o["round_us"]
            entry["first_round_cell_median_us"] = o["cell_median_us"]
            entry["first_round_excess_us"] = o["excess_us"]
            entry["first_round_excess_in_pooled_spreads"] = o["excess_iqr"]
            entry["first_round_excess_pct_of_leg"] = (
                100.0 * o["excess_us"] / timed_leg_us if timed_leg_us else 0.0)
            for segment, value in o["segment_excess_us"].items():
                entry[f"first_round_excess_{segment}_us"] = value
        best = entry
        break
    return best or {}


def leg_record(leg: pathlib.Path) -> dict:
    meta = read_meta(leg)
    record: dict = {
        "tag": leg.name,
        "arm": meta.get("label") or meta.get("arm", "?"),
        "binary_arm": meta.get("arm", "?"),
        "tokens": int(meta.get("tokens", 0) or 0),
        "exit": int(meta.get("exit", -1) or -1),
        "trace": meta.get("trace") == "1",
        "sync_head": meta.get("sync_head") == "1",
        "gpu_temp_entry_c": _float(meta, "gpu_temp_entry_c"),
        "gpu_temp_exit_c": _float(meta, "gpu_temp_exit_c"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "staged_worker_sha256": meta.get("staged_worker_sha256"),
        "staged_cli_sha256": meta.get("staged_cli_sha256"),
        "session_sha256": meta.get("session_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "nm_symbols": int(meta.get("nm_symbols", 0) or 0),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
    }
    for key, value in meta.items():
        if key.startswith("nm_require:") or key.startswith("nm_forbid:"):
            record[key] = value

    score_path = leg / "score.json"
    if not score_path.exists():
        record["failed"] = True
        err = leg / "wrapper.err"
        if err.exists():
            record["error_tail"] = err.read_text().splitlines()[-25:]
    else:
        metrics = json.loads(score_path.read_text())["metrics"]
        record["failed"] = False
        for key in (
            "decode_tokens", "mtp_seconds_per_token", "serial_seconds_per_token",
            "mtp_decode_speedup", "effective_mean_draft_len",
            "accepted_draft_rate", "all_tokens_matched",
            "residual_divergence_count", "mtp_depth",
            "public_drift_tripwire_passed", "head_provenance_sha256",
        ):
            if key in metrics:
                record[key] = metrics[key]

    record.update(census_record(leg))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg")
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--session", default="")
    parser.add_argument("--summary")
    parser.add_argument("--summary-key", default="")
    args = parser.parse_args()

    run = resume_run()
    if args.leg:
        leg = pathlib.Path(args.leg)
        record = leg_record(leg)
        record["leg_position"] = args.position
        record["session"] = args.session
        payload = {
            f"leg/{key}": value
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        payload["leg/tag"] = record["tag"]
        payload["leg/arm"] = record["arm"]
        payload["leg/session"] = args.session
        run.log(payload)
        run.summary[f"legs/{record['tag']}"] = record
        print(json.dumps(record, indent=2, default=str))
    if args.summary:
        summary = json.loads(pathlib.Path(args.summary).read_text())
        if args.summary_key:
            summary = {f"{args.summary_key}/{k}": v for k, v in summary.items()}
        run.summary.update(summary)
        print(json.dumps(summary, indent=2, default=str)[:4000])
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
