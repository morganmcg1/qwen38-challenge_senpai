#!/usr/bin/env python3
"""E94: publish one leg session to W&B.

usage:
  research/e94_wandb.py --legs research/e94-artifacts/rung1.json
                        --name e94-rung1-cap-sweep [--notes TEXT]

`--legs` takes the JSON written by `research/e94_legs.py --out`. Every leg is
UNGATED, so `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false` and `official_or_ranked_score=false` travel
with the run verbatim. Nothing here is a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e94-depth-price-cliff-guard"

LEG_COLS = [
    "tag", "arm", "cap", "order", "rounds", "tokens_emitted",
    "mean_chosen_depth", "median_chosen_depth", "mean_verify_width",
    "mean_accepted_per_round", "mean_tokens_per_round",
    "effective_mean_draft_len", "accepted_draft_rate", "all_tokens_matched",
    "mtp_seconds_per_token", "serial_seconds_per_token", "mtp_decode_speedup",
    "round_us_per_token", "mean_round_us", "median_round_us",
    "first_round_us", "gpu_temp_entry_c", "gpu_temp_exit_c",
    "worker_sha256", "started", "finished",
]

HIST_COLS = [
    "tag", "arm", "cap", "depth", "verify_width", "rounds", "fraction",
    "mean_accepted", "median_round_us", "mean_round_us", "round_us_share",
    "tokens_emitted",
]


def leg_row(leg: dict) -> dict:
    meta, score = leg["meta"], leg["score"]

    def temp(key):
        value = meta.get(key)
        return float(value) if value else None

    return {
        "tag": leg["tag"],
        "arm": meta.get("e94_arm"),
        "cap": int(meta["e94_cap"]) if meta.get("e94_cap") else None,
        "order": meta.get("e94_order"),
        "rounds": leg["rounds"],
        "tokens_emitted": leg["tokens_emitted"],
        "mean_chosen_depth": leg["mean_chosen_depth"],
        "median_chosen_depth": leg["median_chosen_depth"],
        "mean_verify_width": leg["mean_verify_width"],
        "mean_accepted_per_round": leg["mean_accepted_per_round"],
        "mean_tokens_per_round": leg["mean_tokens_per_round"],
        "effective_mean_draft_len": score["effective_mean_draft_len"],
        "accepted_draft_rate": score["accepted_draft_rate"],
        "all_tokens_matched": score["all_tokens_matched"],
        "mtp_seconds_per_token": score["mtp_seconds_per_token"],
        "serial_seconds_per_token": score["serial_seconds_per_token"],
        "mtp_decode_speedup": score["mtp_decode_speedup"],
        "round_us_per_token": leg["round_us_per_token"],
        "mean_round_us": leg["mean_round_us"],
        "median_round_us": leg["median_round_us"],
        "first_round_us": leg["first_round_us"],
        "gpu_temp_entry_c": temp("gpu_temp_entry_c"),
        "gpu_temp_exit_c": temp("gpu_temp_exit_c"),
        "worker_sha256": meta.get("worker_sha256"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
    }


def hist_rows(leg: dict) -> list[dict]:
    meta = leg["meta"]
    rows = []
    for depth, cell in leg["depth_histogram"].items():
        row = {
            "tag": leg["tag"],
            "arm": meta.get("e94_arm"),
            "cap": int(meta["e94_cap"]) if meta.get("e94_cap") else None,
            "depth": int(depth),
        }
        row.update({k: cell[k] for k in
                    ["verify_width", "rounds", "fraction", "mean_accepted",
                     "median_round_us", "mean_round_us", "round_us_share",
                     "tokens_emitted"]})
        rows.append(row)
    return rows


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--group", default=EXPERIMENT)
    args = ap.parse_args()

    doc = json.loads(Path(args.legs).read_text())
    legs = doc["legs"]
    first = legs[0]["meta"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        group=args.group,
        job_type="local-iterate-sweep",
        notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "assignment_pr": 97,
            "student": "qwen-edward",
            "harness": "research/e79_trace_leg.sh via research/e94_*",
            "local_mode": "--local-iterate",
            "decode_tokens": first.get("tokens"),
            "base_sha": first.get("base_sha"),
            "branch_commit": first.get("branch_commit"),
            "host": first.get("host"),
            "chip": first.get("chip"),
            "memory_bytes": first.get("memory_bytes"),
            "head_dir": first.get("head_dir"),
            "head_provenance_sha256": legs[0]["score"].get(
                "head_provenance_sha256"),
            "worker_sha256": first.get("worker_sha256"),
            "cli_sha256": first.get("cli_sha256"),
            "metallib_source_fingerprint": first.get(
                "metallib_source_fingerprint"),
            "sandbox": False,
            "trace": True,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "legs": [leg["tag"] for leg in legs],
        },
    )

    rows = [leg_row(leg) for leg in legs]
    run.log({"legs": table(LEG_COLS, rows)})
    run.log({"depth_histogram": table(
        HIST_COLS, [r for leg in legs for r in hist_rows(leg)])})

    for row in rows:
        prefix = f"leg/{row['tag']}"
        run.log({
            f"{prefix}/mean_chosen_depth": row["mean_chosen_depth"],
            f"{prefix}/mean_verify_width": row["mean_verify_width"],
            f"{prefix}/effective_mean_draft_len":
                row["effective_mean_draft_len"],
            f"{prefix}/accepted_draft_rate": row["accepted_draft_rate"],
            f"{prefix}/mtp_seconds_per_token": row["mtp_seconds_per_token"],
            f"{prefix}/round_us_per_token": row["round_us_per_token"],
            f"{prefix}/rounds": row["rounds"],
        })

    print(f"wandb_run_id={run.id}")
    print(f"wandb_run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
