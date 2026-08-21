#!/usr/bin/env python3
"""Publish the E96 GDN recurrent-step ablation to W&B.

    usage:
      research/e96_wandb_log.py legs TAG [TAG ...]
      research/e96_wandb_log.py model --json research/e96-rung1.json

`legs` writes one run per measured leg: the full identity tuple, the local
score metrics, the host thermal record, and the per-round trace statistics of
every `(d, acc)` bucket that leg produced.

`model` writes the analysis run: the arm ladder, the measured cost of the
recurrent step, the fixed-versus-per-timestep split, and the comparison against
the modelled 8,112.6 us per round.

Every E96 leg runs with the local cool gate disabled inside one
counterbalanced session, so `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged false verbatim on every run. The `off`
and `t1` arms emit unverified tokens by construction, so `tokens_verified` is
logged per arm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e96_report as report  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e96-gated-deltanet-recurrent-step"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

# The brief's modelled itemisation of the width-independent term `a` at M = 5.
MODELLED = {
    "gdn_recurrent_step_us_per_round": 8112.6,
    "gdn_family_us_per_round": 9246.3,
    "a_fixed_us_per_round": 10920.0,
    "local_beagle_round_us": 131024.0,
    "state_bytes_per_round": 301.99e6,
    "dram_bound_us_per_round": 1139.6,
}


def git_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def leg_config(leg):
    metrics = leg["score"]
    return {
        "experiment": "e96",
        "harness": "local",
        "local_mode": "--local-iterate",
        "host": HOST,
        "hostname": leg.get("hostname"),
        "base_sha": leg.get("base_sha"),
        "worker_sha256": leg.get("worker_sha256"),
        "head_dir": leg.get("head_dir"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
        "tokens": int(leg["tokens"]) if leg["tokens"] else None,
        "step_mode": leg["step_mode"],
        "threadgroup_x": 32,
        "threadgroup_y": int(leg["tg_y"]) if leg["tg_y"] else None,
        "grid": "32 x 128 x 48",
        "forced_drafts": leg["force_drafts"],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "timing_valid_within_session_only": True,
        "tokens_verified": bool(metrics.get("all_tokens_matched")),
    }


def leg_metrics(leg):
    metrics = leg["score"]
    out = {
        "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": metrics.get("serial_seconds_per_token"),
        "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "all_tokens_matched": int(bool(metrics.get("all_tokens_matched"))),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "public_drift_tripwire_passed": int(
            bool(metrics.get("public_drift_tripwire_passed"))
        ),
        "rounds_kept": leg["rounds_kept"],
        "gpu_temp_entry_c": float(leg["gpu_temp_entry_c"] or "nan"),
        "gpu_temp_exit_c": float(leg["gpu_temp_exit_c"] or "nan"),
    }
    for (d, acc), records in leg["buckets"].items():
        prefix = f"bucket_d{d}_acc{acc}"
        for field in report.FIELDS:
            values = [r[field] for r in records if field in r]
            if not values:
                continue
            out[f"{prefix}_{field}_mean"] = statistics.mean(values)
            out[f"{prefix}_{field}_median"] = statistics.median(values)
            if len(values) > 1:
                out[f"{prefix}_{field}_sd"] = statistics.stdev(values)
        out[f"{prefix}_n"] = len(records)
    return out


def enrich(leg, tag):
    meta = report.parse_meta(tag)
    leg["hostname"] = meta.get("host")
    leg["base_sha"] = meta.get("base_sha")
    leg["worker_sha256"] = meta.get("worker_sha256")
    leg["head_dir"] = meta.get("head_dir")
    return leg


def log_legs(tags):
    for tag in tags:
        leg = enrich(report.summarise(tag), tag)
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            job_type="ablation-leg",
            name=f"e96-{tag}",
            config=leg_config(leg),
            reinit=True,
        )
        run.log(leg_metrics(leg))
        run.finish()


def log_model(path):
    payload = json.loads(pathlib.Path(path).read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="ablation-model",
        name="e96-rung1-model",
        config={
            "experiment": "e96",
            "harness": "local",
            "host": HOST,
            "candidate_sha": git_sha(),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "modelled": MODELLED,
            **payload.get("config", {}),
        },
        reinit=True,
    )
    run.log(payload["metrics"])
    for name, table in payload.get("tables", {}).items():
        run.log({name: wandb.Table(columns=table["columns"], data=table["rows"])})
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    legs = sub.add_parser("legs")
    legs.add_argument("tags", nargs="+")
    model = sub.add_parser("model")
    model.add_argument("--json", required=True)
    args = parser.parse_args()
    if args.command == "legs":
        log_legs(args.tags)
    else:
        log_model(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
