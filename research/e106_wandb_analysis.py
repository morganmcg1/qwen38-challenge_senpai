#!/usr/bin/env python3
"""Publish the E106 unforced census leg and its two reducers to W&B.

    usage:
      research/e106_wandb_analysis.py --tag TAG \
          --width-json research/out/e106/width-hist.json \
          --shape-json research/out/e106/shape-axis.json

The leg answers two questions that the forced census legs cannot. It records
the verify widths the shipped draft policy actually reaches, and it carries
the legal-shape warmup that runs every projection at rows 1 through 9.

A census leg is NEVER a timing leg. The census serialises every command
buffer, so host wall clock is invalid and only Metal's GPU clock counts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e106-dispatch-fixed-cost-law"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"


def meta(tag):
    path = pathlib.Path("research/out") / tag / "meta.txt"
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def score(tag):
    path = pathlib.Path("research/out") / tag / "score.json"
    return json.loads(path.read_text()).get("metrics", {}) if path.exists() \
        else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width-json", required=True)
    ap.add_argument("--shape-json", required=True)
    args = ap.parse_args()

    tag = args.tag
    width = json.loads(pathlib.Path(args.width_json).read_text())[tag]
    shape = json.loads(pathlib.Path(args.shape_json).read_text())[tag]
    info, metrics = meta(tag), score(tag)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type="census",
        name=f"e106-{tag}",
        config={
            "experiment": "e106-dispatch-fixed-cost-and-the-n5120-anomaly",
            "rung": 0, "host": HOST, "chip": "apple-m4-pro",
            "gpu_cores": 20, "memory_gib": 48, "os": "macos-26.5.2",
            "swift": "6.3.3", "reducer_commit": head,
            "timing_valid": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            **{f"leg_{k}": v for k, v in info.items()},
        },
    )

    hist = width["width_histogram"]
    run.log({
        "mean_verify_width": width["mean_verify_width"],
        "mtp_rounds": width["mtp_rounds"],
        "effect_b_pct_of_mtp_round": width["weighted_pct_of_round"],
        "effect_b_pct_of_serial_round": width["serial_pct_of_round"],
        "effect_b_ratio_gain_pct": width["ratio_gain_pct"],
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
        "mtp_decode_speedup": metrics.get("mtp_decode_speedup"),
        "width_histogram": wandb.Table(
            columns=["width", "rounds", "share"],
            data=[[int(w), n, n / sum(hist.values())]
                  for w, n in sorted(hist.items(), key=lambda kv: int(kv[0]))]),
        "per_width_prize": wandb.Table(
            columns=["width", "rounds", "share", "effect_b_us", "source",
                     "removable_us", "round_us", "pct_of_round"],
            data=[[r["width"], r["count"], r["share"], r.get("effect_b_us"),
                   r.get("source"), r.get("removable_us"), r.get("round_us"),
                   r.get("pct_of_round")] for r in width["per_width"]]),
    })

    for phase, block in shape.items():
        run.log({f"shape_axis_{phase}": wandb.Table(
            columns=["tensor", "N", "K", "rows", "threadgroups", "weight_mb",
                     "us", "gb_per_s", "law_us", "excess_pct", "x_kb",
                     "implied_miss_pct"],
            data=[[c["tensor"], c["N"], c["K"], c["rows"], c["threadgroups"],
                   c["weight_mb"], c["us"], c["gb_per_s"], c["law_us"],
                   c["excess_pct"], c["x_kb"], c.get("implied_miss_pct")]
                  for c in block["cells"]])})
        run.log({f"per_width_law_{phase}": wandb.Table(
            columns=["rows", "families", "F_us", "S_us_per_gb", "S_gb_per_s",
                     "r2"],
            data=[[int(m), law["n"], law["F_us"], law["S_us_per_gb"],
                   law["S_gb_per_s"], law["r2"]]
                  for m, law in sorted(block["fits"].items(),
                                       key=lambda kv: int(kv[0])) if law])})

    print(f"run_id={run.id}")
    print(f"url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
