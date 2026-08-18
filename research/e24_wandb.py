#!/usr/bin/env python3
"""E24: publish the Phase 1 dispatch-cost microbenchmark to W&B.

One run per arm carrying the full N sweep as a step series, plus one analysis
run carrying the derived per-cast costs and the pre-registered decision.

usage:
  research/e24_wandb.py [--json research/results/e24-phase1.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
GROUP = "qwen38-r1-e24-constant-scalar-dispatch-tax"
BASE_SHA = "55c727e959e26cf24333d3e8c0896f7d97ab1224"
PR_NUMBER = 28

ARM_ROLE = {
    "A_homogeneous": "launch-floor",
    "B_cast_plus_filler": "cast-in-mixed-stream",
    "C_filler_only": "subtrahend-control",
}


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config() -> dict:
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_cores": sh("sysctl", "-n", "hw.ncpu"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_base_sha": BASE_SHA,
        "git_dirty_files": len(sh("git", "status", "--porcelain").splitlines()),
        "experiment": GROUP,
        "assignment_revision": "r1",
        "pr_number": PR_NUMBER,
        "phase": "phase1-microbenchmark",
        # This host is an M4 Pro; the ranked runner is M5. Every number here is
        # directional for the ranked host, decisive only for the local decision.
        "ranked_host": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/results/e24-phase1.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = json.loads(Path(args.json).read_text())
    base = host_config()
    base["mlx_device"] = report.get("device")
    base["reps"] = report.get("reps")
    base["warmup_reps"] = report.get("warmup_reps")

    if args.dry_run:
        print(json.dumps(report["projection"], indent=2, sort_keys=True))
        return 0

    import wandb

    for name, arm in sorted(report["arms"].items()):
        config = dict(base)
        config.update({"arm": name, "arm_role": ARM_ROLE.get(name, "unknown")})
        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            job_type="phase1-arm", name=f"e24-phase1-{name}",
            config=config, reinit=True,
        )
        for row in arm["rows"]:
            run.log({
                "n_units": row["n"],
                "dispatches": row["dispatches"],
                "build_seconds": row["build_seconds"],
                "eval_seconds": row["eval_seconds"],
                "eval_iqr_seconds": row["eval_iqr_seconds"],
                "total_seconds": row["total_seconds"],
            })
        run.summary.update({
            "slope_total_seconds_per_unit": arm["slope_total_seconds_per_unit"],
            "slope_total_microseconds_per_unit":
                arm["slope_total_seconds_per_unit"] * 1e6,
            "intercept_total_seconds": arm["intercept_total_seconds"],
            "r2_total": arm["r2_total"],
            "slope_eval_seconds_per_unit": arm["slope_eval_seconds_per_unit"],
            "r2_eval": arm["r2_eval"],
        })
        run.finish()

    proj = report["projection"]
    headline = proj["armB_minus_C_realistic"]
    threshold = report["prereg_threshold_pct"]
    config = dict(base)
    config.update({
        "arm": "analysis",
        "prereg_threshold_pct": threshold,
        "prereg_doc": "research/e24-prereg.md",
        **{f"model_{k}": v for k, v in report["model"].items()},
    })
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="phase1-analysis", name="e24-phase1-analysis",
        config=config, reinit=True,
    )
    summary = {}
    for label, block in proj.items():
        for key, value in block.items():
            summary[f"{label}_{key}"] = value
    summary.update({
        "headline_microseconds_per_cast": headline["microseconds_per_cast"],
        "headline_projected_pct_of_local_true_decode":
            headline["projected_pct_of_local_true_decode"],
        "prereg_threshold_pct": threshold,
        "passes_prereg_threshold":
            headline["projected_pct_of_local_true_decode"] >= threshold,
    })
    run.summary.update(summary)
    run.finish()
    print("logged", len(report["arms"]) + 1, "runs to", f"{ENTITY}/{PROJECT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
