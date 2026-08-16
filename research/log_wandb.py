#!/usr/bin/env python3
"""Research-only: publish one --local-iterate arm (or the pooled curve) to W&B.

Every measurement this experiment reports has to be reachable from a run URL,
so each arm gets its own run and the pooled depth curve gets one summary run
that carries the table the cost model is derived from.

usage:
  research/log_wandb.py arm  research/out/<arm> [--group G] [--notes ...]
  research/log_wandb.py curve research/out --arms d0 d1 ... [--group G]
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_cost_curve import (PHASES, load_legs, read_meta,  # noqa: E402
                             summarize)

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
CURVE_COLUMNS = ["depth", "n", "c_us", "median_us", "stddev_us", "marginal_us",
                 "h", "c_over_c0", "us_per_token", "eval_wall_us", "host_us"]


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def host_config():
    return {
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_cores": sh("sysctl", "-n", "hw.ncpu"),
        "host_memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "host_os": platform.mac_ver()[0],
        "swift_version": sh("swift", "--version").splitlines()[0] if sh("swift", "--version") else "",
        "git_head": sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain")),
    }


def flatten_score(score):
    if not score:
        return {}
    out = {"directional_score": score.get("score")}
    for key, value in (score.get("metrics") or {}).items():
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
    return out


def log_arm(args):
    arm_dir = args.path
    meta = read_meta(arm_dir)
    score_path = arm_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else None
    legs = load_legs(arm_dir, args.warmup)

    config = dict(host_config())
    config.update({f"arm_{k}": v for k, v in meta.items()})
    config.update({
        "experiment": "qwen38-r1-e1-depth-cost-curve",
        "arm": arm_dir.name,
        "warmup_rounds_dropped": args.warmup,
        "local_mode": "local-iterate",
    })

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name or arm_dir.name,
                     group=args.group, job_type="local-iterate", config=config,
                     notes=args.notes, tags=["qwen38-r1-e1", "depth-cost-curve"])

    summary = flatten_score(score)
    for leg_index, leg in enumerate(legs):
        rows = leg["steady_rows"]
        tag = f"leg{leg_index}_d{'_'.join(map(str, leg['depths_seen']))}"
        summary[f"{tag}/rounds_total"] = leg["rounds_total"]
        summary[f"{tag}/dropped_warmup"] = leg["dropped_warmup"]
        summary[f"{tag}/dropped_partial"] = leg["dropped_partial"]
        summary[f"{tag}/acc_mean"] = leg["acc_mean"]
        if leg["begin"]:
            summary[f"{tag}/seed_build_us"] = leg["begin"]["build_us"]
            summary[f"{tag}/seed_eval_wall_us"] = leg["begin"]["eval_wall_us"]
        stats = summarize(rows)
        if stats:
            for key, value in stats.items():
                summary[f"{tag}/round_{key}"] = value
            for phase in PHASES:
                summary[f"{tag}/{phase}_mean"] = summarize(rows, phase)["mean_us"]
        # Per-round series so a thermal drift or a warm-up tail is visible.
        for row in rows:
            wandb.log({f"{tag}/round_us": row["round_us"],
                       f"{tag}/acc": row["acc"], f"{tag}/d": row["d"],
                       **{f"{tag}/{p}": row[p] for p in PHASES}},
                      step=row["round"])
    run.summary.update(summary)
    print(f"{arm_dir.name}: {run.url}")
    run.finish()


def log_curve(args):
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "depth_cost_curve.py"),
         str(args.path), "--warmup", str(args.warmup),
         "--json", str(args.path / "curve.json")]
        + (["--arms", *args.arms] if args.arms else []),
        capture_output=True, text=True)
    print(proc.stdout, proc.stderr, sep="\n")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    data = json.loads((args.path / "curve.json").read_text())
    config = dict(host_config())
    config.update({"experiment": "qwen38-r1-e1-depth-cost-curve",
                   "arm": "curve", "warmup_rounds_dropped": args.warmup,
                   "c0_us": data["c0_us"],
                   "arms_pooled": sorted(data["arms"])})
    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name or "depth-cost-curve",
                     group=args.group, job_type="analysis", config=config,
                     notes=args.notes, tags=["qwen38-r1-e1", "depth-cost-curve"])

    table = wandb.Table(columns=CURVE_COLUMNS)
    summary = {"c0_us": data["c0_us"]}
    for depth_key in sorted(data["curve"], key=int):
        entry = data["curve"][depth_key]
        stats = entry["steady"]
        table.add_data(entry["depth"], stats["n"], stats["mean_us"],
                       stats["median_us"], stats["stddev_us"],
                       entry["marginal_us"], entry["h"], entry["c_over_c0"],
                       entry["us_per_token"],
                       entry["phases"]["eval_wall_us"]["mean_us"],
                       entry["host_us"])
        d = entry["depth"]
        summary.update({
            f"curve/C_{d}_us": stats["mean_us"],
            f"curve/n_{d}": stats["n"],
            f"curve/stddev_{d}_us": stats["stddev_us"],
            f"curve/marginal_{d}_us": entry["marginal_us"],
            f"curve/h_{d}": entry["h"],
            f"curve/us_per_token_{d}": entry["us_per_token"],
        })
    run.log({"depth_cost_curve": table})
    run.summary.update(summary)
    artifact = wandb.Artifact("depth-cost-curve", type="analysis")
    artifact.add_file(str(args.path / "curve.json"))
    run.log_artifact(artifact)
    print(f"curve: {run.url}")
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["arm", "curve"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    (log_arm if args.kind == "arm" else log_curve)(args)


if __name__ == "__main__":
    main()
