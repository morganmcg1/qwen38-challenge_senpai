#!/usr/bin/env python3
"""Publish one E56 leg to W&B as soon as it finishes.

The leg's own trace is the mechanism evidence: a schedule change must move the
verify-width histogram, and a ratio that moves without the histogram moving is
a different effect wearing this experiment's name.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
from collections import Counter

import wandb

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e56-stream-aware-draft-depth-schedule"

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def parse_meta(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_trace(path: pathlib.Path) -> dict:
    """Width histogram of the MTP leg, split off the serial reference leg."""
    if not path.exists():
        return {}
    legs, current, last_round = [], [], -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROUND_RE.search(line)
        if not match:
            continue
        index, depth, accepted = (int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)))
        if index <= last_round and current:
            legs.append(current)
            current = []
        last_round = index
        current.append((depth, accepted))
    if current:
        legs.append(current)
    if not legs:
        return {}
    # The MTP leg is the one that drafts; the serial control never does.
    drafting = [leg for leg in legs if any(depth > 0 for depth, _ in leg)]
    leg = max(drafting or legs, key=len)
    widths = Counter(depth + 1 for depth, _ in leg)
    rounds = len(leg)
    drafted = sum(depth for depth, _ in leg)
    accepted = sum(count for _, count in leg)
    out = {
        "trace_rounds": rounds,
        "trace_drafted": drafted,
        "trace_accepted": accepted,
        "trace_mean_verify_width": sum(depth + 1 for depth, _ in leg) / rounds,
        "trace_accept_ratio": accepted / drafted if drafted else 0.0,
        "trace_non_drafting_rounds": widths.get(1, 0),
        "trace_legs_seen": len(legs),
    }
    for width in range(1, 10):
        out[f"trace_width_share_{width}"] = widths.get(width, 0) / rounds
        out[f"trace_width_count_{width}"] = widths.get(width, 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    out_dir = ROOT / "research" / "out" / args.tag
    meta = parse_meta(out_dir / "meta.txt")
    score_path = out_dir / "score.json"
    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}
    metrics = dict(score.get("metrics", {}))
    metrics["mtp_decode_speedup"] = score.get("score")
    metrics.update(parse_trace(out_dir / "trace.txt"))

    config = {
        "experiment": "qwen38-r1-e56-stream-aware-draft-depth-schedule",
        "arm": args.arm,
        "tag": args.tag,
        "host": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "cool_gate_env": os.environ.get("MLXFAST_LOCAL_COOL_GATE", "default"),
        **meta,
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=f"e56-{args.tag}", job_type="local-iterate",
                     config=config, reinit=True)
    run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    run.summary.update({k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool))})
    run.finish()
    print(f"e56_log_leg: logged {args.tag} ({args.arm}) as {run.id}")


if __name__ == "__main__":
    main()
