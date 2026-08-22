#!/usr/bin/env python3
"""Publish one E122 session to W&B.

    usage: research/e122_wandb_log.py --session NAME --job-type TYPE
               [--artifact FILE ...] [--summary KEY=VALUE ...] [--note TEXT]

One run per session, group `e122-target-margin-conditioned-depth`,
`harness=local` on every config. Each named JSON artifact is logged whole into
the run config under its file stem, and its scalar leaves are flattened into
the run summary so a number is readable from the run page.

NOTHING HERE IS A SCORE. Rung 0 runs the per-round phase trace, which writes a
file inside the round, so no seconds figure from this session measures
anything. `timing_valid`, `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` are logged verbatim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e122-target-margin-conditioned-depth"
BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}/{key}", child, out)
    elif isinstance(value, (bool, int, float, str)):
        out[prefix] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--job-type", required=True)
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--summary", action="append", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--timing-valid", action="store_true")
    args = ap.parse_args()

    config: dict[str, object] = {
        "harness": "local",
        "experiment": "e122-target-margin-conditioned-draft-depth",
        "session": args.session,
        "base_sha": BASE_SHA,
        "git_head": git_head(),
        "host": HOST,
        "note": args.note,
        "timing_valid": bool(args.timing_valid),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }

    summary: dict[str, object] = {}
    for name in args.artifact:
        path = pathlib.Path(name)
        payload = json.loads(path.read_text())
        config[path.stem.replace("-", "_")] = payload
        flatten(path.stem.replace("-", "_"), payload, summary)
    for pair in args.summary:
        key, _, value = pair.partition("=")
        try:
            summary[key] = float(value)
        except ValueError:
            summary[key] = value

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     job_type=args.job_type,
                     name=f"e122-{args.session}", config=config)
    run.summary.update(summary)
    for name in args.artifact:
        run.save(name, policy="now")
    print(f"wandb_run_id={run.id}")
    print(f"wandb_run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
