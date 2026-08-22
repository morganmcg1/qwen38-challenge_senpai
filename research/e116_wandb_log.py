#!/usr/bin/env python3
"""Publish one E116 session to W&B.

    usage: research/e116_wandb_log.py --session NAME --job-type TYPE
               [--artifact FILE ...] [--summary KEY=VALUE ...] [--note TEXT]

One run per session, group `e116-measured-transfer`, `harness=local` on every
config. Each named JSON artifact is logged whole into the run config under its
file stem and its scalar leaves are also flattened into the run summary, so a
number is readable from the run page without opening a file.

NOTHING HERE IS A SCORE. Every E116 leg is ungated by the standing
counterbalanced exception, so `cool_gate_passed_real_gate`,
`gate_qualified_for_timing` and `official_or_ranked_score` are logged false
verbatim on every run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e116-measured-transfer"
BASE_SHA = "67fedb4adb4cb0ec757f870ec8093617ca1e5620"
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
    args = ap.parse_args()

    config: dict[str, object] = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "session": args.session,
        "base_sha": BASE_SHA,
        "git_head": git_head(),
        "host": HOST,
        "note": args.note,
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
                     name=f"e116-{args.session}", config=config)
    run.summary.update(summary)
    for name in args.artifact:
        run.save(name, policy="now")
    print(f"wandb_run_id={run.id}")
    print(f"wandb_run_url={run.url}")
    run.finish()


if __name__ == "__main__":
    main()
