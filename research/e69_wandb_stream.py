#!/usr/bin/env python3
"""Log every E69 rung-1 leg to W&B as the leg finishes, not at session end.

`research/e69_cell_ab` writes one `LEG` line per timed leg to stdout and flushes
it. This reads that stream and calls `wandb.log` per line, so a session that
dies mid-way still leaves every completed leg on the board. The lines are echoed
unchanged so the session log keeps the same text.

Line kinds:
  ARM     one per compiled arm: occupancy and threadgroup-memory facts
  PARITY  one per (shape, arm): differing bf16 elements against `plain`
  LEG     one per timed leg
  SHAPE   one per shape: inner count and entry/exit GPU temperature

  research/e69_cell_ab ... | python3 research/e69_wandb_stream.py --name NAME
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def parse(line: str) -> tuple[str, dict[str, object]] | None:
    parts = line.split()
    if not parts or parts[0] not in {"ARM", "PARITY", "LEG", "SHAPE"}:
        return None
    fields: dict[str, object] = {}
    for token in parts[1:]:
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        try:
            fields[key] = int(value)
        except ValueError:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
    return parts[0], fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="e69-rung1")
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--tags", nargs="*", default=[])
    args = ap.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     group=args.group, job_type="e69-cell",
                     tags=["e69", "qmv-wide", "rung1", *args.tags],
                     config=config)

    step = 0
    arms: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    shapes: list[dict[str, object]] = []
    for raw in sys.stdin:
        sys.stdout.write(raw)
        sys.stdout.flush()
        parsed = parse(raw.strip())
        if parsed is None:
            continue
        kind, fields = parsed
        if kind == "ARM":
            arms.append(fields)
        elif kind == "PARITY":
            parity.append(fields)
            run.log({f"parity/{fields['shape']}/{fields['arm']}":
                     fields["differing"]}, step=step)
        elif kind == "SHAPE":
            shapes.append(fields)
            run.log({f"thermal/{fields['shape']}/entry_gpu_temp_c":
                     fields["entry_gpu_temp_c"],
                     f"thermal/{fields['shape']}/exit_gpu_temp_c":
                     fields["exit_gpu_temp_c"]}, step=step)
        elif kind == "LEG":
            shape, arm = fields["shape"], fields["arm"]
            run.log({
                f"leg/{shape}/{arm}/seconds_per_dispatch":
                    fields["seconds_per_dispatch"],
                f"leg/{shape}/{arm}/gbps": fields["gbps"],
                f"leg/{shape}/{arm}/gpu_seconds": fields["gpu_seconds"],
                "leg/rep": fields["rep"],
                "leg/position": fields["position"],
                "leg/na": fields["na"],
            }, step=step)
            step += 1

    run.summary["arms"] = arms
    run.summary["parity"] = parity
    run.summary["shape_thermal"] = shapes
    run.summary["legs_logged"] = step
    run.summary["parity_bit_identical"] = all(
        entry["differing"] == 0 for entry in parity)
    print(f"e69_wandb_stream: run_id={run.id} url={run.url} legs={step}",
          file=sys.stderr)
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
