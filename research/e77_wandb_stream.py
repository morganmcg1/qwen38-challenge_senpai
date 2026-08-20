#!/usr/bin/env python3
"""Log every E77 rung-1 leg to W&B as the leg finishes, not at session end.

`e77_sweep` writes one `LEG` line per timed leg to stdout and flushes it. This
reads that stream and calls `wandb.log` per line, so a session that dies
mid-way still leaves every completed leg on the board. The lines are echoed
unchanged so the session log keeps the same text.

The sweep's x-axis is the per-thread register count, which only the offline
oracle knows, so `--regs` attaches the rung-0 census to every leg. A leg
therefore carries its own `registers` and `frame_bytes` and needs no join.

Line kinds:
  ARM     one per compiled arm: partition facts and the pipeline occupancy
  PARITY  one per (shape, arm): differing bf16 elements and empty rows
  LEG     one per timed leg
  SHAPE   one per shape: inner count and entry/exit GPU temperature

  e77_sweep ... | python3 research/e77_wandb_stream.py --name NAME
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
LOCAL_ARCH = "applegpu_g16s"


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
    ap.add_argument("--group", default="e77-rung1")
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--regs", type=pathlib.Path)
    ap.add_argument("--tags", nargs="*", default=[])
    args = ap.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    regs: dict[str, dict] = {}
    if args.regs:
        census = json.loads(args.regs.read_text())
        regs = {key.removeprefix("e77_"): row
                for key, row in census["sweep"][LOCAL_ARCH].items()}
        config["rung0_registers_local"] = {
            arm: row["registers"] for arm, row in regs.items()}
        config["rung0_source_sha256"] = census["sweep_source_sha256"]

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     group=args.group, job_type="e77-occupancy",
                     tags=["e77", "qmv-crossrow", "occupancy", "registers",
                           "rung1", *args.tags],
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
            row = regs.get(str(fields["arm"]), {})
            fields["registers"] = row.get("registers")
            fields["frame_bytes"] = row.get("spill_bytes")
            arms.append(fields)
        elif kind == "PARITY":
            parity.append(fields)
            run.log({f"parity/{fields['shape']}/{fields['arm']}/differing":
                     fields["differing"],
                     f"parity/{fields['shape']}/{fields['arm']}/zero_rows":
                     fields["zero_rows"]}, step=step)
        elif kind == "SHAPE":
            shapes.append(fields)
            run.log({f"thermal/{fields['shape']}/entry_gpu_temp_c":
                     fields["entry_gpu_temp_c"],
                     f"thermal/{fields['shape']}/exit_gpu_temp_c":
                     fields["exit_gpu_temp_c"]}, step=step)
        elif kind == "LEG":
            shape, arm = fields["shape"], fields["arm"]
            row = regs.get(str(arm), {})
            record = {
                f"leg/{shape}/{arm}/seconds_per_dispatch":
                    fields["seconds_per_dispatch"],
                f"leg/{shape}/{arm}/gbps": fields["gbps"],
                f"leg/{shape}/{arm}/gpu_seconds": fields["gpu_seconds"],
                "leg/rep": fields["rep"],
                "leg/position": fields["position"],
                "leg/m": fields["m"],
                "leg/ipg": fields["ipg"],
                "leg/groups": fields["groups"],
                "leg/pressure": fields["pressure"],
                "leg/kind_is_control": int(fields["kind"] == "q"),
            }
            if row:
                record["leg/registers"] = row["registers"]
                record["leg/frame_bytes"] = row["spill_bytes"]
                record[f"leg/{shape}/registers"] = row["registers"]
            run.log(record, step=step)
            step += 1

    run.summary["arms"] = arms
    run.summary["parity"] = parity
    run.summary["shape_thermal"] = shapes
    run.summary["legs_logged"] = step
    run.summary["parity_bit_identical"] = all(
        entry["differing"] == 0 for entry in parity)
    run.summary["parity_no_empty_rows"] = all(
        entry["zero_rows"] == 0 for entry in parity)
    print(f"e77_wandb_stream: run_id={run.id} url={run.url} legs={step}",
          file=sys.stderr)
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
