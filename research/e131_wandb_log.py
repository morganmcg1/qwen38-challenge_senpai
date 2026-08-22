#!/usr/bin/env python3
"""Publish the E131 occupancy census to W&B.

    usage: research/e131_wandb_log.py --rung 0 [--dry]

Every E131 artifact is a compile-only census. It runs the real AGX backend for
`applegpu_g16s` and `applegpu_g17s` through `xcrun metal-tt` and never touches
the GPU, so no run here is a timing measurement or a score. Each run logs
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e131-scored-path-occupancy-cliff-census"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e131-artifacts")

ARCHES = ("applegpu_g16s", "applegpu_g17s")

RUNGS = {
    "0": {
        "run_name": "e131-rung0-entry-point-body-census",
        "file": "rung0-census.json",
        "question":
            "which bodies can the wide-QMV entry point inline, which one sets "
            "its register maximum, and what becomes the maximum when the M=5 "
            "call site is removed",
        "command": "python3 research/e131_census.py --outdir research/e131-artifacts",
    },
    "1": {
        "run_name": "e131-rung1-scored-entry-point-census",
        "file": "rung1-census.json",
        "question":
            "which other scored entry points sit within a few registers of an "
            "occupancy step, and how much round-cost-weighted residency is "
            "recoverable from small register reductions",
        "command": "python3 research/e131_rung1.py --outdir research/e131-artifacts",
    },
    "2": {
        "run_name": "e131-rung2-top-cell-scouting",
        "file": "rung2-scouting.json",
        "question":
            "what allocates the marginal registers in the three highest-ranked "
            "cells, read from the ISA rather than the source",
        "command": "python3 research/e131_isa_scout.py --outdir research/e131-artifacts",
    },
    "3": {
        "run_name": "e131-rung3-presubmit-cliff-gate",
        "file": "rung3-gate.json",
        "question":
            "does a standing pre-submit gate fail on the E121 regression "
            "commit and pass on its revert, in under thirty seconds and with "
            "no GPU",
        "command": "senpai/entry-point-cliff-census.sh --base <ref>",
    },
}


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def census_table(payload: dict, section: str) -> wandb.Table | None:
    rows = payload.get(section)
    if not rows:
        return None
    table = wandb.Table(columns=[
        "cell", "arch", "registers", "spill_bytes", "text_bytes", "text_sha8",
        "simdgroups"])
    for cell, per_arch in rows.items():
        for arch in ARCHES:
            record = per_arch.get(arch)
            if not record:
                continue
            table.add_data(cell, arch, record["registers"],
                           record["spill_bytes"], record["text_bytes"],
                           record["text_sha8"], record["simdgroups"])
    return table


def ranked_table(payload: dict) -> wandb.Table | None:
    rows = payload.get("ranked")
    if not rows:
        return None
    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(c) for c in columns])
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    payload = json.loads((ART / spec["file"]).read_text())

    summary: dict = {}
    for key in ("base_sha", "toolchain", "wall_seconds", "harness",
                "timing_valid", "gpu_used", "official_or_ranked_score",
                "scored_cell"):
        if key in payload:
            summary[key] = payload[key]
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "host": HOST,
        "rung": int(args.rung),
        "question": spec["question"],
        "command": spec["command"],
    })
    for section in ("answers", "metric", "gate", "law", "provenance",
                    "simdgroup_budget"):
        flatten(section, payload.get(section, {}), summary)

    if args.dry:
        print(json.dumps(summary, indent=2))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type="census",
                     config={"experiment": "E131", "rung": int(args.rung),
                             "arches": list(ARCHES),
                             "command": spec["command"],
                             "question": spec["question"],
                             "base_sha": payload.get("base_sha"),
                             "toolchain": payload.get("toolchain")})
    for section in ("bodies", "entries", "cells"):
        table = census_table(payload, section)
        if table is not None:
            run.log({"%s_census" % section: table})
    table = ranked_table(payload)
    if table is not None:
        run.log({"ranked_recoverable_residency": table})
    run.summary.update(summary)
    artifact = wandb.Artifact("e131-rung%s" % args.rung, type="census")
    artifact.add_file(str(ART / spec["file"]))
    run.log_artifact(artifact)
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
