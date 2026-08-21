#!/usr/bin/env python3
"""E89: stream the host-state session documents to W&B.

One run per E89 report. It carries the experiment identity tuple, the rung 0a
arm-blind gate re-analysis, the multiplier-versus-added-work model, and the
rung 0b probe session with its per-leg host-state stratum.

Every leg is UNGATED by design, so cool_gate_passed_real_gate=false and
gate_qualified_for_timing=false travel with the run verbatim. Nothing here is
a score.

usage: research/e89_wandb.py --name NAME --doc KEY=PATH [--doc KEY=PATH ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e89-find-and-remove-the-per-drafting-round-binary-host-state"


def cell(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def table(rows: list[dict]) -> wandb.Table:
    columns = sorted({k for r in rows for k in r})
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[cell(row.get(c)) for c in columns])
    return t


def log_scalars(run, prefix: str, obj, depth: int = 0) -> None:
    if depth > 3 or not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            run.log({f"{prefix}/{k}": v})
        elif isinstance(v, dict):
            log_scalars(run, f"{prefix}/{k}", v, depth + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--doc", action="append", required=True, metavar="KEY=PATH")
    ap.add_argument("--notes", default="")
    ap.add_argument("--config", default="{}",
                    help="extra JSON merged into the run config")
    args = ap.parse_args()

    docs = {}
    for spec in args.doc:
        key, _, path = spec.partition("=")
        docs[key] = json.loads(Path(path).read_text())

    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "local_mode": "--local-iterate",
        "cool_gate": 0,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "ranked_host": False,
        "host": "ip-10-231-2-22.ec2.internal",
        "chip": "Apple M4 Pro",
        "memory_bytes": 51539607552,
        "documents": sorted(docs),
    }
    config.update(json.loads(args.config))

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     job_type="host-state-session", notes=args.notes,
                     config=config)

    for key, doc in docs.items():
        log_scalars(run, key, doc)
        legs = doc.get("legs")
        if isinstance(legs, list) and legs and isinstance(legs[0], dict):
            run.log({f"{key}/legs": table(
                [{kk: vv for kk, vv in l.items() if kk != "meta"} for l in legs])})
            metas = [l["meta"] for l in legs if isinstance(l.get("meta"), dict)]
            if metas:
                run.log({f"{key}/leg_meta": table(metas)})
        for field in ("discriminator", "stuck_leg_rate", "clean_median_by_arm",
                      "mid_leg_transition"):
            if isinstance(doc.get(field), dict):
                log_scalars(run, f"{key}/{field}", doc[field])
        pairs = doc.get("pairs")
        if isinstance(pairs, list):
            for p in pairs:
                tag = p["stuck"]
                for group in ("host", "pipe"):
                    rows = [{"phase": ph, **vals} for ph, vals in p[group].items()]
                    run.log({f"{key}/{tag}/{group}": table(rows)})
                for k, v in p.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        run.log({f"{key}/{tag}/{k}": v})
        run.log({f"{key}/raw": wandb.Table(
            columns=["json"], data=[[json.dumps(doc, indent=1)]])})

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
