#!/usr/bin/env python3
"""E87: stream the coarse-shortlist decision to W&B.

One run holds every fact another agent needs to reproduce or overturn it:

  build      the arm-G head build report, including the byte delta the price
             list is charged against.
  validate   the rung-0 positive control. The offline exact argmax must equal
             the proposal the runtime returned, the shipped g64 shortlist must
             miss zero times, and a deliberately damaged scorer must miss often.
  screen     the rung-1 `m` tables for arm G and every arm-C cell, with the
             paired discordance against the shipped shortlist per work and per
             domain, the centroid and stage-2 byte columns, and the predicted
             score change on the WORST domain.
  timed      the rung-2 ungated ABABA session. Every leg carries
             cool_gate_passed_real_gate=false and gate_qualified_for_timing
             =false verbatim. It is directional causal evidence inside one
             session, never a score.

usage:
  research/e87_wandb.py --name e87-coarse-shortlist \
      [--build research/e87-build-e87-coarse-g128.json] \
      [--validate research/e87-validate.json] \
      [--screen research/e87-screen.json] \
      [--timed research/e87-timing.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e87-coarse-draft-shortlist-traffic"


def cell(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def table(columns, rows):
    t = wandb.Table(columns=list(columns))
    for row in rows:
        t.add_data(*[cell(row.get(c)) for c in columns])
    return t


def log_build(run, path: Path) -> None:
    report = json.loads(path.read_text())
    run.log({"build/raw": table(["json"], [{"json": json.dumps(report, indent=2)}])})
    flat = {f"build/{k}": v for k, v in report.items() if isinstance(v, (int, float))}
    run.log(flat)


def log_validate(run, path: Path) -> None:
    report = json.loads(path.read_text())
    run.log({
        "validate/samples": report["samples"],
        "validate/proposal_match": report["proposal_match"],
        "validate/m_shipped_g64": report["m_shipped_g64"]["p"],
        "validate/m_shipped_g64_hi": report["m_shipped_g64"]["hi"],
        "validate/m_damaged_control": report["m_damaged_control"]["p"],
        "validate/raw": table(["json"], [{"json": json.dumps(report, indent=2)}]),
    })


def log_screen(run, path: Path) -> None:
    doc = json.loads(path.read_text())
    cells = doc["cells"]
    columns = ["arm", "n", "misses", "m", "m_lo", "m_hi", "worse_than_shipped",
               "better_than_shipped", "net_miss_vs_shipped", "net_miss_lo",
               "net_miss_hi", "net_miss_worst_domain", "net_miss_worst_work",
               "centroid_bytes", "stage2_bytes", "read_bytes", "removed_bytes",
               "head_pct", "score_gain_pct", "breakeven_m", "predicted_score_pct",
               "predicted_worst_pct", "predicted_worst_domain_pct",
               "by_domain", "by_work"]
    run.log({"screen/cells": table(columns, cells)})
    run.log({"screen/samples": doc["samples"]})

    # One row per (cell, group) so the worst domain is queryable, not buried.
    group_rows = []
    for c in cells:
        for kind in ("by_domain", "by_work"):
            for label, v in c.get(kind, {}).items():
                group_rows.append({
                    "arm": c["arm"], "group_kind": kind[3:], "group": label,
                    "n": v["n"], "m": v["m"], "net_miss_vs_shipped": v["net"],
                    "score_gain_pct": c["score_gain_pct"],
                    "predicted_pct_on_group":
                        c["score_gain_pct"] - 206.6 * v["net"],
                })
    run.log({"screen/by_group": table(sorted({k for r in group_rows for k in r}),
                                      group_rows)})

    best = max(cells, key=lambda c: c["predicted_worst_domain_pct"])
    run.log({
        "screen/best_arm": best["arm"],
        "screen/best_predicted_worst_domain_pct": best["predicted_worst_domain_pct"],
        "screen/best_predicted_score_pct": best["predicted_score_pct"],
    })
    for c in cells:
        if c["arm"] in ("shipped-g64", "armG-g128"):
            run.log({
                f"screen/{c['arm']}/m": c["m"],
                f"screen/{c['arm']}/net_miss_vs_shipped": c["net_miss_vs_shipped"],
                f"screen/{c['arm']}/net_miss_worst_domain": c["net_miss_worst_domain"],
                f"screen/{c['arm']}/predicted_score_pct": c["predicted_score_pct"],
                f"screen/{c['arm']}/predicted_worst_domain_pct":
                    c["predicted_worst_domain_pct"],
            })


def log_timed(run, path: Path) -> None:
    doc = json.loads(path.read_text())
    leg_cols = ["tag", "arm", "rep", "started", "candidate_mtp_seconds_per_token",
                "serial_seconds_per_token", "local_ratio", "rounds",
                "rows_per_token", "mean_d", "mean_acc", "accepted_draft_rate",
                "draft_build_us_per_round", "verify_build_us_per_round",
                "round_us_total", "all_tokens_matched",
                "residual_divergence_count", "head_provenance_sha256",
                "head_loaded_bytes", "gpu_temp_entry_c", "gpu_temp_exit_c",
                "cool_gate_passed_real_gate", "gate_qualified_for_timing",
                "base_sha", "worker_sha256"]
    run.log({"timed/legs": table(leg_cols, doc["legs"])})
    run.log({
        "timed/session_null_pct": doc["session_null_pct"],
        "timed/gpu_temp_entry_spread_c": doc["gpu_temp_entry_spread_c"],
    })
    arm_rows = []
    for arm, s in doc["summary"].items():
        arm_rows.append({"arm": arm, **{k: cell(v) for k, v in s.items()}})
        run.log({
            f"timed/{arm}/candidate_seconds_per_token": s["spt_mean"],
            f"timed/{arm}/spt_delta_pct_vs_base": s["spt_delta_pct_vs_base"],
            f"timed/{arm}/local_ratio": s["ratio_mean"],
            f"timed/{arm}/draft_build_us_per_round": s["draft_build_us_per_round"],
            f"timed/{arm}/head_loaded_bytes": s["head_loaded_bytes"],
        })
        if s.get("predicted_pct") is not None:
            run.log({f"timed/{arm}/predicted_pct": s["predicted_pct"]})
    run.log({"timed/by_arm": table(sorted({k for r in arm_rows for k in r}), arm_rows)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--build")
    ap.add_argument("--validate")
    ap.add_argument("--screen")
    ap.add_argument("--timed")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name, notes=args.notes,
        job_type="e87", tags=["e87", "qwen38-mtp", "harness:local"],
        config={
            "experiment": EXPERIMENT,
            "student": "qwen-thorfinn",
            "pr": 89,
            "harness": "local",
            "host": "Apple M4 Pro 48GB (not the ranked M5)",
            "declared_head_tensor_bytes": 427_738_112,
            "coarse_stage_bytes": 157_337_600,
            "bytes_to_score_pct": 0.0815,
            "miss_to_score_pct": 206.6,
            "official_or_ranked_score": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
        },
    )
    for flag, fn in (("build", log_build), ("validate", log_validate),
                     ("screen", log_screen), ("timed", log_timed)):
        path = getattr(args, flag)
        if path:
            fn(run, Path(path))
            print(f"logged {flag} from {path}")
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
