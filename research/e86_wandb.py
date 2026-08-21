#!/usr/bin/env python3
"""E86: stream the decode-ladder session to W&B.

One run per session document produced by `research/e86_ladder_report.py`. It
carries every fact another agent needs to reproduce or overturn the decision:
the experiment identity tuple, the per-leg table, the per-arm ranking against
the session null, and the rung-0 host-encode / GPU-execute decomposition.

Every leg is UNGATED and ABBA-counterbalanced by design, so
cool_gate_passed_real_gate=false and gate_qualified_for_timing=false travel
with the run verbatim. The session is directional causal evidence, never a
score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "qwen38-r1-e86-decode-asynceval-ladder-and-host-gpu-split"

LEG_COLS = ["tag", "position", "arm", "ladder", "rep", "started", "sync_head",
            "candidate_mtp_seconds_per_token", "serial_seconds_per_token",
            "local_ratio", "rounds", "rows_per_token", "mean_d", "mean_acc",
            "d_submit2_us_med", "verify_build_us_med", "eval_wall_us_med",
            "round_us_med", "round_us_total", "host_phase_sum_us_med",
            "host_state_stuck", "frac_rounds_host_stuck",
            "readout_us_med", "commit_us_med",
            "upkeep_us_med", "all_tokens_matched", "residual_divergence_count",
            "accepted_draft_rate", "effective_mean_draft_len",
            "head_provenance_sha256", "head_loaded_bytes", "head_loaded_files",
            "gpu_temp_entry_c", "gpu_temp_exit_c", "cool_gate_passed_real_gate",
            "gate_qualified_for_timing", "base_sha", "worker_sha256"]

PAIRED_COLS = ["arm", "median_round_us", "ci_lo_us", "ci_hi_us", "mean_round_us",
               "median_vpipe_us", "pairs", "pct_of_round", "pct_ci_lo", "pct_ci_hi"]


def cell(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def arm_mean_position(doc: dict) -> dict:
    """Mean session position per arm. Equal means say the design is balanced."""
    pos: dict[str, list[int]] = {}
    for r in doc["legs"]:
        if "position" in r:
            pos.setdefault(r["arm"], []).append(r["position"])
    return {a: sum(p) / len(p) for a, p in pos.items()}


def table(columns, rows):
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[cell(row.get(c)) for c in columns])
    return t


def log_session(run, doc: dict, prefix: str) -> None:
    run.log({f"{prefix}/legs": table(LEG_COLS, doc["legs"])})

    arm_rows = []
    for arm, s in doc["summary"].items():
        arm_rows.append({"arm": arm, **{k: cell(v) for k, v in s.items()}})
        for k, v in s.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                run.log({f"{prefix}/{arm}/{k}": v})
    run.log({f"{prefix}/by_arm": table(sorted({k for r in arm_rows for k in r}), arm_rows)})

    run.log({
        f"{prefix}/session_null_pct": doc["session_null_pct"],
        f"{prefix}/bit_exact_work": doc["bit_exact_work"],
    })

    # The paired per-round comparison is the decision instrument. The leg-total
    # deltas in `by_arm` are kept only so the contamination stays visible.
    p = doc.get("paired")
    if p:
        rows = [{"arm": a, **v} for a, v in p["arms"].items()]
        if p.get("null"):
            rows.append({"arm": f"NULL ({p['reference_arm']} vs itself)", **p["null"]})
        run.log({f"{prefix}/paired": table(PAIRED_COLS, rows)})
        for a, v in p["arms"].items():
            run.log({f"{prefix}/paired/{a}/median_round_us": v["median_round_us"],
                     f"{prefix}/paired/{a}/pct_of_round": v["pct_of_round"],
                     f"{prefix}/paired/{a}/median_vpipe_us": v["median_vpipe_us"]})
        if p.get("null"):
            run.log({f"{prefix}/paired/null_pct_of_round": p["null"]["pct_of_round"],
                     f"{prefix}/paired/null_position_balanced":
                         p["null"]["position_balanced"]})

    d = doc.get("decomposition")
    if not d:
        return
    for k, v in d.items():
        if isinstance(v, (int, float)):
            run.log({f"{prefix}/decomposition/{k}": v})
    rows = [{"arm": a, **p} for a, p in d["per_arm"].items()]
    run.log({f"{prefix}/decomposition/per_arm":
             table(sorted({k for r in rows for k in r}), rows)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e86-decode-ladder-session")
    ap.add_argument("--session", action="append", required=True,
                    metavar="PREFIX=PATH",
                    help="log this session document under the given key")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    sessions = {}
    for spec in args.session:
        key, _, path = spec.partition("=")
        sessions[key] = json.loads(Path(path).read_text())

    first = next(iter(sessions.values()))
    leg0 = first["legs"][0]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.name,
        job_type="timed-session", notes=args.notes,
        config={
            "experiment": EXPERIMENT,
            "harness": "local",
            "local_mode": "--local-iterate",
            "timed": True,
            "decode_tokens": leg0["decode_tokens"],
            "offered_depth": 8,
            "sessions": {k: {"prefix": v["prefix"],
                             "arms": sorted(v["summary"]),
                             "n_legs": len(v["legs"]),
                             "reference_arm": v["reference_arm"],
                             "leg_order": v.get("leg_order"),
                             "arm_mean_position": arm_mean_position(v)}
                         for k, v in sessions.items()},
            "leg_order": "palindrome (ABBA-counterbalanced); see "
                         "sessions.*.arm_mean_position for the position balance",
            "cool_gate": 0,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "host": "ip-10-231-2-22.ec2.internal",
            "chip": "Apple M4 Pro",
            "memory_bytes": 51539607552,
            "ranked_host": False,
            "base_sha": leg0["base_sha"],
            "worker_sha256": leg0["worker_sha256"],
            "head_provenance_sha256": leg0["head_provenance_sha256"],
            "head_loaded_bytes": leg0["head_loaded_bytes"],
            "shipped_ladder": [0, 1, 9, 19, 29, 39, 49, 57],
        },
    )

    for key, doc in sessions.items():
        log_session(run, doc, key)
        run.log({f"{key}/raw": wandb.Table(
            columns=["json"], data=[[json.dumps(doc, indent=2)]])})

    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
