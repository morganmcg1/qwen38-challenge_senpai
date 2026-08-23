#!/usr/bin/env python3
"""E151 R2: log the double-buffer arm's offline artifact set to W&B.

HARNESS LABELLING, same three-way split as `research/e151_wandb_log.py`:

  harness=offline  the R2 compile gate and the threadgroup race model. Zero GPU
                   seconds, no timing.
  harness=local    NOT PRESENT IN THIS RUN. R2's runtime gate chain has not run
                   yet. When it does it will prove only that the non-NAX path
                   is unbroken, because `is_nax_available()` is false on this
                   host and the pipelined kernel never executes locally.
  harness=ranked   the pre-registered predictions and the published reference
                   rows they are anchored on. No senpai receipt for R2 exists.

The predictions here were pre-registered in interim 6 on PR #151, before the
arm was written, and are not revised after the fact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
TRACK_ID = "qwen3.8-27b-mtp-v1"

BASE_SHA = "de8ce44c7bc133c3c6c079957240782664afd287"
GROWTH_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"

# harness=ranked. Pre-registered in interim 6, before the arm was written.
PREDICTION = {
    "e151_r2_predicted_ranked_prefill_pct": -4.0,
    "e151_r2_predicted_ranked_prefill_pct_low": -5.0,
    "e151_r2_predicted_ranked_prefill_pct_high": -3.0,
    "e151_r1r2_predicted_ranked_prefill_pct": -6.5,
    "e151_r1r2_predicted_ranked_prefill_pct_low": -8.5,
    "e151_r1r2_predicted_ranked_prefill_pct_high": -5.0,
    "e151_predicted_decode_channel_pct": 0.0,
    "e151_predicted_decode_channel_pct_band": 0.15,
    "e151_r2_minimum_useful_prefill_pct": -0.5,
    "e151_r1r2_must_beat_prefill_pct": -4.9722,
}

# harness=ranked. Published reference rows for the two mechanisms, from
# advisor FINDING 268. Neither is a senpai receipt.
RANKED_REFERENCES = {
    "e151_ref_43925f29_prefill_pct": -4.1181,
    "e151_ref_43925f29_prefill_sd": 0.0936,
    "e151_ref_5cdc9c17_prefill_pct": -4.9721,
    "e151_ref_5cdc9c17_prefill_sd": 0.1142,
    "e151_ref_0cf1637e_prefill_pct": -0.0084,
    "e151_ref_0cf1637e_prefill_sd": 0.2132,
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True
    ).stdout.strip()


def blob_digest(rev: str, path: str) -> str:
    p = subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True
    )
    import hashlib

    return hashlib.sha256(p.stdout).hexdigest()


def load(path: str):
    f = pathlib.Path(path)
    if not f.is_file():
        raise SystemExit(f"missing required artifact {path}")
    return json.loads(f.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-gate", default="research/e151-r2-compile-gate.json")
    ap.add_argument("--race-model", default="research/e151-r2-race-model.json")
    ap.add_argument("--r1-sha", required=True)
    ap.add_argument("--r2-sha", required=True)
    ap.add_argument("--name", default="e151-r2-nax-qmm-double-buffer")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cg = load(args.compile_gate)
    rm = load(args.race_model)

    surface = [
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h",
        "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
    ]

    config = {
        "experiment": "E151",
        "rung": "R2",
        "track_id": TRACK_ID,
        "mechanism": "software-pipelined double-buffered k-loop in affine_qmm_t_nax",
        "base_sha": BASE_SHA,
        "growth_enforced_base": GROWTH_BASE_SHA,
        "r1_standalone_sha": args.r1_sha,
        "r1r2_composed_sha": args.r2_sha,
        "head_sha": git("rev-parse", "HEAD"),
        "worktree_clean": git("status", "--porcelain") == "",
        "scored_instantiation": cg.get("scored_instantiation"),
        "jit_concatenation_order": cg.get("jit_concatenation_order"),
        "submitted_surface": surface,
        "submitted_surface_digest_r1": {
            p: blob_digest(args.r1_sha, p) for p in surface
        },
        "submitted_surface_digest_r1r2": {
            p: blob_digest(args.r2_sha, p) for p in surface
        },
        "host_is_nax_available": False,
        "host_chip_family": "applegpu_g16s",
        "ranked_runner_chip_family": "applegpu_g17s",
        "r2_runtime_gate_chain_status": "not run at log time",
        "e151_r2_tgp_table": cg.get("e151_r2_tgp_table"),
        "e151_r2_race_model_rows": rm.get("e151_r2_race_model_rows"),
        "race_model_why_not_executed": rm.get("why_not_executed"),
        "tile_boundary_note": rm.get("tile_boundary_note"),
    }

    metrics: dict[str, float] = {}
    metrics.update({k: float(v) for k, v in PREDICTION.items()})
    metrics.update({k: float(v) for k, v in RANKED_REFERENCES.items()})

    for key, value in cg.items():
        if key.startswith("e151_r2_") and isinstance(value, bool):
            metrics[key] = float(value)
    for key, value in rm.items():
        if key.startswith("e151_r2_") and isinstance(value, bool):
            metrics[key] = float(value)

    for label, air in (cg.get("e151_r2_air_bytes") or {}).items():
        if air is not None:
            metrics[f"e151_r2_air_bytes_{label}"] = float(air)
    for label, delta in (cg.get("e151_r2_armoff_air_delta_bytes") or {}).items():
        metrics[f"e151_r2_armoff_air_delta_bytes_{label}"] = float(delta)

    scored_tgp = next(
        (
            r
            for r in cg.get("e151_r2_tgp_table", [])
            if r.get("T") in ("bfloat16_t", "bfloat16") and r.get("retile") == "on"
        ),
        None,
    )
    if scored_tgp:
        metrics["e151_r2_tgp_bytes"] = float(scored_tgp["double_buffer_bytes"])
        metrics["e151_r2_tgp_delta_bytes"] = float(scored_tgp["delta_bytes"])
        metrics["e151_r2_tgp_limit_bytes"] = float(scored_tgp["limit_bytes"])

    metrics["e151_r2_compile_gate_pass"] = float(
        bool(cg.get("e151_r2_compile_gate_pass"))
    )
    metrics["e151_r2_race_model_pass"] = float(bool(rm.get("e151_r2_race_model_pass")))
    metrics["e151_r2_offline_gates_all_green"] = float(
        bool(cg.get("e151_r2_compile_gate_pass"))
        and bool(rm.get("e151_r2_race_model_pass"))
    )

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.name,
        config=config,
        mode="offline" if args.offline else "online",
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    print(f"e151 r2 wandb run {run.id} {run.url}")
    for key in sorted(metrics):
        print(f"  {key} = {metrics[key]}")
    run.finish()


if __name__ == "__main__":
    main()
