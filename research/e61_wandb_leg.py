#!/usr/bin/env python3
"""Log one E61 whole-leg arm to W&B, immediately after that arm finishes.

The advisor requires a W&B record per leg while the session is timing, not one
aggregate record written at session end, so a session that dies mid-way still
leaves every completed leg on the board.

  python3 research/e61_wandb_leg.py --tag m6 --arm t6 [--group GROUP]

Reads the leg's own artifacts under `.mlxfast-private/e61/runs/<tag>/`:
`meta.txt` for the identity tuple and `score-N.json` for the measured metrics.
Writes the created run's id and URL back to `wandb.json` in the same directory
so the result report can collect every link without querying W&B.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
RUNS = pathlib.Path(".mlxfast-private/e61/runs")
ARMS = pathlib.Path(".mlxfast-private/e61/arms")

# Metrics carried from every leg's score JSON. Correctness fields travel with
# the timing fields on purpose: a speed number whose leg did not match every
# token is not a result.
METRIC_KEYS = (
    "mtp_seconds_per_token",
    "serial_seconds_per_token",
    "mtp_decode_speedup",
    "effective_mean_draft_len",
    "accepted_draft_rate",
    "residual_divergence_count",
    "decode_tokens",
    "mtp_depth",
    "all_tokens_matched",
    "public_drift_tripwire_passed",
    "official_score",
    "rankable",
)


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--group", default="e61-single-stream-qmv-m6")
    ap.add_argument("--discarded", action="store_true",
                    help="declared warm-up leg; recorded, never used as evidence")
    args = ap.parse_args()

    run_dir = RUNS / args.tag
    meta = read_meta(run_dir / "meta.txt")
    scores = sorted(run_dir.glob("score-*.json"))
    legs = [json.loads(p.read_text())["metrics"] for p in scores]
    if not legs:
        raise SystemExit(f"e61_wandb_leg: {run_dir} has no score JSON")

    arm_path = ARMS / f"{args.tag}-arm.json"
    arm = json.loads(arm_path.read_text()) if arm_path.exists() else {}

    config = {
        "experiment": "e61",
        "arm": args.arm,
        "tag": args.tag,
        "discarded_warmup": args.discarded,
        "arm_doc": arm.get("doc"),
        "arm_family": arm.get("family"),
        "arm_cell": arm.get("cell"),
        "arm_never_submit": arm.get("never_submit"),
        "twin_sha256": arm.get("sha256"),
        "host": "Apple M4 Pro",
        "physical_memory_bytes": 51539607552,
        # Local ungated protocol: these three flags stay verbatim.
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "official_or_ranked_score": meta.get("official_or_ranked_score"),
        # Ranked-parity levers and the one known local-to-ranked difference.
        "darkbloom_startup_memory_profile":
            meta.get("darkbloom_startup_memory_profile"),
        "mlx_max_mb_per_buffer": meta.get("mlx_max_mb_per_buffer"),
        "mlx_max_ops_per_buffer": meta.get("mlx_max_ops_per_buffer"),
        "geometry_lever_verified_by": meta.get("geometry_lever_verified_by"),
        "wired_residency_active": meta.get("wired_residency_active"),
    }
    for key in ("head_sha", "base_sha", "dirty", "twin_digests", "fixture",
                "fixture_sha256", "tokens", "offered_depth", "legs",
                "head_safetensors_sha256", "thermal_before", "started",
                "e61_binary_assert_m6_na", "e61_binary_assert_m9_na",
                "e61_binary_assert_wide_bound",
                "e61_binary_assert_worker_text_sha256"):
        config[key] = meta.get(key)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        job_type="whole-leg",
        name=f"e61-{args.tag}",
        config=config,
    )

    for index, metrics in enumerate(legs, start=1):
        run.log({f"leg/{k}": metrics.get(k) for k in METRIC_KEYS}
                | {"leg/index": index})

    summary = {}
    for key in METRIC_KEYS:
        values = [m.get(key) for m in legs if isinstance(m.get(key), (int, float))]
        if values:
            summary[key] = statistics.fmean(values)
            if len(values) > 1:
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
    summary["all_tokens_matched_every_leg"] = all(
        bool(m.get("all_tokens_matched")) for m in legs)
    summary["legs_measured"] = len(legs)
    run.summary.update(summary)

    (run_dir / "wandb.json").write_text(
        json.dumps({"id": run.id, "url": run.url, "group": args.group,
                    "name": run.name}, indent=2) + "\n")
    print(f"e61_wandb_leg: {args.tag} -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
