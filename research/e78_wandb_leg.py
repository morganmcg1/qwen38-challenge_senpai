#!/usr/bin/env python3
"""Log one E78 timed leg group to W&B, immediately after that arm finishes.

  python3 research/e78_wandb_leg.py --tag a1 --arm a_ship [--group GROUP]

Reads the leg group's own artefacts under `.mlxfast-private/e78/runs/<tag>/`:
`meta.txt` for the identity tuple, `score-N.json` for the measured metrics and
`reports/leg-N/04-mtp-timed.json` for the per-round width histogram. Writes the
created run's id and URL back to `wandb.json` in the same directory so the
result report can collect every link without querying W&B.

The width histogram travels with the timing numbers because it is the check
that separates a dispatch-table effect from a schedule effect: every arm in
this experiment must present the identical distribution of rows per round, so
any timing difference is the cost of the same work done differently.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
RUNS = pathlib.Path(".mlxfast-private/e78/runs")
ARMS = pathlib.Path(".mlxfast-private/e78/arms")

# Correctness fields travel with the timing fields on purpose: a speed number
# whose leg did not match every token is not a result.
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

META_KEYS = (
    "head_sha", "base_sha", "dirty", "twin_digests", "fixture",
    "fixture_sha256", "tokens", "offered_depth", "legs",
    "head_safetensors_sha256", "thermal_before", "thermal_after", "started",
    "finished", "cool_gate_requested", "cool_gate_passed_real_gate",
    "gate_qualified_for_timing", "official_or_ranked_score",
    "darkbloom_startup_memory_profile", "mlx_max_mb_per_buffer",
    "mlx_max_ops_per_buffer", "geometry_lever_verified_by",
    "wired_residency_active", "cli_sha256", "worker_sha256",
    "metallib_fingerprint", "status",
)


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            # Later lines win: e42-run.sh appends the authoritative gate verdict
            # after the placeholder it writes in the header.
            meta[key.strip()] = value.strip()
    return meta


def width_histogram(report: pathlib.Path) -> dict[str, int] | None:
    """Rounds per row count M, read from the MTP leg of one timed pair."""
    if not report.exists():
        return None
    doc = json.loads(report.read_text())
    if doc.get("is_serial_control"):
        return None
    lengths = doc.get("effective_draft_lengths")
    if not lengths:
        return None
    counts = collections.Counter(int(x) + 1 for x in lengths)
    return {str(k): counts[k] for k in sorted(counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--group", default="e78-width-dependent-inner-group-count")
    ap.add_argument("--discarded", action="store_true",
                    help="declared warm-up leg; recorded, never used as evidence")
    args = ap.parse_args()

    run_dir = RUNS / args.tag
    meta = read_meta(run_dir / "meta.txt")
    scores = sorted(run_dir.glob("score-*.json"))
    legs = [json.loads(p.read_text())["metrics"] for p in scores]
    if not legs:
        raise SystemExit(f"e78_wandb_leg: {run_dir} has no score JSON")

    arm_path = ARMS / f"{args.tag}-arm.json"
    arm = json.loads(arm_path.read_text()) if arm_path.exists() else {}

    config = {
        "experiment": "e78",
        "rung": "2b",
        "harness": "local",
        "arm": args.arm,
        "tag": args.tag,
        "discarded_warmup": args.discarded,
        "arm_doc": arm.get("doc"),
        "arm_cells": arm.get("cells"),
        "arm_cutoff": arm.get("cutoff"),
        "arm_na_bound": arm.get("na_bound"),
        "arm_never_submit": arm.get("never_submit"),
        "twin_sha256": arm.get("sha256"),
        "twin_source_bytes": arm.get("source_bytes"),
        "host": "Apple M4 Pro",
        "gpu_core_count": 20,
        "gpu_family": "applegpu_g16s",
        "physical_memory_bytes": 51539607552,
    }
    config.update({key: meta.get(key) for key in META_KEYS})
    for key, value in meta.items():
        if key.startswith("e78_binary_assert_") or key.startswith("leg"):
            config[key] = value

    histograms = []
    for index in range(1, len(legs) + 1):
        histograms.append(
            width_histogram(run_dir / f"reports/leg-{index}/04-mtp-timed.json"))
    config["width_histogram"] = histograms[0] if histograms else None
    config["width_histogram_stable_across_legs"] = (
        len({json.dumps(h, sort_keys=True) for h in histograms}) == 1)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        job_type="whole-leg",
        name=f"e78-{args.tag}",
        config=config,
    )

    for index, metrics in enumerate(legs, start=1):
        payload = {f"leg/{k}": metrics.get(k) for k in METRIC_KEYS}
        payload["leg/index"] = index
        for width, rounds in (histograms[index - 1] or {}).items():
            payload[f"leg/width_rounds_m{width}"] = rounds
        run.log(payload)

    summary = {}
    for key in METRIC_KEYS:
        values = [m.get(key) for m in legs
                  if isinstance(m.get(key), (int, float))
                  and not isinstance(m.get(key), bool)]
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
                    "name": run.name, "arm": args.arm,
                    "discarded_warmup": args.discarded}, indent=2) + "\n")
    print(f"e78_wandb_leg: {args.tag} -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
