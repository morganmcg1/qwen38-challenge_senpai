#!/usr/bin/env python3
"""Publish the E213 rows-per-simdgroup evidence to W&B.

    usage: research/e213_wandb.py GATE_DIR SESSION_DIR [SESSION_DIR ...]
           research/e213_wandb.py --dry-run GATE_DIR SESSION_DIR ...

One run per timed session plus one run for the Stage-0 numerics gate, all in
group `qwen38-r1-e213-rows-per-simd-g1`.

harness=local, Apple M4 Pro. The timed legs ran with the cool gate off under
the standing counterbalanced-session allowance, so every run carries
`coolGatePassedRealGate=false` and `gateQualifiedForTiming=false` verbatim.
Both legs of a local pair use the same candidate build and candidate-generated
reference rows, so no run here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e213-rows-per-simd-g1"
CAMPAIGN_BASE = "e0c7a026aebc9dbdd579577964ad413a11ede6fc"
# FINDING 545 same-host anchor for absolute candidate decode time.
ANCHOR_MTP_SPT = 0.02945630


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def identity(extra: dict) -> dict:
    return {
        "harness": "local",
        "experiment": "e213-rows-per-simd-g1",
        "campaignBaseSha": CAMPAIGN_BASE,
        "host": "aws-mac ec2 Apple M4 Pro, 20 GPU cores, 48 GiB",
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "officialOrRankedScore": False,
        "referenceRows": "candidate-generated, local fixture",
        **extra,
    }


def gate_spec(gate_dir: pathlib.Path) -> tuple[dict, dict]:
    meta = read_meta(gate_dir / "meta.txt")
    gate = json.loads((gate_dir / "gate.json").read_text())
    sweep = json.loads((gate_dir / "plain-vs-table.json").read_text())["rows"]

    broken = [r for r in sweep if r["differing"]]
    control = [
        r for r in gate["positive_control"]
        if r["shipped_geometry_differing"] > 0
    ]
    summary = {
        "gateTotalElements": gate["total_elements"],
        "gateTotalDiffering": gate["total_differing"],
        "gateWorstMaxUlp": gate["worst_max_ulp"],
        "gateParameterizationDiffering": sum(
            r["differing"] for r in gate["parameterization_identity"]),
        "gatePositiveControlTripped": len(control),
        "gatePositiveControlRows": len(gate["positive_control"]),
        "sweepRows": len(sweep),
        "sweepDifferingRows": len(broken),
        "sweepDifferingVariants": ",".join(
            sorted({r["variant"] for r in broken})),
        "scoredPlainWidthsClean": all(
            r["differing"] == 0 for r in sweep if r["scored_plain_width"]),
        "rows4InstantiationsClean": all(
            r["differing"] == 0 for r in sweep if r["rows_per_simd"] == 4),
    }
    for row in gate["cells"]:
        key = "arm/%s/m%d/ipg%d/rows%d/table%s" % (
            row["arm"], row["m"], row["arm_ipg"], row["arm_rows_per_simd"],
            row["use_table"])
        summary["%s/differing" % key] = row["differing"]
    config = identity({
        "stage": "stage0-numerics-gate",
        "commitSha": meta["base_sha"],
        "dirtyCandidatePaths": int(meta["dirty_candidate_paths"]),
        "metallibSourceFingerprint": meta["metallib_source_fingerprint"],
        "cells": "7 fused decode cells plus the n=4104 edge cell",
        "gpuUsed": True,
        "tolerance": gate["tolerance"],
        "deliverable": "falsify the moved instantiations before any timed leg",
    })
    return config, summary


def session_spec(session_dir: pathlib.Path) -> tuple[dict, dict, str]:
    blob = json.loads((session_dir / "analysis.json").read_text())
    arm = next(k for k in blob["arms"])
    stats = blob["arms"][arm]
    absolute = stats["whole_leg_absolute"]

    summary = {
        "armMtpSecondsPerToken": absolute["arm_mean"],
        "offMtpSecondsPerToken": absolute["off_mean"],
        "deltaSecondsPerToken": absolute["delta_seconds_per_token"],
        "relativeGain": absolute["relative_gain"],
        "anchorMtpSecondsPerToken": ANCHOR_MTP_SPT,
        "offVersusAnchorPct":
            100.0 * (absolute["off_mean"] - ANCHOR_MTP_SPT) / ANCHOR_MTP_SPT,
        "roundWeightedMsPerRound":
            stats["round_weighted_ms_per_round_recovered"],
        "trajectoriesIdentical": blob["trajectories_identical"],
        "roundCount": blob["round_count"],
        "gpuTempEntrySpreadC": blob["gpu_temp_entry_spread_c"],
        "problems": ",".join(blob["problems"]) or "none",
    }
    for name in ("paired_all_rounds", "paired_moved_widths",
                 "paired_untouched_widths"):
        cell = stats["%s_ms_per_round_recovered" % name]
        stem = name.replace("paired_", "paired/")
        summary["%s/meanMsPerRound" % stem] = cell["mean"]
        summary["%s/ci95Lo" % stem] = cell["ci95_lo"]
        summary["%s/ci95Hi" % stem] = cell["ci95_hi"]
        summary["%s/n" % stem] = cell["n"]
    for width, cell in stats["paired_per_width_ms_per_round_recovered"].items():
        summary["perWidth/m%s/meanMsPerRound" % width] = cell["mean"]
        summary["perWidth/m%s/ci95Lo" % width] = cell["ci95_lo"]
        summary["perWidth/m%s/ci95Hi" % width] = cell["ci95_hi"]
        summary["perWidth/m%s/rounds" % width] = cell["n"]
    for phase, cell in blob["attribution_candidate_side"][arm].items():
        summary["attribution/%s/deltaUsMovedWidths" % phase] = \
            cell["delta_us_moved_widths"]
    for leg in blob["legs"]:
        stem = "leg%d_%s" % (leg["leg"], leg["arm"])
        summary["%s/mtpSecondsPerToken" % stem] = leg["mtp_seconds_per_token"]
        summary["%s/decodeSeconds" % stem] = leg["decode_seconds"]
        summary["%s/gpuTempEntryC" % stem] = leg["gpu_temp_entry_c"]
        summary["%s/gpuTempExitC" % stem] = leg["gpu_temp_exit_c"]
        summary["%s/allTokensMatched" % stem] = leg["all_tokens_matched"]
        summary["%s/residualDivergenceCount" % stem] = \
            leg["residual_divergence_count"]
        summary["%s/declaredRowsTotal" % stem] = leg["declared_rows_total"]
        summary["%s/referenceCheckedRowTotal" % stem] = \
            leg["reference_checked_row_total"]
        summary["%s/observedPlans" % stem] = leg["observed_plans"]

    config = identity({
        "stage": "stage1-timed-session",
        "arm": arm,
        "commitSha": blob["candidate_sha"],
        "sessionOrder": " ".join(blob["session_order"]),
        "movedWidths": blob["arm_widths"][arm],
        "tokens": int(blob["tokens"]),
        "metallibSourceFingerprint": blob["metallib_source_fingerprint"],
        "workerSha256": blob["worker_sha256"],
        "timingSource": blob["timing_source"],
        "gpuUsed": True,
        "deliverable": {
            "g1": "price G = 1 at m = 7, 8 and 9, bought with rows_per_simd 1",
            "probe": "price rows_per_simd 1 alone at m = 9, G held at 2",
        }[arm],
    })
    return config, summary, arm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("gate", type=pathlib.Path)
    parser.add_argument("sessions", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    specs = {}
    config, summary = gate_spec(args.gate)
    specs["gate"] = ("e213-numerics-gate", config, summary)
    for session in args.sessions:
        config, summary, arm = session_spec(session)
        specs[arm] = ("e213-session-%s" % arm, config, summary)

    if args.dry_run:
        print(json.dumps(
            {k: {"name": n, "config": c, "summary": s}
             for k, (n, c, s) in specs.items()},
            indent=1, sort_keys=True, default=str))
        return 0

    import wandb

    published = {}
    for key, (name, config, summary) in specs.items():
        run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                         job_type="local-measurement", name=name,
                         config=config)
        run.summary.update(summary)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    print(json.dumps(published, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
