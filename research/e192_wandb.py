#!/usr/bin/env python3
"""Publish the E192 legs to W&B.

    python3 research/e192_wandb.py --legs e192s0-barrier-alt e192s0-tape-alt \
        --stage stage0 [--dry-run]

harness=local. One run per leg with the full identity tuple in `config`, the
per-round series logged by round index, and the paired arm contrast in
`summary`. One roll-up run carries the account verdicts.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e192_stage0_report import (  # noqa: E402
    INT_FIELDS, PAIRED_FIELDS, full_acceptance, leg_report, read_rounds,
)

SERIES = [f for f in INT_FIELDS if f not in ("round",)]


def identity(report: dict, base_sha: str) -> dict:
    meta_path = pathlib.Path("research/out") / report["tag"] / "meta.txt"
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    return {
        "harness": "local",
        "experiment": "e192-persistent-rollback-slot",
        "leg": report["tag"],
        "armEnv": report["arm_env"],
        "armKey": report["arm_key"],
        "instrument": meta.get("e192_instrument", ""),
        "baseSha": meta.get("base_sha", base_sha),
        "assignmentBaseSha": "55b341555913b9166edc5711c6c2fc7d373cd411",
        "workerSha256": meta.get("worker_sha256", ""),
        "postRunWorkerSha256": meta.get("post_run_worker_sha256", ""),
        "cliSha256": meta.get("cli_sha256", ""),
        "host": meta.get("host", os.uname().nodename),
        "chip": meta.get("chip", ""),
        "memoryBytes": meta.get("memory_bytes", ""),
        "tokens": meta.get("tokens", ""),
        "localMode": meta.get("local_mode", ""),
        "headDir": pathlib.Path(meta.get("head_dir", "")).name,
        "sandbox": meta.get("sandbox", ""),
        "ladder": meta.get("ladder", ""),
        "metallibSourceFingerprint": meta.get("metallib_source_fingerprint", ""),
        "dirtyCandidatePaths": meta.get("dirty_candidate_paths", ""),
        "coolGate": meta.get("cool_gate", ""),
        "coolGatePassedRealGate": meta.get("cool_gate_passed_real_gate", ""),
        "gateQualifiedForTiming": meta.get("gate_qualified_for_timing", ""),
        "officialOrRankedScore": meta.get("official_or_ranked_score", ""),
        "gpuTempEntryC": meta.get("gpu_temp_entry_c", ""),
        "gpuTempExitC": meta.get("gpu_temp_exit_c", ""),
        "startedUtc": meta.get("started", ""),
        "finishedUtc": meta.get("finished", ""),
        "referenceSource": "candidate-generated local rows",
    }


def flat_summary(report: dict) -> dict:
    out = {
        "traceRounds": report["trace_rounds"],
        "draftingRounds": report["drafting_rounds"],
        "fullAcceptanceRounds": report["full_acceptance_rounds"],
        "rejectingRounds": report["rejecting_rounds"],
        "pairCount": report["paired"].get("pair_count", 0),
    }
    for arm, entry in report["per_arm"].items():
        for key, value in entry.items():
            out[f"{arm}/{key}"] = value
    for field in PAIRED_FIELDS:
        stats = report["paired"].get(field)
        if stats:
            for key, value in stats.items():
                out[f"paired/{field}/{key}"] = value
    return out


def verdicts(reports: list[dict]) -> dict:
    """Reduce the two legs to the three competing accounts of FINDING 495.

    in-flight    the release blocks on device work, so a persistent slot
                 inherits the wait. Falsified when the thread-CPU clock tracks
                 the wall clock across the same release.
    allocator    the release is real host work at the clear site.
    relocation   the work is real but is only deferred, so removing the
                 references moves it rather than removing it.
    """
    by_key = {r["arm_key"]: r for r in reports}
    out = {}

    barrier = by_key.get("e192_barrier_arm")
    if barrier:
        arm0 = barrier["per_arm"]["arm0"]
        wall = arm0.get("clear_release_us_median")
        cpu = arm0.get("clear_release_cpu_us_median")
        if wall:
            out["verdict/cpuOverWallUnbarriered"] = cpu / wall
            out["verdict/inFlightAccount"] = (
                "falsified" if cpu / wall > 0.9 else "supported")
        rel = barrier["paired"].get("clear_release_us", {})
        bar = barrier["paired"].get("clear_barrier_us", {})
        if rel and bar:
            out["verdict/barrierRecoveredUs"] = -rel["mean_arm1_minus_arm0"]
            out["verdict/barrierCostUs"] = bar["mean_arm1_minus_arm0"]
            out["verdict/barrierNetUs"] = (
                bar["mean_arm1_minus_arm0"] + rel["mean_arm1_minus_arm0"])

    tape = by_key.get("e192_tape_suppressed_arm")
    if tape:
        rel = tape["paired"].get("clear_release_us", {})
        rnd = tape["paired"].get("round_us", {})
        if rel:
            out["verdict/clearSiteRemovedUs"] = -rel["mean_arm1_minus_arm0"]
        if rnd:
            out["verdict/roundDeltaUs"] = rnd["mean_arm1_minus_arm0"]
            out["verdict/roundDeltaTwoSigmaUs"] = rnd["two_sigma"]
            out["verdict/roundDeltaExcludesZero"] = rnd["excludes_zero"]
            out["verdict/roundLevelCeilingUs"] = -(
                rnd["mean_arm1_minus_arm0"] - rnd["two_sigma"])
            out["verdict/relocationAccount"] = (
                "supported" if not rnd["excludes_zero"] else "rejected")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="+", required=True)
    ap.add_argument("--stage", default="stage0")
    ap.add_argument("--group", default="qwen38-r1-e192-rollback-release-cost")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    reports = [leg_report(tag) for tag in args.legs]

    artifacts = pathlib.Path("research/e192-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / f"{args.stage}-report.json").write_text(
        json.dumps(reports, indent=2) + "\n")
    print(f"wrote {artifacts / f'{args.stage}-report.json'}")

    if args.dry_run:
        for report in reports:
            print(json.dumps(flat_summary(report), indent=1))
        return

    import wandb

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai")
    urls = []

    for report in reports:
        run = wandb.init(entity=entity, project=project, group=args.group,
                         name=f"e192-{report['tag']}", reinit=True,
                         config=identity(report, base_sha) | {"stage": args.stage})
        for row in full_acceptance(read_rounds(report["tag"])):
            run.log({k: v for k, v in row.items() if k in SERIES}
                    | {"round": row["round"]}, step=row["round"])
        run.summary.update(flat_summary(report))
        urls.append((report["tag"], run.id, run.url))
        run.finish()

    roll = wandb.init(entity=entity, project=project, group=args.group,
                      name=f"e192-{args.stage}-rollup", reinit=True,
                      config={"experiment": "e192-persistent-rollback-slot",
                              "harness": "local", "stage": args.stage,
                              "baseSha": base_sha,
                              "legs": [r["tag"] for r in reports]})
    summary = {}
    for report in reports:
        for key, value in flat_summary(report).items():
            summary[f"{report['tag']}/{key}"] = value
    summary.update(verdicts(reports))
    roll.summary.update(summary)
    urls.append(("rollup", roll.id, roll.url))
    roll.finish()

    print("\nW&B runs:")
    for tag, rid, url in urls:
        print(f"  {tag:24s} {rid}  {url}")
    (artifacts / f"{args.stage}-wandb.json").write_text(
        json.dumps([{"leg": t, "runId": i, "url": u} for t, i, u in urls],
                   indent=1) + "\n")


if __name__ == "__main__":
    main()
