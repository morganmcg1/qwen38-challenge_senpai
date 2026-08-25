#!/usr/bin/env python3
"""Publish the E205 stage-A legs and the stage-B curve to W&B.

    python3 research/e205_wandb.py --legs e205a-alt4 e205a-alt1 e205a-alt2 \
        [--dry-run]

harness=local for stage A, harness=ranked for the stage-B desk curve. One run
per timed leg carries the full identity tuple in `config`, the per-round series
by round index, and the paired forced-rejection contrast in `summary`. One
roll-up run carries the FINDING 494 verdicts and the stage-B reading table.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e205_report import (  # noqa: E402
    BAR_US, INT_FIELDS, MUE_US, PAIRED_FIELDS, classify, leg_report,
    premium_by_replayed_rows, read_meta, read_rounds, verdicts,
)

SERIES = [f for f in INT_FIELDS if f != "round"]
ARTIFACTS = pathlib.Path("research/e205-artifacts")
ASSIGNMENT_BASE = "2a748cf054e495fe087815636424c7bf754acd24"
RECEIPT_A = 3.70784519415395


def identity(report: dict, base_sha: str) -> dict:
    meta = read_meta(report["tag"])
    return {
        "harness": "local",
        "experiment": "e205-rejection-round-cost",
        "stage": "A",
        "leg": report["tag"],
        "armEnv": report["arm_env"],
        "instrument": meta.get("e205_instrument", ""),
        "baseSha": meta.get("base_sha", base_sha),
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "workerSha256": meta.get("worker_sha256", ""),
        "postRunWorkerSha256": meta.get("post_run_worker_sha256", ""),
        "sessionWorkerSha256": meta.get("e205_worker_sha256_session", ""),
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
        "forcedScheduleIsOpenLoop": True,
        "pricedAtRoundEndpoint": True,
    }


def flat_summary(report: dict) -> dict:
    out = {
        "traceRounds": report["trace_rounds"],
        "committedTokensFromTrace": report["committed_tokens_from_trace"],
        "exit": report["exit"],
    }
    for kind, count in report["census"].items():
        out[f"census/{kind}"] = count
    for depth, count in report["depth_hist"].items():
        out[f"depth/{depth}"] = count
    for applied, count in report["applied_hist"].items():
        out[f"appliedJ/{applied}"] = count
    for path, count in report["repair_path_hist"].items():
        out[f"repairPath/{path}"] = count
    blocks = [("all", report["paired_all"])]
    blocks += [(f"j{k}", v) for k, v in report["paired_by_applied_j"].items()]
    blocks += [(f"rows{k}", v)
               for k, v in report["paired_by_replayed_rows"].items()]
    for label, block in blocks:
        if not block.get("pair_count"):
            continue
        out[f"{label}/pairCount"] = block["pair_count"]
        out[f"{label}/baselineRoundUs"] = block.get("baseline_round_us_mean")
        out[f"{label}/replayedRowsMean"] = block.get("replayed_rows_mean")
        for field in PAIRED_FIELDS:
            stats = block.get(field)
            if not stats:
                continue
            out[f"{label}/{field}/meanUs"] = stats["mean"]
            out[f"{label}/{field}/twoSigmaUs"] = stats["two_sigma"]
            out[f"{label}/{field}/excludesZero"] = stats["excludes_zero"]
    return out


def stage_b_summary(curve: dict) -> dict:
    """Flatten the stage-B curve JSON into scalar summary keys."""
    model = curve["premium_model"]
    anchor = curve["anchor"]
    out = {
        "stageB/harness": curve["harness"],
        "stageB/premiumModelKind": model["kind"],
        "stageB/premiumConstMs": model["const_ms"],
        "stageB/premiumSlopeMsPerRow": model["slope_ms_per_row"],
        "stageB/premiumInferred": model["inferred"],
        "stageB/premiumSource": model["source"],
        "stageB/transferRatio": curve["transfer_ratio"],
        "stageB/publicFixtureA": curve["public_fixture_a"],
        "stageB/receiptA": anchor["receipt_A"],
        "stageB/crown": anchor["crown"],
        "stageB/cap7AtMeasured": anchor["cap7_at_measured_acceptance"],
        "stageB/anchorDelta": (anchor["cap7_at_measured_acceptance"]
                               - anchor["receipt_A"]),
        "stageB/cap8Flat": anchor["cap8_flat_at_measured_acceptance"],
        "stageB/cap8Step": anchor["cap8_step_at_measured_acceptance"],
        "stageB/e197Flat": anchor["e197_flat"],
        "stageB/e197Step": anchor["e197_step"],
    }
    for row in curve["reading_table"]:
        key = f"{row['observed']:.4f}".replace(".", "p")
        out[f"stageB/reading/{key}"] = row["reading"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="+", required=True)
    ap.add_argument("--group", default="qwen38-r1-e205-rejection-round-cost")
    ap.add_argument("--stage-b", dest="stage_b",
                    default="research/e205-artifacts/stageB-curve.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    reports = [leg_report(tag) for tag in args.legs]
    verdict = verdicts(reports)
    points = premium_by_replayed_rows(reports)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "stageA-report.json").write_text(json.dumps(
        {"legs": reports, "verdicts": verdict,
         "premium_by_replayed_rows": points,
         "source": "e205 stage A forced-truncation paired endpoint, "
                   + ",".join(args.legs)}, indent=2) + "\n")
    print(f"wrote {ARTIFACTS / 'stageA-report.json'}")

    curve = None
    stage_b_path = pathlib.Path(args.stage_b)
    if stage_b_path.exists():
        curve = json.loads(stage_b_path.read_text())

    if args.dry_run:
        for report in reports:
            print(json.dumps(flat_summary(report), indent=1))
        print(json.dumps(verdict, indent=1))
        if curve:
            print(json.dumps(stage_b_summary(curve), indent=1))
        return

    import wandb

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai")
    urls = []

    for report in reports:
        run = wandb.init(entity=entity, project=project, group=args.group,
                         name=f"e205-{report['tag']}", reinit=True,
                         config=identity(report, base_sha))
        for row in read_rounds(report["tag"]):
            if row.get("d", 0) == 0:
                continue
            kind = classify(row)
            payload = {k: v for k, v in row.items() if k in SERIES}
            payload["round"] = row["round"]
            payload["isForcedRejection"] = 1 if kind == "trunc" else 0
            payload["isFullAcceptance"] = 1 if kind == "full" else 0
            payload["replayedRows"] = row.get("acc", 0) + 1
            run.log(payload, step=row["round"])
        run.summary.update(flat_summary(report))
        urls.append((report["tag"], run.id, run.url))
        run.finish()

    roll_config = {
        "experiment": "e205-rejection-round-cost",
        "stage": "A+B",
        "harness": "local+ranked",
        "baseSha": base_sha,
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "legs": [r["tag"] for r in reports],
        "compositionBarUs": BAR_US,
        "minimumUsefulEffectUs": MUE_US,
        "forcedScheduleIsOpenLoop": True,
        "pricedAtRoundEndpoint": True,
    }
    roll = wandb.init(entity=entity, project=project, group=args.group,
                      name="e205-rollup", reinit=True, config=roll_config)
    summary = {}
    for report in reports:
        for key, value in flat_summary(report).items():
            summary[f"{report['tag']}/{key}"] = value
    for key, value in verdict.items():
        for field, item in value.items():
            summary[f"verdict/{key}/{field}"] = item
    for point in points:
        rows = point["replayed_rows"]
        summary[f"premiumByRows/{rows}/ms"] = point["endpoint_premium_ms"]
        summary[f"premiumByRows/{rows}/pairs"] = point["weight"]
    if curve:
        summary.update(stage_b_summary(curve))
    roll.summary.update(summary)
    urls.append(("rollup", roll.id, roll.url))
    roll.finish()

    print("\nW&B runs:")
    for tag, rid, url in urls:
        print(f"  {tag:24s} {rid}  {url}")
    (ARTIFACTS / "wandb.json").write_text(
        json.dumps([{"leg": t, "runId": i, "url": u} for t, i, u in urls],
                   indent=1) + "\n")


if __name__ == "__main__":
    main()
