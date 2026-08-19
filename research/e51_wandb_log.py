#!/usr/bin/env python3
"""Log the E51 exactness-wall dose ladder to W&B, one arm at a time.

The advisor's operational instruction is to log while measuring, never once at
session end: a workspace retag at 15:02Z destroyed another student's end-of-
session legs. So this script resumes one run id and can be called after every
arm.

  research/e51_wandb_log.py --stage step0            # AIR + kernel leg parity
  research/e51_wandb_log.py --stage e2e --arms research/out/e51-r0-a ...

The run id is kept in research/e51-artifacts/wandb-run-id.txt so every call
after the first appends to the same run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e51_row_fingerprint as FP  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e51-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

LADDER = ARTIFACTS / "leg-parity-ladder.json"
MDE = ARTIFACTS / "leg-parity-mde.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e51-exactness-wall-dose-ladder",
        "revision_id": "r1",
        "pr_number": 55,
        "base_sha": "0df93e9f1c8ad2d0d3ecb41f0c799a4c716c1563",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_memory_gb": 48,
        "metal_toolchain": "Apple metal version 32023.883",
        "ranked_toolchain_pin": "com.apple.dt.toolchain.Metal.32023.883",
        "harness": "local",
        "local_mode": "--local-iterate",
        "decode_tokens": 64,
        "patched_twins": [
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h:1029",
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp:1042",
        ],
        "scored_source_form": "JIT string in mlx-generated/quantized.cpp",
        "r0_line": "sums[m] += xm[0] + xm[1] + xm[2] + xm[3];",
        "r1_line": "sums[m] += (xm[0] + xm[1]) + (xm[2] + xm[3]);",
        "r2_line": "sums[m] += xc[0] + xc[1] + xc[2] + xc[3];",
    }


def start_run(resume: str | None) -> wandb.sdk.wandb_run.Run:
    run_id = resume
    if run_id is None and RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip() or None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow" if run_id else None,
        name="e51-exactness-wall-dose-ladder",
        group="e51-exactness-wall",
        job_type="exactness-ladder",
        tags=["e51", "exactness", "qmv-affine4-g64", "dose-ladder",
              "one-sided-sensor", "qwen-alphonse"],
        config=identity(),
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


def log_step0(run) -> None:
    """Kernel-level leg parity, both the ladder and the MDE calibration."""
    tables: dict[str, wandb.Table] = {}
    summary: dict[str, float | bool | str] = {}

    for label, path in (("ladder", LADDER), ("mde", MDE)):
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        columns = ["arm", "shape", "k", "n", "m", "self_stable",
                   "legs_equal", "legs_mismatch_fraction", "legs_max_ulp",
                   "legs_max_rel", "wide_vs_ref_equal", "serial_vs_ref_equal"]
        table = wandb.Table(columns=columns)
        per_arm: dict[str, list] = {}
        for row in blob["entries"]:
            legs = row["legs"]
            table.add_data(
                row["arm"], row["shape"], row["k"], row["n"], row["m"],
                row["self_stable"], legs["equal"], legs["mismatch_fraction"],
                legs["max_ulp_delta"], legs["max_rel_delta"],
                row["wide_vs_ref"]["equal"], row["serial_vs_ref"]["equal"])
            per_arm.setdefault(row["arm"], []).append(row)

        tables[f"step0a/{label}"] = table
        for arm, rows in sorted(per_arm.items()):
            cells = len(rows)
            diverged = sum(0 if r["legs"]["equal"] else 1 for r in rows)
            fractions = [r["legs"]["mismatch_fraction"] for r in rows
                         if not r["legs"]["equal"]]
            key = f"step0a/{label}/{arm}"
            summary[f"{key}/cells"] = cells
            summary[f"{key}/cells_diverged"] = diverged
            summary[f"{key}/legs_bit_identical"] = diverged == 0
            summary[f"{key}/serial_vs_ref_all_equal"] = all(
                r["serial_vs_ref"]["equal"] for r in rows)
            summary[f"{key}/max_ulp_delta"] = max(
                r["legs"]["max_ulp_delta"] for r in rows)
            if fractions:
                summary[f"{key}/median_mismatch_fraction"] = statistics.median(
                    fractions)
                summary[f"{key}/max_mismatch_fraction"] = max(fractions)
        summary[f"step0a/{label}/device"] = blob["device"]
        summary[f"step0a/{label}/repeats"] = blob["repeats"]

    # Step 0b, the AIR dose check. These are the published canonical-line
    # differences under each math mode, with safe math being the scored one.
    summary.update({
        "step0b/scored_math_mode": "safe (setFastMathEnabled(false))",
        "step0b/fast_math_call_site":
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
            "device.cpp:631",
        "step0b/r0_vs_r0b_safe_differs": False,
        "step0b/r0_vs_r1_safe_differs": True,
        "step0b/r0_vs_r1_safe_lines": 40,
        "step0b/r0_vs_r1_isolated_tree_lines": 12,
        "step0b/r0_vs_r1_default_lines": 4,
        "step0b/r0_vs_r1_fast_lines": 18,
        "step0b/r0_vs_r2_safe_lines": 11017,
        "step0b/fadd_bfloat_entry_r0": 84,
        "step0b/fadd_bfloat_entry_r1": 84,
        "step0b/fadd_bfloat_entry_r2": 57,
        "step0b/fadd_float_entry_r0": 48,
        "step0b/fadd_float_entry_r2": 75,
        "step0b/bf16_tree_survives_compiler": True,
        "step0b/advisor_prediction_1_refuted": True,
        "gates/invariants_base_result": "PASS 11/11",
        "gates/invariants_r1_result": "FAIL 2/11",
        "gates/invariants_r1_rows_fired":
            "wide-cell present-check for the pinned bias line, in both twins",
        "gates/invariants_serial_rows_still_green": True,
        "gates/twin_audit_base": "OK (comment-only case-8 waiver)",
        "gates/twin_audit_r1": "STALE (waiver digests no longer match)",
        "gates/twin_audit_r1_is_bookkeeping_only": True,
    })

    run.log(tables) if tables else None
    run.summary.update(summary)


def log_e2e(run, arms: list[pathlib.Path]) -> None:
    records = [FP.collect(p) for p in arms]
    columns = ["arm", "row_count", "distinct_positions",
               "row_evidence_fingerprint", "schedule_fingerprint",
               "all_tokens_matched", "residual_divergence_count",
               "effective_mean_draft_len", "accepted_draft_rate", "score",
               "serial_seconds_per_token", "mtp_seconds_per_token",
               "base_sha", "started", "finished", "exit"]

    def get(rec: dict, name: str):
        for key in (name, f"metrics.{name}", f"meta.{name}"):
            if key in rec:
                return rec[key]
        return None

    table = wandb.Table(columns=columns)
    for rec in records:
        table.add_data(*[rec.get(c) if c in rec else get(rec, c)
                         for c in columns])

    summary: dict = {}
    reference = next((r for r in records if r["arm"].endswith("r0-a")),
                     records[0])
    for rec in records:
        arm = rec["arm"]
        summary[f"e2e/{arm}/row_count"] = rec["row_count"]
        summary[f"e2e/{arm}/distinct_positions"] = rec["distinct_positions"]
        summary[f"e2e/{arm}/row_evidence_fingerprint"] = \
            rec["row_evidence_fingerprint"]
        summary[f"e2e/{arm}/schedule_fingerprint"] = rec["schedule_fingerprint"]
        for name in ("all_tokens_matched", "residual_divergence_count",
                     "effective_mean_draft_len", "accepted_draft_rate",
                     "score", "serial_seconds_per_token",
                     "mtp_seconds_per_token"):
            value = get(rec, name)
            if value is not None:
                summary[f"e2e/{arm}/{name}"] = value
        if rec is not reference:
            moved = (rec["row_evidence_fingerprint"]
                     != reference["row_evidence_fingerprint"])
            summary[f"e2e/{arm}/row_evidence_moved_vs_r0"] = moved
            summary[f"e2e/{arm}/schedule_moved_vs_r0"] = (
                rec["schedule_fingerprint"]
                != reference["schedule_fingerprint"])

    summary["e2e/primary_signal"] = (
        "canonical per-position row-evidence digest over ^mtp-row: lines")
    summary["e2e/preregistered_signal_status"] = (
        "ordered-line digest failed its own A/A control; retained as the "
        "schedule fingerprint and reported for every arm")
    summary["e2e/sensor_is_one_sided"] = True

    run.log({"e2e/arms": table})
    run.summary.update(summary)


def log_margin(run, paths: list[pathlib.Path]) -> None:
    """Decisive-margin distribution for whichever arm moved the sensor."""
    columns = ["arm", "pos", "top1_id_ref", "top1_id_cand", "top1_id_flipped",
               "top2_id_flipped", "margin", "delta_logit", "ratio",
               "values_moved"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for path in paths:
        blob = json.loads(path.read_text())
        arm = blob["summary"]["candidate"]
        for row in blob["per_position"]:
            table.add_data(arm, *[row[c] for c in columns[1:]])
        for key, value in blob["summary"].items():
            if isinstance(value, (int, float, bool)) or value is None:
                summary[f"margin/{arm}/{key}"] = value
        summary[f"margin/{arm}/flipped_positions"] = str(
            blob["summary"]["flipped_positions"])
        summary[f"margin/{arm}/top1_id_flips"] = sum(
            1 for r in blob["per_position"] if r["top1_id_flipped"])
        summary[f"margin/{arm}/top2_only_id_flips"] = sum(
            1 for r in blob["per_position"]
            if r["top2_id_flipped"] and not r["top1_id_flipped"])
    run.log({"margin/per_position": table})
    run.summary.update(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["step0", "e2e", "margin"])
    parser.add_argument("--arms", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument("--resume")
    args = parser.parse_args()

    run = start_run(args.resume)
    if args.stage == "step0":
        log_step0(run)
    elif args.stage == "margin":
        if not args.arms:
            raise SystemExit("--stage margin needs --arms (margin JSON paths)")
        log_margin(run, args.arms)
    else:
        if not args.arms:
            raise SystemExit("--stage e2e needs --arms")
        log_e2e(run, args.arms)
    print(f"wandb run: {run.url}")
    print(f"run id: {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
