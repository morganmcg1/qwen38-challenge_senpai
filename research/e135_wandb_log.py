#!/usr/bin/env python3
"""Publish one E135 session to W&B.

    usage: research/e135_wandb_log.py --label s1 [--dry]

Every leg in an E135 session runs with `MLXFAST_LOCAL_COOL_GATE=0` under the
standing counterbalanced-arm exception, so each run logs
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false. Nothing here is a ranked score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e135_report as report  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e135-tight-qmv-launch-grid"

QUESTION = (
    "does deleting the no-op threadgroups from the wide QMV launch grid make "
    "the candidate leg faster, and how much of the ranked 3388.5 us per unit "
    "verify width is launch cost rather than work"
)


def pipeline_key_digest(label: str) -> tuple[str | None, bool]:
    """Digest the compiled-pipeline key set of both witness legs.

    Returns the shared digest and whether the two arms agreed. The grid
    selector is a host-side `dispatchThreadgroups` argument, so an identical
    key set is the observable form of "both arms run the same compiled code".
    """
    digests = {}
    for arm in report.ARMS:
        path = pathlib.Path(f"research/out/e135{label}w{arm}/pipelines.json")
        if not path.exists():
            return None, False
        by_key = json.loads(path.read_text()).get("by_key", {})
        blob = json.dumps(by_key, sort_keys=True).encode()
        digests[arm] = hashlib.sha256(blob).hexdigest()
    shared = set(digests.values())
    return digests[report.ARMS[0]], len(shared) == 1


def per_width_table(label: str) -> dict:
    path = pathlib.Path(f"research/e135-artifacts/{label}-per-width.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s1")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = report.legs(args.label)
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]
    if not complete:
        print("e135_wandb_log: no scored legs")
        return 1

    fit = report.ols_arm_and_drift(complete, "mtp_seconds_per_token")
    ratio_fit = report.ols_arm_and_drift(complete, "mtp_decode_speedup")
    serial_fit = report.ols_arm_and_drift(complete, "serial_seconds_per_token")

    headline = -100 * fit["contrast"] / fit["mean"]
    by_arm = {}
    for arm in report.ARMS:
        v = [report.fnum(r["metrics"]["mtp_seconds_per_token"])
             for r in complete if r["arm"] == arm]
        if v:
            by_arm[arm] = statistics.fmean(v)

    divergences = sum(
        int(r["metrics"].get("residual_divergence_count", 0) or 0)
        for r in complete)
    matched = all(r["metrics"].get("all_tokens_matched") is True
                  for r in complete)
    drafts = sorted({report.fnum(r["metrics"].get("effective_mean_draft_len"))
                     for r in complete} - {None})

    meta = complete[0]["meta"]
    key_digest, keys_agree = pipeline_key_digest(args.label)
    config = {
        "experiment": "e135-tight-qmv-launch-grid",
        "question": QUESTION,
        "harness": "local",
        "session_label": args.label,
        "decode_tokens": int(meta.get("tokens", 0)),
        "local_mode": meta.get("local_mode"),
        "arms": "wide (shipped default) vs tight (MLX_E120_QMV_GRID=tight)",
        "design": "W T T W palindrome, arm code orthogonal to centred leg index",
        "legs": len(complete),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_bytes": meta.get("memory_bytes"),
        "sandbox": meta.get("sandbox"),
        "head_dir": meta.get("head_dir"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "qmv_table": "onepass67",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "reproduce": f"research/e135_grid_abba.sh 2 512 {args.label} 1",
        # Rung 0 identity. The grid selector is a host-side dispatch argument,
        # so both arms compile the same pipelines and allocate the same
        # registers. That is what rules out the FINDING 181 clamp mechanism.
        "pipeline_by_key_sha256": key_digest,
        "pipeline_by_key_identical_across_arms": keys_agree,
        "grid_changes_any_pipeline_cache_key": False,
        "grid_changes_any_entry_point_register_count": False,
        "threadgroup_columns_per_round": 519040,
        "qmv_dispatches_per_round": 257,
        "swift_test_issues": 40,
        "swift_test_named_failures": 9,
        "swift_test_campaign_added_failures": 0,
    }

    metrics = {
        "e135_tight_grid_candidate_leg_pct": headline,
        "e135_tight_grid_candidate_leg_pct_se": 100 * fit["se"] / fit["mean"],
        "e135_exact_token_divergences": divergences,
        "e135_all_tokens_matched": matched,
        "e135_mtp_seconds_per_token_wide": by_arm.get("wide"),
        "e135_mtp_seconds_per_token_tight": by_arm.get("tight"),
        "e135_mtp_contrast_seconds_per_token": fit["contrast"],
        "e135_residual_sd_seconds_per_token": fit["sigma"],
        "e135_drift_per_leg_seconds_per_token": fit["drift_per_leg"],
        "e135_schedule_identical": len(drafts) == 1,
        "e135_effective_mean_draft_len": drafts[0] if drafts else None,
    }
    if ratio_fit:
        metrics["e135_local_ratio_contrast"] = ratio_fit["contrast"]
        metrics["e135_local_ratio_pct"] = (
            100 * ratio_fit["contrast"] / ratio_fit["mean"])
        metrics["e135_local_ratio_se_pct"] = (
            100 * ratio_fit["se"] / ratio_fit["mean"])
    if serial_fit:
        # Same faster-is-positive convention as the headline, so the two read
        # the same way. The serial leg must not move: it shares the candidate
        # binary but never dispatches the QMV grid under test.
        metrics["e135_serial_leg_pct"] = (
            -100 * serial_fit["contrast"] / serial_fit["mean"])
        metrics["e135_serial_leg_se_pct"] = (
            100 * serial_fit["se"] / serial_fit["mean"])

    extra = per_width_table(args.label)
    for key in ("e135_launch_cost_us_per_column",
                "e135_launch_cost_us_per_column_se",
                "e135_launch_cost_r2",
                "e135_ranked_launch_share_pct",
                "e135_onepass_gain_under_tight_pct"):
        if key in extra:
            metrics[key] = extra[key]

    print(json.dumps({"config": config, "metrics": metrics}, indent=2,
                     default=str))
    if args.dry:
        return 0

    import wandb

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        id=args.run_id, name=args.name
        or f"e135-{args.label}-tight-vs-wide-launch-grid",
        job_type="local-abba-session", config=config)
    for r in rows:
        m = r["metrics"]
        run.log({
            "leg_index": r["idx"],
            "leg_grid_is_tight": 1 if r["arm"] == "tight" else 0,
            "leg_mtp_seconds_per_token":
                report.fnum(m.get("mtp_seconds_per_token")),
            "leg_serial_seconds_per_token":
                report.fnum(m.get("serial_seconds_per_token")),
            "leg_local_ratio": report.fnum(m.get("mtp_decode_speedup")),
            "leg_gpu_temp_entry_c":
                report.fnum(r["meta"].get("gpu_temp_entry_c")),
            "leg_gpu_temp_exit_c":
                report.fnum(r["meta"].get("gpu_temp_exit_c")),
            "leg_effective_mean_draft_len":
                report.fnum(m.get("effective_mean_draft_len")),
            "leg_residual_divergence_count":
                m.get("residual_divergence_count"),
        })
    run.summary.update(metrics)
    if extra.get("per_width"):
        run.summary["e135_per_width"] = extra["per_width"]
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
