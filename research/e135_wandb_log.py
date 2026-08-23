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

SESSION_META = {
    "grid": {
        "group": "e135-tight-qmv-launch-grid",
        "experiment": "e135-tight-qmv-launch-grid",
        "question": (
            "does deleting the no-op threadgroups from the wide QMV launch "
            "grid make the candidate leg faster, and how much of the ranked "
            "3388.5 us per unit verify width is launch cost rather than work"),
        "arms": "wide (shipped default) vs tight (MLX_E120_QMV_GRID=tight)",
        "script": "research/e135_grid_abba.sh",
        "name": "tight-vs-wide-launch-grid",
        "config": {
            "qmv_table": "onepass67",
            # Rung 0 identity. The grid selector is a host-side dispatch
            # argument, so both arms compile the same pipelines and allocate
            # the same registers. That is what rules out the FINDING 181 clamp
            # mechanism.
            "grid_changes_any_pipeline_cache_key": False,
            "grid_changes_any_entry_point_register_count": False,
            "threadgroup_columns_per_round": 519040,
            "qmv_dispatches_per_round": 257,
            "swift_test_issues": 40,
            "swift_test_named_failures": 9,
            "swift_test_campaign_added_failures": 0,
        },
    },
    "e87": {
        "group": "e135-e87-probe-select",
        "experiment": "e135-e87-probe-select",
        "question": (
            "does the restored E87 one-dispatch probe select make the "
            "candidate leg faster than the incumbent argPartition plus "
            "probe-sort chain, at bit-identical draft lengths"),
        "arms": "incumbent (MLX_E87_SELECT=0) vs select (MLX_E87_SELECT=1)",
        "script": "research/e135_e87_select_abba.sh",
        "name": "e87-select-vs-argpartition",
        "config": {
            # Unlike the grid selector, this arm DOES change the compiled
            # pipeline set: one arm JITs `qwen_mtp_e87_probe_select` and the
            # other JITs `qwen_mtp_probe_sort`. Rule 128 says the interaction
            # with anything sharing the JIT library cache must be measured, so
            # the key digests are expected to differ and are logged as such.
            "select_changes_pipeline_cache_key": True,
            "probe_arm": "p15",
            "probe_leaves": 12292,
            "probe_count": 1844,
            "select_verify_trials": 64,
            "select_verify_mismatches": 0,
            "select_positive_control_detected": True,
            "select_rung_sweep_mismatches": 0,
        },
    },
}


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


def arm_witnesses(label: str) -> dict[str, list[str]]:
    """Name the pipelines each witness leg actually compiled.

    Rule 101 wants every witness to have a failing polarity. Recording the
    named pipelines per arm makes that checkable after the fact: an arm that
    claims to run a mechanism must show its pipeline, and must not show the
    other arm's.
    """
    out = {}
    for arm in report.ARMS:
        path = pathlib.Path(f"research/out/e135{label}w{arm}/pipelines.json")
        if not path.exists():
            continue
        by_key = json.loads(path.read_text()).get("by_key", {})
        out[arm] = sorted(
            k for k in by_key if not k.startswith("e135_default_probe"))
    return out


def per_width_table(label: str) -> dict:
    path = pathlib.Path(f"research/e135-artifacts/{label}-per-width.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s1")
    ap.add_argument("--session", default="grid", choices=sorted(SESSION_META))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    report.configure(args.session)
    spec = SESSION_META[args.session]

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
        "experiment": spec["experiment"],
        "question": spec["question"],
        "harness": "local",
        "session_label": args.label,
        "decode_tokens": int(meta.get("tokens", 0)),
        "local_mode": meta.get("local_mode"),
        "arms": spec["arms"],
        "design": "R C C R palindrome, arm code orthogonal to centred leg index",
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
        "reproduce": (f"{spec['script']} {max(1, len(complete) // 4)} "
                      f"{int(meta.get('tokens', 0))} {args.label} 1"),
        "pipeline_by_key_sha256": key_digest,
        "pipeline_by_key_identical_across_arms": keys_agree,
        "arm_witness_pipelines": arm_witnesses(args.label),
        **spec["config"],
    }

    metrics = {
        report.HEADLINE: headline,
        f"{report.HEADLINE}_se": 100 * fit["se"] / fit["mean"],
        "e135_exact_token_divergences": divergences,
        "e135_all_tokens_matched": matched,
        **{f"e135_mtp_seconds_per_token_{arm}": by_arm.get(arm)
           for arm in report.ARMS},
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

    if report.PER_ROUND:
        # The mechanism saves a fixed amount once per drafting round, so its
        # percentage depends on how long a round is. The local fixture drafts
        # deeper than ranked beagle, and a round is 84 % fixed cost, so the
        # same saving reads slightly larger at beagle depth.
        beagle = headline * report.LOCAL_TO_BEAGLE_ROUND
        metrics.update({
            "e135_local_tokens_per_round": (1.0 + drafts[0]) if drafts else None,
            "e135_beagle_tokens_per_round": report.BEAGLE_TOKENS_PER_ROUND,
            "e135_depth_correction": report.LOCAL_TO_BEAGLE_ROUND,
            "e135_beagle_equivalent_pct": beagle,
            "e135_beagle_equivalent_se_pct":
                100 * fit["se"] / fit["mean"] * report.LOCAL_TO_BEAGLE_ROUND,
            "e135_per_round_bar_pct": report.PER_ROUND_BAR_PCT,
            "e135_fraction_of_bar_pct": 100 * beagle / report.PER_ROUND_BAR_PCT,
        })

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
        entity=ENTITY, project=PROJECT, group=spec["group"],
        id=args.run_id, name=args.name
        or f"e135-{args.label}-{spec['name']}",
        job_type="local-abba-session", config=config)
    for r in rows:
        m = r["metrics"]
        run.log({
            "leg_index": r["idx"],
            "leg_is_candidate_arm": 1 if r["arm"] == report.ARMS[1] else 0,
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
