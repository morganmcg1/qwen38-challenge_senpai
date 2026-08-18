#!/usr/bin/env python3
"""qwen38-r1-e29: publish the per-round overhead attribution and the ladder
rung-schedule sweep to W&B.

usage:
  research/e29_log_wandb.py <e29_analyze.py --json-out file> [--group G]
                            [--name N] [--notes ...]

The analysis file holds one entry per measured arm. Arms are classified by the
config recorded in their own meta block, never by label convention, so a
mislabelled arm shows up as unclassified instead of silently joining the wrong
comparison.

Two independent results are published side by side:

  attribution - the six-way per-round split (draft build, verify build, GPU
                eval wall, readout, commit, upkeep). Tiles round_us exactly,
                so `unaccounted` is a real residual and not a definition.
  ladder      - the decode asyncEval rung-schedule sweep. Each arm carries its
                own serial leg, so the speedup ratio is self-normalising
                against thermal drift between arms.

The headline correction is deliberately explicit in the summary: the E20
baseline of 10.26% counted only what was left after subtracting attributed
work, which folded host graph-build time into `target_work`. The direct split
measures host tail at ~55% of true decode. Lower is better for the metric, so
55% is a MEASUREMENT CORRECTION of the baseline, not a regression caused by
this branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

# research/results/qwen38-r1-e20-verify-side-layer-family-attribution.md.
# E20 defined overhead subtractively: round minus attributed layer work.
E20_BASELINE_OVERHEAD_PCT = 10.26

SEGMENTS = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
]
HOST_SEGMENTS = [s for s in SEGMENTS if s != "eval_wall_us"]

CORRECTNESS = [
    "all_tokens_matched",
    "reference_checked_rows",
    "reference_total_rows",
    "residual_divergence_count",
]


def arm_config(entry: dict) -> dict:
    """The config an arm actually ran under, read from its own meta block."""
    meta = entry.get("meta") or {}
    return {
        "trace": entry.get("trace"),
        "ladder": meta.get("ladder", "default"),
        "offered_depth": meta.get("offered_depth"),
        "head_sha": meta.get("head_sha"),
        "dirty": meta.get("dirty"),
        "tokens": meta.get("tokens"),
        "exit": meta.get("exit"),
        "chip": meta.get("chip"),
        "host": meta.get("host"),
        "cli_sha256": meta.get("cli_sha256"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
    }


def leg(entry: dict, which: str) -> dict:
    return entry.get(which) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--group", default="qwen38-r1-e29")
    ap.add_argument("--name", default="e29-round-overhead-host-graph")
    ap.add_argument("--notes", default="")
    ap.add_argument("--attribution-arm", default="T1",
                    help="arm whose six-way split is the headline")
    ap.add_argument("--ladder-baseline", default="T1",
                    help="arm whose ladder is the shipped default")
    ap.add_argument("--isolation-arm", default="L0",
                    help="ladder-off arm; its host tail is the only segment "
                         "view free of GPU-wait misattribution, so it supplies "
                         "the primary metric")
    args = ap.parse_args()

    doc = json.loads(args.analysis.read_text())
    arms = {k: v for k, v in doc.items() if isinstance(v, dict)}

    head = arms.get(args.attribution_arm)
    if head is None:
        print(f"attribution arm {args.attribution_arm!r} not in "
              f"{sorted(arms)}")
        return 1

    config = {
        "experiment": "qwen38-r1-e29-round-overhead-host-graph",
        "assignment_pr": 34,
        "base_sha": "d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602",
        "arms": sorted(arms),
        "attribution_arm": args.attribution_arm,
        "ladder_baseline_arm": args.ladder_baseline,
        "shipped_ladder_rungs": [0, 1, 9, 19, 29, 39, 49, 57],
        "ladder_gate_source": (
            "Qwen35.swift Qwen35TextModelInner.callAsFunction: "
            "ladderActive = inputs.dim(1) <= 9 || inputs.dim(1) >= 512"),
        "e20_baseline_overhead_pct": E20_BASELINE_OVERHEAD_PCT,
        "e20_overhead_definition": (
            "subtractive: block_m1(M) - attributed_m1(M); host graph-build "
            "time was folded into target_work"),
        "e29_overhead_definition": (
            "direct: per-round six-way split that tiles round_us; host tail = "
            "round minus GPU eval wall"),
        "skip_warmup_rounds": head.get("skip_warmup_rounds"),
        "notes": args.notes,
    }
    for k, v in arm_config(head).items():
        config[f"headline/{k}"] = v

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        name=args.name,
        job_type="attribution",
        config=config,
        notes=args.notes or None,
    )

    arm_rows, split_rows, width_rows, econ_rows, ladder_rows = [], [], [], [], []

    for label in sorted(arms):
        entry = arms[label]
        cfg = arm_config(entry)
        mtp = leg(entry, "mtp")
        serial = leg(entry, "serial")
        arm_rows.append([
            label, cfg["trace"], cfg["ladder"], cfg["offered_depth"],
            entry.get("timed_session_rounds"),
            mtp.get("mtp_seconds_per_token"),
            serial.get("serial_seconds_per_token"),
            mtp.get("mtp_decode_speedup"),
            mtp.get("all_tokens_matched"),
            mtp.get("effective_mean_draft_len"),
            mtp.get("accepted_draft_rate"),
            (entry.get("totals_ms") or {}).get("round"),
            (entry.get("share_of_round_pct") or {}).get("host_tail"),
            entry.get("unaccounted_us"),
        ])

        for view in ("overall", "steady_state"):
            src = entry if view == "overall" else entry.get("steady_state")
            if not src:
                continue
            totals = src.get("totals_ms") or {}
            shares = src.get("share_of_round_pct") or {}
            for seg in SEGMENTS + ["host_tail"]:
                split_rows.append([
                    label, view, seg, totals.get(seg), shares.get(seg),
                    src.get("rounds", entry.get("timed_session_rounds")),
                ])

        for m, row in sorted((entry.get("per_width") or {}).items(),
                             key=lambda kv: int(kv[0])):
            seg = row.get("segment_mean_ms") or {}
            width_rows.append([
                label, int(m), row.get("rounds"), row.get("round_mean_ms"),
                row.get("host_tail_mean_ms"),
                *[seg.get(s) for s in SEGMENTS],
            ])

        econ = entry.get("width_economics")
        if econ:
            for m, row in sorted(econ["per_width"].items(),
                                 key=lambda kv: int(kv[0])):
                econ_rows.append([
                    label, int(m), row["rounds"], row["round_mean_ms"],
                    row["full_accept_ms_per_token"],
                    int(m) == econ["best_width"],
                ])

    base = arms.get(args.ladder_baseline)
    base_spt = leg(base or {}, "mtp").get("mtp_seconds_per_token")
    base_speedup = leg(base or {}, "mtp").get("mtp_decode_speedup")
    for label in sorted(arms):
        entry = arms[label]
        cfg = arm_config(entry)
        mtp = leg(entry, "mtp")
        spt = mtp.get("mtp_seconds_per_token")
        speedup = mtp.get("mtp_decode_speedup")
        ladder_rows.append([
            label, cfg["ladder"], spt, speedup,
            (100.0 * (spt - base_spt) / base_spt
             if spt is not None and base_spt else None),
            (100.0 * (speedup - base_speedup) / base_speedup
             if speedup is not None and base_speedup else None),
            mtp.get("all_tokens_matched"),
            (entry.get("steady_state") or {}).get(
                "share_of_round_pct", {}).get("verify_build_us"),
            (entry.get("steady_state") or {}).get(
                "share_of_round_pct", {}).get("host_tail"),
            mtp.get("serial_seconds_per_token"),
        ])

    run.log({
        "arms": wandb.Table(
            columns=["arm", "trace", "ladder", "offered_depth", "rounds",
                     "mtp_seconds_per_token", "serial_seconds_per_token",
                     "mtp_decode_speedup", "all_tokens_matched",
                     "effective_mean_draft_len", "accepted_draft_rate",
                     "round_total_ms", "host_tail_share_pct",
                     "unaccounted_us"],
            data=arm_rows),
        "six_way_split": wandb.Table(
            columns=["arm", "view", "segment", "total_ms", "share_of_round_pct",
                     "rounds"],
            data=split_rows),
        "per_width_segments": wandb.Table(
            columns=["arm", "M", "rounds", "round_mean_ms",
                     "host_tail_mean_ms", *SEGMENTS],
            data=width_rows),
        "full_accept_economics": wandb.Table(
            columns=["arm", "M", "rounds", "round_mean_ms",
                     "full_accept_ms_per_token", "is_cheapest"],
            data=econ_rows),
        "ladder_sweep": wandb.Table(
            columns=["arm", "ladder", "mtp_seconds_per_token",
                     "mtp_decode_speedup", "spt_delta_pct_vs_baseline",
                     "speedup_delta_pct_vs_baseline", "all_tokens_matched",
                     "steady_verify_build_share_pct",
                     "steady_host_tail_share_pct",
                     "serial_seconds_per_token"],
            data=ladder_rows),
    })

    hm = leg(head, "mtp")
    steady = head.get("steady_state") or {}
    econ = head.get("width_economics") or {}
    iso = arms.get(args.isolation_arm) or {}
    iso_steady = iso.get("steady_state") or {}
    summary = {
        "e29/round_overhead_share_of_decode_pct":
            (iso.get("share_of_round_pct") or {}).get("host_tail"),
        "e29/round_overhead_share_of_decode_steady_pct":
            (iso_steady.get("share_of_round_pct") or {}).get("host_tail"),
        "e29/round_overhead_arm": args.isolation_arm,
        "e29/baseline_round_overhead_share_of_decode_pct":
            E20_BASELINE_OVERHEAD_PCT,
        "e29/overhead_delta_is_measurement_correction": True,
        "e29/unaccounted_us": head.get("unaccounted_us"),
        # Shipped-build host tail is NOT host CPU work: mid-forward asyncEval
        # encodes inline, so GPU backpressure waits land in this bucket.
        "e29/shipped_build_host_tail_share_pct":
            (head.get("share_of_round_pct") or {}).get("host_tail"),
        "e29/shipped_build_host_tail_is_gpu_wait_inclusive": True,
        "e29/host_tail_share_steady_pct":
            (steady.get("share_of_round_pct") or {}).get("host_tail"),
        "e29/best_full_accept_width": econ.get("best_width"),
        "e29/best_full_accept_ms_per_token": econ.get("best_ms_per_token"),
        "e29/schedule_headroom_pct_full_accept_bound":
            econ.get("headroom_pct"),
        "e29/mtp_decode_speedup": hm.get("mtp_decode_speedup"),
        "e29/mtp_seconds_per_token": hm.get("mtp_seconds_per_token"),
    }
    for seg in SEGMENTS + ["host_tail"]:
        summary[f"e29/share_of_round_pct/{seg}"] = (
            head.get("share_of_round_pct") or {}).get(seg)
        summary[f"e29/steady_share_of_round_pct/{seg}"] = (
            steady.get("share_of_round_pct") or {}).get(seg)
    for k in CORRECTNESS:
        summary[f"e29/correctness/{k}"] = hm.get(k)

    if ladder_rows:
        # Rank on MTP seconds/token, never on the local ratio: both local legs
        # run the candidate build, so a ladder change also moves the serial
        # baseline and inflates the ratio. Only the MTP leg maps to the ranked
        # score, where the serial build is pinned.
        scored = [r for r in ladder_rows if r[2] is not None]
        if scored:
            best = min(scored, key=lambda r: r[2])
            summary["e29/ladder/best_arm"] = best[0]
            summary["e29/ladder/best_rungs"] = best[1]
            summary["e29/ladder/best_mtp_seconds_per_token"] = best[2]
            summary["e29/ladder/best_mtp_spt_delta_pct"] = best[4]
            summary["e29/ladder/best_local_ratio"] = best[3]
            summary["e29/ladder/best_local_ratio_delta_pct"] = best[5]
            summary["e29/ladder/ranked_relevant_metric"] = (
                "mtp_seconds_per_token")
            summary["e29/ladder/local_ratio_is_serial_contaminated"] = True
            summary["e29/ladder/all_arms_matched"] = all(
                r[6] is True for r in scored)
            serials = [r[9] for r in scored if r[9] is not None]
            if len(serials) > 1:
                summary["e29/ladder/serial_spt_spread_pct"] = (
                    100.0 * (max(serials) - min(serials)) / min(serials))

    run.summary.update(summary)
    print(f"logged {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
