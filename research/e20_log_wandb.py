#!/usr/bin/env python3
"""qwen38-r1-e20: publish the verify-side layer-family attribution to W&B.

usage:
  research/e20_log_wandb.py <e20_analyze.py --json-out file> [--group G]
                            [--notes ...] [--name N]

Everything published here is post-hoc analysis of arms that were already run;
the pre-registered prediction lives in
research/results/qwen38-r1-e20-verify-side-layer-family-attribution.md and is
logged as config so the comparison is visible without leaving the run.
"""

from __future__ import annotations

import argparse
import json
import os

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"

# Pre-registered at commit 42ad911, before any timed run. Shares of verify-side
# decode work, from the 14.77 GB weight-traffic model in section 1.3.
PREREGISTERED = {
    "gdn": 0.28,
    "full_attention": 0.08,
    "mlp": 0.59,
    "head_and_top_two": 0.05,
}

FAMILIES = ["gdn", "full_attention", "mlp", "head_and_top_two", "embed"]

CORRECTNESS = [
    "all_tokens_matched",
    "parity_all_ok",
    "residual_divergence_count",
    "max_rejected_tail_logit_delta",
    "accepted_draft_rate",
    "accepted_draft_total",
    "rejected_draft_total",
    "round_count",
    "declared_rows_total",
    "reference_checked_row_total",
    "emitted_token_total",
    "target_tail_total",
    "uses_native_mtp_head",
]

TIMING = [
    "decode_seconds",
    "seed_prefill_seconds",
    "decode_seconds_ex_prefill",
    "sec_per_token_ex_prefill",
    "prefill_share_of_charged_window",
    "parent_measured_seconds_per_token",
    "decode_token_count",
    "seed_token_count",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--group", default="qwen38-r1-e20")
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--base-sha", default="c0f7e370921a14f348fa1872f2176b1b43028752")
    args = ap.parse_args()

    data = json.load(open(args.analysis))
    any_meta = next(iter(data.values()))["meta"] if data else {}

    config = {
        "experiment": "qwen38-r1-e20-verify-side-layer-family-attribution",
        "assignment_id": "qwen38-r1-e20-verify-side-layer-family-attribution",
        "revision_id": "r1",
        "pr_number": 24,
        "base_sha": args.base_sha,
        "head_sha": any_meta.get("head_sha"),
        "dirty": any_meta.get("dirty"),
        "host": any_meta.get("host"),
        "chip": any_meta.get("chip"),
        "head_dir": any_meta.get("head_dir"),
        "head_provenance_sha256": any_meta.get("head_provenance_sha256"),
        "head_bytes": any_meta.get("head_bytes"),
        "head_dtype": any_meta.get("head_dtype"),
        "tokens": any_meta.get("tokens"),
        "local_mode": "--local-iterate",
        # Carried verbatim, per the assignment: this host idles above the 40C
        # gate, so no arm here is cool-gate qualified.
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "no_sandbox": any_meta.get("no_sandbox"),
        "arms": sorted(data),
        "notes": args.notes,
    }
    for fam, share in PREREGISTERED.items():
        config[f"prereg_share/{fam}"] = share

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=args.group,
        name=args.name or "e20-verify-side-layer-family-attribution",
        job_type="attribution",
        config=config,
        notes=args.notes,
    )

    arm_rows = []
    width_rows = []
    fam_rows = []
    summary: dict = {}

    for label in sorted(data):
        arm = data[label]
        meta = arm["meta"]
        mtp = arm.get("mtp") or {}
        serial = arm.get("serial") or {}
        attrib = arm.get("attrib") or {}

        arm_rows.append(
            [
                label,
                meta.get("build"),
                meta.get("attrib_mode"),
                meta.get("offered_depth"),
                meta.get("tokens"),
                meta.get("thermal_before"),
                meta.get("thermal_after"),
                meta.get("cli_sha256"),
                meta.get("worker_sha256"),
                meta.get("exit"),
                mtp.get("decode_seconds"),
                mtp.get("seed_prefill_seconds"),
                mtp.get("decode_seconds_ex_prefill"),
                serial.get("decode_seconds_ex_prefill"),
                mtp.get("all_tokens_matched"),
                mtp.get("parity_all_ok"),
                mtp.get("residual_divergence_count"),
                mtp.get("max_rejected_tail_logit_delta"),
            ]
        )

        for key in TIMING + CORRECTNESS:
            if key in mtp:
                summary[f"{label}/mtp/{key}"] = mtp[key]
            if key in serial:
                summary[f"{label}/serial/{key}"] = serial[key]
        if mtp.get("decode_seconds_ex_prefill") and serial.get(
            "decode_seconds_ex_prefill"
        ):
            summary[f"{label}/local_ratio_ex_prefill"] = (
                serial["decode_seconds_ex_prefill"]
                / mtp["decode_seconds_ex_prefill"]
            )

        hist = mtp.get("ledger_width_histogram") or {}
        for m, n in hist.items():
            width_rows.append([label, "ledger", int(m), n])
            summary[f"{label}/ledger_width/M{m}"] = n

        by_m = attrib.get("by_m") or {}
        for m, e in sorted(by_m.items(), key=lambda kv: int(kv[0])):
            width_rows.append([label, "instrumented", int(m), e["forwards"]])
            row = [label, int(m), e["forwards"], (e.get("total_ns_mean") or 0) / 1e3]
            for fam in FAMILIES:
                row.append(e.get(f"{fam}_share"))
                summary[f"{label}/M{m}/{fam}_share"] = e.get(f"{fam}_share")
                summary[f"{label}/M{m}/{fam}_us"] = (
                    e.get(f"{fam}_ns_mean", 0.0) / 1e3
                )
            row.append(e.get("residual_frac"))
            summary[f"{label}/M{m}/residual_frac"] = e.get("residual_frac")
            summary[f"{label}/M{m}/forwards"] = e["forwards"]
            summary[f"{label}/M{m}/total_us"] = (e.get("total_ns_mean") or 0) / 1e3
            fam_rows.append(row)

        pooled = attrib.get("pooled")
        if pooled:
            for fam in FAMILIES:
                summary[f"{label}/pooled/{fam}_share"] = pooled.get(f"{fam}_share")
            summary[f"{label}/pooled/forwards"] = pooled["forwards"]
            summary[f"{label}/pooled/widths"] = str(pooled["widths"])

    run.log(
        {
            "arms": wandb.Table(
                columns=[
                    "label",
                    "build",
                    "attrib_mode",
                    "offered_depth",
                    "tokens",
                    "thermal_before",
                    "thermal_after",
                    "cli_sha256",
                    "worker_sha256",
                    "exit",
                    "mtp_decode_s",
                    "mtp_prefill_s",
                    "mtp_net_s",
                    "serial_net_s",
                    "all_tokens_matched",
                    "parity_all_ok",
                    "residual_divergence_count",
                    "max_rejected_tail_logit_delta",
                ],
                data=arm_rows,
            ),
            "width_histograms": wandb.Table(
                columns=["label", "source", "M", "count"], data=width_rows
            ),
            "family_shares_by_width": wandb.Table(
                columns=["label", "M", "forwards", "total_us"]
                + [f"{f}_share" for f in FAMILIES]
                + ["residual_frac"],
                data=fam_rows,
            ),
        }
    )
    run.summary.update(summary)
    print(f"wandb run: {run.url}")
    print(f"run id   : {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
