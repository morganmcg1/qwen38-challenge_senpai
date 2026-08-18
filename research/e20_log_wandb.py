#!/usr/bin/env python3
"""qwen38-r1-e20: publish the verify-side layer-family attribution to W&B.

usage:
  research/e20_log_wandb.py <e20_analyze.py --json-out file> [--group G]
                            [--notes ...] [--name N]

Everything published here is post-hoc analysis of arms that were already run;
the pre-registered prediction lives in
research/results/qwen38-r1-e20-verify-side-layer-family-attribution.md and is
logged as config so the comparison is visible without leaving the run.

Three views of the split are published side by side and never mixed:
  scored    - verify forwards inside the timed window. The answer.
  warmup    - the harness's shape-warming forwards. Context only.
  corrected - scored, minus the fitted per-boundary instrumentation cost.
"""

from __future__ import annotations

import argparse
import json

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

# The assignment's four families, then the two buckets that belong to neither a
# layer family nor the readout and are therefore reported on their own.
FAMILIES = ["gdn", "full_attention", "mlp", "head_and_top_two"]
EXTRA = ["embed", "drain"]
ALL_SHARES = FAMILIES + EXTRA
LAYER_GROUPS = ["gdn_layer", "full_attention_layer"]

# Which families issue crossrow QMV dispatches, derived from quantized.h:1822
# at the base commit and constant across M in 3..9. Logged as config so the
# assignment's extra column travels with the numbers.
CROSSROW_QMV = {
    "gdn": {"wide": 144, "narrow": 0, "non_crossrow_qmv": 96},
    "full_attention": {"wide": 32, "narrow": 32, "non_crossrow_qmv": 0},
    "mlp": {"wide": 192, "narrow": 0, "non_crossrow_qmv": 0},
    "head_and_top_two": {"wide": 1, "narrow": 0, "non_crossrow_qmv": 0},
}

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
    "effective_mean_draft_len",
    "effective_max_draft_len",
    "non_drafting_round_count",
]


def arms_of(data: dict) -> dict:
    return {k: v for k, v in data.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("--group", default="qwen38-r1-e20")
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--base-sha", default="c0f7e370921a14f348fa1872f2176b1b43028752")
    args = ap.parse_args()

    data = json.load(open(args.analysis))
    arms = arms_of(data)
    any_meta = next(iter(arms.values()))["meta"] if arms else {}

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
        "arms": sorted(arms),
        "notes": args.notes,
    }
    for fam, share in PREREGISTERED.items():
        config[f"prereg_share/{fam}"] = share
    for fam, counts in CROSSROW_QMV.items():
        for kind, n in counts.items():
            config[f"crossrow_qmv/{fam}/{kind}"] = n

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
    corrected_rows = []
    fit_rows = []
    agree_rows = []
    summary: dict = {}

    for label in sorted(arms):
        arm = arms[label]
        meta = arm["meta"]
        mtp = arm.get("mtp") or {}
        serial = arm.get("serial") or {}

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

        for source in ("attrib_scored", "attrib_warmup"):
            a = arm.get(source) or {}
            tag = source.removeprefix("attrib_")
            by_m = sorted((a.get("by_m") or {}).items(), key=lambda kv: int(kv[0]))
            for m, e in by_m:
                width_rows.append([label, f"instrumented_{tag}", int(m), e["forwards"]])
                row = [
                    label,
                    tag,
                    int(m),
                    e["forwards"],
                    (e.get("total_ns_median") or 0) / 1e3,
                    e.get("evals_median"),
                ]
                for fam in ALL_SHARES + LAYER_GROUPS:
                    row.append(e.get(f"{fam}_share"))
                    if tag == "scored":
                        summary[f"{label}/M{m}/{fam}_share"] = e.get(f"{fam}_share")
                        summary[f"{label}/M{m}/{fam}_us"] = (
                            e.get(f"{fam}_ns_mean", 0.0) / 1e3
                        )
                row.append(e.get("residual_frac"))
                fam_rows.append(row)
                if tag == "scored":
                    summary[f"{label}/M{m}/residual_frac"] = e.get("residual_frac")
                    summary[f"{label}/M{m}/forwards"] = e["forwards"]
                    summary[f"{label}/M{m}/total_us"] = (
                        e.get("total_ns_median") or 0
                    ) / 1e3

            pooled = a.get("pooled")
            if pooled and tag == "scored":
                for fam in FAMILIES + ["embed"]:
                    summary[f"{label}/pooled/{fam}_share"] = pooled.get(f"{fam}_share")
                summary[f"{label}/pooled/forwards"] = pooled["forwards"]
                summary[f"{label}/pooled/widths"] = str(pooled["widths"])

        corrected = sorted(
            (arm.get("corrected") or {}).items(), key=lambda kv: int(kv[0])
        )
        for m, r in corrected:
            corrected_rows.append(
                [
                    label,
                    int(m),
                    r["forwards"],
                    r["per_eval_ns"] / 1e3,
                    r["raw_total_ns"] / 1e3,
                    r["corrected_total_ns"] / 1e3,
                    (r.get("unperturbed_total_ns") or 0) / 1e3,
                    r.get("max_abs_resid_frac"),
                ]
                + [r.get(f"{fam}_share") for fam in ALL_SHARES]
            )
            for fam in ALL_SHARES:
                summary[f"{label}/M{m}/corrected/{fam}_share"] = r.get(f"{fam}_share")

    for source, fit in (data.get("_boundary_fits") or {}).items():
        for m, f in sorted((fit.get("by_m") or {}).items(), key=lambda kv: int(kv[0])):
            fit_rows.append(
                [
                    source.removeprefix("attrib_"),
                    int(m),
                    f["n_boundary_counts"],
                    f["per_eval_ns"] / 1e3,
                    f["gpu_ns_intercept"] / 1e3,
                    f.get("max_abs_resid_frac"),
                    len(f["points"]),
                ]
            )
            if source == "attrib_scored":
                summary[f"fit/M{m}/per_eval_us"] = f["per_eval_ns"] / 1e3
                summary[f"fit/M{m}/unperturbed_total_us"] = f["gpu_ns_intercept"] / 1e3
                summary[f"fit/M{m}/max_abs_resid_frac"] = f.get("max_abs_resid_frac")
                summary[f"fit/M{m}/n_boundary_counts"] = f["n_boundary_counts"]

    agree = data.get("_layer_group_agreement") or {}
    for m, groups in sorted((agree.get("by_m") or {}).items(), key=lambda kv: int(kv[0])):
        for grp, g in groups.items():
            agree_rows.append(
                [int(m), grp, g["mode1_share"], g["mode3_share"], g["abs_delta"]]
            )
            summary[f"agreement/M{m}/{grp}/abs_delta"] = g["abs_delta"]
    if agree:
        summary["agreement/max_abs_share_delta"] = agree.get("max_abs_share_delta")
        summary["agreement/source"] = agree.get("source")

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
                columns=["label", "phase", "M", "forwards", "total_us", "evals"]
                + [f"{f}_share" for f in ALL_SHARES + LAYER_GROUPS]
                + ["residual_frac"],
                data=fam_rows,
            ),
            "corrected_shares_by_width": wandb.Table(
                columns=[
                    "label",
                    "M",
                    "forwards",
                    "per_eval_us",
                    "raw_total_us",
                    "corrected_total_us",
                    "unperturbed_total_us",
                    "max_abs_resid_frac",
                ]
                + [f"{f}_share" for f in ALL_SHARES],
                data=corrected_rows,
            ),
            "boundary_overhead_fits": wandb.Table(
                columns=[
                    "source",
                    "M",
                    "n_boundary_counts",
                    "per_eval_us",
                    "unperturbed_total_us",
                    "max_abs_resid_frac",
                    "points",
                ],
                data=fit_rows,
            ),
            "mode1_vs_mode3_agreement": wandb.Table(
                columns=["M", "layer_group", "mode1_share", "mode3_share",
                         "abs_delta"],
                data=agree_rows,
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
