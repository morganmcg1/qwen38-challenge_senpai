#!/usr/bin/env python3
"""qwen38-r1-e20: publish the verify-side layer-family attribution to W&B.

usage:
  research/e20_log_wandb.py <e20_analyze.py --json-out file>... [--group G]
                            [--notes ...] [--name N]

Accepts one analysis file per session. The first is the headline session; the
rest are merged so cross-session views (the perturbation ladder, the fixed vs
per-row fit anchors) can be computed from logged data rather than asserted.
Sessions are tagged by the common prefix of their arm labels (S, D, N).

Everything published here is post-hoc analysis of arms that were already run;
the pre-registered prediction lives in
research/results/qwen38-r1-e20-verify-side-layer-family-attribution.md and is
logged as config so the comparison is visible without leaving the run.

Views of the split are published side by side and never mixed:
  headline    - mode-2 forward wall time reconciled against the parent's own
                per-round block clock, then summed over the shipped width
                histogram. The answer.
  scored      - raw shares from verify forwards inside the timed window.
  warmup      - the harness's shape-warming forwards. Context only.
  corrected   - scored, minus the fitted per-boundary cost. FALSIFIED: the
                linear-in-boundary-count model leaves 26-52% residual. Kept
                only so the falsification is auditable.
  apportioned - superseded diagnostic that apportioned the whole window by
                pooled shares. Kept for audit, never quoted.
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


def session_tag(labels) -> str:
    return os.path.commonprefix(sorted(labels)) or "?"


def cross_session_fit(sessions, min_rounds: int = 5) -> dict:
    """Fixed vs per-row cost anchored on the two best-supported widths.

    Those widths sit in different sessions (M=3 only occurs at offered depth 2,
    M=9 only at depth 8), so the widest lever arm the campaign actually
    measured is unavailable to any single-session fit.
    """
    best: dict = {}
    for sess, _path, d in sessions:
        for m, r in ((d.get("_headline") or {}).get("by_m") or {}).items():
            n = r.get("rounds_per_window") or 0
            if n < min_rounds:
                continue
            m = int(m)
            if m not in best or n > best[m][1]:
                best[m] = (sess, n, r)
    if len(best) < 2:
        return {}
    lo_m, hi_m = min(best), max(best)
    lo_sess, lo_n, lo = best[lo_m]
    hi_sess, hi_n, hi = best[hi_m]
    span = hi_m - lo_m

    out: dict = {}
    for fam in ALL_SHARES + ["target_work"]:
        lo_ms = lo.get(f"{fam}_s", 0.0) * 1e3
        hi_ms = hi.get(f"{fam}_s", 0.0) * 1e3
        per_row = (hi_ms - lo_ms) / span
        fixed = lo_ms - lo_m * per_row
        out[fam] = {
            "fixed_ms": fixed,
            "per_row_ms": per_row,
            "fixed_share_at_hi": fixed / hi_ms if hi_ms else None,
            "share_of_marginal": None,
            "lo_session": lo_sess,
            "hi_session": hi_sess,
            "lo_m": lo_m,
            "hi_m": hi_m,
            "lo_rounds": lo_n,
            "hi_rounds": hi_n,
        }
    total = out["target_work"]["per_row_ms"]
    if total:
        for fam in ALL_SHARES:
            out[fam]["share_of_marginal"] = out[fam]["per_row_ms"] / total
    return out


def perturbation_ladder(arms: dict) -> dict:
    """Cost of the instrument itself, per offered depth, relative to BASE.

    Grouped by (offered depth, build, attribution mode) so the inert mode-2
    path is separable from the mode-1/mode-3 logging paths, and so the serial
    control is compared against the same build as its MTP leg.
    """
    groups: dict = {}
    for label, arm in sorted(arms.items()):
        meta = arm.get("meta") or {}
        key = (
            str(meta.get("offered_depth")),
            str(meta.get("build")),
            str(meta.get("attrib_mode")),
        )
        g = groups.setdefault(key, {"labels": [], "mtp": [], "serial": []})
        g["labels"].append(label)
        mtp = (arm.get("mtp") or {}).get("decode_seconds_ex_prefill")
        if mtp is not None:
            g["mtp"].append(mtp)
        ser = (arm.get("serial") or {}).get("sec_per_token_ex_prefill")
        if ser is not None:
            g["serial"].append(ser * 1e3)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    out: dict = {}
    for key, g in groups.items():
        mtp_mean = mean(g["mtp"])
        spread = None
        if mtp_mean and len(g["mtp"]) > 1:
            spread = (max(g["mtp"]) - min(g["mtp"])) / mtp_mean * 100
        out[key] = {
            "n": len(g["labels"]),
            "arms": ",".join(g["labels"]),
            "mtp_mean_s": mtp_mean,
            "mtp_spread_pct": spread,
            "mtp_vs_base_pct": None,
            "serial_mean_ms_per_tok": mean(g["serial"]),
            "serial_vs_base_pct": None,
        }

    for key, v in out.items():
        base = out.get((key[0], "BASE", "0"))
        if base is None:
            continue
        if base["mtp_mean_s"] and v["mtp_mean_s"] is not None:
            v["mtp_vs_base_pct"] = (v["mtp_mean_s"] / base["mtp_mean_s"] - 1) * 100
        b_ser = base["serial_mean_ms_per_tok"]
        if b_ser and v["serial_mean_ms_per_tok"] is not None:
            v["serial_vs_base_pct"] = (v["serial_mean_ms_per_tok"] / b_ser - 1) * 100
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", nargs="+")
    ap.add_argument("--group", default="qwen38-r1-e20")
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--base-sha", default="c0f7e370921a14f348fa1872f2176b1b43028752")
    args = ap.parse_args()

    sessions = []
    arms: dict = {}
    arm_session: dict = {}
    for path in args.analysis:
        d = json.load(open(path))
        a = arms_of(d)
        sess = session_tag(a)
        sessions.append((sess, path, d))
        for label, arm in a.items():
            arms[label] = arm
            arm_session[label] = sess

    data = sessions[0][2]
    primary = sessions[0][0]
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
        "sessions": [s for s, _p, _d in sessions],
        "headline_session": primary,
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
                arm_session[label],
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

        hist = mtp.get("report_width_histogram") or {}
        for m, n in hist.items():
            width_rows.append([label, "report", int(m), n])
            summary[f"{label}/report_width/M{m}"] = n
        for key in ("block_seconds_sum", "block_round_count", "block_closure_delta_s",
                    "first_block_seconds"):
            if key in mtp:
                summary[f"{label}/mtp/{key}"] = mtp[key]

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

    for sess, _path, d in sessions:
        for source, fit in (d.get("_boundary_fits") or {}).items():
            by_m = sorted((fit.get("by_m") or {}).items(), key=lambda kv: int(kv[0]))
            for m, f in by_m:
                fit_rows.append(
                    [
                        sess,
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
                    k = f"fit/{sess}/M{m}"
                    summary[f"{k}/per_eval_us"] = f["per_eval_ns"] / 1e3
                    summary[f"{k}/unperturbed_total_us"] = f["gpu_ns_intercept"] / 1e3
                    summary[f"{k}/max_abs_resid_frac"] = f.get("max_abs_resid_frac")
                    summary[f"{k}/n_boundary_counts"] = f["n_boundary_counts"]

        for source, agree in (d.get("_layer_group_agreement") or {}).items():
            src = source.removeprefix("attrib_")
            by_m = sorted((agree.get("by_m") or {}).items(), key=lambda kv: int(kv[0]))
            for m, groups in by_m:
                for grp, g in groups.items():
                    agree_rows.append(
                        [sess, src, int(m), grp, g["mode1_share"], g["mode3_share"],
                         g["abs_delta"]]
                    )
                    k = f"agreement/{sess}/{src}/M{m}/{grp}"
                    summary[f"{k}/abs_delta"] = g["abs_delta"]
            summary[f"agreement/{sess}/{src}/max_abs_share_delta"] = agree.get(
                "max_abs_share_delta"
            )
    # The stopping rule in section 1.7 is evaluated against the worst mode-1 vs
    # mode-3 disagreement anywhere in the timed window, not per session. Warmup
    # forwards are outside that window and are bounded separately.
    for src in ("scored", "warmup"):
        deltas = [r[6] for r in agree_rows if r[1] == src and r[6] is not None]
        if deltas:
            summary[f"agreement/{src}/max_abs_share_delta"] = max(deltas)
    summary["agreement/max_abs_share_delta"] = summary.get(
        "agreement/scored/max_abs_share_delta"
    )

    # Superseded diagnostic: whole-window apportionment by pooled shares. It
    # ignores that the split moves with width, so it is logged for audit only.
    diag_rows = []
    ap = data.get("_apportioned") or {}
    for m, r in sorted((ap.get("by_m") or {}).items(), key=lambda kv: int(kv[0])):
        diag_rows.append(
            [int(m), r["forwards"], r["unperturbed_ns_per_forward"] / 1e6,
             r["unperturbed_seconds_at_width"]]
            + [r.get(f"{f}_seconds") for f in ALL_SHARES]
            + [r.get(f"{f}_share") for f in ALL_SHARES]
            + [r["share_spread_pp"], ",".join(r["share_arms"])]
        )
        summary[f"diagnostic_apportioned/M{m}/share_spread_pp"] = r["share_spread_pp"]
    for k, v in (ap.get("verify_proper_shares") or {}).items():
        summary[f"diagnostic_apportioned/verify_proper/{k}_share"] = v

    # HEADLINE. Mode-2 forward time reconciled against the parent's own block
    # clock at each width, then summed over the shipped width histogram.
    headline_rows = []
    ident_rows = []
    fit_rows_marginal = []
    for sess, _path, d in sessions:
        h = d.get("_headline") or {}
        if not h.get("by_m"):
            continue
        is_primary = sess == primary
        headline_rows.append(
            [sess, h["window_seconds"], h["verify_side_seconds"],
             h["verify_side_share_of_window"]]
            + [h["family_seconds"].get(f) for f in ALL_SHARES]
            + [h["family_share_of_window"].get(f) for f in ALL_SHARES]
            + [h["family_share_of_verify"].get(f) for f in FAMILIES + ["embed"]]
        )
        pre = "headline" if is_primary else f"headline/{sess}"
        summary[f"{pre}/window_seconds"] = h["window_seconds"]
        summary[f"{pre}/verify_side_seconds"] = h["verify_side_seconds"]
        summary[f"{pre}/verify_side_share_of_window"] = h["verify_side_share_of_window"]
        summary[f"{pre}/source"] = h.get("source")
        for f in ALL_SHARES:
            summary[f"{pre}/{f}_seconds"] = h["family_seconds"].get(f)
            summary[f"{pre}/{f}_share_of_window"] = h["family_share_of_window"].get(f)
            if f in h["family_share_of_verify"]:
                summary[f"{pre}/{f}_share_of_verify"] = h["family_share_of_verify"][f]
        for fam in FAMILIES:
            got = h["family_share_of_verify"].get(fam)
            if got is not None and is_primary:
                summary[f"prereg_error_pp/{fam}"] = (got - PREREGISTERED[fam]) * 100

        for m, r in sorted(h["by_m"].items(), key=lambda kv: int(kv[0])):
            ident_rows.append(
                [sess, int(m), r["rounds_per_window"], r["block_m1_s"] * 1e3,
                 r["attributed_m1_s"] * 1e3, r["instrument_overhead_s"] * 1e3,
                 r["block_m2_s"] * 1e3, r["target_work_s"] * 1e3,
                 r["instrument_inflation"], r["mode1_occupancy"],
                 r["mode1_arms"], r["mode2_arms"], r["mode1_forwards"]]
                + [r.get(f"{f}_s", 0.0) * 1e3 for f in ALL_SHARES]
                + [r.get(f"{f}_share_of_verify") for f in FAMILIES + ["embed"]]
            )
            k = f"{pre}/M{m}"
            summary[f"{k}/rounds"] = r["rounds_per_window"]
            summary[f"{k}/block_m2_ms"] = r["block_m2_s"] * 1e3
            summary[f"{k}/target_work_ms"] = r["target_work_s"] * 1e3
            summary[f"{k}/instrument_inflation"] = r["instrument_inflation"]
            summary[f"{k}/mode1_occupancy"] = r["mode1_occupancy"]
            for f in FAMILIES + ["embed"]:
                summary[f"{k}/{f}_share_of_verify"] = r.get(f"{f}_share_of_verify")
                summary[f"{k}/{f}_ms"] = r.get(f"{f}_s", 0.0) * 1e3

        mf = h.get("marginal_fit") or {}
        for fam, v in sorted((mf.get("by_family") or {}).items()):
            fit_rows_marginal.append(
                [sess, "within_session", fam, v["fixed_s"] * 1e3,
                 v["per_row_s"] * 1e3, v["fixed_share_at_hi"],
                 v.get("share_of_marginal")]
            )
            if is_primary:
                summary[f"marginal_fit/{fam}/fixed_ms"] = v["fixed_s"] * 1e3
                summary[f"marginal_fit/{fam}/per_row_ms"] = v["per_row_s"] * 1e3
                summary[f"marginal_fit/{fam}/fixed_share_at_hi"] = v[
                    "fixed_share_at_hi"
                ]
                summary[f"marginal_fit/{fam}/share_of_marginal"] = v.get(
                    "share_of_marginal"
                )

    # Preferred fit: anchored on the two widest-support widths in the whole
    # campaign, which live in different sessions (M=3 in D, M=9 in S).
    for fam, v in sorted(cross_session_fit(sessions).items()):
        fit_rows_marginal.append(
            [f"{v['lo_session']}+{v['hi_session']}", "cross_session", fam,
             v["fixed_ms"], v["per_row_ms"], v["fixed_share_at_hi"],
             v.get("share_of_marginal")]
        )
        summary[f"marginal_fit/cross_session/{fam}/fixed_ms"] = v["fixed_ms"]
        summary[f"marginal_fit/cross_session/{fam}/per_row_ms"] = v["per_row_ms"]
        summary[f"marginal_fit/cross_session/{fam}/fixed_share_at_hi"] = v[
            "fixed_share_at_hi"
        ]
        summary[f"marginal_fit/cross_session/{fam}/share_of_marginal"] = v.get(
            "share_of_marginal"
        )

    ladder_rows = []
    for key, v in sorted(perturbation_ladder(arms).items()):
        ladder_rows.append(
            [key[0], key[1], key[2], v["n"], v["arms"], v["mtp_mean_s"],
             v["mtp_spread_pct"], v["mtp_vs_base_pct"], v["serial_mean_ms_per_tok"],
             v["serial_vs_base_pct"]]
        )
        k = f"ladder/d{key[0]}/{key[1]}_mode{key[2]}"
        summary[f"{k}/mtp_mean_s"] = v["mtp_mean_s"]
        summary[f"{k}/mtp_spread_pct"] = v["mtp_spread_pct"]
        summary[f"{k}/mtp_vs_base_pct"] = v["mtp_vs_base_pct"]
        summary[f"{k}/serial_ms_per_tok"] = v["serial_mean_ms_per_tok"]
        summary[f"{k}/serial_vs_base_pct"] = v["serial_vs_base_pct"]

    run.log(
        {
            "headline_split": wandb.Table(
                columns=[
                    "session",
                    "window_s",
                    "verify_side_s",
                    "verify_side_share_of_window",
                ]
                + [f"{f}_seconds" for f in ALL_SHARES]
                + [f"{f}_share_of_window" for f in ALL_SHARES]
                + [f"{f}_share_of_verify" for f in FAMILIES + ["embed"]],
                data=headline_rows,
            ),
            "headline_accounting_identity": wandb.Table(
                columns=[
                    "session",
                    "M",
                    "rounds",
                    "block_m1_ms",
                    "attributed_m1_ms",
                    "instrument_overhead_ms",
                    "block_m2_ms",
                    "target_work_ms",
                    "instrument_inflation",
                    "mode1_occupancy",
                    "mode1_arms",
                    "mode2_arms",
                    "mode1_forwards",
                ]
                + [f"{f}_ms" for f in ALL_SHARES]
                + [f"{f}_share_of_verify" for f in FAMILIES + ["embed"]],
                data=ident_rows,
            ),
            "headline_marginal_fit": wandb.Table(
                columns=[
                    "session",
                    "kind",
                    "family",
                    "fixed_ms",
                    "per_row_ms",
                    "fixed_share_at_hi",
                    "share_of_marginal",
                ],
                data=fit_rows_marginal,
            ),
            "perturbation_ladder": wandb.Table(
                columns=[
                    "offered_depth",
                    "build",
                    "attrib_mode",
                    "n",
                    "arms",
                    "mtp_mean_s",
                    "mtp_spread_pct",
                    "mtp_vs_base_pct",
                    "serial_mean_ms_per_tok",
                    "serial_vs_base_pct",
                ],
                data=ladder_rows,
            ),
            "diagnostic_apportioned_family_seconds": wandb.Table(
                columns=["M", "forwards", "forward_ms", "seconds_at_width"]
                + [f"{f}_seconds" for f in ALL_SHARES]
                + [f"{f}_share" for f in ALL_SHARES]
                + ["share_spread_pp", "share_arms"],
                data=diag_rows,
            ),
            "arms": wandb.Table(
                columns=[
                    "label",
                    "session",
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
                    "session",
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
                columns=["session", "source", "M", "layer_group", "mode1_share",
                         "mode3_share", "abs_delta"],
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
