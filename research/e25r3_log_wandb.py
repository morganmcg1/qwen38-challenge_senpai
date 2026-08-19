#!/usr/bin/env python3
"""Research-only (qwen38-r1-e25-per-row-draft-price, r3).

Publish the post-E27 re-measurement of the per-row draft price.

r3 is measurement only: no arm, no cap, no policy. It answers three things.
(1) Did E27's weight-stream fix move `c_3`, the price that made arm D a hard
DEEP_CAP=3 in r2?  (2) What is the full price curve now, on both instruments?
(3) After E27, how much of the r2 headroom is left for a price rule to win?

MATCHING.  Every pre/post pair here is the SAME eight prompts, the same forced
depth cycle, the same declared head, the same 512-token window and the same
host. The pre-E27 tapes were archived before the post-E27 legs overwrote them,
so the comparison is per-prompt, not against a remembered number.

CLOCKS.  A re-cost charges ONE build's measured T(d) on both sides of the
ratio. Pricing a delta on the post-E27 curve while keeping the pre-E27 taped
wall time as the denominator would put two builds' clocks in one ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e25-per-row-draft-price"

BASE_SHA = "329d3644dc96972d6843ecfe759141b8b0ab539d"
R2_BASE_SHA = "d7619a7f4606c2a0e1c46e04d8fae2e4e0e96602"
R1_BASE_SHA = "0d2eef9cac75d890de06a5eef4fd686c3c34c1ef"

# Re-anchored per r3: companygardener 11863aa on organizer 5068eb8d.
PROMOTED_SUBMISSION_ID = "11863aa9-0dc0-4703-b7a4-eacd473810cb"
PROMOTED_FRONTIER_SHA = "5068eb8d0bae032faca6e901de398fc732531160"
LIVE_RANKED_BAR = 3.24326223889754

# The advisor branch moved past the pinned base while r3 was measuring. Both
# scored-path deltas are inert with no env override set, so the curve transfers.
ADVISOR_HEAD_AT_SUBMIT = "a88b4d33bbd1ade0638dfb3302551bf61606be8b"

PRIMARY_METRIC = "e25/measured_row_step_ratio_at_depth_3"
PRIMARY_BASELINE = 0.442442

# Registered in research/e25-r3-prereg.md at 150c957, BEFORE any r3 leg ran.
PREREG = {
    "advisor_band_low": 0.13,
    "advisor_band_high": 0.23,
    "advisor_point": 0.18,
    "student_c3_point": 0.1754,
    "student_c4_point": 0.3343,
    "student_mean_depth": 2.30,
    "student_mean_depth_tol": 0.25,
    "student_residual_headroom_pct_max": 1.0,
    "falsifier_c3_ge": 0.30,
    "falsifier_c4_lt": 0.20,
    "falsifier_headroom_pct_gt": 1.5,
}


def curve(report: dict) -> dict[int, dict]:
    return {int(d): v for d, v in report["time_parent_clock"]["round_ms"].items()}


def steps(report: dict) -> dict[int, float]:
    return {int(d): v
            for d, v in report["time_parent_clock"]["step_ratio"].items()}


def prompt_step(report: dict, prompt: str, d: int):
    """c_d refit inside one prompt, so the pooled number has a spread."""
    pp = report["per_prompt_curve"].get(prompt, {}).get("step_ratio", {})
    return pp.get(str(d), pp.get(d))


def prompt_T(report: dict, prompt: str, d: int):
    pp = report["per_prompt_curve"].get(prompt, {}).get("round_ms", {})
    cell = pp.get(str(d), pp.get(d))
    return cell["mean_ms"] if cell else None


def mean_depth(report: dict) -> float:
    hist = report["instrument"]["taken_depth_histogram"]
    n = sum(hist.values())
    return sum(int(d) * c for d, c in hist.items()) / n


def flat_curve(report: dict, prefix: str) -> dict:
    out = {}
    for d, v in curve(report).items():
        out[f"{prefix}/T{d}_mean_ms"] = v["mean_ms"]
        out[f"{prefix}/T{d}_median_ms"] = v["median_ms"]
        out[f"{prefix}/T{d}_sem_ms"] = v["sem_ms"]
        out[f"{prefix}/T{d}_n"] = v["n"]
        out[f"{prefix}/T{d}_weight_passes"] = v["weight_passes"]
    for d, c in steps(report).items():
        out[f"{prefix}/c{d}"] = c
        out[f"{prefix}/c{d}_ceiling"] = 1.0 / (d + 1)
        out[f"{prefix}/c{d}_admissible"] = c < 1.0 / (d + 1)
    return out


def fidelity_rollup(report: dict, prefix: str) -> dict:
    f = report["fidelity"]
    led = report["ledger_crosscheck"]
    return {
        f"{prefix}/all_tokens_matched_all": all(
            v["all_tokens_matched"] for v in f.values()),
        f"{prefix}/parity_all_ok_all": all(
            v["parity_all_ok"] for v in f.values()),
        f"{prefix}/residual_divergence_total": sum(
            v["residual_divergence_count"] for v in f.values()),
        f"{prefix}/rows_closed_all": all(v["rows_closed"] for v in led.values()),
        f"{prefix}/accepted_agrees_all": all(
            v["accepted_agrees"] for v in led.values()),
        f"{prefix}/proposed_agrees_all": all(
            v["proposed_agrees"] for v in led.values()),
        f"{prefix}/head_origin": sorted(
            {v["head_origin"] for v in f.values()})[0],
        f"{prefix}/head_sha256": sorted(
            {v["head_sha256"] for v in f.values()})[0],
        f"{prefix}/mean_round_robin_speedup": statistics.mean(
            v["round_robin_speedup"] for v in f.values()),
    }


def headroom(recost: dict) -> dict:
    """Best priced arm against the shipped anchor, on one build's curve."""
    arms = recost["arms"]
    best_name, best = None, -float("inf")
    for name, a in arms.items():
        if name == "A_shipped_scalar_h0.18":
            continue
        g = a["projection"]["median_of_8_true_decode_gain_pct"]
        if g > best:
            best_name, best = name, g
    b = arms[best_name]
    return {
        "best_arm": best_name,
        "median_of_8_gain_pct": best,
        "pooled_gain_pct": b["pooled"]["true_decode_gain_pct"],
        "mean_depth_base": b["pooled"]["mean_depth_base"],
        "mean_depth_arm": b["pooled"]["mean_depth_arm"],
        "rounds_requesting_more_depth": b["deepening"]["rounds_requesting_more_depth"],
        "cap_depth3_median_of_8_gain_pct":
            arms["B_cap_depth3"]["projection"]["median_of_8_true_decode_gain_pct"],
        "cap_depth3_pooled_gain_pct":
            arms["B_cap_depth3"]["pooled"]["true_decode_gain_pct"],
        "c3_used": recost["curve_step_ratio"].get("3",
                                                  recost["curve_step_ratio"].get(3)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=Path, required=True,
                    help="e25r2_refit report over the archived pre-E27 tapes")
    ap.add_argument("--post-force", type=Path, required=True)
    ap.add_argument("--post-base", type=Path, required=True)
    ap.add_argument("--recost-pre", type=Path, required=True,
                    help="e25_price on the r1 tape, pre-E27 curve")
    ap.add_argument("--recost-post", type=Path, required=True,
                    help="e25_price on the r1 tape, post-E27 curve")
    ap.add_argument("--name", default="e25-r3-post-e27-price-curve")
    ap.add_argument("--notes", default=(
        "E25 r3 (measurement only): E27 removed the M=5 weight-stream pass and "
        "with it the M=4->5 cliff. c_3 falls 0.442442 -> ~0.176, back under its "
        "1/4 admissibility ceiling, so the depth-3 wall that made arm D a hard "
        "DEEP_CAP=3 in r2 is gone. Only T(4) moved; T(0..3) and T(5..7) are "
        "flat. The residual headroom for a price rule is now under 1%, so this "
        "lever is closed by E27 rather than by a new rule."))
    args = ap.parse_args()

    pre = json.loads(args.pre.read_text())
    post_f = json.loads(args.post_force.read_text())
    post_b = json.loads(args.post_base.read_text())
    rc_pre = json.loads(args.recost_pre.read_text())["recost_on_cost_curve"]
    rc_post = json.loads(args.recost_post.read_text())["recost_on_cost_curve"]

    c3_post = steps(post_f)[3]
    c4_post = steps(post_f)[4]
    hr_pre, hr_post = headroom(rc_pre), headroom(rc_post)

    config = {
        "assignment": "qwen38-r1-e25-per-row-draft-price",
        "revision": "r3",
        "pr": 29,
        "student": "qwen-edward",
        "scope": "measurement only (no arm D, no row-5 cap, no policy arm)",
        "credit": "thorfinn E22 follow-up #1 (two-piece boundary-aware price, PR #26)",
        "base_sha": BASE_SHA,
        "r2_base_sha": R2_BASE_SHA,
        "r1_base_sha": R1_BASE_SHA,
        "advisor_head_at_submit": ADVISOR_HEAD_AT_SUBMIT,
        "advisor_head_transfers": True,
        "advisor_head_transfer_reason":
            "both scored-path deltas inert with no env override: "
            "MLX_QWEN_MTP_TRACE_SYNC_HEAD probe off by default, and the ladder "
            "rung set is unchanged ([0,1,9,19,29,39,49,57]); "
            "`ladderActive = inputs.dim(1) <= 9` is a CONTEXT line in that "
            "diff, already present at 329d3644",
        "promoted_submission_id": PROMOTED_SUBMISSION_ID,
        "promoted_frontier_sha": PROMOTED_FRONTIER_SHA,
        "live_ranked_bar": LIVE_RANKED_BAR,
        "prompts": post_f["prompts"],
        "n_prompts": len(post_f["prompts"]),
        "decode_tokens_per_leg": 512,
        "seed_tokens_per_leg": 512,
        "offered_depth": 8,
        "forced_depth_cycle": [0, 1, 2, 3, 4, 5, 6, 7],
        "primary_timer": "parent block_request_seconds x effective_draft_lengths",
        "host": "Apple M4 Pro / Mac16,11 / applegpu_g16s (NOT the ranked M5)",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "cool_gate_bypassed": True,
        "counterbalanced": "ABBA probe pairs within session",
        "unmeasurable_floor_pct": 0.05,
        "prereg": PREREG,
        "prereg_committed_before_measurement": "150c957",
    }

    run = wandb.init(entity=ENTITY, project=PROJECT, name=args.name,
                     group=GROUP, job_type="analysis", config=config,
                     notes=args.notes)

    summary = {
        PRIMARY_METRIC: c3_post,
        "e25/primary_metric_baseline": PRIMARY_BASELINE,
        "e25/primary_metric_delta": c3_post - PRIMARY_BASELINE,
        "e25/primary_metric_direction": "minimize",

        "e25r3/c3_pre_e27": steps(pre)[3],
        "e25r3/c3_post_e27_force": c3_post,
        "e25r3/c3_post_e27_base": steps(post_b).get(3),
        "e25r3/c4_post_e27_force": c4_post,
        "e25r3/c3_ceiling": 0.25,
        "e25r3/c3_admissible_pre": steps(pre)[3] < 0.25,
        "e25r3/c3_admissible_post": c3_post < 0.25,
        "e25r3/depth3_wall_removed": (steps(pre)[3] >= 0.25) and (c3_post < 0.25),

        # --- adjudication of the pre-registered predictions ---
        "e25r3/prereg_advisor_band_hit":
            PREREG["advisor_band_low"] <= c3_post <= PREREG["advisor_band_high"],
        "e25r3/prereg_student_c3_abs_error": abs(c3_post - PREREG["student_c3_point"]),
        "e25r3/prereg_student_c4_abs_error": abs(c4_post - PREREG["student_c4_point"]),
        "e25r3/prereg_falsifier_c3_ge_0p30": c3_post >= PREREG["falsifier_c3_ge"],
        "e25r3/prereg_falsifier_c4_lt_0p20": c4_post < PREREG["falsifier_c4_lt"],
        "e25r3/prereg_falsifier_headroom":
            hr_post["median_of_8_gain_pct"] > PREREG["falsifier_headroom_pct_gt"],

        # --- what actually moved ---
        "e25r3/T4_pre_ms": curve(pre)[4]["mean_ms"],
        "e25r3/T4_post_ms": curve(post_f)[4]["mean_ms"],
        "e25r3/T4_delta_ms": curve(post_f)[4]["mean_ms"] - curve(pre)[4]["mean_ms"],
        "e25r3/max_abs_pct_change_excluding_T4": max(
            abs(100.0 * (curve(post_f)[d]["mean_ms"] / curve(pre)[d]["mean_ms"] - 1.0))
            for d in curve(pre) if d != 4 and d in curve(post_f)),

        # --- remaining headroom (deliverable c) ---
        "e25r3/headroom_pre_e27_median_of_8_pct": hr_pre["median_of_8_gain_pct"],
        "e25r3/headroom_post_e27_median_of_8_pct": hr_post["median_of_8_gain_pct"],
        "e25r3/headroom_pre_e27_pooled_pct": hr_pre["pooled_gain_pct"],
        "e25r3/headroom_post_e27_pooled_pct": hr_post["pooled_gain_pct"],
        "e25r3/headroom_closed_pct": (hr_pre["median_of_8_gain_pct"]
                                      - hr_post["median_of_8_gain_pct"]),
        "e25r3/headroom_best_arm_pre": hr_pre["best_arm"],
        "e25r3/headroom_best_arm_post": hr_post["best_arm"],
        "e25r3/cap3_headroom_post_median_of_8_pct":
            hr_post["cap_depth3_median_of_8_gain_pct"],
        "e25r3/lever_closed":
            hr_post["median_of_8_gain_pct"] < PREREG["student_residual_headroom_pct_max"],

        # --- realised depth on the shipped rule, current base ---
        "e25r3/realised_mean_depth_base": mean_depth(post_b),
        "e25r3/prereg_mean_depth_abs_error":
            abs(mean_depth(post_b) - PREREG["student_mean_depth"]),
        "e25r3/prereg_mean_depth_hit":
            abs(mean_depth(post_b) - PREREG["student_mean_depth"])
            <= PREREG["student_mean_depth_tol"],
        "e25r3/realised_depth_histogram_base": json.dumps(
            post_b["instrument"]["taken_depth_histogram"]),
        "e25r3/realised_depth_histogram_force": json.dumps(
            post_f["instrument"]["taken_depth_histogram"]),

        "e25r3/rounds_analysed_post_force": post_f["rounds_analysed"],
        "e25r3/rounds_analysed_post_base": post_b["rounds_analysed"],
        "e25r3/rounds_analysed_pre": pre["rounds_analysed"],
    }
    summary.update(flat_curve(pre, "e25r3/pre"))
    summary.update(flat_curve(post_f, "e25r3/post_force"))
    summary.update(flat_curve(post_b, "e25r3/post_base"))
    summary.update(fidelity_rollup(pre, "e25r3/fid_pre"))
    summary.update(fidelity_rollup(post_f, "e25r3/fid_post_force"))
    summary.update(fidelity_rollup(post_b, "e25r3/fid_post_base"))

    run.summary.update(summary)

    price = wandb.Table(columns=[
        "depth", "weight_passes", "T_pre_ms", "T_post_force_ms", "delta_ms",
        "pct_change", "c_pre", "c_post_force", "c_ceiling",
        "admissible_pre", "admissible_post"])
    cp, cf = curve(pre), curve(post_f)
    sp, sf = steps(pre), steps(post_f)
    for d in sorted(cp):
        if d not in cf:
            continue
        price.add_data(
            d, cp[d]["weight_passes"], cp[d]["mean_ms"], cf[d]["mean_ms"],
            cf[d]["mean_ms"] - cp[d]["mean_ms"],
            100.0 * (cf[d]["mean_ms"] / cp[d]["mean_ms"] - 1.0),
            sp.get(d), sf.get(d), 1.0 / (d + 1) if d in sp else None,
            (sp[d] < 1.0 / (d + 1)) if d in sp else None,
            (sf[d] < 1.0 / (d + 1)) if d in sf else None)

    per_prompt = wandb.Table(columns=[
        "prompt", "c3_pre_force", "c3_post_force", "c3_post_base",
        "T4_pre_ms", "T4_post_ms", "matched_pre", "matched_post"])
    for p in post_f["prompts"]:
        per_prompt.add_data(
            p, prompt_step(pre, p, 3), prompt_step(post_f, p, 3),
            prompt_step(post_b, p, 3), prompt_T(pre, p, 4), prompt_T(post_f, p, 4),
            pre["fidelity"][p]["all_tokens_matched"],
            post_f["fidelity"][p]["all_tokens_matched"])

    hr = wandb.Table(columns=[
        "curve", "arm", "pooled_gain_pct", "median_of_8_gain_pct",
        "mean_depth_base", "mean_depth_arm", "rounds_requesting_more_depth"])
    for label, rc in (("pre_e27", rc_pre), ("post_e27", rc_post)):
        for name, a in rc["arms"].items():
            hr.add_data(label, name, a["pooled"]["true_decode_gain_pct"],
                        a["projection"]["median_of_8_true_decode_gain_pct"],
                        a["pooled"]["mean_depth_base"], a["pooled"]["mean_depth_arm"],
                        a["deepening"]["rounds_requesting_more_depth"])

    run.log({"e25r3/price_curve": price,
             "e25r3/per_prompt": per_prompt,
             "e25r3/headroom": hr})
    print(f"{run.url}\n{PRIMARY_METRIC} = {c3_post:.6f} "
          f"(baseline {PRIMARY_BASELINE}, delta {c3_post - PRIMARY_BASELINE:+.6f})")
    run.finish()


if __name__ == "__main__":
    main()
