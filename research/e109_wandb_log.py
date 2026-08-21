#!/usr/bin/env python3
"""Publish the E109 evidence to W&B.

    usage: research/e109_wandb_log.py [--protocol TAG] [--shape FAMILY ...]

  `e109-rung0-protocol`   the blocked, counterbalanced, round-paired A/B
                          protocol: its measured resolution against a
                          byte-identical null control, the recovery of two
                          known injected doses, the noise split between
                          per-round and per-leg terms, and the wall-clock cost
                          of one two-arm decision.
  `e109-rung1-<family>`   one threadgroup-shape sweep of a live kernel family,
                          with time, threadgroup count, waves over the local
                          cores, registers, spill and machine-code digest for
                          both the local g16s and the ranked g17s at every
                          point, and the named mechanism the curve supports.

Every timing leg here runs with `MLXFAST_LOCAL_COOL_GATE=0`, which this host
requires because it asymptotes at 40.55 C and cannot reach the 40 C gate. Legs
are counterbalanced inside one session, entry and exit GPU temperature are
recorded per leg, and every run logs `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` verbatim. No leg here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e109-resolve-the-bar-then-latency-residual"
OUT = pathlib.Path("research/out")

BASE_SHA = "05d88b8fc0976ea3ff17c42f13890c1b8c7f0297"
SUBMIT_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
FRONTIER = "51b9bf85"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
GPU_CORES = 20
BAR_LOCAL_PCT = 0.20

# E105 rung 1 residuals, the reason rung 1 exists.
RESIDUAL = {
    "prework": {"us_per_dispatch_total": 11.36, "us_per_dispatch_residual": 6.92,
                "us_per_round": 332.2, "pct_of_round": 0.245},
    "qkrope": {"us_per_dispatch_total": 9.17, "us_per_dispatch_residual": 5.80,
               "us_per_round": 92.8, "pct_of_round": 0.069},
}


def identity() -> dict:
    return {
        "experiment": "e109-resolve-the-bar-then-latency-residual",
        "harness": "local",
        "base_sha": BASE_SHA,
        "submit_base_sha": SUBMIT_BASE_SHA,
        "board_frontier": FRONTIER,
        "host": HOST,
        "gpu_cores": GPU_CORES,
        "chip": "Apple M4 Pro",
        "device_class": "AGXG16SDevice",
        "ranked_runner_chip": "M5",
        "promotion_bar_local_pct": BAR_LOCAL_PCT,
    }


def gate_flags(kind: str) -> dict:
    return {
        "leg_kind": kind,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }


def log_protocol(tag: str) -> str:
    report = json.loads((OUT / tag / "ab-report.json").read_text())
    res = report["resolution"]
    noise = report["noise"]
    integ = report["integrity"]
    design = report["design"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="measurement-protocol",
        name="e109-rung0-protocol",
        config={
            "rung": "0",
            "question": (
                "can a local protocol resolve 0.20 % of a decode-only round, "
                "and what does one decision cost"
            ),
            "session": f"research/out/{tag}",
            "session_command": (
                f"E109_BLOCKS={design['estimate_blocks']}"
                f" E109_TOKENS={design['tokens']}"
                f" research/e109_ab_session.sh {tag} <arms...>"
            ),
            "estimator": "round-aligned pairing, then block pairing",
            "endpoint": "block_request_seconds (parent-measured decode round)",
            "decode_tokens": design["tokens"],
            "mtp_depth": design["offered_depth"],
            "estimate_blocks": design["estimate_blocks"],
            "conditioning_blocks": design["conditioning_blocks"],
            "arms": report["arms"],
            "control_arm": design["control_arm"],
            **identity(),
            **gate_flags("counterbalanced A/B timing legs, GPU"),
        },
        reinit=True,
    )

    legs = wandb.Table(
        columns=["dir", "block", "arm", "arm_env", "rounds_used",
                 "round_us_mean", "round_us_median", "round_us_sd",
                 "seconds_per_token", "mean_draft", "matched",
                 "entry_c", "exit_c", "wall_seconds"])
    for leg in report["legs"]:
        legs.add_data(leg["dir"], leg["block"], leg["arm"], leg["arm_env"],
                      leg["rounds_used"], leg["round_us_mean"],
                      leg["round_us_median"], leg["round_us_sd"],
                      leg["seconds_per_token"], leg["mean_draft"],
                      leg["matched"], leg["entry_c"], leg["exit_c"],
                      leg["wall_seconds"])

    contrasts = wandb.Table(
        columns=["arm", "byte_identical_to_control", "round_alignment_verified",
                 "effect_us_per_round", "effect_pct", "half_width_us",
                 "half_width_pct", "ci95_lo_us", "ci95_hi_us",
                 "significant_at_95", "round_diff_sd_us", "blocks"])
    for arm, c in report["contrasts"].items():
        contrasts.add_data(
            arm, c["byte_identical_to_control"], c["round_alignment_verified"],
            c["effect_us_per_round"], c["effect_pct"], c["half_width_us"],
            c["half_width_pct"], c["ci95_us"][0], c["ci95_us"][1],
            c["significant_at_95"], c["round_diff_sd_us"], c["blocks"])

    dose = wandb.Table(
        columns=["arm", "units", "us_per_unit", "us_per_unit_lo",
                 "us_per_unit_hi", "predicts"])
    for arm, d in report["known_dose"].items():
        dose.add_data(arm, d["units"], d["us_per_unit"],
                      d["us_per_unit_ci95"][0], d["us_per_unit_ci95"][1],
                      json.dumps(d["predicts_other_arms_us"]))

    run.log({"protocol/legs": legs, "protocol/contrasts": contrasts,
             "protocol/known_dose": dose})
    run.summary.update({
        "resolution_half_width_us": res["half_width_us"],
        "resolution_half_width_pct": res["half_width_pct"],
        "clears_0p20_bar": bool(res["half_width_pct"] < BAR_LOCAL_PCT),
        "control_round_us": report["control_round_us"],
        "bar_us": report["bar_us"],
        "sigma_pairdiff_us": noise["sigma_pairdiff_us"],
        "within_leg_sem_us": noise["within_leg_sem_us_median"],
        "sigma_leg_us": noise["sigma_leg_us"],
        "leg_share_of_pair_variance": noise["leg_share_of_pair_variance"],
        "raw_round_sd_us": noise["round_sd_us_median"],
        "rounds_per_leg": noise["rounds_per_leg"],
        "minutes_per_two_arm_decision": report["cost"]["minutes_per_two_arm_decision"],
        "mean_leg_wall_seconds": report["cost"]["mean_leg_wall_seconds"],
        "legs": report["cost"]["legs"],
        "all_legs_matched": integ["all_legs_matched"],
        "entry_c_spread": integ["entry_c_spread"],
    })
    url = run.url
    run.finish()
    return url


def log_shape(spec: str) -> str:
    """`spec` is `family` or `family:suffix`, naming a sweep output directory.

    The width-stratified sweeps (`prework:-s5`, `prework:-s8`, ...) are the ones
    that decided rung 1: the `g1nored` lever wins at the nominal width S=5 and
    loses at the modal realised width S=8, so a sweep logged without its width
    is not interpretable.
    """
    family, _, suffix = spec.partition(":")
    directory = f"e109-shape-{family}{suffix}"
    report = json.loads((OUT / directory / "verdict.json").read_text())
    v = report["verdict"]
    rows = report.get("rows")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="kernel-shape-sweep",
        name=f"e109-rung1-{family}{suffix}",
        config={
            "rung": "1a" if family == "prework" else "1b",
            "question": (
                f"what carries the intra-kernel residual of {report['live_kernel']}"
                ": occupancy, a dependent chain, or per-threadgroup granularity"
            ),
            "live_kernel": report["live_kernel"],
            "dispatch": report["dispatch"],
            "sweep_dir": f"research/out/{directory}",
            "sweep_command": (
                f"research/e109_shape_probe.sh {family}"
                + (f" --rows {rows}" if rows is not None else "")
            ),
            "verify_rows": rows,
            "reps": report["reps"],
            "inner": report["inner"],
            **RESIDUAL[family],
            **identity(),
            **gate_flags("isolated Metal shape sweep, GPU"),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=["arm", "variant", "function", "fold_factor", "threadgroups",
                 "simdgroups_per_threadgroup", "threads_per_threadgroup",
                 "waves_over_cores", "us_per_dispatch_median",
                 "us_per_dispatch_min", "us_per_dispatch_sd",
                 "max_total_threads", "thread_execution_width",
                 "threadgroup_memory_bytes",
                 "g16s_registers", "g16s_spill_bytes", "g16s_text_bytes",
                 "g16s_text_sha8",
                 "g17s_registers", "g17s_spill_bytes", "g17s_text_bytes",
                 "g17s_text_sha8",
                 "shipped", "exact_vs_arm0"])
    for a in report["arms"]:
        g16 = a["isa"].get("applegpu_g16s", {})
        g17 = a["isa"].get("applegpu_g17s", {})
        table.add_data(
            a["name"], a["variant"], a["function"], a["fold_factor"],
            a["threadgroups"],
            a["simdgroups_per_threadgroup"], a["threads_per_threadgroup"],
            a["waves_over_cores"], a["us_per_dispatch_median"],
            a["us_per_dispatch_min"], a["us_per_dispatch_sd"],
            a["max_total_threads"], a["thread_execution_width"],
            a["threadgroup_memory_bytes"],
            g16.get("registers"), g16.get("spill_bytes"),
            g16.get("text_bytes"), g16.get("text_sha8"),
            g17.get("registers"), g17.get("spill_bytes"),
            g17.get("text_bytes"), g17.get("text_sha8"),
            a["shipped"], a["exact_vs_arm0"])
    payload = {f"shape/{family}": table}

    controls = report.get("positive_controls")
    if controls:
        # A bit-exactness claim is worthless without evidence that the
        # comparison can fail. Each control perturbs one input channel that the
        # lever's own surface reads, and must move at least one output byte.
        control_table = wandb.Table(
            columns=["control", "perturbed_buffer", "perturbed_element",
                     "fired", "mismatch_bytes", "output_bytes",
                     "mismatch_by_buffer"])
        for c in controls:
            control_table.add_data(
                c["name"], c["perturbation"]["buffer"],
                c["perturbation"]["element"], c["mismatched"],
                c["mismatch_bytes"], c["output_bytes"],
                json.dumps(c["mismatch_by_buffer"]))
        payload[f"shape/{family}_positive_controls"] = control_table

    run.log(payload)

    for a in report["arms"]:
        run.log({
            "shape/fold_factor": a["fold_factor"],
            "shape/threadgroups": a["threadgroups"],
            "shape/us_per_dispatch": a["us_per_dispatch_median"],
            "shape/waves_over_cores": a["waves_over_cores"],
        })

    run.summary.update({
        "verify_rows": rows,
        "positive_control_fired": bool(
            controls and any(c["mismatched"] for c in controls)),
        "mechanism": v["mechanism"],
        "mechanism_reason": v["reason"],
        "log_log_slopes": report["log_log_slopes"],
        "shipped_arm": v["shipped_arm"],
        "shipped_us_per_dispatch": v["shipped_us_per_dispatch"],
        "best_arm": v["best_arm"],
        "best_us_per_dispatch": v["best_us_per_dispatch"],
        "fractional_gain_vs_shipped": v["fractional_gain_vs_shipped"],
        "min_useful_gain": v["min_useful_gain"],
        "actionable": v["actionable"],
        "all_folds_bit_exact": v["all_folds_bit_exact"],
        "levers": json.dumps(report["levers"]),
    })
    url = run.url
    run.finish()
    return url


def log_v2(report_path: pathlib.Path,
           offsets_path: pathlib.Path | None = None) -> str:
    """Rung 0 v2: the same dose resolved from inside a single leg.

    v1 contrasted whole legs and failed the bar at 833.5 us, because a per-leg
    offset with SD 697 us carried 97.9 percent of the pair variance. v2 moves
    the contrast between neighbouring rounds of one leg, where that offset is
    common to both members and cancels.
    """
    report = json.loads(report_path.read_text())
    legs = report["legs"]
    # Classify by what the leg was ASKED to do. A timing leg carries no
    # witness, on purpose, so presence of the worker's own accounting cannot
    # be the classifier.
    dosed = [leg for leg in legs if leg["dose_requested"]]
    null = [leg for leg in legs if not leg["dose_requested"]]
    session = report["session"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="measurement-protocol",
        name="e109-rung0-protocol-v2",
        config={
            "rung": "0-v2",
            "question": (
                "does a within-leg alternating dose resolve 0.20 % of a "
                "decode-only round in ONE leg, where whole-leg contrasts "
                "needed 180 legs and 4.6 hours"
            ),
            "estimator": (
                "mean difference over non-overlapping neighbouring round pairs "
                "at equal verify width, opposite dose state, round 0 dropped"
            ),
            "drift_control": (
                "equal-width DUD/UDU triples, (x0 + x2) / 2 - x1, which cancels "
                "a linear within-leg drift that the phase-locked pair "
                "estimator absorbs as -drift"
            ),
            "endpoint": "block_request_seconds (parent-measured decode round)",
            "v1_half_width_us": 833.5,
            "v1_half_width_pct": 0.470,
            "v1_sigma_leg_us": 697.0,
            "v1_leg_share_of_pair_variance": 0.979,
            "predicted_half_width_us": 313.0,
            "report": str(report_path),
            **identity(),
            **gate_flags("within-leg alternating-dose timing legs, GPU"),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=["leg", "arm", "arm_env", "matched", "round_count", "pairs",
                 "pairs_by_width", "control_round_us", "effect_us",
                 "half_width_us", "half_width_pct", "drift_us_per_round",
                 "triples", "drift_cancelled_us", "drift_cancelled_half_us",
                 "dose_requested", "entry_c", "exit_c", "wall_seconds",
                 "worker_sha256", "git_head"])
    for leg in legs:
        table.add_data(
            leg["leg"], leg["arm_label"], leg["arm_env"],
            leg["all_tokens_matched"], leg["round_count"], leg["pairs"],
            json.dumps(leg["pairs_by_width"]), leg["e109_control_round_us"],
            leg["paired_difference_mean_us"], leg["half_width_us"],
            leg["half_width_percent"], leg["within_leg_drift_us_per_round"],
            leg["drift_cancelled_triples"], leg["drift_cancelled_mean_us"],
            leg["drift_cancelled_half_width_us"], leg["dose_requested"],
            leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            leg["leg_wall_seconds"], leg["worker_sha256"], leg["git_head"])
    payload = {"v2/legs": table}

    # The recipe another student runs: how many legs buy how much resolution,
    # what that costs, and how often a bar-sized arm actually reads non-zero.
    cost = wandb.Table(
        columns=["legs", "half_width_us", "half_width_pct", "minutes",
                 "clears_bar", "power_at_one_bar"])
    for point in session["cost_curve"]:
        cost.add_data(point["legs"], point["half_width_us"],
                      point["half_width_percent"], point["minutes"],
                      point["clears_bar"], point["power_at_one_bar"])
    payload["v2/cost_curve"] = cost

    # Why `round_alignment_verified` is false: the dose alternates over
    # QUALIFYING FORWARDS and the forward stream is not one per round.
    diagnoses = report.get("forward_stream_diagnosis") or []
    if diagnoses:
        stream = wandb.Table(
            columns=["leg", "tokens", "rounds", "qualifying_forwards",
                     "forwards_per_round", "width_one_forwards",
                     "wide_forwards_after_warmup", "replayed_verify_blocks",
                     "rejected_drafts", "alternation_exact",
                     "one_forward_per_round", "tail_width_fingerprint_matched",
                     "width_histogram_observed",
                     "width_histogram_expected_for_rounds"])
        for d in diagnoses:
            stream.add_data(
                d["leg"], d["tokens"], d["round_count"],
                d["qualifying_forwards"], d["forwards_per_round"],
                d["width_one_forwards"], d["wide_forwards_after_warmup"],
                d["verify_block_replayed_round_count"],
                d["rejected_draft_total"], d["alternation_exact"],
                d["one_forward_per_round"],
                d["tail_width_fingerprint_matched"],
                json.dumps(d["width_histogram_observed"]),
                json.dumps(d["width_histogram_expected_for_rounds"]))
        payload["v2/forward_stream"] = stream

    injection = report.get("synthetic_injection")
    if injection:
        table_inj = wandb.Table(
            columns=["leg", "injected_us", "pair_us", "pair_half_us",
                     "pair_covers", "triple_us", "triple_half_us",
                     "triple_covers"])
        for leg in injection["legs"]:
            table_inj.add_data(
                leg["leg"], injection["injected_us"], leg["pair_mean_us"],
                leg["pair_half_width_us"], leg["pair_covers_delta"],
                leg["triple_mean_us"], leg["triple_half_width_us"],
                leg["triple_covers_delta"])
        payload["v2/synthetic_injection"] = table_inj

    run.log(payload)

    halves = sorted(leg["half_width_us"] for leg in dosed)
    median_half = statistics.median(halves) if halves else float("nan")
    control_us = statistics.fmean(
        leg["e109_control_round_us"] for leg in legs)
    # The null legs run the identical estimator over a dose-free series. If
    # any of their intervals excludes zero, the estimator is manufacturing an
    # effect and no dosed reading from the same session can be believed.
    null_clean = all(
        abs(leg["paired_difference_mean_us"]) <= leg["half_width_us"]
        for leg in null) if null else False

    run.summary.update({
        "dosed_legs": len(dosed),
        "null_legs": len(null),
        "single_leg_half_width_us_median": median_half,
        "single_leg_half_width_us_min": halves[0] if halves else None,
        "single_leg_half_width_us_max": halves[-1] if halves else None,
        "single_leg_half_width_pct_median": 100.0 * median_half / control_us,
        "clears_0p20_bar": bool(median_half <= report["bar_us"]),
        "bar_us": report["bar_us"],
        "control_round_us": control_us,
        "null_covers_zero": null_clean,
        "dosed_effect_us_median": statistics.median(
            [leg["paired_difference_mean_us"] for leg in dosed]) if dosed else None,
        "dosed_effect_us_drift_cancelled_median": statistics.median(
            [leg["drift_cancelled_mean_us"] for leg in dosed]) if dosed else None,
        "all_legs_matched": all(leg["all_tokens_matched"] for leg in legs),
        "improvement_over_v1": 833.5 / median_half if median_half else None,
        # The frame, under the name the advisor asked every consumer to use.
        "e109_v2_control_round_us": session["e109_v2_control_round_us"],
        "arms": json.dumps(session["arms"]),
        "round_alignment_verified": session["resolution"][
            "round_alignment_verified"],
        "dose_minus_null_us": session["pair"]["dose_minus_null_us"],
        "dose_minus_null_half_width_us": session["pair"]["half_width_us"],
        "dose_minus_null_excludes_zero": session["pair"]["excludes_zero"],
        "dose_minus_null_triple_us": session["triple"]["dose_minus_null_us"],
        "dose_minus_null_triple_half_width_us": session["triple"][
            "half_width_us"],
        "null_leg_estimate_scatter_us": session["detection"][
            "null_leg_estimate_scatter_us"],
        "legs_to_reach_bar": session["legs_to_reach_bar"],
        "minutes_to_reach_bar": session["minutes_to_reach_bar"],
        "seconds_per_leg": session["seconds_per_leg"],
        "synthetic_injection_us": (
            report["synthetic_injection"]["injected_us"]
            if report.get("synthetic_injection") else None),
        "synthetic_injection_pair_bias_us": (
            report["synthetic_injection"]["pair_bias_us"]
            if report.get("synthetic_injection") else None),
        "synthetic_injection_all_cover": (
            report["synthetic_injection"]["all_legs_cover_delta"]
            if report.get("synthetic_injection") else None),
        # Why v2 and not simply more v1 legs: the v1 leg offset is
        # exchangeable, so shortening a leg buys almost nothing on a host
        # where the leg is dominated by fixed setup rather than by decode.
        **({f"v1_offset_{k}": (json.dumps(v) if isinstance(v, dict) else v)
            for k, v in json.loads(offsets_path.read_text()).items()}
           if offsets_path else {}),
    })
    url = run.url
    run.finish()
    return url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--v2", type=pathlib.Path)
    parser.add_argument("--offsets", type=pathlib.Path,
                        help="e109_offset_diagnosis.py --json output")
    parser.add_argument("--shape", action="append", default=[],
                        help="FAMILY or FAMILY:SUFFIX, e.g. prework:-s8")
    args = parser.parse_args()

    urls = {}
    if args.protocol:
        urls["e109-rung0-protocol"] = log_protocol(args.protocol)
    if args.v2:
        urls["e109-rung0-protocol-v2"] = log_v2(args.v2, args.offsets)
    for spec in args.shape:
        urls[f"e109-rung1-{spec.replace(':', '')}"] = log_shape(spec)
    for name, url in urls.items():
        print(f"{name}  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
