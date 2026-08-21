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
            "mtp_depth": design["depth"],
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


def log_shape(family: str) -> str:
    verdict_path = OUT / f"e109-shape-{family}" / "verdict.json"
    report = json.loads(verdict_path.read_text())
    v = report["verdict"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="kernel-shape-sweep",
        name=f"e109-rung1-{family}",
        config={
            "rung": "1a" if family == "prework" else "1b",
            "question": (
                f"what carries the intra-kernel residual of {report['live_kernel']}"
                ": occupancy, a dependent chain, or per-threadgroup granularity"
            ),
            "live_kernel": report["live_kernel"],
            "dispatch": report["dispatch"],
            "sweep_command": f"research/e109_shape_probe.sh {family}",
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
    run.log({f"shape/{family}": table})

    for a in report["arms"]:
        run.log({
            "shape/fold_factor": a["fold_factor"],
            "shape/threadgroups": a["threadgroups"],
            "shape/us_per_dispatch": a["us_per_dispatch_median"],
            "shape/waves_over_cores": a["waves_over_cores"],
        })

    run.summary.update({
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--shape", action="append", default=[])
    args = parser.parse_args()

    urls = {}
    if args.protocol:
        urls["e109-rung0-protocol"] = log_protocol(args.protocol)
    for family in args.shape:
        urls[f"e109-rung1-{family}"] = log_shape(family)
    for name, url in urls.items():
        print(f"{name}  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
