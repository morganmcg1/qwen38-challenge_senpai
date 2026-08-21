#!/usr/bin/env python3
"""Publish the E105 latency-class dispatch-family evidence to W&B.

    usage: research/e105_wandb_log.py [--only RUN] [--census TAG] [--dose TAG]

  `e105-rung0-census`   the in-situ dispatch census at the scored cell: which
                        kernels the GDN prework, q/k norm+RoPE and KV write
                        families actually dispatch, their grid and threadgroup
                        shapes, threadgroup counts, waves against 20 GPU cores,
                        dispatches and microseconds per round, and the real
                        achieved bandwidth against DRAM peak and a stream
                        roofline.
  `e105-rung12-dose`    the counterbalanced in-situ dispatch-dose ladder: the
                        marginal cost F of one dispatch boundary, measured by
                        injecting a known number of extra dependent dispatches
                        per decoder layer, at three dose shapes, and the
                        resulting fusion ceiling priced against four named
                        local-round denominators.

Rung 0 is an instrumented census leg and never a timing leg. The dose legs are
timing legs run with `MLXFAST_LOCAL_COOL_GATE=0`, which this host requires
because it asymptotes at 40.55 C and cannot reach the 40 C gate. They are
counterbalanced inside one session, entry and exit GPU temperature are
recorded per leg, and every run logs `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` verbatim. No leg here is a score.

Advisor feedback e105-f1 retired the Finding 22 LATENCY transfer multiplier of
2.40x. Launch-overhead-bound dispatch work transfers at about 0.95x, so every
number here is a local percent and no multiplier is applied.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e105-latency-class-dispatch-family"
OUT = pathlib.Path("research/out")

BASE_SHA = "f556bd5f9bcf0ddb3c8c6ccd33490cb0d2000e03"
SUBMIT_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
FRONTIER = "51b9bf85"
PROMOTED = "f04b102e"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
GPU_CORES = 20

BAR_LOCAL_PCT = 0.20  # e105-f1 promotion bar
RUNG0_FLOOR_PCT = 0.25  # e105-f1 rung 0 stop rule
PUB_FLOOR_PCT = 0.277
SF_FLOOR_PCT = 0.160

# The advisor's family table is anchored on the older E96 census round.
ADVISOR_ROUND_US = 127533.0


def identity() -> dict:
    return {
        "experiment": "e105-latency-class-dispatch-family",
        "harness": "local",
        "base_sha": BASE_SHA,
        "submit_base_sha": SUBMIT_BASE_SHA,
        "board_frontier": FRONTIER,
        "our_promoted_row": PROMOTED,
        "host": HOST,
        "gpu_cores": GPU_CORES,
        "chip": "Apple M4 Pro",
        "device_class": "AGXG16SDevice",
        "ranked_runner_chip": "M5",
        "transfer_law": "e105-f1: ranked % ~= 0.95 x local %, no multiplier",
        "promotion_bar_local_pct": BAR_LOCAL_PCT,
        "rung0_floor_local_pct": RUNG0_FLOOR_PCT,
        "published_detection_floor_pct": PUB_FLOOR_PCT,
        "serial_free_detection_floor_pct": SF_FLOOR_PCT,
    }


def gate_flags(kind: str) -> dict:
    return {
        "leg_kind": kind,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }


def read_meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    if not path.exists():
        return {}
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def log_census(tag: str) -> str:
    report = json.loads((OUT / tag / "census-report.json").read_text())
    meta = read_meta(tag)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="dispatch-census",
        name="e105-rung0-census",
        config={
            "rung": "0",
            "question": (
                "do the GDN prework, q/k norm+RoPE and KV write families "
                "reach the scored cell, are they editable, and is their low "
                "census bandwidth real or a cache-served artefact"
            ),
            "leg": f"research/out/{tag}",
            "leg_command": f"research/e96_census_leg.sh {tag} 4 64 0",
            "buffer_limit_ops": 0,
            "buffer_limit_mb": 1,
            "forced_draft_depth": 4,
            "decode_tokens": 64,
            "scored_width": report["width"],
            "scored_phase": report["phase"],
            "w5_rounds": report["rounds"],
            "dram_peak_gb_s": report["dram_peak_gb_s"],
            "stream_roofline_gb_s": report["stream_roofline_gb_s"],
            "worker_sha256": meta.get("worker_sha256_pre", ""),
            **identity(),
            **gate_flags("E58 one-dispatch-per-buffer census, GPU"),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=[
            "family",
            "signature",
            "grid",
            "threadgroup",
            "threadgroup_count",
            "waves_over_20_cores",
            "dispatches_per_round",
            "us_per_dispatch",
            "us_per_dispatch_min",
            "us_per_dispatch_max",
            "us_per_round",
            "bytes_per_dispatch",
            "achieved_gb_s",
            "pct_of_dram_peak",
            "roofline_us_per_dispatch",
            "roofline_share_of_measured_pct",
        ]
    )
    for f in report["families"]:
        table.add_data(
            f["label"],
            f["signature"],
            "x".join(str(d) for d in f["grid"]),
            "x".join(str(d) for d in f["threadgroup"]),
            f["threadgroup_count"],
            f["waves_over_cores"],
            f["dispatches_per_round"],
            f["us_per_dispatch"],
            f["us_per_dispatch_min"],
            f["us_per_dispatch_max"],
            f["us_per_round"],
            f["bytes_per_dispatch"],
            f["achieved_gb_s"],
            f["pct_of_dram_peak"],
            f["roofline_us_per_dispatch"],
            f["roofline_share_of_measured"],
        )
    run.log({"census/families": table})

    anchors = wandb.Table(columns=["width", "rounds", "gpu_busy_us_per_round"])
    for width, rounds, us in report.get("width_anchors", []):
        anchors.add_data(width, rounds, us)
    run.log({"census/width_gpu_busy": anchors})

    census_round = report.get("census_round_us")
    subtotal = report["total_us_per_round"]
    summary = {
        "rung0/dispatches_per_round": report["total_dispatches_per_round"],
        "rung0/addressable_us_per_round": subtotal,
        "rung0/advisor_round_us": ADVISOR_ROUND_US,
        "rung0/addressable_pct_advisor_round": 100.0 * subtotal / ADVISOR_ROUND_US,
    }
    if census_round:
        summary["rung0/census_round_us"] = census_round
        summary["rung0/addressable_pct_census_round"] = (
            100.0 * subtotal / census_round
        )
        summary["rung0/clears_rung0_floor"] = (
            100.0 * subtotal / census_round >= RUNG0_FLOOR_PCT
        )
    run.summary.update(summary)

    url = run.url
    print(f"e105-rung0-census  {run.id}  {url}")
    run.finish()
    return url


def log_dose(prefix: str) -> str:
    report = json.loads((OUT / prefix / "dose-report.json").read_text())
    legs = report["legs"]
    meta = read_meta(legs[0]["tag"])
    temps = [float(leg["entry_c"]) for leg in legs if leg["entry_c"]]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="dispatch-dose-ladder",
        name="e105-rung12-dose",
        config={
            "rung": "1+2",
            "question": (
                "what is the marginal cost F of one dispatch boundary in "
                "situ, and does N x F clear the 0.20 % local promotion bar "
                "for N = 80 removable dispatches"
            ),
            "session": f"research/out/{prefix}",
            "session_command": f"research/e105_dose_session.sh {prefix}",
            "design": "palindrome A0 Z0@32 B1 C4 D4tiny E4op | reversed",
            "decoder_layers": 64,
            "doses_per_layer": sorted({leg["dose"] for leg in legs}),
            "dose_shapes": sorted({leg["shape"] for leg in legs}),
            "token_counts": sorted({leg["tokens"] for leg in legs}),
            "all_tokens_matched": all(leg["matched"] for leg in legs),
            "entry_temp_min_c": min(temps) if temps else None,
            "entry_temp_max_c": max(temps) if temps else None,
            "entry_temp_spread_c": (max(temps) - min(temps)) if temps else None,
            "worker_sha256": meta.get("worker_sha256_pre", ""),
            **identity(),
            **gate_flags("in-situ dispatch-dose ladder, GPU timing"),
        },
        reinit=True,
    )

    leg_table = wandb.Table(
        columns=[
            "tag",
            "dose_per_layer",
            "shape",
            "decode_tokens",
            "added_dispatches_per_forward",
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "mtp_round_us",
            "serial_round_us",
            "speedup",
            "mean_draft",
            "all_tokens_matched",
            "gpu_temp_entry_c",
            "gpu_temp_exit_c",
            "worker_sha256",
        ]
    )
    for leg in legs:
        leg_table.add_data(
            leg["tag"],
            leg["dose"],
            leg["shape"],
            leg["tokens"],
            leg["added_dispatches_per_forward"],
            leg["mtp_spt"],
            leg["serial_spt"],
            leg["mtp_round_us"],
            leg["serial_round_us"],
            leg["speedup"],
            leg["mean_draft"],
            leg["matched"],
            leg["entry_c"],
            leg["exit_c"],
            leg["worker_sha256"],
        )
    run.log({"dose/legs": leg_table})

    slope_table = wandb.Table(
        columns=[
            "shape",
            "pass",
            "F_us_per_dispatch",
            "points_round_us_by_added_dispatches",
            "pairwise_us_per_dispatch",
        ]
    )
    ceiling_table = wandb.Table(
        columns=[
            "shape",
            "pass",
            "dispatches_removed",
            "ceiling_us_per_round",
            "denominator",
            "denominator_us",
            "local_pct",
            "multiple_of_0p20_bar",
            "clears_bar",
        ]
    )
    required_table = wandb.Table(columns=["shape", "pass", "case", "required_F_us"])

    summary: dict = {}
    for shape in sorted(k for k in ("tiny", "prework", "op") if k in report):
        for pas in ("serial", "mtp"):
            block = report[shape].get(pas)
            if block is None:
                continue
            slope_table.add_data(
                shape,
                pas,
                block["F_us_per_dispatch"],
                json.dumps(block["points"]),
                json.dumps({k: round(v, 3) for k, v in block["pairwise"].items()}),
            )
            for case in ("N80_real_max", "N96_absolute"):
                c = block[case]
                for dname, dval in block["denominators_us"].items():
                    q = c[dname]
                    ceiling_table.add_data(
                        shape,
                        pas,
                        c["dispatches_removed"],
                        c["ceiling_us_per_round"],
                        dname,
                        dval,
                        q["local_pct"],
                        q["multiple_of_bar"],
                        q["clears_0p20_bar"],
                    )
            for case, val in block["required_F_us"].items():
                required_table.add_data(shape, pas, case, val)
            summary[f"F/{shape}_{pas}_us_per_dispatch"] = block["F_us_per_dispatch"]
            n80 = block["N80_real_max"]
            summary[f"ceiling/{shape}_{pas}_N80_us"] = n80["ceiling_us_per_round"]
            summary[f"ceiling/{shape}_{pas}_N80_decode_only_pct"] = n80[
                "decode_only_round"
            ]["local_pct"]
            summary[f"ceiling/{shape}_{pas}_N80_clears_bar"] = n80[
                "decode_only_round"
            ]["clears_0p20_bar"]

    run.log({"dose/slopes": slope_table})
    run.log({"dose/ceilings": ceiling_table})
    run.log({"dose/required_F": required_table})

    dec = report.get("decode_only")
    if dec:
        dtab = wandb.Table(
            columns=[
                "pass",
                "spt_at_n_lo_s",
                "spt_at_n_hi_s",
                "fixed_seed_and_warmup_s",
                "marginal_spt_s",
                "fixed_share_of_reported_pct",
            ]
        )
        for label in ("serial", "mtp"):
            b = dec[label]
            dtab.add_data(
                label,
                b["spt_at_n_lo"],
                b["spt_at_n_hi"],
                dec["fixed_seed_and_warmup_s"],
                b["marginal_spt_s"],
                100.0 * b["fixed_share_of_reported_at_n_hi"],
            )
        run.log({"dose/decode_only_decomposition": dtab})
        summary["denominator/decode_only_round_us"] = dec["decode_only_round_us"]
        for k, v in dec["seed_model_check"].items():
            summary[f"denominator/seed_model_{k}"] = v

    contrast = report.get("shape_contrast")
    if contrast:
        for k, v in contrast.items():
            if not isinstance(v, dict):
                summary[f"contrast/{k}"] = v

    # Repeatability of the identical mirrored legs. The ceiling this experiment
    # is chasing has to be read against this, not against the slope alone.
    for key in ("serial_round_us", "mtp_round_us"):
        diffs = []
        seen: dict[tuple, list[float]] = {}
        for leg in legs:
            seen.setdefault((leg["shape"], leg["dose"], leg["tokens"]), []).append(
                leg[key]
            )
        for vals in seen.values():
            if len(vals) == 2:
                diffs.append(abs(vals[0] - vals[1]))
        if diffs:
            diffs.sort()
            summary[f"noise/{key}_median_mirror_diff_us"] = diffs[len(diffs) // 2]
            summary[f"noise/{key}_max_mirror_diff_us"] = diffs[-1]

    summary["dose/all_tokens_matched"] = all(leg["matched"] for leg in legs)
    if temps:
        summary["thermal/entry_spread_c"] = max(temps) - min(temps)
    run.summary.update(summary)

    url = run.url
    print(f"e105-rung12-dose  {run.id}  {url}")
    run.finish()
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("census", "dose"))
    ap.add_argument("--census", default="e105r0-d4-ops0")
    ap.add_argument("--dose", default="e105r12")
    args = ap.parse_args()

    if args.only in (None, "census"):
        log_census(args.census)
    if args.only in (None, "dose"):
        log_dose(args.dose)


if __name__ == "__main__":
    main()
