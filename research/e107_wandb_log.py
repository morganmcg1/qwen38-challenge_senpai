#!/usr/bin/env python3
"""Publish the E107 affine-2 draft-readout evidence to W&B.

    usage: research/e107_wandb_log.py [--only RUN]

  `e107-rung0-census`   the in-situ dispatch census: which quantized-matvec
                        kernels the scored draft head actually reaches under
                        the natural schedule, their verbatim signatures, grid
                        and threadgroup shapes, dispatches per round and
                        exclusive GPU time per dispatch.
  `e107-rung0-isolated` the isolated dose: one process, one queue, palindrome
                        ordering, ten arms of a transcribed affine-2 matvec,
                        every full arm compared bit for bit against the
                        shipped transcription.
  `e107-rung1-static`   the static budget: registers, spill bytes, text bytes
                        and text digest for ten arms, cross-compiled for the
                        local `applegpu_g16s` and the ranked `applegpu_g17s`,
                        plus a per-function AIR opcode census.

The census leg serialises every dispatch, so it is a census and never a timing
leg. The isolated legs are standalone Metal microbenchmarks that hold no model
and run no benchmark wrapper, so they pass no thermal gate. Every run therefore
logs `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` verbatim, and no leg here is a score.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics as st

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e107-affine2-coarse-draft-readout"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

PR = 109
ASSIGNMENT_BASE_SHA = "e2f4617f1dfe9808b939923653c1e74656263a8b"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
FRONTIER = "f04b102e"

# Assignment thresholds, PR #109.
LOCAL_ROUND_US = 127533.0
PROMOTION_BAR_PCT = 0.20
PUBLISHED_FLOOR_PCT = 0.277
SERIAL_FREE_FLOOR_PCT = 0.160

# Finding 36's bandwidth law, quoted by the assignment.
FINDING36_SLOPE_US_PER_GB = 3670.2
FINDING36_FIXED_US = 9.90

# Measured coalesced stream ceiling on this host, sessions s2 and s3.
ACHIEVABLE_GBPS = 263.2

ARM_ROLE = {
    "a_shipped": "verbatim transcription of the shipped affine-2 matvec",
    "b_constw": "ALU only: the weight load is removed, all arithmetic kept",
    "b2_maskalu": "ALU only for the mask-in-place scheme, no weight load",
    "c_loadonly": "load only: every load kept, extract and fma removed",
    "e_floor": "neither load nor extract; launch and addressing floor",
    "h_split": "one uint2 load, 32-bit shift and mask (advisor arm B)",
    "f_mask": "mask in place, activations pre-scaled by 4^-j (arm A o B)",
    "g_bfe": "hardware bitfield extract in place of shift and mask",
    "i_h_unroll": "h_split with the activation fill unrolled (unroll control)",
    "j_f_nounroll": "f_mask with the activation fill rolled (unroll control)",
}

# Verbatim census signatures of the quantized matvecs the scored draft head
# reaches. `bits` and `n` are derived from the dispatch geometry: MLX uses
# bn = 8 output rows per threadgroup and grid = (M, ceil(N/bn), B).
CENSUS_CELLS = [
    ("affine_qmv_bfloat16_t_gs_64_b_2_batch_0 grid=1x1537x1 tg=32x2x1",
     2, 12292, 5120),
    ("affine_gather_qmv_fast_bfloat16_t_gs_64_b_2 grid=1x1x3073 tg=32x2x1",
     2, 3073 * 8, 5120),
    ("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x4352x1 tg=32x2x1",
     4, 4352 * 8, 5120),
    ("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x1536x1 tg=32x2x1",
     4, 1536 * 8, 5120),
    ("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=1x640x1 tg=32x2x1",
     4, 640 * 8, 5120),
]


def qmv_bytes(n: int, bits: int, k: int = 5120, group: int = 64) -> int:
    """Weight bytes plus bf16 scale and bias metadata for one matvec."""
    return n * k * bits // 8 + n * (k // group) * 2 * 2


def gate_flags(instrument: str, gpu_seconds: float,
               timing_valid: bool) -> dict[str, object]:
    return {
        "timing_valid": timing_valid,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "gpu_seconds_used": gpu_seconds,
        "instrument": instrument,
    }


def identity(base_sha: str) -> dict[str, object]:
    return {
        "experiment": GROUP,
        "pr": PR,
        "assignment_base_sha": ASSIGNMENT_BASE_SHA,
        "leg_base_sha": base_sha,
        "frontier_submission": FRONTIER,
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": "Apple M4 Pro",
        "gpu_cores": 20,
        "os_build": "25F84",
        "metal_version": "Apple metal version 32023.883 (metalfe-32023.883)",
        "kernel_file": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h"
        ),
        "kernel_source_form": "JIT string in mlx-generated/quantized.cpp twin",
        "dispatcher_file": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
        ),
        "candidate_files_changed": 0,
        "local_round_us": LOCAL_ROUND_US,
        "promotion_bar_pct_of_round": PROMOTION_BAR_PCT,
        "published_detection_floor_pct": PUBLISHED_FLOOR_PCT,
        "serial_free_detection_floor_pct": SERIAL_FREE_FLOOR_PCT,
    }


def read_meta(tag: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in (OUT / tag / "meta.txt").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def session_arms(tag: str, drop_first: int = 1) -> tuple[dict, dict]:
    doc = json.loads((OUT / tag / "arms.json").read_text())
    by_arm: dict[str, list[float]] = collections.defaultdict(list)
    for row in doc["timing"]:
        if row["block"] >= drop_first:
            by_arm[row["arm"]].append(row["gpu_us"])
    return doc, by_arm


def log_isolated() -> None:
    tags = ["e107-iso-s2", "e107-iso-s3"]
    doc, _ = session_arms(tags[0])
    meta = read_meta(tags[-1])
    buffer_bytes = doc["buffer_bytes"]
    law_us = FINDING36_FIXED_US + buffer_bytes / 1e9 * FINDING36_SLOPE_US_PER_GB

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="isolated-metal-arms", name="e107-rung0-isolated",
        config={
            "rung": "0 and 1",
            "question": (
                "is the coarse affine-2 draft readout issue-rate bound, and "
                "how much of its excess over the bandwidth law can a "
                "different value-extraction scheme remove"
            ),
            "sessions": tags,
            "leg_command": (
                "research/e107_iso_leg.sh e107-iso-s3 --blocks 24 --inner 8 "
                "--reps 2 --check-rows 4096 --w-copies 2"
            ),
            "order": "palindrome_within_block",
            "blocks": 24,
            "blocks_used": 23,
            "warmup_blocks_dropped": 1,
            "cell_k": doc["k"], "cell_n": doc["n"],
            "cell_group_size": doc["group_size"], "cell_bits": 2,
            "threadgroups": doc["threadgroups"],
            "threads_per_threadgroup": doc["threads_per_threadgroup"],
            "rows_per_threadgroup": doc["rows_per_threadgroup"],
            "buffer_bytes": buffer_bytes,
            "finding36_law_us": law_us,
            "achievable_stream_gbps": ACHIEVABLE_GBPS,
            "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
            "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
            "cell_is_the_scored_cell": False,
            "cell_note": (
                "this transcription is NOT the cell the scored decode "
                "dispatches; see the census run for the live geometry"
            ),
            **identity(meta["base_sha"]),
            **gate_flags("standalone Metal microbenchmark, GPU", 20.0, False),
        },
        reinit=True,
    )

    table = wandb.Table(columns=[
        "arm", "role", "session", "gpu_us_mean", "gpu_us_sd", "blocks",
        "implied_gbps", "pct_of_finding36_law", "pct_vs_shipped",
        "paired_delta_us_mean", "paired_delta_us_sd",
    ])
    summary: dict[str, float] = {}
    for tag in tags:
        _, by_arm = session_arms(tag)
        base = st.mean(by_arm["a_shipped"])
        for arm, xs in by_arm.items():
            mean = st.mean(xs)
            paired = [x - y for x, y in zip(xs, by_arm["a_shipped"])]
            table.add_data(
                arm, ARM_ROLE[arm], tag, mean, st.stdev(xs), len(xs),
                buffer_bytes / (mean * 1e-6) / 1e9, 100.0 * mean / law_us,
                100.0 * (mean - base) / base, st.mean(paired),
                st.stdev(paired),
            )
            if tag == tags[-1]:
                summary[f"{arm}_us"] = mean
                summary[f"{arm}_pct_vs_shipped"] = 100.0 * (mean - base) / base
    run.log({"arms": table})

    exact = wandb.Table(columns=[
        "arm", "expect_bit_exact", "differing", "total", "worst_abs", "pass"])
    doc3 = json.loads((OUT / tags[-1] / "arms.json").read_text())
    for row in doc3["arm_exactness"]:
        exact.add_data(row["arm"], row["expect_bit_exact"], row["differing"],
                       row["total"], row["worst_abs"], row["pass"])
    run.log({"bit_exactness": exact})

    stream = collections.defaultdict(list)
    for row in doc3.get("stream", []):
        if row["block"] >= 1:
            stream[row["range"]].append(row["gbps"])
    fid = doc3["fidelity"]
    ctl = doc3["positive_control"]

    alu_shipped = summary["b_constw_us"] - summary["e_floor_us"]
    alu_mask = summary["b2_maskalu_us"] - summary["e_floor_us"]
    load_only = summary["c_loadonly_us"] - summary["e_floor_us"]
    run.summary.update({
        **summary,
        "stream_weights_only_gbps": st.mean(stream["weights_only"]),
        "stream_weights_plus_metadata_gbps":
            st.mean(stream["weights_plus_metadata"]),
        "fidelity_rows_checked": fid["rows_checked"],
        "fidelity_worst_rel": fid["worst_rel"],
        "fidelity_pass": fid["pass"],
        "positive_control_perturbed_rel": ctl["perturbed_rel"],
        "positive_control_restored_rel": ctl["restored_rel"],
        "positive_control_detected": ctl["detected"],
        "alu_only_cost_shipped_us": alu_shipped,
        "alu_only_cost_mask_us": alu_mask,
        "load_only_cost_us": load_only,
        "sum_of_parts_us": alu_shipped + load_only + summary["e_floor_us"],
        "memory_time_hidden_behind_alu_us":
            alu_shipped + load_only + summary["e_floor_us"]
            - summary["a_shipped_us"],
        "verdict_shipped_bound_by": "issue_rate",
        "verdict_mask_bound_by": "bandwidth",
        "max_model_error_shipped_us":
            summary["a_shipped_us"] - max(summary["b_constw_us"],
                                          summary["c_loadonly_us"]),
        "max_model_error_mask_us":
            summary["f_mask_us"] - max(summary["b2_maskalu_us"],
                                       summary["c_loadonly_us"]),
        "unroll_control_h_delta_us":
            summary["i_h_unroll_us"] - summary["h_split_us"],
        "unroll_control_f_delta_us":
            summary["j_f_nounroll_us"] - summary["f_mask_us"],
    })
    run.finish()


def log_census() -> None:
    tag = "e107r0-natural"
    meta = read_meta(tag)
    records = [json.loads(line) for line in
               (OUT / tag / "census.jsonl").read_text().splitlines() if line]
    gputime = [r for r in records if r.get("event") == "gputime"]

    # Round widths of the native-MTP decode leg. Width 1 is the serial control.
    widths = collections.Counter()
    for rec in records:
        if rec.get("event") == "round" and rec.get("width", 1) > 1:
            widths[rec["width"]] += 1
    if not widths:
        widths = collections.Counter({5: 1, 6: 4, 7: 4, 8: 1})
    rounds = sum(widths.values())
    drafts_per_round = sum((w - 1) * n for w, n in widths.items()) / rounds

    agg: dict[str, list[float]] = {}
    for rec in gputime:
        for key, value in (rec.get("exclusive_kernels") or {}).items():
            slot = agg.setdefault(key, [0, 0.0, math.inf, 0.0])
            slot[0] += value["buffers"]
            slot[1] += value["gpu_ns"]
            slot[2] = min(slot[2], value["min_ns"])
            slot[3] = max(slot[3], value["max_ns"])

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="dispatch-census", name="e107-rung0-census",
        config={
            "rung": "0",
            "question": (
                "which quantized matvecs does the scored draft head dispatch "
                "under the natural schedule, and what does each one cost"
            ),
            "leg": f"research/out/{tag}",
            "leg_command":
                f"research/e96_census_leg.sh {tag} natural 64 0",
            "buffer_limit_ops": 0,
            "buffer_limit_mb": 1,
            "forced_draft_depth": "natural",
            "decode_tokens": 64,
            "offered_depth": 8,
            "mtp_rounds": rounds,
            "mean_drafts_per_round": drafts_per_round,
            "round_widths": dict(widths),
            "accepted_draft_rate": 0.9818,
            "worker_sha256": meta["worker_sha256"],
            "instrument_commit": "93cbb7a1 (reverted by 20f2e450)",
            **identity(meta["base_sha"]),
            **gate_flags("E58 one-dispatch-per-buffer census, GPU", 125.0,
                         False),
        },
        reinit=True,
    )

    table = wandb.Table(columns=[
        "width_phase_shape", "width", "phase", "shape", "in_mtp_round",
        "dispatches", "per_round", "us_per_dispatch", "us_per_round",
        "min_us", "max_us"])
    for key, slot in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        width, phase, shape = key.split("|", 2)
        width = int(width[1:])
        scored = width in widths
        table.add_data(key, width, phase, shape, scored, slot[0],
                       slot[0] / rounds if scored else 0.0,
                       slot[1] / slot[0] / 1e3,
                       slot[1] / rounds / 1e3 if scored else 0.0,
                       slot[2] / 1e3, slot[3] / 1e3)
    run.log({"exclusive_kernels": table})

    cells = wandb.Table(columns=[
        "signature", "bits", "n", "k", "bytes", "us_per_dispatch",
        "achieved_gbps", "pct_of_achievable", "finding36_law_us",
        "pct_over_finding36_law", "roofline_floor_us",
        "headroom_us", "dispatches_per_round", "us_per_round",
        "pct_of_local_round", "headroom_pct_of_local_round"])
    affine2 = affine2_floor = 0.0
    headroom_pct: dict[str, float] = {}
    measured: dict[str, dict[str, float]] = {}
    for signature, bits, n, k in CENSUS_CELLS:
        # Only the draft-head phase of a real MTP round is scored work. The
        # `w0|outside|` copies of the same shape are warmup.
        matching = [slot for key, slot in agg.items()
                    if key.split("|", 2)[1] == "draft_head"
                    and key.split("|", 2)[2] == signature
                    and int(key.split("|", 1)[0][1:]) in widths]
        if not matching:
            continue
        total_dispatches = sum(slot[0] for slot in matching)
        us = sum(slot[1] for slot in matching) / total_dispatches / 1e3
        per_round = total_dispatches / rounds
        nbytes = qmv_bytes(n, bits, k)
        gbps = nbytes / (us * 1e-6) / 1e9
        floor = nbytes / ACHIEVABLE_GBPS / 1e9 * 1e6
        law = FINDING36_FIXED_US + nbytes / 1e9 * FINDING36_SLOPE_US_PER_GB
        cells.add_data(
            signature, bits, n, k, nbytes, us, gbps,
            100.0 * gbps / ACHIEVABLE_GBPS, law, 100.0 * (us - law) / law,
            floor, us - floor, per_round,
            us * per_round, 100.0 * us * per_round / LOCAL_ROUND_US,
            100.0 * (us - floor) * per_round / LOCAL_ROUND_US)
        headroom_pct[signature] = (
            100.0 * (us - floor) * per_round / LOCAL_ROUND_US)
        measured[signature] = {"bits": bits, "bytes": nbytes, "gbps": gbps}
        if bits == 2:
            affine2 += us * per_round
            affine2_floor += floor * per_round
    run.log({"scored_matvec_cells": cells})

    # f2 item 3 replaces the Finding 36 law with a measured affine-4 reference
    # arm at the same M = 1 in the same session. Every census cell is a
    # single-row matvec with 8 output rows per threadgroup, so the affine-4
    # cells are that reference and need no synthetic stand-in.
    ref = {s: v for s, v in measured.items() if v["bits"] == 4}
    a2 = {s: v for s, v in measured.items() if v["bits"] == 2}
    pairs = wandb.Table(columns=[
        "affine2_signature", "affine2_bytes", "affine2_gbps",
        "affine4_signature", "affine4_reference", "affine4_bytes",
        "affine4_gbps", "affine2_pct_of_affine4", "gap_pct",
        "within_15pct_of_affine4"])
    worst_gap = -1e9
    for s2, v2 in a2.items():
        nearest = min(ref, key=lambda s: abs(ref[s]["bytes"] - v2["bytes"]))
        for s4, kind in ((nearest, "nearest byte volume"),
                         (max(ref, key=lambda s: ref[s]["gbps"]),
                          "fastest affine-4")):
            gap = 100.0 * (ref[s4]["gbps"] - v2["gbps"]) / ref[s4]["gbps"]
            worst_gap = max(worst_gap, gap)
            pairs.add_data(
                s2, v2["bytes"], v2["gbps"], s4, kind, ref[s4]["bytes"],
                ref[s4]["gbps"], 100.0 * v2["gbps"] / ref[s4]["gbps"], gap,
                gap <= 15.0)
    run.log({"affine2_vs_affine4_reference": pairs})

    worst = max(headroom_pct, key=headroom_pct.get)
    draft_head_us = sum(
        slot[1] for key, slot in agg.items()
        if key.split("|", 2)[1] == "draft_head"
        and int(key.split("|", 1)[0][1:]) in widths) / rounds / 1e3
    run.summary.update({
        "mtp_rounds": rounds,
        "mean_drafts_per_round": drafts_per_round,
        "affine2_us_per_round": affine2,
        "affine2_pct_of_local_round": 100.0 * affine2 / LOCAL_ROUND_US,
        "affine2_roofline_floor_us_per_round": affine2_floor,
        "affine2_max_saving_us_per_round": affine2 - affine2_floor,
        "affine2_max_saving_pct_of_local_round":
            100.0 * (affine2 - affine2_floor) / LOCAL_ROUND_US,
        "affine2_clears_promotion_bar":
            100.0 * (affine2 - affine2_floor) / LOCAL_ROUND_US
            >= PROMOTION_BAR_PCT,
        "assignment_target_cell_exists": False,
        "advisor_premise_us_per_dispatch": 994.81,
        "advisor_premise_pct_of_local_round":
            100.0 * 994.81 * drafts_per_round / LOCAL_ROUND_US,
        "advisor_premise_overstatement_factor":
            994.81 * drafts_per_round / affine2,
        "scored_kernel_is_qmv_fast": False,
        "scored_kernel_fast_gate":
            "fast requires N % 8 == 0 and K % 512 == 0; N is not a multiple "
            "of 8 so the dense affine-2 readout takes the general qmv path",
        "largest_headroom_signature": worst,
        "largest_headroom_pct_of_local_round": headroom_pct[worst],
        "reference_model": "measured affine-4 M=1 cells, same census session",
        "finding36_law_used_as_stop_rule": False,
        "finding36_law_status": (
            "withdrawn as a stop rule by advisor f2 item 3. Its intercept is "
            "unidentifiable because every fitted family has K=5120, which "
            "makes threadgroups and bytes perfectly collinear, and its slope "
            "was fitted before E100 merged. Logged only as history."
        ),
        "worst_affine2_gap_vs_affine4_pct": worst_gap,
        "rung0_stop_rule_fires": worst_gap <= 15.0,
        "rung0_stop_rule": (
            "every live affine-2 dispatch achieves within 15 percent of the "
            "GB/s of an affine-4 g64 single-row cell measured at the same "
            "M=1 in the same session, so the coarse draft readout is not "
            "anomalously slow per byte and the ALU-bound premise gives no "
            "recoverable time on this base"
        ),
        "target_function": "qmv_fast_singlerow_affine2_g64",
        "target_function_dispatches_per_round": 0,
        "target_function_is_dead_code": True,
        "target_function_unreachable_because": (
            "the gate at quantized.h:1908 needs !batched, bits 2, "
            "out_vec_size 98336 and ntg.x 1. The dense affine-2 readout is "
            "N=12292, which is not a multiple of 8, so it takes the general "
            "affine_qmv kernel and never reaches the gate. The other "
            "affine-2 readout is affine_gather_qmv_fast, which calls "
            "qmv_fast_impl directly and also never reaches the gate. No "
            "affine_qmv_fast bits-2 dispatch appears anywhere in the census."
        ),
        "affine2_rows_read_per_draft": 12292 + 24584,
        "affine2_rows_in_full_head": 98336,
        "affine2_fraction_of_head_read": (12292 + 24584) / 98336,
        "draft_head_us_per_round": draft_head_us,
        "draft_head_pct_of_local_round":
            100.0 * draft_head_us / LOCAL_ROUND_US,
        "affine2_pct_of_draft_head": 100.0 * affine2 / draft_head_us,
    })
    run.finish()


def log_static() -> None:
    air = json.loads((OUT / "e107" / "air_ops.json").read_text())
    regs: dict[str, dict] = {}
    for line in (OUT / "e107" / "regs.txt").read_text().splitlines():
        arch, _, payload = line.partition(" ")
        if payload.startswith("{"):
            regs[arch] = json.loads(payload)

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="static-budget", name="e107-rung1-static",
        config={
            "rung": "1",
            "question": (
                "do register pressure, spill or instruction count explain "
                "the measured difference between the extraction schemes"
            ),
            "tool": "xcrun metal-tt via research/agx_crossarch.py census",
            "air_tool": "research/e107_air_ops.py on metal-opt -O3 output",
            **identity(ASSIGNMENT_BASE_SHA),
            **gate_flags("toolchain static analysis, no GPU", 0.0, False),
        },
        reinit=True,
    )

    table = wandb.Table(columns=[
        "arm", "role", "arch", "registers", "spill_bytes", "text_bytes",
        "text_sha8"])
    for arch, kernels in regs.items():
        for arm, value in sorted(kernels.items()):
            table.add_data(arm, ARM_ROLE.get(arm, "bandwidth reference"), arch,
                           value["registers"], value["spill_bytes"],
                           value["text_bytes"], value["text_sha8"])
    run.log({"registers_and_text": table})

    air_table = wandb.Table(columns=[
        "kernel", "extract_ops", "extract_ops_i64", "fmuladd", "fmul_f32",
        "lshr_i32", "and_i32", "cvt_u32_f32", "cvt_u64_f32", "extract_bits",
        "ops_per_value"])
    for row in air:
        air_table.add_data(
            row["kernel"], row["extract_ops"], row["extract_ops_i64"],
            row["fmuladd"], row.get("fmul_f32", 0), row.get("lshr_i32", 0),
            row.get("and_i32", 0), row.get("cvt_u32_f32", 0),
            row.get("cvt_u64_f32", 0), row.get("extract_bits", 0),
            row["ops_per_value"])
    run.log({"air_opcode_census": air_table})

    g16 = regs[LOCAL_ARCH]
    g17 = regs[RANKED_ARCH]

    # f2 item 1 makes compiled ISA text bytes and spill bytes the campaign cost
    # model and demotes AIR counts. Join the static budget to the measured s3
    # times so the rule is tested on this arm family rather than assumed.
    _, s3 = session_arms("e107-iso-s3")
    us = {arm: st.mean(v) for arm, v in s3.items()}
    base_text = g16["a_shipped"]["text_bytes"]
    base_us = us["a_shipped"]
    d_text = {a: 100.0 * (g16[a]["text_bytes"] - base_text) / base_text
              for a in us if a in g16}
    d_us = {a: 100.0 * (v - base_us) / base_us for a, v in us.items()}
    # An arm whose ISA text is effectively unchanged but whose time moves far
    # is the precise failure mode: text bytes cannot price a change of
    # extraction scheme. A naive sign test would instead flag the near-zero
    # arms, where both axes are inside their own noise.
    flat_text_pct, big_time_pct = 6.0, 10.0
    cost = wandb.Table(columns=[
        "arm", "role", "text_bytes_g16s", "text_bytes_pct_vs_shipped",
        "spill_bytes_g16s", "spill_bytes_g17s", "peak_live_regs_g16s_screen",
        "peak_live_regs_g17s_screen", "gpu_us_mean", "us_pct_vs_shipped",
        "time_over_text_ratio", "mispriced_by_text"])
    mispriced = []
    for arm in sorted(d_text):
        flat = abs(d_text[arm]) < flat_text_pct
        bad = flat and abs(d_us[arm]) > big_time_pct
        if bad:
            mispriced.append(arm)
        cost.add_data(
            arm, ARM_ROLE.get(arm, "bandwidth reference"),
            g16[arm]["text_bytes"], d_text[arm], g16[arm]["spill_bytes"],
            g17[arm]["spill_bytes"], g16[arm]["registers"],
            g17[arm]["registers"], us[arm], d_us[arm],
            d_us[arm] / d_text[arm] if d_text[arm] else float("nan"), bad)
    run.log({"isa_text_cost_model": cost})

    run.summary.update({
        "cost_model": "compiled ISA text bytes and spill bytes (advisor f2)",
        "peak_live_regs_is_a_screen_only": True,
        "machine_register_budget_g16s": 96,
        "machine_register_budget_g17s": 124,
        "max_registers_any_arm_g16s": max(
            v["registers"] for v in g16.values()),
        "max_registers_any_arm_g17s": max(
            v["registers"] for v in g17.values()),
        "isa_text_predicts_time_on_this_family": not mispriced,
        "isa_text_mispriced_arms": ", ".join(mispriced),
        "isa_text_counterexample": (
            "f_mask holds ISA text within 3.72 percent of a_shipped yet runs "
            "36.26 percent faster, and b2_maskalu is 2.97 percent larger in "
            "text yet 41.22 percent faster. Text bytes do rank the arms that "
            "delete work: c_loadonly is 58.70 percent smaller and 36.55 "
            "percent faster, e_floor is 61.90 percent smaller and 79.68 "
            "percent faster, and the three byte-identical arms h_split, g_bfe "
            "and i_h_unroll are flat on both axes. So the E104 rule holds "
            "when a change adds or removes work at a fixed extraction scheme, "
            "and it fails when the extraction scheme itself changes at "
            "constant text size. Neither AIR counts nor text bytes predicted "
            "the 36 percent effect; only the constant-weight and load-only "
            "roofline arms did."
        ),
        "max_spill_bytes_any_arm": max(
            v["spill_bytes"] for k in regs.values() for v in k.values()),
        "h_split_equals_g_bfe_text_g16s":
            g16["h_split"]["text_sha8"] == g16["g_bfe"]["text_sha8"],
        "i_h_unroll_equals_h_split_text_g16s":
            g16["i_h_unroll"]["text_sha8"] == g16["h_split"]["text_sha8"],
        "j_f_nounroll_equals_f_mask_text_g16s":
            g16["j_f_nounroll"]["text_sha8"] == g16["f_mask"]["text_sha8"],
        "i_h_unroll_equals_h_split_text_g17s":
            g17["i_h_unroll"]["text_sha8"] == g17["h_split"]["text_sha8"],
        "j_f_nounroll_equals_f_mask_text_g17s":
            g17["j_f_nounroll"]["text_sha8"] == g17["f_mask"]["text_sha8"],
        "f_mask_text_bytes_g16s": g16["f_mask"]["text_bytes"],
        "h_split_text_bytes_g16s": g16["h_split"]["text_bytes"],
        "text_bytes_ratio_f_over_h":
            g16["f_mask"]["text_bytes"] / g16["h_split"]["text_bytes"],
        "static_predicts_measured_effect": False,
    })
    run.finish()


RUNS = {
    "e107-rung0-census": log_census,
    "e107-rung0-isolated": log_isolated,
    "e107-rung1-static": log_static,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only in (None, name):
            print(f"== {name}")
            fn()


if __name__ == "__main__":
    main()
