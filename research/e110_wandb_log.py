#!/usr/bin/env python3
"""Publish the E110 rung-1 extraction-schedule study to W&B.

    usage: research/e110_wandb_log.py [--only RUN ...]

Rung 0 refuted H1 and H2: the activation stream costs nothing on its own and
the register ladder is far too shallow to explain the one-group deficit. Rung 1
therefore attacks the extraction schedule inside the k-block. Three runs:

  `e110-rung1-census`
      Static evidence for every arm on both GPU generations: AIR device and
      threadgroup load counts after -O2, registers, spill, machine text,
      compiled threadgroup memory and the occupancy each tile implies.
  `e110-rung1-timing`
      The pooled isolated sweep. Counterbalanced sessions over five scored
      shapes and NA = 2, 3, 4, 5, the round-weighted headline over the realised
      verify-width histogram, the rule 36b roofline pair, exactness and the
      pre-registered KILL and ADVANCE decisions.
  `e110-rung2-insitu`
      The advanced arm inside the real kernel: full-window exactness, matched
      ABBA absolute candidate MTP seconds per token against a fresh unchanged
      base in the same session, and the driver-read threadgroup memory that
      killed the staging arm.

Every timed leg here ran with no thermal gate, so `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are logged false
verbatim. No number here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e110-one-group-wide-qmv-activation-reread"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
OUT = pathlib.Path("research/out")

BASE_SHA = "05321b0f5867e82d73c479638a5f9a1cf503ec2f"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false> _wide body, one x-group"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
BASE_ARM = "a_base"
WARMUP_BLOCKS = 1

# Rule 34 round frame. Decode-only M = 5 on the current tree.
DRAM_PEAK_GBS = 273.0
ROUND_US = 102_864.0
STREAM_GB = 14.4123
STREAM_RATE_GBS = 179.9
STREAM_US = STREAM_GB / STREAM_RATE_GBS * 1e6
PROMOTION_BAR_PCT = 0.20
RANKED_TRANSFER = 0.95

# The realised verify-width histogram of the same fixture, from the advisor.
# NA = 4 carries two thirds of the streaming time; NA = 5 carries 3.4 %.
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# Pre-registered rung-1 decision bars, round-weighted.
KILL_PCT = -0.30
ADVANCE_PCT = -0.50

# Threadgroup memory pool per core, and 64 threads = 2 simdgroups per group.
TG_POOL_BYTES = 32768
SIMDGROUPS_PER_THREADGROUP = 2
REGISTER_FILE = {LOCAL_ARCH: 384 * 1024, RANKED_ARCH: 496 * 1024}

# Sessions pooled into the timing run. Every session carries `a_base`, and the
# palindrome makes each session internally counterbalanced.
SESSIONS = ("e110-rung1", "e110-rung1b", "e110-rung1c")
CENSUS = OUT / "e110/rung1c-census.json"

ROOFLINE_LOAD = "l_loadonly"
ROOFLINE_ALU = "b_constw"
DIAGNOSTIC_ARMS = (ROOFLINE_LOAD, ROOFLINE_ALU)

ARM_NOTE = {
    "a_base": "shipped one-group body, unmodified",
    "l_loadonly": "every device load kept, the 160xNA scalar body dropped; "
                  "the LOAD half of the rule 36b roofline pair",
    "b_constw": "both operand streams constant, full arithmetic kept; "
                "the ALU half of the rule 36b roofline pair",
    "b_barrier": "a_base plus the two staging barriers and nothing else",
    "xs_stage": "the NA x 512 activation tile staged once per threadgroup",
    "mo_stage": "mo_swap plus xs_stage",
    "xv4": "the four scalar activation reads folded into one 8-byte vec<T,4>",
    "xv8": "one 16-byte uint4 covers 8 values; i becomes two halves of two",
    "xv4_stage": "xs_stage plus the vec<T,4> read of the staged tile",
    "mo_swap": "the i-outer/m-inner nest swapped to m-outer/i-inner",
    "mu_swap": "mo_swap with the m loop force-unrolled",
    "mo_hoist": "i-outer kept, the nibble extraction hoisted before the m loop",
}


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            meta[key] = value
    return meta


def start(name: str, job_type: str, question: str, step: int, config: dict,
          meta_dir: str = SESSIONS[0]):
    meta = read_meta(OUT / meta_dir / "meta.txt")
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "rung": 1, "step": step,
            "question": question, "entry_cell": ENTRY_CELL,
            "host": HOST, "hostname": meta.get("host"),
            "chip": meta.get("chip"), "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "metal_toolchain": meta.get("metal_toolchain"),
            "git_head": meta.get("git_head"),
            "base_sha": BASE_SHA, "advisor_branch": ADVISOR_BRANCH,
            "local_arch": LOCAL_ARCH, "ranked_arch": RANKED_ARCH,
            "round_frame_us": ROUND_US,
            "streaming_term_us": STREAM_US,
            "round_weights": {str(k): v for k, v in ROUND_WEIGHTS.items()},
            "ranked_transfer_factor": RANKED_TRANSFER,
            "promotion_bar_pct_of_round": PROMOTION_BAR_PCT,
            "kill_bar_weighted_pct": KILL_PCT,
            "advance_bar_weighted_pct": ADVANCE_PCT,
            **config, **gate_flags(),
        },
        reinit=True,
    )


def attach(run, *paths: pathlib.Path) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="analysis")
    for path in present:
        artifact.add_file(str(path), name=f"{path.parent.name}/{path.name}")
    run.log_artifact(artifact)


def med(values) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def weighted(per_width: dict[int, float]) -> float:
    total = sum(ROUND_WEIGHTS[w] for w in per_width if w in ROUND_WEIGHTS)
    if not total:
        return float("nan")
    return sum(ROUND_WEIGHTS[w] * v for w, v in per_width.items()
               if w in ROUND_WEIGHTS) / total


def load_sessions() -> tuple[dict, list[str]]:
    """Per-block timings keyed by (session, shape, na), plus the arm union."""
    cells: dict[tuple[str, str, int], dict] = {}
    arms: list[str] = []
    for session in SESSIONS:
        path = OUT / session / "rate.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for arm in doc["arms"]:
            if arm not in arms:
                arms.append(arm)
        for row in doc["measurements"]:
            if row["kind"] != "timing" or row["block"] < WARMUP_BLOCKS:
                continue
            cell = cells.setdefault((session, row["shape"], row["m"]), {
                "bytes": row["read_bytes"], "flops": row["flops"],
                "temp": row["gpu_temp_entry_c"], "blocks": []})
            cell["blocks"].append(row["seconds"])
    return cells, arms


def paired_pct(cells: dict, arm: str) -> dict[int, list[float]]:
    """Per-block percent change against `a_base`, pooled over sessions."""
    out: dict[int, list[float]] = {}
    for (_, _, width), cell in cells.items():
        for block in cell["blocks"]:
            if arm not in block or BASE_ARM not in block:
                continue
            out.setdefault(width, []).append(
                100.0 * (block[arm] / block[BASE_ARM] - 1.0))
    return out


def log_census() -> None:
    census = json.loads(CENSUS.read_text())
    widths = [int(w) for w in census["widths"]]
    arms = list(census["arms"])

    run = start(
        "e110-rung1-census", "census",
        "What do the static instruments say about each extraction schedule, "
        "and do they rank the arms correctly?",
        1, {"arms": arms, "widths": widths,
            "instrument": "AIR load counts after -O2 plus metal-tt registers, "
                          "spill, machine text and threadgroup memory on both "
                          "architectures",
            "isa_disassembly_available": False,
            "isa_note": "metal-objdump on this toolchain has no AGX "
                        "instruction printer, so the load census is read from "
                        "post-optimization AIR and corroborated by machine "
                        "text size, register counts and threadgroup memory"})

    air = wandb.Table(columns=[
        "arm", "na", "device_loads", "threadgroup_loads",
        "threadgroup_stores", "threadgroup_bytes", "air_lines", "note"])
    for arm in arms:
        for width in widths:
            rec = census["arms"][arm]["air"][str(width)]
            air.add_data(arm, width, rec["device_loads"],
                         rec["threadgroup_loads"], rec["threadgroup_stores"],
                         rec["threadgroup_bytes"], rec["air_lines"],
                         ARM_NOTE.get(arm, ""))
    run.log({"air_load_census": air})

    text = wandb.Table(columns=[
        "arm", "arch", "na", "registers", "spill_bytes", "text_bytes",
        "text_sha8", "register_limited_simdgroups"])
    for arm in arms:
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for width in widths:
                rec = census["arms"][arm].get(arch, {}).get(str(width))
                if rec is None:
                    continue
                text.add_data(arm, arch, width, rec["registers"],
                              rec["spill_bytes"], rec["text_bytes"],
                              rec.get("text_sha8"),
                              REGISTER_FILE[arch] // (128 * rec["registers"]))
    run.log({"machine_text_census": text})

    tg = wandb.Table(columns=[
        "arm", "na", "threadgroup_bytes", "threadgroups_per_core",
        "tile_limited_simdgroups"])
    for arm in arms:
        for width in widths:
            size = census["arms"][arm]["air"][str(width)]["threadgroup_bytes"]
            groups = TG_POOL_BYTES // size if size else None
            tg.add_data(arm, width, size, groups,
                        groups * SIMDGROUPS_PER_THREADGROUP if groups
                        else None)
    run.log({"threadgroup_memory": tg})

    summary: dict[str, object] = {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        tag = arch.replace("applegpu_", "")
        for arm in arms:
            per_arch = census["arms"][arm].get(arch, {})
            regs = [per_arch[str(w)]["registers"] for w in widths
                    if str(w) in per_arch]
            if not regs:
                continue
            summary[f"register_ceiling/{arm}_{tag}"] = max(regs)
            summary[f"simdgroups_at_ceiling/{arm}_{tag}"] = (
                REGISTER_FILE[arch] // (128 * max(regs)))
            for width in widths:
                rec = per_arch.get(str(width))
                if rec is None:
                    continue
                summary[f"registers/{arm}_{tag}_na{width}"] = rec["registers"]
                summary[f"text_bytes/{arm}_{tag}_na{width}"] = (
                    rec["text_bytes"])
                summary[f"spill_bytes/{arm}_{tag}_na{width}"] = (
                    rec["spill_bytes"])
    for arm in arms:
        for width in widths:
            rec = census["arms"][arm]["air"][str(width)]
            summary[f"device_loads/{arm}_na{width}"] = rec["device_loads"]
            summary[f"threadgroup_bytes/{arm}_na{width}"] = (
                rec["threadgroup_bytes"])
    run.summary.update(summary)
    attach(run, CENSUS)
    run.finish()


def log_timing() -> None:
    cells, arms = load_sessions()
    census = json.loads(CENSUS.read_text())
    sessions = sorted({s for s, _, _ in cells})
    shapes = sorted({s for _, s, _ in cells})
    widths = sorted({w for _, _, w in cells})
    others = [a for a in arms if a != BASE_ARM]

    metas = {s: read_meta(OUT / s / "meta.txt") for s in sessions}
    run = start(
        "e110-rung1-timing",
        "timing",
        "Which change to the k-block extraction schedule lowers the one-group "
        "wide QMV time at the realised verify-width operating point?",
        2, {"arms": arms, "arm_notes": ARM_NOTE, "sessions": sessions,
            "shapes": shapes, "widths": widths,
            "warmup_blocks_dropped": WARMUP_BLOCKS,
            "cells": len(cells), "dram_peak_gbs": DRAM_PEAK_GBS,
            "session_heads": {s: metas[s].get("git_head") for s in sessions},
            "session_entry_c": {s: metas[s].get("gpu_temp_entry_c")
                                for s in sessions},
            "session_exit_c": {s: metas[s].get("gpu_temp_exit_c")
                               for s in sessions}})

    rates = wandb.Table(columns=[
        "session", "shape", "na", "arm", "median_us", "read_bytes", "gbs",
        "pct_dram_peak", "gflops", "vs_base_pct", "gpu_temp_entry_c"])
    for (session, shape, width), cell in sorted(cells.items()):
        base = med(b[BASE_ARM] for b in cell["blocks"])
        for arm in arms:
            present = [b[arm] for b in cell["blocks"] if arm in b]
            if not present:
                continue
            t = med(present)
            rate = cell["bytes"] / t / 1e9
            rates.add_data(session, shape, width, arm, t * 1e6, cell["bytes"],
                           rate, 100.0 * rate / DRAM_PEAK_GBS,
                           cell["flops"] / t / 1e9,
                           100.0 * (t - base) / base, cell["temp"])
    run.log({"absolute_rates": rates})

    paired = wandb.Table(columns=[
        "arm", "na", "median_pct", "min_pct", "max_pct", "spread_pct",
        "blocks", "blocks_faster", "note"])
    ladders: dict[str, dict[int, float]] = {}
    for arm in others:
        per_width = paired_pct(cells, arm)
        for width in sorted(per_width):
            pct = per_width[width]
            paired.add_data(arm, width, med(pct), min(pct), max(pct),
                            max(pct) - min(pct), len(pct),
                            sum(1 for v in pct if v < 0.0),
                            ARM_NOTE.get(arm, ""))
        ladders[arm] = {w: med(v) for w, v in per_width.items()}
    run.log({"paired_effect": paired})

    headline = wandb.Table(columns=[
        "arm", "na2_pct", "na3_pct", "na4_pct", "na5_pct", "weighted_pct",
        "round_pct", "ranked_pct", "diagnostic_only", "verdict", "note"])
    verdicts: dict[str, str] = {}
    survivors: list[tuple[float, str]] = []
    for arm in others:
        ladder = ladders[arm]
        w_pct = weighted(ladder)
        round_pct = w_pct * STREAM_US / ROUND_US
        diag = arm in DIAGNOSTIC_ARMS
        if diag:
            verdict = "diagnostic"
        elif w_pct <= ADVANCE_PCT:
            verdict = "ADVANCE"
            survivors.append((w_pct, arm))
        elif w_pct <= KILL_PCT:
            verdict = "SURVIVES"
            survivors.append((w_pct, arm))
        else:
            verdict = "KILL"
        verdicts[arm] = verdict
        headline.add_data(
            arm, ladder.get(2), ladder.get(3), ladder.get(4), ladder.get(5),
            w_pct, round_pct, round_pct * RANKED_TRANSFER, diag, verdict,
            ARM_NOTE.get(arm, ""))
    run.log({"round_weighted_headline": headline})

    roofline = wandb.Table(columns=[
        "na", "load_us", "alu_us", "roofline_us", "a_base_us", "gap_pct"])
    gaps: dict[int, float] = {}
    for width in widths:
        load, alu, base = [], [], []
        for (_, _, w), cell in cells.items():
            if w != width:
                continue
            for block in cell["blocks"]:
                if ROOFLINE_LOAD not in block or ROOFLINE_ALU not in block:
                    continue
                load.append(block[ROOFLINE_LOAD])
                alu.append(block[ROOFLINE_ALU])
                base.append(block[BASE_ARM])
        if not base:
            continue
        l_us, a_us, b_us = med(load) * 1e6, med(alu) * 1e6, med(base) * 1e6
        roof = max(l_us, a_us)
        gaps[width] = 100.0 * (b_us - roof) / roof
        roofline.add_data(width, l_us, a_us, roof, b_us, gaps[width])
    run.log({"roofline_pair": roofline})

    exact = wandb.Table(columns=[
        "session", "shape", "na", "arm", "exact_required", "differing",
        "total", "bit_identical"])
    control = wandb.Table(columns=[
        "session", "shape", "na", "arm", "differing", "total", "fired"])
    checks = failures = fired = controls = 0
    for session in sessions:
        doc = json.loads((OUT / session / "rate.json").read_text())
        for row in doc["measurements"]:
            if row["kind"] == "fidelity":
                for arm in row["arms"]:
                    required = arm["arm"] not in DIAGNOSTIC_ARMS
                    exact.add_data(session, row["shape"], row["m"],
                                   arm["arm"], required, arm["differing"],
                                   arm["total"], arm["bit_identical"])
                    if required:
                        checks += 1
                        failures += 0 if arm["bit_identical"] else 1
            elif row["kind"] == "positive_control":
                controls += 1
                fired += 1 if row["detected"] else 0
                control.add_data(session, row["shape"], row["m"], row["arm"],
                                 row["differing"], row["total"],
                                 row["detected"])
    run.log({"bit_exactness": exact, "positive_controls": control})

    per_shape = wandb.Table(columns=[
        "arm", "shape", "na", "base_us", "arm_us", "change_pct", "base_gbs",
        "arm_gbs", "base_pct_dram_peak"])
    for arm in others:
        for shape in shapes:
            for width in widths:
                base_t, arm_t, nbytes = [], [], None
                for (_, s, w), cell in cells.items():
                    if s != shape or w != width:
                        continue
                    nbytes = cell["bytes"]
                    for block in cell["blocks"]:
                        if arm not in block:
                            continue
                        base_t.append(block[BASE_ARM])
                        arm_t.append(block[arm])
                if not arm_t:
                    continue
                b, a = med(base_t), med(arm_t)
                per_shape.add_data(
                    arm, shape, width, b * 1e6, a * 1e6,
                    100.0 * (a - b) / b, nbytes / b / 1e9, nbytes / a / 1e9,
                    100.0 * (nbytes / b / 1e9) / DRAM_PEAK_GBS)
    run.log({"per_shape_effect": per_shape})

    entry, exits = [], []
    for session in sessions:
        doc = json.loads((OUT / session / "rate.json").read_text())
        for row in doc["measurements"]:
            if row["kind"] == "thermal":
                entry.append(row["gpu_temp_entry_c"])
                exits.append(row["gpu_temp_exit_c"])

    survivors.sort()
    summary: dict[str, object] = {
        "sessions": len(sessions),
        "exactness_checks": checks,
        "exactness_failures": failures,
        "positive_control_cells": controls,
        "positive_control_cells_fired": fired,
        "gpu_temp_entry_min_c": min(entry), "gpu_temp_entry_max_c": max(entry),
        "gpu_temp_entry_spread_c": max(entry) - min(entry),
        "gpu_temp_exit_min_c": min(exits), "gpu_temp_exit_max_c": max(exits),
        "advanced_arm": survivors[0][1] if survivors else None,
        "advanced_weighted_pct": survivors[0][0] if survivors else None,
    }
    for arm in others:
        w_pct = weighted(ladders[arm])
        round_pct = w_pct * STREAM_US / ROUND_US
        summary[f"weighted_pct/{arm}"] = w_pct
        summary[f"round_pct/{arm}"] = round_pct
        summary[f"ranked_pct/{arm}"] = round_pct * RANKED_TRANSFER
        summary[f"verdict/{arm}"] = verdicts[arm]
        for width, value in ladders[arm].items():
            summary[f"paired_median_pct/{arm}_na{width}"] = value
    for width, gap in gaps.items():
        summary[f"roofline_gap_pct/na{width}"] = gap
    if gaps:
        summary["roofline_gap_pct_weighted"] = weighted(gaps)
    for arm in others:
        rec = census["arms"].get(arm, {}).get(RANKED_ARCH, {})
        regs = [rec[str(w)]["registers"] for w in widths if str(w) in rec]
        if regs:
            summary[f"ranked_register_ceiling/{arm}"] = max(regs)
    run.summary.update(summary)

    attach(run, CENSUS, *[OUT / s / "rate.json" for s in sessions],
           *[OUT / s / "meta.txt" for s in sessions],
           OUT / "e110/rung1-summary.json")
    run.finish()


def log_tgmem(run) -> None:
    """Driver-read threadgroup memory for each staging placement.

    The occupancy column is the reason the staging arm is dead: the tile can
    only be declared at the entry point, so every dispatch of that kernel pays
    it, including M = 1.
    """
    path = OUT / "e110/rung2-tgmem.json"
    if not path.exists():
        print(f"[wandb] no {path}; run the driver probe first")
        return
    doc = json.loads(path.read_text())
    pool = doc["max_threadgroup_memory_length"]

    table = wandb.Table(columns=[
        "variant", "compiled", "entry_point", "static_threadgroup_memory",
        "max_threads", "threads_per_simd", "threadgroups_per_pool",
        "simdgroups_per_pool", "compile_error"])
    for variant in doc["variants"]:
        if not variant["compiled"]:
            table.add_data(variant["label"], False, None, None, None, None,
                           None, None, variant.get("compile_error"))
            continue
        for entry in variant["entry_points"]:
            tg_bytes = entry["static_threadgroup_memory"]
            groups = pool // tg_bytes if tg_bytes else None
            table.add_data(
                variant["label"], True, entry["name"], tg_bytes,
                entry["max_threads"], entry.get("threads_per_simd"), groups,
                groups * SIMDGROUPS_PER_THREADGROUP if groups else None, None)
    run.log({"threadgroup_memory": table})

    def bytes_of(label: str):
        for variant in doc["variants"]:
            if variant["label"] == label and variant["compiled"]:
                return max(e["static_threadgroup_memory"]
                           for e in variant["entry_points"])
        return None

    entry_bytes = bytes_of("tile_entry")
    run.summary.update({
        "tgmem_device": doc["device"],
        "tgmem_pool_bytes": pool,
        "tgmem_shipped_bytes": bytes_of("shipped"),
        "tgmem_tile_wide_compiles": bytes_of("tile_wide") is not None,
        "tgmem_tile_entry_bytes": entry_bytes,
        "tgmem_tile_entry_simdgroup_cap":
            pool // entry_bytes * SIMDGROUPS_PER_THREADGROUP,
    })


def log_rung2() -> None:
    """The advanced arm inside the real kernel, in situ."""
    path = OUT / "e110/rung2-insitu.json"
    if not path.exists():
        print(f"[wandb] no {path}; run the in-situ leg first")
        return
    doc = json.loads(path.read_text())

    run = start(
        "e110-rung2-insitu", "timing",
        "Does the advanced arm lower absolute candidate MTP seconds per token "
        "in the real worker, with full-window exactness intact?",
        3, {"rung": 2,
            "arm": doc["arm"], "candidate_commit": doc["candidate_commit"],
            "base_commit": doc["base_commit"],
            "worker_fingerprint": doc.get("worker_fingerprint"),
            "token_window": doc.get("token_window"),
            "order": doc.get("order"), "replicates": doc.get("replicates"),
            "reproduction": doc.get("reproduction")},
        meta_dir="e110-rung2")

    legs = wandb.Table(columns=[
        "replicate", "position", "tree", "tag", "seconds_per_token",
        "serial_seconds_per_token", "local_ratio", "rounds",
        "effective_mean_draft_len", "accepted_draft_rate", "all_tokens_matched",
        "gpu_temp_entry_c", "gpu_temp_exit_c", "worker_sha256"])
    for leg in doc["legs"]:
        legs.add_data(leg["replicate"], leg["position"], leg["tree"],
                      leg.get("tag"), leg["seconds_per_token"],
                      leg.get("serial_seconds_per_token"),
                      leg.get("local_ratio"), leg.get("rounds"),
                      leg.get("effective_mean_draft_len"),
                      leg.get("accepted_draft_rate"),
                      leg.get("all_tokens_matched"),
                      leg.get("gpu_temp_entry_c"), leg.get("gpu_temp_exit_c"),
                      leg.get("worker_sha256"))
    run.log({"abba_legs": legs})

    per_replicate = wandb.Table(columns=[
        "replicate", "mtp_spt_base", "mtp_spt_xv4", "mtp_spt_pct",
        "serial_spt_base", "serial_spt_xv4", "serial_spt_pct",
        "ratio_base", "ratio_xv4", "ratio_pct", "base_pair_drift_pct"])
    for rec in doc.get("per_replicate", []):
        per_replicate.add_data(
            rec["replicate"], rec["mtp_spt_base"], rec["mtp_spt_xv4"],
            rec["mtp_spt_pct"], rec["serial_spt_base"], rec["serial_spt_xv4"],
            rec["serial_spt_pct"], rec["ratio_base"], rec["ratio_xv4"],
            rec["ratio_pct"], rec["base_pair_drift_pct"])
    run.log({"per_replicate": per_replicate})

    log_tgmem(run)

    if doc.get("exactness"):
        exact = wandb.Table(columns=[
            "check", "rows", "expected_digest", "observed_digest", "passed"])
        for rec in doc["exactness"]:
            exact.add_data(rec["check"], rec.get("rows"),
                           rec.get("expected_digest"),
                           rec.get("observed_digest"), rec["passed"])
        run.log({"exactness": exact})

    run.summary.update(doc["summary"])
    attach(run, path, OUT / "e110/rung2-exactness.json",
           OUT / "e110/rung2-tgmem.json", OUT / "e110-rung2/meta.txt")
    run.finish()


RUNS = {"census": log_census, "timing": log_timing, "rung2": log_rung2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS), nargs="*")
    args = ap.parse_args()
    for name in (args.only or list(RUNS)):
        print(f"[wandb] {name}")
        RUNS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
