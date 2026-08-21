#!/usr/bin/env python3
"""Publish the E110 activation-re-read experiment to W&B.

    usage: research/e110_wandb_log.py [--only RUN ...]

Two runs:

  `e110-dce-control`
      The mandatory first item. Does the compiled output of E104's
      `l_loadonly` arm still contain its activation loads, or did the
      optimizer delete them? Answered from AIR device-load counts after
      `-O2` and from machine text on both GPU generations, not from source.
  `e110-rung0-roofline`
      Rung 0. The isolated one-group roofline sweep over five scored shapes
      and NA = 2..6, the bit-exactness matrix, the 2 x 2 that separates the
      activation stream from the arithmetic it sits next to, and the
      pre-registered stop rule for H1.

Every timed leg ran with no thermal gate, so `timing_valid`,
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

# The pre-registered decision, copied from the assignment.
STOP_RULE_NA = 5
STOP_RULE_PCT = 5.0
STOP_RULE_ARM = "xs_stage"

# What each arm removes relative to `a_base`, for the reader of the table.
ARM_NOTE = {
    "a_base": "shipped one-group body, unmodified",
    "l_loadonly": "keeps every activation load, drops the 160xNA scalar body",
    "z_loadxconst": "l_loadonly with the activations replaced by constants",
    "w_only": "full arithmetic, activations replaced by constants",
    "x_only": "activation stream only, weight stream removed",
    "b_barrier": "a_base plus one threadgroup barrier, price of the barrier",
    "xs_stage": "a_base with the activations staged once in threadgroup memory",
}
EXACT_REQUIRED = ("b_barrier", "xs_stage")


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
          meta_dir: str = "e110-rung0"):
    meta = read_meta(OUT / meta_dir / "meta.txt")
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={
            "experiment": GROUP, "rung": 0, "step": step,
            "question": question, "entry_cell": ENTRY_CELL,
            "host": HOST, "hostname": meta.get("host"),
            "chip": meta.get("chip"), "memory_gib": meta.get("memory_gib"),
            "toolchain": meta.get("toolchain"),
            "metal_toolchain": meta.get("metal_toolchain"),
            "git_head": meta.get("git_head"),
            "base_sha": BASE_SHA, "advisor_branch": ADVISOR_BRANCH,
            "local_arch": LOCAL_ARCH, "ranked_arch": RANKED_ARCH,
            "shipped_source_touched": False,
            "round_frame_us": ROUND_US,
            "streaming_term_us": STREAM_US,
            "promotion_bar_pct_of_round": PROMOTION_BAR_PCT,
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


def load_census() -> dict:
    return json.loads((OUT / "e110/rung0-census.json").read_text())


def load_rate() -> dict:
    return json.loads((OUT / "e110-rung0/rate.json").read_text())


def cells_of(doc: dict) -> dict:
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        cell = cells.setdefault((row["shape"], row["m"]), {
            "bytes": row["read_bytes"], "flops": row["flops"],
            "temp": row["gpu_temp_entry_c"], "blocks": []})
        cell["blocks"].append(row["seconds"])
    return cells


def med(values) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def log_dce_control() -> None:
    """The advisor's mandatory first item, answered from compiled output."""
    census = load_census()
    widths = [int(w) for w in census["widths"]]
    arms = list(census["arms"])

    run = start(
        "e110-dce-control", "census",
        "Did the optimizer delete the activation loads from E104's "
        "`l_loadonly` arm, or is that arm valid evidence?",
        1, {"arms": arms, "widths": widths,
            "instrument": "AIR device-load count after -O2, plus metal-tt "
                          "machine text on both architectures",
            "isa_disassembly_available": False,
            "isa_note": "metal-objdump on this toolchain has no AGX "
                        "instruction printer, so the load census is read "
                        "from post-optimization AIR and corroborated by "
                        "machine text size and register counts"})

    air = wandb.Table(columns=[
        "arm", "na", "device_loads", "threadgroup_loads",
        "threadgroup_stores", "air_lines", "keeps_activation_loads", "note"])
    base_loads = {w: census["arms"][BASE_ARM]["air"][str(w)]["device_loads"]
                  for w in widths}
    for arm in arms:
        for width in widths:
            rec = census["arms"][arm]["air"][str(width)]
            air.add_data(arm, width, rec["device_loads"],
                         rec["threadgroup_loads"], rec["threadgroup_stores"],
                         rec["air_lines"],
                         rec["device_loads"] >= base_loads[width],
                         ARM_NOTE.get(arm, ""))
    run.log({"air_load_census": air})

    text = wandb.Table(columns=["arm", "arch", "na", "registers",
                                "spill_bytes", "text_bytes", "text_sha8"])
    for arm in arms:
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for width in widths:
                rec = census["arms"][arm].get(arch, {}).get(str(width))
                if rec is None:
                    continue
                text.add_data(arm, arch, width, rec["registers"],
                              rec["spill_bytes"], rec["text_bytes"],
                              rec.get("text_sha8"))
    run.log({"machine_text_census": text})

    summary: dict[str, object] = {}
    kept = all(census["arms"]["l_loadonly"]["air"][str(w)]["device_loads"]
               == base_loads[w] for w in widths)
    summary["dce_loads_survived_in_l_loadonly"] = kept
    summary["e104_l_loadonly_is_valid_evidence"] = kept
    for width in widths:
        for arm in arms:
            rec = census["arms"][arm]["air"][str(width)]
            summary[f"device_loads/{arm}_na{width}"] = rec["device_loads"]
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        tag = arch.replace("applegpu_", "")
        for arm in arms:
            rec = census["arms"][arm].get(arch, {}).get(str(STOP_RULE_NA))
            if rec is None:
                continue
            summary[f"registers_na5/{arm}_{tag}"] = rec["registers"]
            summary[f"text_bytes_na5/{arm}_{tag}"] = rec["text_bytes"]
            summary[f"spill_bytes_na5/{arm}_{tag}"] = rec["spill_bytes"]

    # H2, occupancy, was pre-registered as near-refuted. The register counts
    # settle it: the arm that is 29 % faster allocates the same registers.
    base_reg = census["arms"][BASE_ARM][LOCAL_ARCH][str(STOP_RULE_NA)]["registers"]
    load_reg = census["arms"]["l_loadonly"][LOCAL_ARCH][str(STOP_RULE_NA)]["registers"]
    summary["h2_occupancy_registers_equal_at_na5"] = base_reg == load_reg
    summary["h2_occupancy_refuted"] = base_reg == load_reg

    # The ranked architecture allocates registers per entry point, so the
    # staging rider only pays if it lowers the ceiling over every live width.
    ranked = census["arms"]["xs_stage"][RANKED_ARCH]
    base_ranked = census["arms"][BASE_ARM][RANKED_ARCH]
    summary["ranked_registers_base"] = [base_ranked[str(w)]["registers"]
                                        for w in widths]
    summary["ranked_registers_xs_stage"] = [ranked[str(w)]["registers"]
                                            for w in widths]
    summary["ranked_register_ceiling_base"] = max(
        base_ranked[str(w)]["registers"] for w in widths)
    summary["ranked_register_ceiling_xs_stage"] = max(
        ranked[str(w)]["registers"] for w in widths)
    run.summary.update(summary)
    attach(run, OUT / "e110/rung0-census.json")
    run.finish()


def log_rung0() -> None:
    doc = load_rate()
    census = load_census()
    meta = read_meta(OUT / "e110-rung0/meta.txt")
    arms = list(doc["arms"])
    others = [a for a in arms if a != BASE_ARM]
    cells = cells_of(doc)
    shapes = sorted({s for s, _ in cells})
    widths = sorted({w for _, w in cells})
    blocks = len(next(iter(cells.values()))["blocks"])

    run = start(
        "e110-rung0-roofline", "timing",
        "Does the `_wide` activation re-read explain why one x-group runs at "
        "66 % of the 273 GB/s DRAM peak?",
        2, {"arms": arms, "arm_notes": ARM_NOTE,
            "shapes": shapes, "widths": widths,
            "blocks_per_cell": doc["pairs"],
            "warmup_blocks_dropped": WARMUP_BLOCKS,
            "blocks_kept_per_cell": blocks,
            "order": doc["order"], "cells": len(cells),
            "device": doc["device"], "architecture": doc["architecture"],
            "dram_peak_gbs": DRAM_PEAK_GBS,
            "stop_rule_na": STOP_RULE_NA,
            "stop_rule_bar_pct": STOP_RULE_PCT,
            "stop_rule_arm": STOP_RULE_ARM,
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c")})

    # Roofline pair, rule 36b: absolute rate and the paired ratio side by side.
    roofline = wandb.Table(columns=[
        "shape", "na", "arm", "median_us", "read_bytes", "gbs",
        "pct_dram_peak", "flops", "gflops", "vs_base_pct",
        "gpu_temp_entry_c"])
    for (shape, width), cell in sorted(cells.items()):
        base = med(b[BASE_ARM] for b in cell["blocks"])
        for arm in arms:
            t = med(b[arm] for b in cell["blocks"])
            rate = cell["bytes"] / t / 1e9
            roofline.add_data(
                shape, width, arm, t * 1e6, cell["bytes"], rate,
                100.0 * rate / DRAM_PEAK_GBS, cell["flops"],
                cell["flops"] / t / 1e9, 100.0 * (t - base) / base,
                cell["temp"])
    run.log({"roofline": roofline})

    # The estimator that decides: the ratio is formed inside one palindrome
    # block, so the drift the palindrome cannot cancel drops out.
    paired = wandb.Table(columns=[
        "arm", "na", "median_pct", "min_pct", "max_pct", "spread_pct",
        "blocks", "blocks_faster", "note"])
    pooled: dict[tuple[str, int], list[float]] = {}
    for arm in others:
        for width in widths:
            ratios = [b[arm] / b[BASE_ARM]
                      for shape in shapes
                      if (shape, width) in cells
                      for b in cells[(shape, width)]["blocks"]]
            pct = [100.0 * (r - 1.0) for r in ratios]
            pooled[(arm, width)] = pct
            paired.add_data(arm, width, med(pct), min(pct), max(pct),
                            max(pct) - min(pct), len(pct),
                            sum(1 for v in pct if v < 0.0),
                            ARM_NOTE.get(arm, ""))
    run.log({"paired_effect": paired})

    # The 2 x 2. Rows are the activation stream, columns are the arithmetic.
    # A bandwidth or residency cost would price about the same in both
    # columns; the interaction term measures how far from true that is.
    factorial = wandb.Table(columns=[
        "na", "full_arith_pct", "light_arith_pct", "interaction_pct",
        "interaction_share_of_full_pct", "blocks", "contaminated",
        "contamination_note"])
    spill_na = {}
    for width in widths:
        spill = census["arms"]["w_only"][LOCAL_ARCH].get(str(width), {})
        spill_na[width] = spill.get("spill_bytes", 0) or 0
    for width in widths:
        full, light = [], []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is None:
                continue
            for block in cell["blocks"]:
                base = block[BASE_ARM]
                full.append(100.0 * (base - block["w_only"]) / base)
                light.append(
                    100.0 * (block["l_loadonly"] - block["z_loadxconst"])
                    / base)
        f_pct, l_pct = med(full), med(light)
        bad = spill_na[width] > 0
        factorial.add_data(
            width, f_pct, l_pct, f_pct - l_pct,
            100.0 * (f_pct - l_pct) / f_pct if f_pct else float("nan"),
            len(full), bad,
            f"w_only spills {spill_na[width]} B on {LOCAL_ARCH}" if bad
            else "")
    run.log({"activation_arithmetic_factorial": factorial})

    # Exactness. Only the two arms that must not change a bit are graded.
    exact = wandb.Table(columns=["shape", "na", "arm", "exact_required",
                                 "differing", "total", "bit_identical"])
    checks = failures = 0
    for row in doc["measurements"]:
        if row["kind"] != "fidelity":
            continue
        for arm in row["arms"]:
            required = arm["arm"] in EXACT_REQUIRED
            exact.add_data(row["shape"], row["m"], arm["arm"], required,
                           arm["differing"], arm["total"],
                           arm["bit_identical"])
            if required:
                checks += 1
                failures += 0 if arm["bit_identical"] else 1
    run.log({"bit_exactness": exact})

    control = wandb.Table(columns=["shape", "na", "arm", "differing", "total",
                                   "fired"])
    fired = total_controls = 0
    for row in doc["measurements"]:
        if row["kind"] != "positive_control":
            continue
        total_controls += 1
        fired += 1 if row["detected"] else 0
        control.add_data(row["shape"], row["m"], row["arm"], row["differing"],
                         row["total"], row["detected"])
    run.log({"positive_controls": control})

    per_shape = wandb.Table(columns=[
        "shape", "k", "n", "base_us", "stage_us", "change_pct",
        "base_gbs", "base_pct_dram_peak"])
    for shape in shapes:
        cell = cells.get((shape, STOP_RULE_NA))
        if cell is None:
            continue
        base = med(b[BASE_ARM] for b in cell["blocks"])
        stage = med(b[STOP_RULE_ARM] for b in cell["blocks"])
        parts = shape.split("_")
        k = next((int(p[1:]) for p in parts if p.startswith("k")
                  and p[1:].isdigit()), None)
        n = next((int(p[1:]) for p in parts if p.startswith("n")
                  and p[1:].isdigit()), None)
        rate = cell["bytes"] / base / 1e9
        per_shape.add_data(shape, k, n, base * 1e6, stage * 1e6,
                           100.0 * (stage - base) / base, rate,
                           100.0 * rate / DRAM_PEAK_GBS)
    run.log({"stop_rule_per_shape": per_shape})

    thermal = [r for r in doc["measurements"] if r["kind"] == "thermal"]
    entry = [r["gpu_temp_entry_c"] for r in thermal]
    exits = [r["gpu_temp_exit_c"] for r in thermal]

    effect = med(pooled[(STOP_RULE_ARM, STOP_RULE_NA)])
    n_blocks = len(pooled[(STOP_RULE_ARM, STOP_RULE_NA)])
    moved = abs(effect) > STOP_RULE_PCT
    saved_us = STREAM_US * (1.0 - 1.0 / (1.0 - effect / 100.0))

    summary: dict[str, object] = {
        "stop_rule_effect_pct": effect,
        "stop_rule_blocks": n_blocks,
        "stop_rule_blocks_faster": sum(
            1 for v in pooled[(STOP_RULE_ARM, STOP_RULE_NA)] if v < 0.0),
        "stop_rule_bar_pct": STOP_RULE_PCT,
        "stop_rule_cleared": moved,
        "h1_activation_reread_supported": moved,
        "h1_verdict": "H1 LIVE" if moved else "H1 DEAD",
        "rung1_authorised": moved,
        "exactness_checks": checks,
        "exactness_failures": failures,
        "positive_control_cells": total_controls,
        "positive_control_cells_fired": fired,
        "gpu_temp_entry_min_c": min(entry), "gpu_temp_entry_max_c": max(entry),
        "gpu_temp_entry_spread_c": max(entry) - min(entry),
        "gpu_temp_exit_min_c": min(exits), "gpu_temp_exit_max_c": max(exits),
        # Indicative only: the staging proxy applied to the rule 34 streaming
        # term, not a score and not a promotion claim. Negative is faster.
        "indicative_streaming_change_us": -saved_us,
        "indicative_round_change_pct": -100.0 * saved_us / ROUND_US,
        "indicative_over_promotion_bar": (
            saved_us > 0.0 and 100.0 * saved_us / ROUND_US
            > PROMOTION_BAR_PCT),
    }
    for (arm, width), pct in pooled.items():
        summary[f"paired_median_pct/{arm}_na{width}"] = med(pct)
    for shape in shapes:
        cell = cells.get((shape, STOP_RULE_NA))
        if cell is None:
            continue
        base = med(b[BASE_ARM] for b in cell["blocks"])
        stage = med(b[STOP_RULE_ARM] for b in cell["blocks"])
        summary[f"stage_pct_na5/{shape}"] = 100.0 * (stage - base) / base
    run.summary.update(summary)

    attach(run, OUT / "e110-rung0/rate.json", OUT / "e110-rung0/meta.txt",
           OUT / "e110/rung0-summary.json", OUT / "e110/rung0-census.json")
    run.finish()


RUNS = {"dce": log_dce_control, "rung0": log_rung0}


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
