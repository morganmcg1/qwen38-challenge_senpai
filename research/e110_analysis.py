#!/usr/bin/env python3
"""E110 rung 0: read the roofline triple and apply the pre-registered stop rule.

    research/e110_analysis.py research/out/e110-rung0/rate.json \
        --census research/out/e110/rung0-census.json

Every arm is timed inside one counterbalanced palindrome block, so the paired
per-block ratio is the estimator: it cancels the drift the palindrome cannot,
and its spread across blocks is the instrument's own noise at that cell.

Round frame, rule 34: the decode-only M = 5 frame is 102,864 us on the current
tree, of which the four clean streaming families are 14.4123 GB moving at a
measured 179.9 GB/s, which is 80,113 us. Nothing here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

DRAM_PEAK_GBS = 273.0
ROUND_US = 102_864.0
STREAM_GB = 14.4123
STREAM_RATE_GBS = 179.9
STREAM_US = STREAM_GB / STREAM_RATE_GBS * 1e6

ARMS = ("a_base", "l_loadonly", "z_loadxconst", "w_only", "x_only",
        "b_barrier", "xs_stage")

# The pre-registered decision. If the staging proxy does not move the isolated
# one-group NA = 5 rate by more than this, H1 is dead and the axis goes back.
STOP_RULE_NA = 5
STOP_RULE_PCT = 5.0


def collect(doc: dict, warmup_blocks: int) -> dict:
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup_blocks:
            continue
        cell = cells.setdefault((row["shape"], row["m"]), {
            "bytes": row["read_bytes"], "flops": row["flops"],
            "temp": [], "blocks": []})
        cell["temp"].append(row["gpu_temp_entry_c"])
        cell["blocks"].append(row["seconds"])
    return cells


def paired(cell: dict, arm: str, ref: str = "a_base") -> list[float]:
    """Per-block ratio arm/ref, one value per counterbalanced block."""
    return [b[arm] / b[ref] for b in cell["blocks"] if b.get(ref)]


def med(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def fidelity(doc: dict) -> tuple[list[str], list[str]]:
    failures, controls = [], []
    for row in doc["measurements"]:
        if row["kind"] == "fidelity":
            for arm in row["arms"]:
                if arm["exact_required"] and not arm["bit_identical"]:
                    failures.append("%s M=%d %s: %d/%d differ" % (
                        row["shape"], row["m"], arm["arm"], arm["differing"],
                        arm["total"]))
        elif row["kind"] == "positive_control":
            controls.append("%s M=%d %s: %d/%d differ, detected=%s" % (
                row["shape"], row["m"], row["arm"], row["differing"],
                row["total"], row["detected"]))
    return failures, controls


def thermal(doc: dict) -> None:
    entry = [r["gpu_temp_entry_c"] for r in doc["measurements"]
             if r["kind"] == "thermal"]
    exits = [r["gpu_temp_exit_c"] for r in doc["measurements"]
             if r["kind"] == "thermal"]
    if not entry:
        return
    print("=== thermal, ungated by design ===")
    print("  entry C: min %.1f max %.1f spread %.1f    exit C: min %.1f max %.1f"
          % (min(entry), max(entry), max(entry) - min(entry), min(exits),
             max(exits)))
    print("  cool_gate_passed_real_gate=false  gate_qualified_for_timing=false")
    print()


def factorial(cells: dict, shapes: list[str], widths: list[int]) -> None:
    """The 2 x 2 the four diagnostic arms happen to form.

    Rows are the activation stream, columns are the arithmetic. `a_base` and
    `w_only` carry the full 160 x NA scalar FP body; `l_loadonly` and
    `z_loadxconst` carry the 64 x NA reduced body. `a_base` and `l_loadonly`
    read the activations; `w_only` and `z_loadxconst` replace them with
    compile-time constants and read only the weight stream.

    If the activation stream cost bandwidth or cache residency, its price would
    be about the same in both columns. The interaction term measures how far
    from true that is.
    """
    print("\n=== 2 x 2: activation stream against arithmetic, pooled over "
          "shapes ===")
    print("  price of the activation loads, as %% of a_base, in each "
          "arithmetic column")
    print("  %3s %14s %14s %14s %9s" % (
        "NA", "full arith", "light arith", "interaction", "n"))
    for width in widths:
        full, light = [], []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is None:
                continue
            for block in cell["blocks"]:
                base = block["a_base"]
                full.append(100.0 * (base - block["w_only"]) / base)
                light.append(
                    100.0 * (block["l_loadonly"] - block["z_loadxconst"]) / base)
        if not full:
            continue
        f_pct, l_pct = med(full), med(light)
        print("  %3d %13.2f%% %13.2f%% %13.2f%% %9d" % (
            width, f_pct, l_pct, f_pct - l_pct, len(full)))
    print("  A cost that appears only next to the full arithmetic is not a "
          "bandwidth cost.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_json")
    ap.add_argument("--census")
    ap.add_argument("--warmup-blocks", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()

    doc = json.loads(pathlib.Path(args.rate_json).read_text())
    arms = [a for a in doc["arms"]]
    cells = collect(doc, args.warmup_blocks)
    shapes, widths = [], []
    for shape, width in cells:
        if shape not in shapes:
            shapes.append(shape)
        if width not in widths:
            widths.append(width)
    widths.sort()
    blocks = len(next(iter(cells.values()))["blocks"])

    print("device: %s (%s)" % (doc["device"], doc["architecture"]))
    print("arms: %s" % ", ".join(arms))
    print("blocks kept per cell: %d of %d, first %d discarded as cold start"
          % (blocks, doc["pairs"], args.warmup_blocks))
    print()

    failures, controls = fidelity(doc)
    print("=== exactness ===")
    for line in controls:
        print("  positive control  %s" % line)
    if failures:
        for line in failures:
            print("  *** EXACTNESS FAILURE *** %s" % line)
    else:
        print("  every arm required to be exact is bit-identical to a_base at "
              "every cell")
    print()

    thermal(doc)

    for shape in shapes:
        print("=== %s ===" % shape)
        print(" %3s %-13s %9s %9s %8s %9s" % (
            "NA", "arm", "us", "GB/s", "%peak", "vs base"))
        for width in widths:
            cell = cells.get((shape, width))
            if cell is None:
                continue
            base = med([b["a_base"] for b in cell["blocks"]])
            for arm in arms:
                t = med([b[arm] for b in cell["blocks"]])
                rate = cell["bytes"] / t / 1e9
                print(" %3d %-13s %9.1f %9.1f %7.1f%% %+8.2f%%" % (
                    width, arm, t * 1e6, rate, 100.0 * rate / DRAM_PEAK_GBS,
                    100.0 * (t - base) / base))
        print()

    print("=== paired per-block effect, the estimator that decides ===")
    print("  each block is one ABC..CBA palindrome; the ratio is formed inside "
          "the block")
    for arm in arms:
        if arm == "a_base":
            continue
        print("\n  %s against a_base" % arm)
        print("  %3s %11s %9s %9s %9s %7s" % (
            "NA", "median %", "min %", "max %", "spread", "n"))
        for width in widths:
            ratios = []
            for shape in shapes:
                cell = cells.get((shape, width))
                if cell is None:
                    continue
                ratios.extend(paired(cell, arm))
            if not ratios:
                continue
            pct = [100.0 * (r - 1.0) for r in ratios]
            print("  %3d %+11.2f %+9.2f %+9.2f %9.2f %7d" % (
                width, med(pct), min(pct), max(pct), max(pct) - min(pct),
                len(pct)))

    factorial(cells, shapes, widths)

    print("\n=== per shape at NA=%d ===" % STOP_RULE_NA)
    print("  %-32s %10s %10s %10s" % (
        "shape", "base us", "stage us", "change %"))
    for shape in shapes:
        cell = cells.get((shape, STOP_RULE_NA))
        if cell is None:
            continue
        base = med([b["a_base"] for b in cell["blocks"]])
        stage = med([b["xs_stage"] for b in cell["blocks"]])
        print("  %-32s %10.1f %10.1f %+10.2f" % (
            shape, base * 1e6, stage * 1e6, 100.0 * (stage - base) / base))

    stop_ratios = []
    for shape in shapes:
        cell = cells.get((shape, STOP_RULE_NA))
        if cell is not None:
            stop_ratios.extend(paired(cell, "xs_stage"))
    verdict = "inconclusive"
    if stop_ratios:
        effect = 100.0 * (med(stop_ratios) - 1.0)
        moved = abs(effect) > STOP_RULE_PCT
        verdict = ("H1 LIVE: proxy moves NA=%d by %+.2f%%, past the %.0f%% bar"
                   % (STOP_RULE_NA, effect, STOP_RULE_PCT)) if moved else (
                   "H1 DEAD: proxy moves NA=%d by %+.2f%%, inside the %.0f%% bar"
                   % (STOP_RULE_NA, effect, STOP_RULE_PCT))
        print("\n=== pre-registered stop rule ===")
        print("  %s" % verdict)
        print("  n=%d paired blocks across %d shapes, sign of effect: "
              "%d of %d blocks faster"
              % (len(stop_ratios), len(shapes),
                 sum(1 for r in stop_ratios if r < 1.0), len(stop_ratios)))
        if moved and effect < 0:
            saved = STREAM_US * (1.0 - 1.0 / (1.0 - effect / 100.0))
            print("  indicative round value, rule 34 frame: the four clean "
                  "streaming families are")
            print("  %.4f GB at %.1f GB/s = %.0f us of the %.0f us decode-only "
                  "M=5 round; a %+.2f%%"
                  % (STREAM_GB, STREAM_RATE_GBS, STREAM_US, ROUND_US, effect))
            print("  change on that term is %.0f us, %.2f%% of the round."
                  % (-saved, -100.0 * saved / ROUND_US))

    if args.census:
        census = json.loads(pathlib.Path(args.census).read_text())
        print("\n=== registers, spill and machine text, both architectures ===")
        for arch in ("applegpu_g16s", "applegpu_g17s"):
            print("  %s" % arch)
            for arm in arms:
                row = census["arms"].get(arm, {}).get(arch, {})
                cells_txt = []
                for width in widths:
                    value = row.get(str(width))
                    if value is None:
                        continue
                    spill = value["spill_bytes"] or 0
                    cells_txt.append("NA%d=%d%s/%dB" % (
                        width, value["registers"],
                        "s%d" % spill if spill else "", value["text_bytes"]))
                print("    %-13s %s" % (arm, "  ".join(cells_txt)))

    if args.out:
        summary = {
            "device": doc["device"], "architecture": doc["architecture"],
            "blocks_kept": blocks, "verdict": verdict,
            "stop_rule": {"na": STOP_RULE_NA, "bar_pct": STOP_RULE_PCT,
                          "median_pct": 100.0 * (med(stop_ratios) - 1.0)
                          if stop_ratios else None,
                          "n_blocks": len(stop_ratios)},
            "cells": [
                {"shape": shape, "na": width,
                 "median_us": {a: med([b[a] for b in cells[(shape, width)]
                                       ["blocks"]]) * 1e6 for a in arms},
                 "median_gbs": {a: cells[(shape, width)]["bytes"]
                                / med([b[a] for b in cells[(shape, width)]
                                       ["blocks"]]) / 1e9 for a in arms}}
                for shape in shapes for width in widths
                if (shape, width) in cells],
        }
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
