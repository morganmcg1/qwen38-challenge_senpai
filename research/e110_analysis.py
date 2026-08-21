#!/usr/bin/env python3
"""E110 rung 1: read the arms and apply the round-weighted kill and advance bars.

    research/e110_analysis.py research/out/e110-rung1/rate.json \
        --census research/out/e110/rung1-census.json \
        --probe-log research/out/e110-rung1/probe.log

Every arm is timed inside one counterbalanced palindrome block, so the paired
per-block ratio is the estimator: it cancels the drift the palindrome cannot,
and its spread across blocks is the instrument's own noise at that cell.

Round frame, rule 34: the decode-only M = 5 frame is 102,864 us on the current
tree, of which the four clean streaming families are 14.4123 GB moving at a
measured 179.9 GB/s, which is 80,113 us. Nothing here is a score.

The headline is round-weighted, not NA = 5. The realised verify-width histogram
of this fixture puts 66.7 % of the streaming time at NA = 4 and only 3.4 % at
NA = 5, so an arm is priced by the weighted sum over its NA ladder.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

DRAM_PEAK_GBS = 273.0
ROUND_US = 102_864.0
STREAM_GB = 14.4123
STREAM_RATE_GBS = 179.9
STREAM_US = STREAM_GB / STREAM_RATE_GBS * 1e6

# Share of the streaming term spent in a one-group pass of each width, from the
# realised width histogram of this fixture weighted by 1 / rate(NA).
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# Rule 34 transfer from a local streaming-term change to a ranked one.
RANKED_TRANSFER = 0.95

# Rung 1 decision bars, on the round-weighted isolated gain.
KILL_PCT = -0.30
ADVANCE_PCT = -0.50

# Rule 36b: the extraction schedule changes, so text bytes are a screen only and
# a measured roofline pair is mandatory. These two arms are its halves.
ROOFLINE_LOAD = "l_loadonly"
ROOFLINE_ALU = "b_constw"

# Apple's threadgroup memory limit, which is also the pool the resident
# threadgroups of a core share.
TG_POOL_BYTES = 32768


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


def roofline(cells: dict, shapes: list[str], widths: list[int]) -> None:
    """Rule 36b: does `max(ALU, load)` predict the shipped kernel's time?

    `l_loadonly` keeps every device load and drops three quarters of the
    arithmetic. `b_constw` keeps every arithmetic operation and drops every
    operand load. If the kernel were a clean roofline, `a_base` would cost
    `max` of the two. The gap is the part neither stream explains on its own,
    and it is the part a schedule change can address.
    """
    print("\n=== rule 36b roofline pair: max(ALU, load) against a_base ===")
    print("  %3s %11s %11s %11s %11s %9s" % (
        "NA", "load us", "ALU us", "max us", "a_base us", "gap %"))
    for width in widths:
        load, alu, base = [], [], []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is None:
                continue
            for block in cell["blocks"]:
                load.append(block[ROOFLINE_LOAD])
                alu.append(block[ROOFLINE_ALU])
                base.append(block["a_base"])
        if not base:
            continue
        l_us, a_us, b_us = med(load) * 1e6, med(alu) * 1e6, med(base) * 1e6
        roof = max(l_us, a_us)
        print("  %3d %11.1f %11.1f %11.1f %11.1f %+8.1f%%" % (
            width, l_us, a_us, roof, b_us, 100.0 * (b_us - roof) / roof))
    print("  A positive gap is time that neither the load stream nor the "
          "arithmetic explains alone.")


def weighted(per_width: dict[int, float]) -> float | None:
    """Round-weighted percent change over the realised width histogram."""
    total = sum(ROUND_WEIGHTS[w] for w in per_width if w in ROUND_WEIGHTS)
    if not total:
        return None
    return sum(ROUND_WEIGHTS[w] * v for w, v in per_width.items()
               if w in ROUND_WEIGHTS) / total


def ladder(cells: dict, shapes: list[str], widths: list[int],
           arm: str) -> dict[int, float]:
    """Paired median percent change against a_base, one value per width."""
    out = {}
    for width in widths:
        ratios = []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is not None:
                ratios.extend(paired(cell, arm))
        if ratios:
            out[width] = 100.0 * (med(ratios) - 1.0)
    return out


def threadgroup_memory(probe_log: pathlib.Path) -> dict[tuple[str, int], int]:
    """Compiled threadgroup memory per arm and width, from the device itself.

    The probe prints `staticThreadgroupMemoryLength` for every pipeline it
    builds, which is what the driver actually reserves, not what one case
    declares.
    """
    line_re = re.compile(
        r"e110_rate_probe:\s+(\S+) e110_iso_na(\d+)\s+max_threads=(\d+)\s+"
        r"tg_mem=(\d+)")
    found = {}
    for line in probe_log.read_text(errors="replace").splitlines():
        hit = line_re.search(line)
        if hit:
            found[(hit.group(1), int(hit.group(2)))] = int(hit.group(4))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_json")
    ap.add_argument("--census")
    ap.add_argument("--probe-log")
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

    roofline(cells, shapes, widths)

    print("\n=== round-weighted headline, rule 34 frame ===")
    print("  weights over the realised width histogram: %s"
          % "  ".join("NA%d=%.3f" % (w, ROUND_WEIGHTS[w])
                      for w in sorted(ROUND_WEIGHTS)))
    print("  %-11s %s %11s %11s %11s" % (
        "arm", " ".join("%8s" % ("NA%d %%" % w) for w in widths),
        "weighted %", "round %", "ranked %"))
    ladders, headline = {}, {}
    for arm in arms:
        if arm == "a_base":
            continue
        ladders[arm] = ladder(cells, shapes, widths, arm)
        value = weighted(ladders[arm])
        headline[arm] = value
        if value is None:
            continue
        round_pct = value * STREAM_US / ROUND_US
        print("  %-11s %s %+10.3f %+10.3f %+10.3f" % (
            arm, " ".join("%+8.2f" % ladders[arm].get(w, float("nan"))
                          for w in widths),
            value, round_pct, round_pct * RANKED_TRANSFER))
    print("  round %% is the weighted change applied to the %.0f us streaming "
          "term of the %.0f us round." % (STREAM_US, ROUND_US))

    print("\n=== rung 1 decision, pre-registered ===")
    print("  KILL above %+.2f%% weighted, ADVANCE at %+.2f%% or better"
          % (KILL_PCT, ADVANCE_PCT))
    exact_arms = [a for a in arms
                  if a not in (ROOFLINE_LOAD, ROOFLINE_ALU, "a_base")]
    survivors = []
    for arm in exact_arms:
        value = headline.get(arm)
        if value is None:
            continue
        state = ("ADVANCE" if value <= ADVANCE_PCT
                 else "SURVIVES" if value <= KILL_PCT else "KILL")
        if value <= KILL_PCT:
            survivors.append((value, arm))
        print("  %-11s %+8.3f%% weighted   %s" % (arm, value, state))
    survivors.sort()
    verdict = "no arm clears the kill bar"
    if survivors:
        best_value, best_arm = survivors[0]
        verdict = "best surviving arm %s at %+.3f%% weighted" % (
            best_arm, best_value)
        singles = [(v, a) for v, a in survivors if a != "mo_stage"]
        if singles and best_arm == "mo_stage":
            single_value, single_arm = singles[0]
            if single_value <= best_value + 0.05:
                verdict += ("; the combination is no better than %s, so ship "
                            "the single arm" % single_arm)
    print("  %s" % verdict)

    if args.probe_log:
        tg = threadgroup_memory(pathlib.Path(args.probe_log))
        if tg:
            print("\n=== compiled threadgroup memory, from the driver ===")
            print("  a threadgroup is 64 threads = 2 simdgroups; the cap is "
                  "floor(%d / tile)" % TG_POOL_BYTES)
            for arm in arms:
                cells_txt = []
                for width in widths:
                    value = tg.get((arm, width))
                    if value is None:
                        continue
                    cells_txt.append("NA%d=%dB%s" % (
                        width, value,
                        "/%dsg" % (2 * (TG_POOL_BYTES // value))
                        if value else ""))
                print("  %-11s %s" % (arm, "  ".join(cells_txt)))

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
            "round_weights": ROUND_WEIGHTS,
            "bars": {"kill_pct": KILL_PCT, "advance_pct": ADVANCE_PCT},
            "ladder_pct": ladders,
            "weighted_pct": headline,
            "round_pct": {a: (v * STREAM_US / ROUND_US if v is not None
                              else None) for a, v in headline.items()},
            "ranked_pct": {a: (v * STREAM_US / ROUND_US * RANKED_TRANSFER
                               if v is not None else None)
                           for a, v in headline.items()},
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
