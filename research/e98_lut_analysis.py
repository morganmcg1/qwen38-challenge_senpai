#!/usr/bin/env python3
"""Aggregate the E98 rung-1b three-arm LUT probe into per-cell effects.

Reads lut.json from research/e98_qmv_ab.m and reports, for every (shape, M)
cell: the block-averaged time of each arm, the indexed saving (a - b), the
no-metadata bound (a - c), the conversion ratio (a - b) / (a - c), the achieved
read bandwidth against the DRAM floor, and the within-block same-arm null.

The stop rule needs three numbers per cell:

  (a) - (c)   what removing ALL metadata traffic is worth here. If this is far
              below the byte share it predicts, the cell is not metadata bound
              and no indexing scheme can pay there.
  (a) - (b)   what the shipped-compatible indexed read is actually worth.
  ratio       how much of the achievable saving the index captures. The pair
              index removes 2 of the 4 metadata bytes per group, so a perfectly
              bandwidth-bound cell with a free LUT gives 0.5.

  python3 research/e98_lut_analysis.py --input research/out/e98-lut-r1b/lut.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

DRAM_PEAK_GB_S = 273.0  # M4 Pro theoretical peak

# quantized.h:1929-1974 wide cross-row dispatch: inputs per weight stream.
CROSSROW_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def weight_streams(m: int) -> int:
    if m <= 2:
        return 1
    if m in CROSSROW_IPG:
        return math.ceil(m / CROSSROW_IPG[m])
    return m


def kernel_family(m: int, n: int) -> str:
    if m == 1:
        return "qmv_fast_impl"
    if n >= 4096:
        return "crossrow_m" if m >= 3 else "crossrow"
    return "crossrow"


def pct(x: float) -> str:
    return "%+7.2f%%" % (100.0 * x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="research/out/e98-lut-r1b/lut.json")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = json.load(open(args.input))
    rows = d["measurements"]
    arms = d["arms"]

    fidelity = [r for r in rows if r["kind"] == "fidelity"]
    controls = [r for r in rows if r["kind"] == "positive_control"]
    timing = [r for r in rows if r["kind"] == "timing"]

    print("=== fidelity: arm (b) must be BIT-IDENTICAL to arm (a) ===")
    print("%-28s %3s %14s %14s %10s %s" % (
        "shape", "M", "a_vs_double_max", "a_vs_double_rms_sig", "differing",
        "bit_identical"))
    all_bit_identical = True
    for r in fidelity:
        all_bit_identical &= bool(r["bit_identical"])
        print("%-28s %3d %14.3e %14.3e %10d %s" % (
            r["shape"], r["m"], r["a_vs_double_max_rel"],
            r["a_vs_double_rms_over_signal"], r["b_vs_a_differing"],
            r["bit_identical"]))

    print()
    print("=== positive control: a perturbed LUT entry MUST be detected ===")
    all_detected = True
    for r in controls:
        all_detected &= bool(r["detected"])
        print("%-28s M=%d differing=%d/%d detected=%s" % (
            r["shape"], r["m"], r["differing"], r["total"], r["detected"]))

    by = defaultdict(list)
    for r in timing:
        by[(r["shape"], r["m"])].append(r)

    print()
    print("=== timing: block-averaged, ABCCBA counterbalanced ===")
    print("%-28s %3s %-14s %2s %9s %9s %9s %9s %9s %7s %8s" % (
        "shape", "M", "kernel", "G", "a_us", "b_us", "c_us", "a-b", "a-c",
        "ratio", "null"))

    cells = []
    for (shape, m), block in sorted(by.items()):
        n = int(shape.split("_n")[-1])
        fam = kernel_family(m, n)
        g = block[0]["weight_streams"]
        a = statistics.mean(r["a_s"] for r in block)
        b = statistics.mean(r["b_s"] for r in block)
        c = statistics.mean(r["c_s"] for r in block)
        # slots 0 and 5 are both arm (a); their gap is the within-block drift
        # this design cancels, reported so the reader can size it.
        nulls = [(r["slots"][5] - r["slots"][0]) / r["slots"][0]
                 for r in block]
        null = statistics.mean(nulls)
        d_ab = (a - b) / a
        d_ac = (a - c) / a
        ratio = (a - b) / (a - c) if (a - c) != 0.0 else float("nan")

        rb = {"a": block[0]["a_read_bytes"], "b": block[0]["b_read_bytes"],
              "c": block[0]["c_read_bytes"]}
        rate = {k: rb[k] / t / 1e9 for k, t in (("a", a), ("b", b), ("c", c))}

        print("%-28s %3d %-14s %2d %9.2f %9.2f %9.2f %9s %9s %7.3f %7s" % (
            shape, m, fam, g, a * 1e6, b * 1e6, c * 1e6, pct(d_ab), pct(d_ac),
            ratio, pct(null)))

        cells.append({
            "shape": shape, "m": m, "kernel_family": fam, "weight_streams": g,
            "blocks": len(block),
            "a_s": a, "b_s": b, "c_s": c,
            "a_read_bytes": rb["a"], "b_read_bytes": rb["b"],
            "c_read_bytes": rb["c"],
            "a_gb_s": rate["a"], "b_gb_s": rate["b"], "c_gb_s": rate["c"],
            "a_util": rate["a"] / DRAM_PEAK_GB_S,
            "b_util": rate["b"] / DRAM_PEAK_GB_S,
            "c_util": rate["c"] / DRAM_PEAK_GB_S,
            "indexed_saving": d_ab,
            "no_metadata_bound": d_ac,
            "capture_ratio": ratio,
            "byte_share_ab": 1.0 - rb["b"] / rb["a"],
            "byte_share_ac": 1.0 - rb["c"] / rb["a"],
            "session_null": null,
            "block_spread_a": (max(r["a_s"] for r in block)
                               - min(r["a_s"] for r in block)) / a,
        })

    print()
    print("=== bandwidth and byte accounting ===")
    print("%-28s %3s %8s %8s %8s %9s %9s %9s" % (
        "shape", "M", "a_GB/s", "b_GB/s", "c_GB/s", "a_util", "share_ab",
        "share_ac"))
    for c in cells:
        print("%-28s %3d %8.1f %8.1f %8.1f %8.1f%% %8s %8s" % (
            c["shape"], c["m"], c["a_gb_s"], c["b_gb_s"], c["c_gb_s"],
            100.0 * c["a_util"], pct(c["byte_share_ab"]),
            pct(c["byte_share_ac"])))
    below = [c for c in cells if c["a_gb_s"] > DRAM_PEAK_GB_S]
    if below:
        print()
        print("NOTE: %d cell(s) exceed the %.0f GB/s DRAM peak on logical "
              "bytes, so those bytes are partly cache-served and overstate "
              "DRAM traffic:" % (len(below), DRAM_PEAK_GB_S))
        for c in below:
            print("  %-28s M=%d  %.1f GB/s" % (c["shape"], c["m"],
                                               c["a_gb_s"]))

    summary = {
        "arms": arms,
        "lut_entries": d["lut_entries"],
        "device": d["device"],
        "architecture": d["architecture"],
        "bit_identical_all": all_bit_identical,
        "positive_control_all_detected": all_detected,
        "cells": cells,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print("\nwrote %s" % args.json_out)

    print()
    print("bit_identical_all=%s positive_control_all_detected=%s"
          % (all_bit_identical, all_detected))
    return 0 if (all_bit_identical and all_detected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
