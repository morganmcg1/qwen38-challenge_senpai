#!/usr/bin/env python3
"""E223 stage 1: measure the float32 activation arm against its desk price.

Reads the `xdtype` phase of an E219 session. Each (cell, NA, G, thermal) cell
holds two arms that differ in ONE line group:

  xbf16  the shipped gather, bfloat16 activations, the anchor
  xf32   the same gather with float32 activations in the shipped [m, k] layout

`activationDTypeExactness` proves the two arms return identical bits, so the
difference is pure cost. The arm removes 12.6 of the 123.2 machine
instructions per column per k-block on applegpu_g17s and doubles the issued
activation bytes.

The measured delta is the IN-KERNEL net: relief minus the extra-byte cost. It
excludes the fill pass a shipped version would need, because the timed arm
reads a float32 slab that already exists. The desk `cost_ms_per_round`
INCLUDES that fill floor, so this script removes the fill term before it
compares the two, and then adds it back for the shipped-net figure.

Stage-0 desk range for xf32flat was -3.72 to +2.00 ms/round pooled. The
op-proportional reading is an upper bound; an issue-limited reading predicts
exactly zero relief. This measurement separates them.

NOT gate-qualified. harness=local-microbench.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e219_analyze import chain_slopes  # noqa: E402

STOP_BAR_MS = 0.3
ADVANCE_BAR_MS = 1.0
RELIEF_CEILING_MS = 23.066
# Stage-0 census, applegpu_g17s, fully unrolled regime (NA 2..5).
INSN_PER_COLUMN_BF16 = 123.2
INSN_PER_COLUMN_F32 = 110.6
# `mlp.down` is reported per cell and never pooled: it is the one cell whose
# activation slab is 17408 wide, so pooling it would hide the effect.
POOLED_CELLS = ("gdn.in_proj", "mlp.gate_up")
ROWS_PER_SIMD = 4
SIMD_WIDTH = 32
BLOCK_SIZE = 512


def lane_k_blocks(k: int, n: int, groups: int) -> int:
    """Lane-k-block products one dispatch executes.

    One simdgroup owns `ROWS_PER_SIMD` output rows, so covering `n` rows takes
    `n / ROWS_PER_SIMD` simdgroups of `SIMD_WIDTH` lanes, and each lane walks
    `k / BLOCK_SIZE` k-blocks. Every group pass repeats that work.
    """
    return (n // ROWS_PER_SIMD) * SIMD_WIDTH * (k // BLOCK_SIZE) * groups


def overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def desk_key(cell: str, groups: int, cold: bool, na: int) -> str:
    return "%s/g%d/%s/na%d" % (cell, groups, "cold" if cold else "hot", na)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xdtype_json", type=pathlib.Path)
    ap.add_argument(
        "--desk", type=pathlib.Path,
        default=pathlib.Path("research/e223-artifacts/e223-desk-pricing.json"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    blob = json.loads(args.xdtype_json.read_text())
    if blob.get("phase") != "xdtype":
        raise SystemExit("not an E223 xdtype phase report: %s"
                         % args.xdtype_json)
    units = chain_slopes(blob)
    desk = json.loads(args.desk.read_text())["pricing"]["xf32flat"]["arms"]

    grid: dict[tuple, dict[str, dict]] = {}
    for unit in units.values():
        f = unit["fields"]
        tag = "xf32" if f["activation_dtype"] == "float32" else "xbf16"
        grid.setdefault(
            (f["cell"], f["na"], f["groups"], f["cold"]), {})[tag] = unit

    arms = []
    for cell_key in sorted(grid):
        pair = grid[cell_key]
        if "xbf16" not in pair or "xf32" not in pair:
            continue
        cell, na, groups, cold = cell_key
        ref, got = pair["xbf16"], pair["xf32"]
        f = ref["fields"]
        inv = f["invocations_per_round"]
        k, n = f["k"], f["n"]

        # Positive delta = the float32 arm is SLOWER. Relief is its negative.
        d_us = got["slope_us"] - ref["slope_us"]
        relief_ms = -d_us * inv / 1000.0
        ci_half_ms = 0.5 * (
            (got["slope_ci95"][1] - got["slope_ci95"][0])
            + (ref["slope_ci95"][1] - ref["slope_ci95"][0])
        ) * inv / 1000.0

        lkb = lane_k_blocks(k, n, groups)
        removed_insn = (INSN_PER_COLUMN_BF16 - INSN_PER_COLUMN_F32) * na * lkb
        extra_mb = (
            got["fields"]["activation_read_bytes"]
            - ref["fields"]["activation_read_bytes"]
        ) * groups / 1048576.0

        row = {
            "cell": cell, "na": na, "groups": groups, "cold": cold,
            "m": f["m"], "k": k, "n": n,
            "invocations_per_round": inv,
            "pooled_cell": cell in POOLED_CELLS,
            "lane_k_blocks_per_dispatch": lkb,
            "removed_instructions_per_invocation": removed_insn,
            "extra_issued_activation_mb_per_invocation": extra_mb,
            "xbf16_us_per_invocation": ref["slope_us"],
            "xbf16_ci95_us": ref["slope_ci95"],
            "xbf16_ms_per_round": ref["slope_us"] * inv / 1000.0,
            "xf32_us_per_invocation": got["slope_us"],
            "xf32_ci95_us": got["slope_ci95"],
            "xf32_ms_per_round": got["slope_us"] * inv / 1000.0,
            "measured_in_kernel_relief_ms_per_round": relief_ms,
            "measured_ci95_half_width_ms_per_round": ci_half_ms,
            "ci95_overlaps_xbf16": overlaps(
                got["slope_ci95"], ref["slope_ci95"]),
            "ratio_xf32_over_xbf16": (
                got["slope_us"] / ref["slope_us"] if ref["slope_us"] else None),
        }

        # If the whole measured relief were instruction relief, at what rate
        # would the removed instructions retire? A rate far above the machine's
        # plausible issue rate falsifies the op-proportional reading.
        if relief_ms > 0:
            row["implied_removed_ops_per_second_if_pure_relief"] = (
                removed_insn / (relief_ms * 1000.0 / inv * 1e-6))
        # If the arm is slower, all of the delta is at most the byte price;
        # relief pushes the other way, so this is an UPPER bound.
        if relief_ms < 0 and extra_mb:
            row["measured_us_per_issued_mb_upper_bound"] = (
                -relief_ms * 1000.0 / inv / extra_mb)

        entry = desk.get(desk_key(cell, groups, cold, na))
        if entry is not None:
            fill_ms = (
                entry["slab_fill_us_floor_per_invocation"] * inv / 1000.0)
            in_kernel_cost = entry["cost_ms_per_round"] - fill_ms
            row["desk"] = {
                "gross_relief_ms_per_round": entry["relief_ms_per_round"],
                "byte_cost_ms_per_round": in_kernel_cost,
                "slab_fill_ms_per_round": fill_ms,
                "predicted_in_kernel_net_ms_per_round":
                    entry["relief_ms_per_round"] - in_kernel_cost,
                "predicted_shipped_net_ms_per_round":
                    entry["net_ms_per_round"],
                "us_per_issued_mb": entry["us_per_issued_mb"],
            }
            row["measured_shipped_net_ms_per_round"] = relief_ms - fill_ms
            row["desk_prediction_error_ms_per_round"] = (
                relief_ms
                - (entry["relief_ms_per_round"] - in_kernel_cost))
        arms.append(row)

    pooled = {}
    for na in sorted({a["na"] for a in arms}):
        for groups in sorted({a["groups"] for a in arms}):
            for cold in (True, False):
                sel = [a for a in arms
                       if a["na"] == na and a["groups"] == groups
                       and a["cold"] == cold and a["pooled_cell"]]
                if len(sel) != len(POOLED_CELLS):
                    continue
                relief = sum(
                    a["measured_in_kernel_relief_ms_per_round"] for a in sel)
                # Independent cells, so half-widths add in quadrature.
                half = sum(
                    a["measured_ci95_half_width_ms_per_round"] ** 2
                    for a in sel) ** 0.5
                entry = {
                    "cells": [a["cell"] for a in sel],
                    "xbf16_ms_per_round": sum(
                        a["xbf16_ms_per_round"] for a in sel),
                    "measured_in_kernel_relief_ms_per_round": relief,
                    "ci95_half_width_ms_per_round": half,
                    "relief_ceiling_ms_per_round": RELIEF_CEILING_MS,
                    "meets_stop_bar": relief >= STOP_BAR_MS,
                    "meets_advance_bar": relief >= ADVANCE_BAR_MS,
                    "separated_from_zero": abs(relief) > half,
                }
                if all("measured_shipped_net_ms_per_round" in a for a in sel):
                    entry["measured_shipped_net_ms_per_round"] = sum(
                        a["measured_shipped_net_ms_per_round"] for a in sel)
                if all("desk" in a for a in sel):
                    entry["desk_predicted_in_kernel_net_ms_per_round"] = sum(
                        a["desk"]["predicted_in_kernel_net_ms_per_round"]
                        for a in sel)
                    entry["desk_gross_relief_ms_per_round"] = sum(
                        a["desk"]["gross_relief_ms_per_round"] for a in sel)
                pooled["na%d/g%d/%s"
                       % (na, groups, "cold" if cold else "hot")] = entry

    best = None
    for tag, entry in pooled.items():
        if best is None or (entry["measured_in_kernel_relief_ms_per_round"]
                            > pooled[best][
                                "measured_in_kernel_relief_ms_per_round"]):
            best = tag

    verdict = {
        "best_pooled_arm": best,
        "best_pooled_in_kernel_relief_ms_per_round": (
            pooled[best]["measured_in_kernel_relief_ms_per_round"]
            if best else None),
        "stop_bar_ms_per_round": STOP_BAR_MS,
        "advance_bar_ms_per_round": ADVANCE_BAR_MS,
        "any_arm_meets_advance_bar": any(
            e["meets_advance_bar"] for e in pooled.values()),
        "any_arm_meets_stop_bar": any(
            e["meets_stop_bar"] for e in pooled.values()),
    }

    out = {
        "experiment": "e223-percolumn-inner-loop",
        "stage": "1 - standalone census screen",
        "harness": "local-microbench",
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "cool_gate_passed_real_gate": blob.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": blob.get("gate_qualified_for_timing"),
        "gpu_temperature_c": blob.get("gpu_temperature_c"),
        "host_chip": "Apple M4 Pro",
        "local_arch": "applegpu_g16s",
        "ranked_arch": "applegpu_g17s",
        "per_column_instructions_g17s": {
            "bfloat16": INSN_PER_COLUMN_BF16, "float32": INSN_PER_COLUMN_F32},
        "source": str(args.xdtype_json),
        "desk_source": str(args.desk),
        "arms": arms,
        "pooled": pooled,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote %s (%d arms, %d pooled)"
          % (args.out, len(arms), len(pooled)))
    if best:
        e = pooled[best]
        print("best pooled %s: in-kernel relief %+.3f +/- %.3f ms/round "
              "(desk predicted %+.3f)"
              % (best, e["measured_in_kernel_relief_ms_per_round"],
                 e["ci95_half_width_ms_per_round"],
                 e.get("desk_predicted_in_kernel_net_ms_per_round",
                       float("nan"))))


if __name__ == "__main__":
    main()
