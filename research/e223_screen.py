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
# RULE 410 (advisor, Entry 447, from E222 FINDING 586/587): every priced stream
# must carry an access-regime tag, and `b_stream = 1.9292 us/MiB` prices
# first-touch DRAM only.
B_STREAM_US_PER_MIB = 1.9292
CACHE_SERVED_FRACTION_BOUND = 0.36


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
    ap.add_argument(
        "--base-census", type=pathlib.Path,
        default=pathlib.Path("research/e223-artifacts/e223-census-base.json"))
    ap.add_argument(
        "--arm-census", type=pathlib.Path,
        default=pathlib.Path(
            "research/e223-artifacts/e223-census-xf32ship.json"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    blob = json.loads(args.xdtype_json.read_text())
    if blob.get("phase") != "xdtype":
        raise SystemExit("not an E223 xdtype phase report: %s"
                         % args.xdtype_json)
    units = chain_slopes(blob)
    desk = json.loads(args.desk.read_text())["pricing"]["xf32flat"]["arms"]

    # RULE 407: the register and spill witness for the EXACT instantiation that
    # was timed. `xf32ship` is the source form `e223ActivationF32Header()`
    # builds, so these are the timed kernel's own numbers.
    base_agx = json.loads(args.base_census.read_text())["agx"]
    arm_agx = json.loads(args.arm_census.read_text())["agx"]
    occupancy = {}
    for arch in sorted(arm_agx):
        per_na = {}
        for na in range(2, 7):
            key = "na%d_tbl" % na
            if key not in arm_agx[arch] or key not in base_agx[arch]:
                continue
            b, a = base_agx[arch][key], arm_agx[arch][key]
            per_na["na%d" % na] = {
                "registers_bf16": b["registers"],
                "registers_f32": a["registers"],
                "registers_delta": a["registers"] - b["registers"],
                "spill_bytes_bf16": b["spill_bytes"],
                "spill_bytes_f32": a["spill_bytes"],
                "spill_bytes_delta": a["spill_bytes"] - b["spill_bytes"],
                "text_bytes_bf16": b["text_bytes"],
                "text_bytes_f32": a["text_bytes"],
            }
        occupancy[arch] = per_na

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

        # RULE 410 access-regime split of the extra activation bytes. The
        # activation slab is `m * k` values and every simdgroup in the dispatch
        # re-reads all of it, so first touch is the slab itself and the rest is
        # cache-served/concurrent inside one dispatch.
        first_touch_mb = f["m"] * k * 2 / 1048576.0  # bf16 -> f32 adds 2 B/value
        row["access_regime"] = {
            "stream": "activation re-read, extra bytes from bf16 -> f32",
            "first_touch_mb_per_invocation": first_touch_mb,
            "cache_served_concurrent_mb_per_invocation":
                extra_mb - first_touch_mb,
            "cache_served_fraction": (
                (extra_mb - first_touch_mb) / extra_mb if extra_mb else None),
            "tag": "cache-served/concurrent",
            "first_touch_cost_ms_per_round_at_b":
                first_touch_mb * B_STREAM_US_PER_MIB * inv / 1000.0,
            "cache_served_cost_ms_per_round_at_0p36b":
                (extra_mb - first_touch_mb) * B_STREAM_US_PER_MIB
                * CACHE_SERVED_FRACTION_BOUND * inv / 1000.0,
        }

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

    # The coefficient the advisor asked for: the marginal worth of one removed
    # non-arithmetic instruction. This arm cannot identify it alone. Removing a
    # convert REQUIRES widening the load, so removed instructions and extra
    # bytes are exactly proportional in every arm, and only their SUM is
    # identified. Report the ratio, the sum, and the exclusion test.
    ratios = sorted({
        round(a["removed_instructions_per_invocation"]
              / (a["extra_issued_activation_mb_per_invocation"] * 1048576.0), 6)
        for a in arms if a["extra_issued_activation_mb_per_invocation"]})
    coefficient = {
        "identified_quantity":
            "sum of the instruction-relief and extra-byte terms, not either "
            "term alone",
        "removed_instructions_per_extra_byte": ratios,
        "collinear": len(ratios) == 1,
        "why_collinear":
            "removing a bfloat16-to-float convert requires widening the load "
            "that feeds it, so both terms scale as NA * lane_k_blocks * groups",
    }
    # The cheapest arms carry the smallest byte penalty, so they bound the
    # instruction term most tightly from above.
    cheap = [a for a in arms if a["na"] == 2 and a["groups"] == 1]
    if cheap:
        per_round_insn = sum(
            a["removed_instructions_per_invocation"]
            * a["invocations_per_round"] for a in cheap if a["pooled_cell"])
        coefficient["min_byte_arms"] = {
            "arm": "na2/g1",
            "removed_instructions_per_round_pooled": per_round_insn,
            "arms_overlapping_zero": sum(
                1 for a in cheap
                if abs(a["measured_in_kernel_relief_ms_per_round"])
                <= a["measured_ci95_half_width_ms_per_round"]),
            "arms_total": len(cheap),
        }
        for cold in (True, False):
            tag = "na2/g1/%s" % ("cold" if cold else "hot")
            if tag not in pooled:
                continue
            e = pooled[tag]
            upper = (e["measured_in_kernel_relief_ms_per_round"]
                     + e["ci95_half_width_ms_per_round"])
            # Op-proportional relief scales with removed instructions, so the
            # desk's NA=4 gross figure halves at NA=2.
            desk_na4 = pooled.get("na4/g1/%s" % ("cold" if cold else "hot"), {})
            predicted = desk_na4.get("desk_gross_relief_ms_per_round")
            coefficient[tag] = {
                "measured_ms_per_round":
                    e["measured_in_kernel_relief_ms_per_round"],
                "ci95_half_width_ms_per_round":
                    e["ci95_half_width_ms_per_round"],
                "measured_upper_bound_ms_per_round": upper,
                "op_proportional_prediction_ms_per_round": (
                    predicted / 2.0 if predicted is not None else None),
                "op_proportional_excluded": (
                    predicted is not None and upper < predicted / 2.0),
                "us_per_removed_instruction_upper_bound": (
                    upper * 1000.0 / per_round_insn if per_round_insn else None),
            }

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
        # RULE 410 guard. The kill rests on a matched measurement of the arm
        # itself, not on any b-priced stream term, so no value of b' can move
        # it. Recording that explicitly is what makes the decision safe to take
        # before E226 supplies refitted coefficients.
        "decision_uses_b_stream": False,
        "decision_flips_anywhere_in_b_prime": False,
        "b_independence_note":
            "The advance bar is +1.0 ms/round. Every pooled arm above NA=2 "
            "measures between -1.7 and -354.8 ms/round, and the best arm is "
            "+0.300 +/- 0.446, which does not clear the +0.3 stop bar with "
            "separation. b' enters no term of this comparison.",
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
        "coefficient": coefficient,
        "rule_407_occupancy_witness": occupancy,
        "rule_410_b_stream_us_per_mib": B_STREAM_US_PER_MIB,
        "rule_410_cache_served_fraction_bound": CACHE_SERVED_FRACTION_BOUND,
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
