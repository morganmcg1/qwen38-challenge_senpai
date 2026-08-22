#!/usr/bin/env python3
"""E126 rung 0: zero-GPU instruction census and price model for Route B.

harness=local. Every number here is a PREDICTION written before the rung 1
session. Nothing in this file measures a GPU.

Three questions:

1. What does each arm actually execute, per lane per k-block, at each width?
2. What does E121 already collect of Route B's prize, and what is left?
3. What leg effect should thorfinn's rung 5e report once his mechanism runs on
   a base that already contains E121?

The arms are built by `research/e126_arms.py`:

  share_off       the pre-E121 shape: private sums tree, full consumer
  share_on        the shipped base: split-ownership tree plus exchange
  n_sums_free     tree AND the `sums * bias` consumer term deleted
  n_nosums_e123   tree deleted, scalar `+ bias_local[r]` retained (E123's arm)
  n_sums_loaded   tree deleted, full consumer, `sums` read from threadgroup
                  memory once per row per k-block (Route B faithful)

Only `share_off` and `share_on` are bit exact. The other three are diagnostic
shapes that price a boundary; they never ship.
"""

from __future__ import annotations

import argparse
import json
import pathlib

WIDTHS = (2, 3, 4, 5)
ROUND_WEIGHT = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# `affine_qmv_fast` routes M=2 to `qmv_fast_crossrow_affine4_g64<T,2>`, which
# has no `sums_xchg`, and no `_m` instantiation produces `wide<2>` (E121 §
# "Headline correction"). NA=5 exists but the shipped SHARE_SUMS gate folds off
# there, so `share_on` is byte-identical to `share_off` at NA=5.
REACHABLE = {2: False, 3: True, 4: True, 5: True}
GATED_OFF = {2: True, 3: False, 4: False, 5: True}

# --- geometry of one wide k block --------------------------------------------
# The wide template runs `for k in 0..in_vec_size step block_size` with
# block_size = values_per_thread * SIMD_SIZE = 16 * 32 = 512, and inside it
# `for i in 0..3` over 4-value chunks and `for m in 0..NA-1` over accumulator
# rows, then a consumer over rows_per_simd = 4 output rows.
CHUNKS = 4          # i
ROWS = 4            # rows_per_simd
TREE_ADDS = 4       # xv[0]+xv[1]+xv[2]+xv[3] is 3 adds, `sums[m] +=` is 1

# --- E123 measured prices, percent of a_base per instruction per k block -----
# a_base in E123 is the pre-E121 shape: g17s R=101, 39 resident simdgroups,
# which the E126 census reproduces exactly for `share_off`.
P_ALU = {2: 0.0408, 3: 0.0748, 4: 0.1350, 5: None}
P_TGLD = {2: -0.0075, 3: 0.1481, 4: 0.4044, 5: None}
P_TGST = {2: 0.0137, 3: 0.0869, 4: 0.0934, 5: None}
P_BAR = {2: 0.1652, 3: 0.0851, 4: 0.0836, 5: None}
# The whole-tree deletion contrast `a_base -> n_nosums`, E123 §8. This is a
# MEASURED deletion price, not the injection price, so it already carries the
# ~2.16x realisation discount that E123 found between census and session.
P_TREE_DELETE = {2: 0.0047, 3: 0.0182, 4: 0.0951, 5: 0.0891}
# E123's own instruction count for that contrast at NA=4. It scales linearly
# in NA in their census, so the per-width count is taken as 16.95 * NA.
E123_TREE_COUNT_NA4 = 67.8

# --- E121 measured, the shipped arm ------------------------------------------
# E121 reports two per-width columns. `per_width_pct` is the plain mean over
# the five scored shapes. `cost_weighted_per_width_pct` weights each shape by
# its share of round cost, and it is the column E121's own corrected headline
# used. The two disagree by about 30 % at NA=4, and the primary E126 metric is
# a difference against this arm, so both are carried.
E121_PER_WIDTH_PCT = {2: 0.687, 3: 0.463, 4: 1.482, 5: -0.118}
E121_COST_WEIGHTED_PCT = {2: 0.785, 3: 0.660, 4: 1.927, 5: -0.098}

# --- thorfinn's Route B grid, PR #121, ledger 269.6 ---------------------------
# Per-matvec gain for a pure group of that width, measured on a share_off base.
THORFINN_GAIN = {3: 2.20, 4: 5.88, 5: 6.52}
THORFINN_GAIN_FIT = {3: 3.63, 4: 5.88, 5: 6.52}
THORFINN_NET = {3: -0.03, 4: 3.46, 5: 4.40}
KERNEL_TO_LEG = 0.6068      # wide-QMV share of the leg
LEG_TO_RANKED = 0.95        # rule 34 dispatch-class transfer


def owned(na: int, lo: bool) -> int:
    """Rows a simdgroup owns under the shipped split. H = NA/2, sg0 owns m<H."""
    h = na // 2
    return h if lo else na - h


def census(na: int) -> dict[str, dict[str, float]]:
    """Per lane, per k block, dynamic instruction counts by arm.

    `alu` counts scalar float operations. A vec<float,NA> operation is NA of
    them. Counts are for the critical path: where the two simdgroups do
    different amounts of work the larger is reported, because the barrier makes
    the slower simdgroup set the block latency.
    """
    full_tree = CHUNKS * TREE_ADDS * na
    if GATED_OFF[na]:
        shared_tree = full_tree
        exchange_ld = exchange_st = barriers = 0
    else:
        shared_tree = CHUNKS * TREE_ADDS * max(owned(na, True), owned(na, False))
        # Each lane stores the rows it owns and loads the rows it does not, so
        # both simdgroups touch threadgroup memory exactly NA times per block.
        exchange_st = max(owned(na, True), owned(na, False))
        exchange_ld = na - min(owned(na, True), owned(na, False))
        barriers = 2

    # Consumer, over ROWS output rows and NA vector components.
    consumer_full = ROWS * na * 4       # 2 mul + 2 add
    consumer_free = ROWS * na * 2       # 1 mul + 1 add
    consumer_e123 = ROWS * na * 3       # 1 mul + 2 add

    return {
        "share_off": {
            "alu": full_tree + consumer_full,
            "tg_load": 0, "tg_store": 0, "barrier": 0},
        "share_on": {
            "alu": shared_tree + consumer_full,
            "tg_load": exchange_ld, "tg_store": exchange_st,
            "barrier": barriers},
        "n_sums_free": {
            "alu": consumer_free, "tg_load": 0, "tg_store": 0, "barrier": 0},
        "n_nosums_e123": {
            "alu": consumer_e123, "tg_load": 0, "tg_store": 0, "barrier": 0},
        # +2 integer operations for the ring slab index.
        "n_sums_loaded": {
            "alu": consumer_full + 2, "tg_load": na, "tg_store": 0,
            "barrier": 0},
    }


def priced(na: int, counts: dict[str, float]) -> float | None:
    """Nominal E123 injection price of one arm's per-block work."""
    if P_ALU[na] is None:
        return None
    return (counts["alu"] * P_ALU[na] + counts["tg_load"] * P_TGLD[na]
            + counts["tg_store"] * P_TGST[na] + counts["barrier"] * P_BAR[na])


def compose(outer: float, inner: float) -> float:
    """Gain of `inner` against `outer` when both are percent against share_off.

    Times are 1 - g/100 of share_off, so the ratio, not the difference, is what
    a rung 1 session measures when it uses `outer` as its own baseline.
    """
    return (inner - outer) / (1.0 - outer / 100.0)


def e123_tree_gain(na: int) -> float:
    """E123's measured share_off -> n_nosums_e123 gain, percent."""
    return P_TREE_DELETE[na] * E123_TREE_COUNT_NA4 * na / 4.0


def build() -> dict:
    rows = {}
    for na in WIDTHS:
        counts = census(na)
        rows[na] = {
            "reachable": REACHABLE[na],
            "gate_folds_off": GATED_OFF[na],
            "round_weight": ROUND_WEIGHT[na],
            "counts": counts,
            "nominal_price_pct": {a: priced(na, c) for a, c in counts.items()},
        }

    # --- model A: E123 measured deletion prices, bottom up -------------------
    # Anchor on E123's measured whole-tree deletion, then add or remove the
    # consumer difference at the same measured ALU deletion price.
    model_a = {}
    for na in WIDTHS:
        counts = census(na)
        p_del = P_TREE_DELETE[na]
        base_alu = counts["share_off"]["alu"]
        g_e123 = e123_tree_gain(na)
        # `n_sums_free` deletes ROWS*NA more ALU than `n_nosums_e123`.
        extra = ROWS * na * p_del
        g_free = g_e123 + extra
        # `n_sums_loaded` restores that ALU, adds NA threadgroup loads and the
        # 2 ring operations.
        tgld = P_TGLD[na] if P_TGLD[na] is not None else P_TGLD[4]
        g_loaded = g_e123 - extra - na * tgld - 2 * p_del
        model_a[na] = {
            "share_off_alu_per_kblock": base_alu,
            "n_nosums_e123_vs_share_off_pct": g_e123,
            "n_sums_free_vs_share_off_pct": g_free,
            "n_sums_loaded_vs_share_off_pct": g_loaded,
        }

    # --- model B: anchored on thorfinn's measured grid -----------------------
    model_b = {na: {"n_sums_free_vs_share_off_pct": THORFINN_GAIN.get(na),
                    "fitted": THORFINN_GAIN_FIT.get(na)} for na in WIDTHS}

    # --- the primary rung 1 metric, predicted --------------------------------
    primary = {}
    for na in WIDTHS:
        g_on = 0.0 if GATED_OFF[na] else E121_PER_WIDTH_PCT[na]
        a = model_a[na]["n_sums_free_vs_share_off_pct"]
        b = THORFINN_GAIN.get(na)
        entry = {
            "share_on_vs_share_off_pct": g_on,
            "model_a_primary_pct": compose(g_on, a),
            "model_a_faithful_primary_pct": compose(
                g_on, model_a[na]["n_sums_loaded_vs_share_off_pct"]),
            "overlap_O_model_a": (g_on / a) if a else None,
        }
        g_on_cw = 0.0 if GATED_OFF[na] else E121_COST_WEIGHTED_PCT[na]
        entry["share_on_cost_weighted_pct"] = g_on_cw
        if b is not None:
            entry["model_b_primary_pct"] = compose(g_on, b)
            entry["model_b_primary_pct_cost_weighted"] = compose(g_on_cw, b)
            entry["overlap_O_model_b"] = g_on / b
            entry["overlap_O_model_b_cost_weighted"] = g_on_cw / b
        primary[na] = entry

    def weighted(pick) -> float:
        total = 0.0
        for na in WIDTHS:
            if not REACHABLE[na]:
                continue
            value = pick(na)
            if value is None:
                continue
            total += ROUND_WEIGHT[na] * value
        return total

    # --- task 4: thorfinn rung 5e on an E121-containing base -----------------
    # Route B pays one replica dispatch whose absolute cost does not change
    # when E121 is present, so it is subtracted after the marginal gain.
    replica = {na: THORFINN_GAIN[na] - THORFINN_NET[na] for na in THORFINN_NET}
    marginal = {}
    for na in THORFINN_GAIN:
        g_on = 0.0 if GATED_OFF[na] else E121_PER_WIDTH_PCT[na]
        marginal[na] = {
            "marginal_gain_pct": compose(g_on, THORFINN_GAIN[na]),
            "marginal_gain_pct_fitted": compose(g_on, THORFINN_GAIN_FIT[na]),
            "replica_cost_pp": replica[na],
        }
        marginal[na]["marginal_net_pct"] = (
            marginal[na]["marginal_gain_pct"] - replica[na])
        marginal[na]["marginal_net_pct_fitted"] = (
            marginal[na]["marginal_gain_pct_fitted"] - replica[na])

    base_net_round = weighted(lambda na: THORFINN_NET.get(na))
    marg_net_round = weighted(
        lambda na: marginal[na]["marginal_net_pct"] if na in marginal else None)
    marg_net_round_fit = weighted(
        lambda na: marginal[na]["marginal_net_pct_fitted"]
        if na in marginal else None)

    task4 = {
        "per_width": marginal,
        "share_off_base_round_net_pct": base_net_round,
        "share_off_base_leg_pct": base_net_round * KERNEL_TO_LEG,
        "e121_base_round_net_pct": marg_net_round,
        "e121_base_leg_pct": marg_net_round * KERNEL_TO_LEG,
        "e121_base_leg_pct_fitted_na3": marg_net_round_fit * KERNEL_TO_LEG,
        "e121_base_ranked_pct": marg_net_round * KERNEL_TO_LEG * LEG_TO_RANKED,
        "fraction_of_route_b_leg_value_removed_by_e121": (
            1.0 - marg_net_round / base_net_round),
        "kernel_to_leg": KERNEL_TO_LEG,
        "leg_to_ranked": LEG_TO_RANKED,
    }

    return {
        "harness": "local",
        "kind": "prediction",
        "widths": list(WIDTHS),
        "round_weight": ROUND_WEIGHT,
        "reachable": REACHABLE,
        "gate_folds_off": GATED_OFF,
        "per_width": rows,
        "model_a_e123_prices": model_a,
        "model_b_thorfinn_grid": model_b,
        "primary_predictions": primary,
        "primary_round_weighted_model_a": weighted(
            lambda na: primary[na]["model_a_primary_pct"]),
        "primary_round_weighted_model_b": weighted(
            lambda na: primary[na].get("model_b_primary_pct")),
        "primary_round_weighted_faithful": weighted(
            lambda na: primary[na]["model_a_faithful_primary_pct"]),
        "primary_round_weighted_model_b_cost_weighted": weighted(
            lambda na: primary[na].get("model_b_primary_pct_cost_weighted")),
        "task4_thorfinn_rung5e": task4,
    }


def fmt(value, width=8, places=3) -> str:
    if value is None:
        return "%*s" % (width, "-")
    return "%*.*f" % (width, places, value)


def report(data: dict) -> None:
    print("E126 rung 0 - zero GPU prediction, harness=local\n")

    print("1. Per lane, per k block, dynamic counts "
          "(alu scalar ops / tg ld / tg st / barriers)")
    header = "  %-14s" % "arm" + "".join("      NA%d" % na for na in WIDTHS)
    print(header)
    for arm in ("share_off", "share_on", "n_sums_free", "n_nosums_e123",
                "n_sums_loaded"):
        cells = []
        for na in WIDTHS:
            c = data["per_width"][na]["counts"][arm]
            cells.append("%4d/%d/%d/%d" % (
                c["alu"], c["tg_load"], c["tg_store"], c["barrier"]))
        print("  %-14s %s" % (arm, "  ".join(cells)))
    print("  NA2 is unreachable (wide<2> is never instantiated); NA5 folds the "
          "gate off,\n  so share_on is byte identical to share_off there.")

    print("\n2. Model A, E123 measured deletion prices, percent versus "
          "share_off")
    print("  %-24s %s" % ("arm", "".join("      NA%d" % na for na in WIDTHS)))
    for key, label in (("n_nosums_e123_vs_share_off_pct", "n_nosums_e123"),
                       ("n_sums_free_vs_share_off_pct", "n_sums_free"),
                       ("n_sums_loaded_vs_share_off_pct", "n_sums_loaded")):
        print("  %-24s %s" % (label, "".join(
            fmt(data["model_a_e123_prices"][na][key]) for na in WIDTHS)))
    print("  %-24s %s" % ("thorfinn measured", "".join(
        fmt(THORFINN_GAIN.get(na)) for na in WIDTHS)))
    print("  %-24s %s" % ("share_on (E121 measured)", "".join(
        fmt(0.0 if GATED_OFF[na] else E121_PER_WIDTH_PCT[na])
        for na in WIDTHS)))

    print("\n3. Primary metric prediction, gain(n_sums_free vs share_on)")
    print("  %-24s %s" % ("source", "".join("      NA%d" % na
                                            for na in WIDTHS)))
    for key, label in (
            ("model_a_primary_pct", "model A (E123 prices)"),
            ("model_b_primary_pct", "model B (thorfinn)"),
            ("model_b_primary_pct_cost_weighted", "model B, cost-weighted on"),
            ("model_a_faithful_primary_pct", "Route B faithful"),
            ("overlap_O_model_a", "overlap O, model A"),
            ("overlap_O_model_b", "overlap O, model B"),
            ("overlap_O_model_b_cost_weighted", "overlap O, B cost-weighted")):
        print("  %-26s %s" % (label, "".join(
            fmt(data["primary_predictions"][na].get(key)) for na in WIDTHS)))
    print("  round weighted, model A %.3f %%   model B %.3f %%   "
          "B cost-weighted %.3f %%   faithful %.3f %%" % (
              data["primary_round_weighted_model_a"],
              data["primary_round_weighted_model_b"],
              data["primary_round_weighted_model_b_cost_weighted"],
              data["primary_round_weighted_faithful"]))

    print("\n4. Task 4, thorfinn rung 5e on an E121-containing base")
    t4 = data["task4_thorfinn_rung5e"]
    print("  %-22s %8s %8s %8s" % ("width", "gain", "replica", "net"))
    for na in sorted(t4["per_width"]):
        row = t4["per_width"][na]
        print("  %-22s %s %s %s" % (
            "NA%d" % na, fmt(row["marginal_gain_pct"]),
            fmt(row["replica_cost_pp"]), fmt(row["marginal_net_pct"])))
    print("  share_off base : round net %.3f %%  leg %.3f %%"
          % (t4["share_off_base_round_net_pct"], t4["share_off_base_leg_pct"]))
    print("  E121 base      : round net %.3f %%  leg %.3f %%  ranked %.3f %%"
          % (t4["e121_base_round_net_pct"], t4["e121_base_leg_pct"],
             t4["e121_base_ranked_pct"]))
    print("  E121 base, fitted phi(3)=3.63 : leg %.3f %%"
          % t4["e121_base_leg_pct_fitted_na3"])
    print("  E121 removes %.1f %% of Route B's leg value."
          % (100.0 * t4["fraction_of_route_b_leg_value_removed_by_e121"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    data = build()
    report(data)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
