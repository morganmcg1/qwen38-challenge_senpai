#!/usr/bin/env python3
"""E123: price the four missing ladder classes, then audit the kernel with them.

    research/e123_analysis.py --rate research/out/e123-full/rate.json \
        --census research/e123-artifacts/census.json \
        --entrypoint research/e123-artifacts/entrypoint-census.json \
        --out research/e123-artifacts/summary.json

Sign convention. `cost(arm)` is the percent by which an arm is SLOWER than
`a_base`, because every price in this experiment is the cost of an added
instruction and a cost should be positive. `research/e118_analysis.py` uses the
opposite sign, so its reduction is negated once, here, and never mixed.

Every price is a RUNG CONTRAST between two arms that differ only in how many
instructions of one class they inject. The contrast cancels the injection
scaffold exactly, which is why no price carries a share of it.

harness=local throughout. The local Mac cannot reach the 40 C cool gate, so no
number here is gate qualified and none is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e118_analysis as e118a  # noqa: E402
import e123_arms as arms  # noqa: E402

WIDTHS = (2, 3, 4, 5)
HEADLINE_NA = 4
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# Pre-registration section 3. Point, low, high, in %/instruction/k-block at
# NA=4. Nothing here may be edited after the first `rate.json` exists.
BANDS = {
    "tgld": (0.45, 0.25, 0.70),
    "tgldc": (0.80, 0.45, 1.40),
    "tgst": (0.38, 0.20, 0.60),
    "bar": (0.15, 0.05, 0.40),
    "sbar": (0.00, -0.05, 0.05),
    "cvt": (0.47, 0.27, 0.72),
    "ssum": (0.20, 0.05, 0.60),
}
CVT_MINUS_TGLD_BAND = (0.02, 0.00, 0.09)
TGLDC_OVER_TGLD_BAND = (1.8, 1.3, 3.0)
BAR8_BAND_PP = 0.3
HOLDOUT_GATE_PP = 1.0
PRIMARY_BASELINE_PP = 0.66
SCAFFOLD_NA4_GATE_PP = 0.3

# Pre-registration section 10.1 and 10.2.
DELETION_RATIO_BAND = (1.35, 1.05, 1.90)
DELETION_DIVISOR_THRESHOLD = 1.15
DELETION_UNITY_BAND = (0.90, 1.15)
TG_ACCESS_PREDICTION = 0.20
EXCHANGE_COST_PP = 0.801
# Alphonse's cross-session numbers, quoted for scoring and never re-derived.
ALPHONSE_HALFSUMS_MEASURED_PCT = 2.266
ALPHONSE_LADDER_PREDICTED_PCT = 3.760

# Pre-registration section 1. E118's prices at NA=4 on this same host and
# architecture, quoted so this session can be checked against that one.
E118_PRICES_NA4 = {"alu": 0.09398, "ld": 0.58330, "shuf": 0.96486}

# Pre-registration section 10.3. Any one of these voids the session.
PEAK_BANDWIDTH_GB_S = 273.0
BANDWIDTH_GATE_FACTOR = 1.2
SCAFFOLD_ANY_WIDTH_GATE_PCT = 0.50

# Section 2: one AGX instruction in this kernel averages 8.25 bytes of machine
# text. Used only to convert a deletion arm's text delta into an instruction
# count, because AIR keeps the `m` loop rolled and cannot see the deletion.
BYTES_PER_INSTRUCTION = 8.25


# --- reduction ----------------------------------------------------------------

def costs(rate: dict) -> dict:
    """(shape, NA) -> arm -> percent SLOWER than `a_base`, per block."""
    out: dict[tuple[str, int], dict[str, list[float]]] = {}
    for key, bucket in e118a.paired_pct(rate).items():
        out[key] = {arm: [-v for v in vals] for arm, vals in bucket.items()}
    return out


def cost_by_width(cells: dict) -> dict:
    """NA -> arm -> summary over shapes, of the per-shape median cost."""
    per: dict[int, dict[str, list[float]]] = {}
    for (_shape, na), bucket in cells.items():
        for arm, vals in bucket.items():
            per.setdefault(na, {}).setdefault(arm, []).append(
                statistics.median(vals))
    return {na: {arm: e118a.summarise(v) for arm, v in b.items()}
            for na, b in per.items()}


def cost(by_width: dict, arm: str, na: int) -> float | None:
    cell = by_width.get(na, {}).get(arm)
    return None if cell is None else cell["median"]


# --- validity gates -----------------------------------------------------------

def validity_gates(rate: dict, by_width: dict, excluded: dict) -> dict:
    """Alphonse's three gates from E121, adopted whole.

    Gate 1 is the runtime check for harness defect 22: an out-of-bounds device
    write faults the command buffer, every later dispatch then retires in 1 to
    3 microseconds, and the harness still exits 0. The only visible symptom is
    an impossible read rate.
    """
    worst = {"gb_s": 0.0, "arm": None, "shape": None, "m": None}
    for row in rate["measurements"]:
        if row.get("kind") != "timing":
            continue
        for arm, seconds in row["seconds"].items():
            if not seconds:
                continue
            gb_s = row["read_bytes"] / seconds / 1e9
            if gb_s > worst["gb_s"]:
                worst = {"gb_s": gb_s, "arm": arm, "shape": row["shape"],
                         "m": row["m"]}
    limit = BANDWIDTH_GATE_FACTOR * PEAK_BANDWIDTH_GB_S
    bandwidth_ok = worst["gb_s"] <= limit

    scaffold = {na: cost(by_width, "q_scaffold", na) for na in WIDTHS}
    scaffold_moves = {na: v for na, v in scaffold.items()
                      if v is not None and abs(v) > SCAFFOLD_ANY_WIDTH_GATE_PCT}
    na4 = scaffold.get(HEADLINE_NA)
    scaffold_na4_ok = na4 is not None and abs(na4) <= SCAFFOLD_NA4_GATE_PP

    # Section 10.3: a positive control failure on a cell the pre-registered
    # spill rule already excludes voids that cell, not the session.
    failures, excused = [], []
    for row in rate["measurements"]:
        if row.get("kind") != "positive_control" or row["detected"]:
            continue
        if row["m"] in excluded.get(row["arm"], set()):
            excused.append({"arm": row["arm"], "shape": row["shape"],
                            "m": row["m"]})
        else:
            failures.append({"arm": row["arm"], "shape": row["shape"],
                             "m": row["m"]})

    return {
        "bandwidth": {"max_implied_gb_s": worst["gb_s"], "at": worst,
                      "limit_gb_s": limit, "passed": bandwidth_ok},
        "null_scaffold": {"cost_pct_by_width": scaffold,
                          "over_050_pct": scaffold_moves,
                          "na4_gate_pp": SCAFFOLD_NA4_GATE_PP,
                          "passed": not scaffold_moves and scaffold_na4_ok},
        "positive_controls": {"failures": failures,
                              "excused_on_excluded_cells": excused,
                              "passed": not failures},
        "session_valid": bandwidth_ok and not scaffold_moves
        and scaffold_na4_ok and not failures,
    }


# --- prices -------------------------------------------------------------------

def spilling(census: dict | None, arm: str, na: int) -> bool:
    if census is None:
        return False
    arch = census["local_arch"]
    cell = (census["arms"].get(arm, {}).get(arch, {}) or {}).get(str(na)) or {}
    return bool(cell.get("spill_bytes"))


def excluded_cells(census: dict | None) -> dict[str, set[int]]:
    if census is None:
        return {}
    out: dict[str, set[int]] = {}
    for arm in arms.ARMS:
        for na in WIDTHS:
            if spilling(census, arm, na):
                out.setdefault(arm, set()).add(na)
    return out


def prices(by_width: dict, census: dict | None) -> dict:
    """Per class per width: the contrast between the two extreme live rungs.

    A rung is dropped wherever the census says that arm spills on the timed
    architecture. E118's spill defect shows spilling changes both the time and
    the correctness of this kernel, so a spilling rung is not a measurement of
    its own class. Dropping the rung and contrasting the survivors keeps the
    class priceable; dropping the whole width would not.
    """
    out: dict[str, dict[int, dict]] = {}
    for klass, ladder in arms.CAL_LADDER.items():
        for na in WIDTHS:
            live = [(a, n) for a, n in ladder
                    if not spilling(census, a, na)
                    and cost(by_width, a, na) is not None]
            dropped = [a for a, n in ladder if (a, n) not in live]
            if len(live) < 2:
                out.setdefault(klass, {})[na] = {
                    "price": None, "dropped_rungs": dropped,
                    "reason": "fewer than two live rungs"}
                continue
            (lo_arm, lo_n), (hi_arm, hi_n) = live[0], live[-1]
            lo, hi = cost(by_width, lo_arm, na), cost(by_width, hi_arm, na)
            record = {
                "price": (hi - lo) / (hi_n - lo_n),
                "low_rung": [lo_arm, lo_n, lo], "high_rung": [hi_arm, hi_n, hi],
                "dropped_rungs": dropped,
                "live_rungs": [[a, n, cost(by_width, a, na)] for a, n in live],
            }
            if len(live) >= 3:
                fit = e118a.ols([n for _, n in live],
                                [cost(by_width, a, na) for a, _ in live])
                record["linearity"] = fit
            out.setdefault(klass, {})[na] = record
    return out


def price_of(priced: dict, klass: str, na: int) -> float | None:
    return (priced.get(klass, {}).get(na) or {}).get("price")


# --- the held-out predictions -------------------------------------------------

def holdouts(by_width: dict, priced: dict) -> dict:
    """Pre-registration section 4, computed exactly as written."""
    out: dict[str, dict[int, dict]] = {}
    for na in WIDTHS:
        p_alu = price_of(priced, "alu", na)
        p_ld = price_of(priced, "ld", na)
        p_shuf = price_of(priced, "shuf", na)
        p_tgld = price_of(priced, "tgld", na)
        p_tgst = price_of(priced, "tgst", na)
        p_bar = price_of(priced, "bar", na)

        preds: dict[str, float | None] = {}
        if None not in (p_alu, cost(by_width, "k_alu8", na)):
            preds["k_hold_alu12"] = cost(by_width, "k_alu8", na) + 4 * p_alu
        if None not in (p_shuf, p_ld, cost(by_width, "k_shuf8", na),
                        cost(by_width, "k_ld8", na)):
            s_sl = 0.5 * ((cost(by_width, "k_shuf8", na) - 8 * p_shuf)
                          + (cost(by_width, "k_ld8", na) - 8 * p_ld))
            preds["k_hold_sl"] = s_sl + 4 * p_shuf + 4 * p_ld
        if None not in (p_tgst, p_tgld, p_bar, p_alu,
                        cost(by_width, "k_tg0", na)):
            preds["k_hold_mix"] = (cost(by_width, "k_tg0", na)
                                   + 8 * p_tgst + 8 * p_tgld
                                   + 2 * p_bar + 4 * p_alu)
        for arm, pred in preds.items():
            measured = cost(by_width, arm, na)
            out.setdefault(arm, {})[na] = {
                "predicted_pct": pred, "measured_pct": measured,
                "error_pp": None if measured is None else abs(pred - measured),
                "signed_error_pp": None if measured is None else pred - measured,
            }
    return out


def primary_metric(hold: dict) -> dict:
    errs = {arm: (hold.get(arm, {}).get(HEADLINE_NA) or {}).get("error_pp")
            for arm in arms.HOLDOUT_ARMS}
    live = [v for v in errs.values() if v is not None]
    weighted = {}
    for arm in arms.HOLDOUT_ARMS:
        num = den = 0.0
        for na, cell in hold.get(arm, {}).items():
            if cell["error_pp"] is not None:
                num += ROUND_WEIGHTS[na] * cell["error_pp"]
                den += ROUND_WEIGHTS[na]
        weighted[arm] = num / den if den else None
    return {
        "e123_ladder_out_of_sample_prediction_error_pp":
            statistics.median(live) if live else None,
        "baseline_pp": PRIMARY_BASELINE_PP,
        "direction": "minimize",
        "per_holdout_na4_error_pp": errs,
        "round_weighted_error_pp": weighted,
        "max_error_pp": max(live) if live else None,
        "kill_rule_gate_pp": HOLDOUT_GATE_PP,
        "kill_rule_fired": bool(live) and max(live) > HOLDOUT_GATE_PP,
    }


# --- injection against deletion -----------------------------------------------

def deleted_instructions(census: dict | None, lo_arm: str, hi_arm: str,
                         na: int) -> dict:
    """Machine-text instruction count that `hi_arm` issues and `lo_arm` does not.

    AIR keeps the `m` loop rolled, so the AIR delta between `a_base` and
    `n_halfsums_free` is zero by construction and cannot supply this count.
    The backend unrolls at the fixed template width, so the machine text can.
    """
    if census is None:
        return {"count": None}
    arch = census["local_arch"]
    lo = census["arms"][lo_arm][arch][str(na)]["text_bytes"]
    hi = census["arms"][hi_arm][arch][str(na)]["text_bytes"]
    return {"text_bytes": hi - lo,
            "count": (hi - lo) / BYTES_PER_INSTRUCTION,
            "arch": arch}


DELETION_CONTRASTS = (
    ("first_half", "a_base", "n_halfsums_free"),
    ("second_half", "n_halfsums_free", "n_nosums"),
    ("whole_tree", "a_base", "n_nosums"),
)


def deletion_ladder(by_width: dict, priced: dict, census: dict | None) -> dict:
    """Every deletion contrast the three sums-tree arms support.

    Section 10.1 pre-registers one of these, `second_half`, and the headline
    ratio must come from it. The other two are reported because they use the
    same three arms and the same session, and because a ratio that depends on
    which half of one add tree you delete is not a property of the class.
    """
    out: dict[str, dict[int, dict]] = {}
    for name, keep_arm, cut_arm in DELETION_CONTRASTS:
        for na in WIDTHS:
            kept, cut = cost(by_width, keep_arm, na), cost(by_width, cut_arm,
                                                           na)
            counted = deleted_instructions(census, cut_arm, keep_arm, na)
            n = counted.get("count")
            inject = price_of(priced, "alu", na)
            record: dict = {"deleted_instructions_from_text": n,
                            "saving_pct": None, "deletion_price": None,
                            "ratio": None, "injection_price": inject}
            if None not in (kept, cut) and n:
                saving = kept - cut
                price = saving / n
                record.update(saving_pct=saving, deletion_price=price,
                              ratio=None if not (inject and price)
                              else inject / price)
            out.setdefault(name, {})[na] = record
    return out


def injection_vs_deletion(by_width: dict, priced: dict,
                          census: dict | None) -> dict:
    """Pre-registration section 10.1, both directions in one session."""
    out: dict[int, dict] = {}
    for na in WIDTHS:
        inject = price_of(priced, "alu", na)
        lo, hi = cost(by_width, "n_nosums", na), cost(by_width,
                                                      "n_halfsums_free", na)
        # Both deletion arms are FASTER than `a_base`, so both costs are
        # negative and the deeper deletion is the more negative one. The
        # contrast is written so that a deletion price comes out positive.
        counted = deleted_instructions(census, "n_nosums", "n_halfsums_free",
                                       na)
        source_count = 20 * arms.halfsums_kept(na)
        record: dict = {
            "injection_price": inject,
            "n_halfsums_free_cost_pct": hi,
            "n_nosums_cost_pct": lo,
            "deleted_instructions_from_text": counted.get("count"),
            "deleted_instructions_from_source": source_count,
        }
        if None not in (inject, lo, hi) and counted.get("count"):
            delete = (hi - lo) / counted["count"]
            record["deletion_price"] = delete
            record["ratio"] = inject / delete if delete else None
        out[na] = record

    ratio = (out.get(HEADLINE_NA) or {}).get("ratio")
    point, low, high = DELETION_RATIO_BAND
    divisor = 1.0
    verdict = "not measured"
    if ratio is not None:
        if ratio > DELETION_DIVISOR_THRESHOLD:
            divisor, verdict = ratio, "injection over-prices deletion"
        elif DELETION_UNITY_BAND[0] <= ratio <= DELETION_UNITY_BAND[1]:
            verdict = "injection price is deletion price on this class"
        else:
            verdict = "deletion costs more than injection"
    return {
        "per_width": out,
        "ratio_na4": ratio,
        "prediction": {"point": point, "band": [low, high],
                       "inside_band": ratio is not None and low <= ratio <= high},
        "alphonse_cross_session_ratio": (ALPHONSE_LADDER_PREDICTED_PCT
                                         / ALPHONSE_HALFSUMS_MEASURED_PCT),
        "rung1_divisor": divisor,
        "verdict": verdict,
    }


def e118_reproduction(priced: dict) -> dict:
    """Do E118's three prices reproduce in a second session on the same host?

    Same chip, same architecture, same entry points, same instrument, same
    width. Only the session differs, so a large drift here bounds how much any
    cross-session comparison in this campaign can be trusted.
    """
    out = {}
    for klass, before in E118_PRICES_NA4.items():
        now = price_of(priced, klass, HEADLINE_NA)
        out[klass] = {
            "e118_pct_per_instruction": before, "e123_pct_per_instruction": now,
            "drift_pct": None if now is None else 100.0 * (now - before) / before,
        }
    return out


def census_calibration(priced: dict, by_width: dict, census: dict | None
                       ) -> dict:
    """Check the priced census against the one deletion this session measured.

    The census prices the sums add tree from the injection ladder. The session
    also measures the tree being deleted for real, twice. If the census method
    works, the two must agree.
    """
    out = {}
    for na in WIDTHS:
        p_alu = price_of(priced, "alu", na)
        counted = deleted_instructions(census, "n_halfsums_free", "a_base", na)
        measured = cost(by_width, "a_base", na)
        half = cost(by_width, "n_halfsums_free", na)
        if None in (p_alu, measured, half) or not counted.get("count"):
            continue
        predicted = counted["count"] * p_alu
        realised = measured - half
        out[na] = {
            "deleted_instructions": counted["count"],
            "census_predicted_saving_pct": predicted,
            "measured_saving_pct": realised,
            "over_prediction_factor": predicted / realised if realised else None,
        }
    return out


def additivity(by_width: dict) -> dict:
    """Pre-registration section 9.3: does `k_hold_sl` add up without a slope?"""
    out = {}
    for na in WIDTHS:
        c0 = cost(by_width, "k_cal0", na)
        c_ld, c_shuf = cost(by_width, "k_ld4", na), cost(by_width, "k_shuf4", na)
        measured = cost(by_width, "k_hold_sl", na)
        if None in (c0, c_ld, c_shuf, measured):
            continue
        pred = c_ld + c_shuf - c0
        out[na] = {"scaffold_pct": c0, "k_ld4_pct": c_ld, "k_shuf4_pct": c_shuf,
                   "additive_prediction_pct": pred, "measured_pct": measured,
                   "excess_pp": measured - pred,
                   "superadditive": measured - pred > HOLDOUT_GATE_PP}
    return out


def free_predictions(priced: dict) -> dict:
    """Pre-registration section 10.2: two predictions bought with no GPU time."""
    p_tgld = price_of(priced, "tgld", HEADLINE_NA)
    p_tgst = price_of(priced, "tgst", HEADLINE_NA)
    p_bar = price_of(priced, "bar", HEADLINE_NA)
    out: dict = {"na": HEADLINE_NA}
    if None not in (p_tgld, p_tgst):
        access = 0.5 * (p_tgld + p_tgst)
        out["threadgroup_access"] = {
            "predicted": TG_ACCESS_PREDICTION, "measured": access,
            "error_pp": abs(TG_ACCESS_PREDICTION - access),
            "inside_gate": abs(TG_ACCESS_PREDICTION - access) <= HOLDOUT_GATE_PP,
        }
    if None not in (p_tgld, p_tgst, p_bar):
        pred = 2 * p_bar + 2 * p_tgld + 2 * p_tgst
        out["exchange_cost"] = {
            "predicted_pp": pred, "measured_pp": EXCHANGE_COST_PP,
            "error_pp": abs(pred - EXCHANGE_COST_PP),
            "inside_gate": abs(pred - EXCHANGE_COST_PP) <= HOLDOUT_GATE_PP,
            "barrier_share_pp": None if p_bar is None else 2 * p_bar,
        }
    return out


# --- rung 1: the priced instruction census ------------------------------------

# group -> (count(NA), ladder class, deletability, note)
#
# `deletability` is one of:
#   a  bit-exact deletable
#   b  deletable only with a numerical change, so unshippable and a ceiling only
#   c  not deletable
CENSUS_GROUPS: tuple[tuple[str, object, str, str, str], ...] = (
    ("weight element loads", lambda na: 16, "ld", "c",
     "the quantised weights the product is computed from"),
    ("metadata loads", lambda na: 8, "ld", "c",
     "already coalesced to 7 device loads at every width; the p_split_meta null "
     "confirms there is nothing left to remove"),
    ("metadata widenings", lambda na: 8, "cvt", "c",
     "bf16 scale and bias must reach float before use"),
    ("activation vec4 loads", lambda na: 4 * na, "ld", "c",
     "one vec4 load per (i, m); the activations must be read"),
    ("activation widenings", lambda na: 16 * na, "cvt", "b",
     "the FMA operand could stay bf16 only by changing the product's precision"),
    ("sums add tree", lambda na: 20 * na, "alu", "a",
     "x_sumshare_min removes half of it bit exactly; n_nosums removes all of it "
     "and is wrong, so the whole group is not class (a) -- only the half is"),
    ("activation register moves", lambda na: 16 * na, "alu", "a",
     "the transpose that lets one vector FMA serve all NA rows; deleting it is "
     "xr_split2, a measured null and stop-listed, so a positive price here is a "
     "finding about vectorisation and not an actionable deletion"),
    ("nibble integer operations", lambda na: 112, "alu", "a",
     "DIRECT_NIBBLES is already a template parameter; both settings are priced"),
    ("nibble integer to float", lambda na: 64, "alu", "c",
     "the 4-bit codes must reach float to multiply the activations"),
    ("lane FMAs", lambda na: 64 * na, "alu", "c",
     "the product itself"),
    ("final accumulate", lambda na: 12 * na, "alu", "c",
     "scale, bias and accumulate per row"),
    ("epilogue simd_sum", lambda na: 0.4 * na, "ssum", "c",
     "8 reductions per row amortised over 10 k-blocks"),
)


def rung1(priced: dict, divisor: float) -> dict:
    groups = []
    for name, count, klass, deletable, note in CENSUS_GROUPS:
        per_width, weighted = {}, 0.0
        for na in WIDTHS:
            price = price_of(priced, klass, na)
            n = count(na)
            value = None if price is None else n * price
            per_width[na] = {"count": n, "price": price, "pct": value}
            if value is not None:
                weighted += ROUND_WEIGHTS[na] * value
        groups.append({
            "group": name, "class": klass, "deletable": deletable,
            "note": note, "per_width": per_width,
            "round_weighted_pct": weighted,
            "round_weighted_pct_deletion_priced": weighted / divisor,
        })

    recon = {}
    for na in WIDTHS:
        total = sum(g["per_width"][na]["pct"] or 0.0 for g in groups)
        missing = [g["group"] for g in groups
                   if g["per_width"][na]["pct"] is None]
        recon[na] = {"total_pct_of_a_base": total, "ratio": total / 100.0,
                     "unpriced_groups": missing}

    ranked = sorted((g for g in groups if g["deletable"] == "a"),
                    key=lambda g: -g["round_weighted_pct_deletion_priced"])
    top = ranked[0]["round_weighted_pct_deletion_priced"] if ranked else 0.0
    return {
        "groups": groups,
        "reconstruction": recon,
        "reconstruction_band": [0.55, 0.85],
        "bit_exact_ranking": [
            {"group": g["group"],
             "round_weighted_pct": g["round_weighted_pct"],
             "round_weighted_pct_deletion_priced":
                 g["round_weighted_pct_deletion_priced"]}
            for g in ranked],
        "deletion_price_divisor": divisor,
        "e123_largest_predicted_bit_exact_deletion_pct_round_weighted": top,
        "rung2_bar_pct": 1.0,
        "rung2_justified": top > 1.0,
    }


# --- report -------------------------------------------------------------------

def band_verdict(value: float | None, band: tuple[float, float, float]) -> dict:
    point, low, high = band
    return {"point": point, "band": [low, high], "measured": value,
            "inside": value is not None and low <= value <= high}


def report(rate_path: pathlib.Path, census_path: pathlib.Path | None,
           entry_path: pathlib.Path | None, out_path: pathlib.Path | None
           ) -> int:
    rate = json.loads(rate_path.read_text())
    census = json.loads(census_path.read_text()) if census_path else None
    entry = json.loads(entry_path.read_text()) if entry_path else None

    cells = costs(rate)
    by_width = cost_by_width(cells)
    excluded = excluded_cells(census)
    gates = validity_gates(rate, by_width, excluded)
    priced = prices(by_width, census)
    hold = holdouts(by_width, priced)
    primary = primary_metric(hold)
    delete = injection_vs_deletion(by_width, priced, census)
    delladder = deletion_ladder(by_width, priced, census)
    repro = e118_reproduction(priced)
    calib = census_calibration(priced, by_width, census)
    add = additivity(by_width)
    free = free_predictions(priced)
    audit = rung1(priced, delete["rung1_divisor"])
    fidelity = e118a.fidelity_rows(rate)
    dispersion = e118a.block_dispersion(rate, cells)
    gap = e118a.forward_reverse_gap(rate)
    spill = e118a.spill_exactness(rate, census)

    print("E123 -- harness=local, cool_gate_passed_real_gate=false, "
          "gate_qualified_for_timing=false, official_or_ranked_score=false")
    print("%s on %s, %d arms, %d pairs"
          % (rate["device"], rate["architecture"], rate["arm_count"],
             rate["pairs"]))

    print("\n== validity gates (any failure voids the session) ==")
    bw = gates["bandwidth"]
    print("  implied bandwidth   %8.1f GB/s  limit %.1f  %s"
          % (bw["max_implied_gb_s"], bw["limit_gb_s"],
             "ok" if bw["passed"] else "VOID"))
    print("  null scaffold       %s  %s"
          % (" ".join("NA%d=%+.3f" % (na, v) for na, v
                      in sorted(gates["null_scaffold"]["cost_pct_by_width"]
                                .items()) if v is not None),
             "ok" if gates["null_scaffold"]["passed"] else "VOID"))
    pc = gates["positive_controls"]
    print("  positive controls   %d failure(s), %d excused on excluded cells  %s"
          % (len(pc["failures"]), len(pc["excused_on_excluded_cells"]),
             "ok" if pc["passed"] else "VOID"))
    print("  SESSION %s" % ("VALID" if gates["session_valid"] else "VOID"))

    print("\n== prices, percent/instruction/k-block ==")
    print("  %-8s %9s %9s %9s %9s   %s"
          % ("class", "NA2", "NA3", "NA4", "NA5", "NA4 band"))
    for klass, ladder in arms.CAL_LADDER.items():
        cellrow = []
        for na in WIDTHS:
            p = price_of(priced, klass, na)
            if p is not None:
                cellrow.append("%9.4f" % p)
                continue
            why = "spill" if any(spilling(census, a, na) for a, _ in ladder) \
                else "unmeas"
            cellrow.append("%9s" % why)
        band = BANDS.get(klass)
        verdict = ""
        if band:
            v = band_verdict(price_of(priced, klass, HEADLINE_NA), band)
            verdict = "%.2f [%.2f, %.2f] %s" % (
                band[0], band[1], band[2], "ok" if v["inside"] else "MISS")
        print("  %-8s %s   %s" % (klass, " ".join(cellrow), verdict))

    p_cvt = price_of(priced, "cvt", HEADLINE_NA)
    p_tgld = price_of(priced, "tgld", HEADLINE_NA)
    p_tgldc = price_of(priced, "tgldc", HEADLINE_NA)
    conversion = None if None in (p_cvt, p_tgld) else p_cvt - p_tgld
    ratio_c = None if not p_tgld else (p_tgldc / p_tgld if p_tgldc else None)
    print("\n  bf16 widening   cvt - tgld = %s   %s"
          % ("%.4f" % conversion if conversion is not None else "n/a",
             "ok" if band_verdict(conversion, CVT_MINUS_TGLD_BAND)["inside"]
             else "MISS"))
    print("  bank conflict   tgldc / tgld = %s   %s"
          % ("%.3f" % ratio_c if ratio_c is not None else "n/a",
             "ok" if band_verdict(ratio_c, TGLDC_OVER_TGLD_BAND)["inside"]
             else "MISS"))
    bar8 = cost(by_width, "k_bar8", HEADLINE_NA)
    print("  barrier merge   k_bar8 = %s pp, predicted 0.0 +/- %.1f   %s"
          % ("%+.3f" % bar8 if bar8 is not None else "n/a", BAR8_BAND_PP,
             "ok" if bar8 is not None and abs(bar8) <= BAR8_BAND_PP
             else "MISS"))

    print("\n== held-out predictions ==")
    for arm in arms.HOLDOUT_ARMS:
        for na in WIDTHS:
            cell = hold.get(arm, {}).get(na)
            if cell is None:
                continue
            print("  %-14s NA%d  pred %+7.3f  meas %+7.3f  err %5.3f pp  %s"
                  % (arm, na, cell["predicted_pct"], cell["measured_pct"],
                     cell["error_pp"],
                     "ok" if cell["error_pp"] <= HOLDOUT_GATE_PP else "MISS"))
    print("  primary  e123_ladder_out_of_sample_prediction_error_pp = %s "
          "(baseline %.2f, minimize)"
          % ("%.3f" % primary["e123_ladder_out_of_sample_prediction_error_pp"]
             if primary["e123_ladder_out_of_sample_prediction_error_pp"]
             is not None else "n/a", PRIMARY_BASELINE_PP))
    print("  kill rule %s (max holdout error %s pp against a %.1f pp gate)"
          % ("FIRED" if primary["kill_rule_fired"] else "not fired",
             "%.3f" % primary["max_error_pp"]
             if primary["max_error_pp"] is not None else "n/a",
             HOLDOUT_GATE_PP))

    if "threadgroup_access" in free:
        t = free["threadgroup_access"]
        print("\n  free prediction: one threadgroup access at NA=4")
        print("    predicted %.2f, measured %.4f, err %.3f pp  %s"
              % (t["predicted"], t["measured"], t["error_pp"],
                 "ok" if t["inside_gate"] else "MISS"))
    if "exchange_cost" in free:
        x = free["exchange_cost"]
        print("  free prediction: the x_sumshare_min exchange cost")
        print("    predicted %.3f pp, measured %.3f pp, err %.3f pp  %s"
              % (x["predicted_pp"], x["measured_pp"], x["error_pp"],
                 "ok" if x["inside_gate"] else "MISS"))
        print("    of which barriers %.3f pp" % (x["barrier_share_pp"] or 0.0))

    print("\n== injection price against deletion price ==")
    for na in WIDTHS:
        r = delete["per_width"][na]
        print("  NA%d  inject %s  delete %s  ratio %s  (deleted %s instr from "
              "text, %d from source)"
              % (na,
                 "%.4f" % r["injection_price"]
                 if r.get("injection_price") is not None else "n/a",
                 "%.4f" % r["deletion_price"]
                 if r.get("deletion_price") is not None else "n/a",
                 "%.3f" % r["ratio"] if r.get("ratio") is not None else "n/a",
                 "%.1f" % r["deleted_instructions_from_text"]
                 if r.get("deleted_instructions_from_text") else "n/a",
                 r["deleted_instructions_from_source"]))
    print("  NA=4 ratio %s, predicted %.2f [%.2f, %.2f]  %s"
          % ("%.3f" % delete["ratio_na4"] if delete["ratio_na4"] is not None
             else "n/a", *DELETION_RATIO_BAND,
             "ok" if delete["prediction"]["inside_band"] else "MISS"))
    print("  verdict: %s; rung 1 divisor %.3f"
          % (delete["verdict"], delete["rung1_divisor"]))

    print("\n  the same three arms, all three contrasts (only the second is "
          "pre-registered)")
    for name, _, _ in DELETION_CONTRASTS:
        row = []
        for na in WIDTHS:
            cell = delladder[name][na]
            row.append("%9.4f" % cell["deletion_price"]
                       if cell["deletion_price"] is not None else "%9s" % "-")
        r4 = delladder[name][HEADLINE_NA]["ratio"]
        print("    %-12s %s   NA4 ratio %s"
              % (name, " ".join(row), "%.3f" % r4 if r4 is not None else "n/a"))

    print("\n== does E118 reproduce in a second session on the same host ==")
    for klass, cell in repro.items():
        print("  p_%-5s E118 %.5f  E123 %s  drift %s"
              % (klass, cell["e118_pct_per_instruction"],
                 "%.5f" % cell["e123_pct_per_instruction"]
                 if cell["e123_pct_per_instruction"] is not None else "n/a",
                 "%+.1f %%" % cell["drift_pct"]
                 if cell["drift_pct"] is not None else "n/a"))

    print("\n== census method checked against a real measured deletion ==")
    for na, cell in sorted(calib.items()):
        print("  NA%d  %.1f instructions: census predicts %+.3f %%, the "
              "session measures %+.3f %%, over-prediction %.2fx"
              % (na, cell["deleted_instructions"],
                 cell["census_predicted_saving_pct"],
                 cell["measured_saving_pct"],
                 cell["over_prediction_factor"] or float("nan")))

    print("\n== cross-class additivity, measured scaffold ==")
    for na, cell in sorted(add.items()):
        print("  NA%d  k_cal0 %+.3f  k_ld4 %+.3f  k_shuf4 %+.3f  -> pred %+.3f"
              "  meas %+.3f  excess %+.3f pp %s"
              % (na, cell["scaffold_pct"], cell["k_ld4_pct"],
                 cell["k_shuf4_pct"], cell["additive_prediction_pct"],
                 cell["measured_pct"], cell["excess_pp"],
                 "SUPERADDITIVE" if cell["superadditive"] else ""))

    print("\n== rung 1: priced instruction census ==")
    print("  %-28s %-5s %-3s %8s %8s %8s %8s %10s"
          % ("group", "class", "del", "NA2 %", "NA3 %", "NA4 %", "NA5 %",
             "weighted"))
    for g in audit["groups"]:
        print("  %-28s %-5s %-3s %s %10.3f"
              % (g["group"], g["class"], g["deletable"],
                 " ".join("%8.3f" % (g["per_width"][na]["pct"] or float("nan"))
                          for na in WIDTHS),
                 g["round_weighted_pct"]))
    for na in WIDTHS:
        r = audit["reconstruction"][na]
        print("  reconstruction NA%d  %.1f %% of a_base, ratio %.3f%s"
              % (na, r["total_pct_of_a_base"], r["ratio"],
                 "  (unpriced: %s)" % ", ".join(r["unpriced_groups"])
                 if r["unpriced_groups"] else ""))
    print("  bit-exact ranking, deletion priced (divisor %.3f):"
          % audit["deletion_price_divisor"])
    for row in audit["bit_exact_ranking"]:
        print("    %-28s %7.3f %%  (undivided %7.3f %%)"
              % (row["group"], row["round_weighted_pct_deletion_priced"],
                 row["round_weighted_pct"]))
    print("  largest bit-exact deletion %.3f %% round weighted; rung 2 %s"
          % (audit[
              "e123_largest_predicted_bit_exact_deletion_pct_round_weighted"],
             "justified" if audit["rung2_justified"]
             else "NOT justified against the +1.0 % bar"))

    if entry is not None:
        print("\n== entry-point census (all widths inlined into one kernel) ==")
        for arch, table in entry["simdgroups"].items():
            print("  %s" % arch)
            for arm, sg in table.items():
                cell = entry["arms"][arm][arch]["0"]
                print("    %-18s %4s regs / s%-4s / %6s B / %s simdgroups"
                      % (arm, cell["registers"], cell["spill_bytes"],
                         cell["text_bytes"], sg))

    print("\n== fidelity ==")
    print("  exact failures: %d" % len(fidelity["exact_failures"]))
    for row in fidelity["exact_failures"][:12]:
        print("    %-14s NA%d %-28s %d/%d differ, max_ulp %d"
              % (row["arm"], row["m"], row["shape"], row["differing"],
                 row["total"], row["max_ulp"]))
    if spill is not None:
        print("  largest spill while exact: %s; smallest spill while wrong: %s;"
              " separates: %s"
              % (spill["max_spill_while_exact"], spill["min_spill_while_wrong"],
                 spill["separates"]))

    summary = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "device": rate["device"], "architecture": rate["architecture"],
        "validity_gates": gates,
        "cost_by_width": {str(na): {a: c for a, c in b.items()}
                          for na, b in by_width.items()},
        "prices": {k: {str(na): v for na, v in b.items()}
                   for k, b in priced.items()},
        "band_verdicts": {k: band_verdict(price_of(priced, k, HEADLINE_NA), b)
                          for k, b in BANDS.items()},
        "conversion_na4": conversion,
        "conversion_band": band_verdict(conversion, CVT_MINUS_TGLD_BAND),
        "bank_conflict_ratio_na4": ratio_c,
        "bank_conflict_band": band_verdict(ratio_c, TGLDC_OVER_TGLD_BAND),
        "barrier_merge_k_bar8_na4_pp": bar8,
        "holdouts": {a: {str(na): v for na, v in b.items()}
                     for a, b in hold.items()},
        "primary_metric": primary,
        "free_predictions": free,
        "injection_vs_deletion": {
            **delete,
            "per_width": {str(na): v for na, v in delete["per_width"].items()}},
        "deletion_ladder": {name: {str(na): v for na, v in per.items()}
                            for name, per in delladder.items()},
        "e118_reproduction": repro,
        "census_calibration": {str(na): v for na, v in calib.items()},
        "additivity": {str(na): v for na, v in add.items()},
        "rung1": audit,
        "secondary_metrics": {
            "e123_threadgroup_load_pct_per_instr_per_kblock_na4": p_tgld,
            "e123_threadgroup_store_pct_per_instr_per_kblock_na4":
                price_of(priced, "tgst", HEADLINE_NA),
            "e123_barrier_pct_per_barrier_per_kblock_na4":
                price_of(priced, "bar", HEADLINE_NA),
            "e123_bf16_to_float_conversion_pct_per_instr_per_kblock_na4":
                conversion,
            "e123_largest_predicted_bit_exact_deletion_pct_round_weighted":
                audit[
                    "e123_largest_predicted_bit_exact_deletion_pct_round_weighted"],
        },
        "fidelity": fidelity,
        "block_dispersion": dispersion,
        "forward_reverse_gap": gap,
        "spill_exactness": spill,
        "entrypoint_census": entry,
        "excluded_cells": {a: sorted(v) for a, v in excluded.items()},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print("\nwrote %s" % out_path)
    return 0 if gates["session_valid"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=pathlib.Path, required=True)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--entrypoint", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    return report(args.rate, args.census, args.entrypoint, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
