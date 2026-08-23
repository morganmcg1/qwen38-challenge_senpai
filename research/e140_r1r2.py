#!/usr/bin/env python3
"""E140 F7 rungs R1 and R2. `harness=local instrument`. Zero GPU.

R1. `pb6` is not one mechanism. `makeBoundaryDepthPrice` holds the TOTAL, so
raising one step forces every other step down. The scheduler never reads the
total; it reads `marginal[d] / cumulative[d]`. Rebuild that table from source,
then split the arm into the two mechanisms it actually contains and price each
one on its own:

    arm S   marginal = [within] * 8                pure global subsidy
    arm P   marginal = [0.18] * 8, marginal[4] = 0.261
                                                   pure boundary spike
    arm pb6 = S composed with P, which is the shipped confound

R1 has a real ranked A/B behind it, which is rare. `572b2cc4` and `e003a86d`
differ by `pb6` alone on every quantity that moves a draft length, so the
replay is scored against a receipt it was never fitted to. The transfer fit
targets `RANKED_PROMPTS[*]["depth"]`, and those eight values ARE `572b2cc4`'s
published draft lengths, so the control arm is already anchored correctly and
`e003a86d` is a clean out-of-sample target.

R2. Campaign Rule 121. A replayed median is only decision-grade if the rank
order of the eight raw ratios is reported next to it. Emit the rank vector and
the median pair for every arm in every table E140 has published, and flag every
arm that reorders the receipt.

Usage:
  python3 e140_r1r2.py --json e140-artifacts/r1r2.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e128_price import (  # noqa: E402
    MAX_DEPTH, RANKED_PROMPTS, load_board_receipt,
)
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct, simulate, walk  # noqa: E402
from e140_cells import (  # noqa: E402
    CELLS, TIER_GRID, install, load_masses, lookahead_walker, lopo_curves,
    price_for, summarise, tier_ratios, transfer_cache,
)
from e140_lookahead import (  # noqa: E402
    CURVE_FORMS, PRICE_CUMULATIVE, PRICE_MARGINAL, SHIPPED_CLIFF,
    SHIPPED_TIER, load_curves, pb6_price,
)
from e140_perturb import ARMS as PERTURB_ARMS  # noqa: E402
from e140_perturb import ALPHONSE_ROUND_SHARE, cliff_step, perturb  # noqa: E402

HEAD_STEP_COST_RATIO = 0.18

# The receipt pair that isolates `pb6` on the official runner. `A` carries the
# tight grid alone; `B` adds pb6 1.45, probe 0.15 and Table.shipped, and the
# advisor reports that neither probe nor Table.shipped has ever moved a draft
# length on any receipt.
CONTROL_RECEIPT = "572b2cc4"
TREATMENT_RECEIPT = "e003a86d"

# F7 section 3, transcribed for an independent check. Depth, ship threshold,
# pb6 threshold, ratio.
ADVISOR_THRESHOLDS = (
    (0, 0.180000, 0.170414, 0.9467),
    (1, 0.152542, 0.145601, 0.9545),
    (2, 0.132353, 0.127096, 0.9603),
    (3, 0.116883, 0.112763, 0.9648),
    (4, 0.104651, 0.146942, 1.4041),
    (5, 0.094737, 0.088354, 0.9326),
    (6, 0.086538, 0.081181, 0.9381),
    (7, 0.079646, 0.075086, 0.9428),
)


# --------------------------------------------------------------- price tables

def prefix_costs(marginal):
    """`Qwen36MTPBlockSession.prefixCosts`, accumulated, not closed form."""
    out = [1.0]
    for value in marginal:
        out.append(out[-1] + value)
    return out


def subsidy_marginal(tier: float = SHIPPED_TIER):
    """`makeBoundaryDepthPrice`'s `within`, with the spike removed."""
    count = MAX_DEPTH
    within = count * HEAD_STEP_COST_RATIO / (count - 1 + tier)
    return [within] * count


def spike_marginal(tier: float = SHIPPED_TIER, cliff: int = SHIPPED_CLIFF):
    """The priced boundary alone. The total is NOT held, by design."""
    marginal = [HEAD_STEP_COST_RATIO] * MAX_DEPTH
    marginal[cliff] = HEAD_STEP_COST_RATIO * tier
    return marginal


def thresholds(price):
    """What `costModelDepth` compares `reach` against, less `(1 + expected)`.

    The dropped factor is common to both arms at the same depth, so a ratio of
    two thresholds at one depth is exact. A threshold LEVEL is not the whole
    story on its own, because `expected` depends on the arm's own trajectory.
    """
    marginal, cumulative = price
    return [marginal[d] / cumulative[d] for d in range(len(marginal))]


R1_ARMS = {
    "A_ship": (list(PRICE_MARGINAL), list(PRICE_CUMULATIVE)),
    "S_subsidy": (subsidy_marginal(), prefix_costs(subsidy_marginal())),
    "P_spike": (spike_marginal(), prefix_costs(spike_marginal())),
    "pb6": pb6_price(SHIPPED_TIER, SHIPPED_CLIFF),
}


# ---------------------------------------------------------------- rank vectors

def raw_ratios(receipt, ratios) -> dict:
    """The eight per-prompt raw ratios `median_pct` sorts, before it sorts."""
    out = {}
    for prompt, entry in receipt["per_prompt"].items():
        ratio = ratios.get(prompt, {}).get("ratio", 1.0)
        out[prompt] = entry["serial"] / (entry["candidate"] * ratio)
    return out


def rank_vector(raws) -> list:
    return [p for p, _ in sorted(raws.items(), key=lambda kv: kv[1])]


def ship_raws(receipt) -> dict:
    return {p: e["serial"] / e["candidate"]
            for p, e in receipt["per_prompt"].items()}


def binding_gap(raws) -> float:
    """Rule 121's smallest binding inter-rank gap, in percent.

    The pair is positions 3 and 4. A move smaller than the gap on either side
    of the pair cannot change which prompts occupy it, so this is the size an
    effect must exceed before the marginal weights stop being exact.
    """
    values = sorted(raws.values())
    return min((values[4] / values[3] - 1.0) * 100.0,
               (values[5] / values[4] - 1.0) * 100.0)


def rank_row(receipt, ratios, ship_order) -> dict:
    raws = raw_ratios(receipt, ratios)
    order = rank_vector(raws)
    return {
        "rank_vector": order,
        "median_pair": [order[3], order[4]],
        "reordered": order != ship_order,
        "pair_changed": set(order[3:5]) != set(ship_order[3:5]),
        "raws": raws,
    }


# ------------------------------------------------------------------ R1 replay

def arm_rows(cache, seeds, curve, windows, price, base_cache, curve_key,
             only=None):
    """One arm on eight prompts, with the depth and round detail R1 asks for."""
    install(curve)
    out = {}
    for prompt in ([only] if only else RANKED_PROMPTS):
        ratios, depths, rounds, nondraft = [], [], [], []
        for seed in seeds:
            entry = cache[(seed, prompt)]
            slot = (curve_key, seed, prompt)
            if slot in base_cache:
                base = base_cache[slot]
            else:
                base = simulate(None, entry["factory"](entry["p_target"]),
                                windows)
                base_cache[slot] = base
            run = simulate(None, entry["factory"](entry["p_target"]), windows,
                           price=price)
            ratios.append(run["us_per_token"] / base["us_per_token"])
            depths.append(run["mean_depth"])
            rounds.append(run["rounds"] / windows)
            nondraft.append(run["depth_counts"][0] / run["rounds"])
        out[prompt] = {
            "ratio": statistics.fmean(ratios),
            "ratio_sd": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            "mean_depth": statistics.fmean(depths),
            "rounds_per_512": statistics.fmean(rounds),
            "non_drafting_share": statistics.fmean(nondraft),
        }
    return out


def run_r1(cache, seeds, curves, masses, control, treatment, windows) -> dict:
    """Every arm on every curve form, scored against the real ranked A/B."""
    measured_ratio = {
        p: treatment["per_prompt"][p]["candidate"]
        / control["per_prompt"][p]["candidate"]
        for p in control["per_prompt"]}
    measured_depth = {p: treatment["per_prompt"][p]["draft_len"]
                      for p in control["per_prompt"]}
    control_depth = {p: control["per_prompt"][p]["draft_len"]
                     for p in control["per_prompt"]}
    measured_median = (treatment["score"] / control["score"] - 1.0) * 100.0
    order_ship = rank_vector(ship_raws(control))

    base_cache = {}
    forms = {}
    for form in CURVE_FORMS:
        lopo = lopo_curves(masses, form) if form in FORMS else None
        arms = {}
        for name, price in R1_ARMS.items():
            rows = arm_rows(cache, seeds, curves[form], windows, price,
                            base_cache, form)
            entry = {"per_prompt": rows,
                     "median_pct": median_pct(control, rows),
                     "rank": rank_row(control, rows, order_ship)}
            if lopo is not None:
                folded = {}
                for prompt in RANKED_PROMPTS:
                    one = arm_rows(cache, seeds, lopo[prompt], windows, price,
                                   base_cache, "lopo|%s|%s" % (form, prompt),
                                   only=prompt)
                    folded[prompt] = one[prompt]
                entry["median_pct_lopo"] = median_pct(control, folded)
                entry["rank_lopo"] = rank_row(control, folded, order_ship)
            arms[name] = entry
        forms[form] = arms
    return {
        "measured_ratio": measured_ratio,
        "measured_depth": measured_depth,
        "control_depth": control_depth,
        "measured_median_pct": measured_median,
        "control_rank_vector": order_ship,
        "control_binding_gap_pct": binding_gap(ship_raws(control)),
        "treatment_rank_vector": rank_vector(ship_raws(treatment)),
        "forms": forms,
    }


def unlock_curve(cache, seeds, curve, windows, levels) -> dict:
    """Non-drafting share against the depth-0 threshold, per prompt.

    At depth 0 the scheduler compares `reach` with `marginal[0] * 1.0 /
    cumulative[0]`, and `cumulative[0]` is 1.0 and `expected` is 0.0, so a
    uniform price of level `m` sets the depth-0 threshold to exactly `m`. F7
    says a cut from 0.180000 to 0.170414 turned 449 of plutarch's 487 rounds
    from non-drafting to drafting. That claim is a statement about how much
    reach mass sits in a 5.33 percent band just below 0.18, and this sweep
    measures that mass directly instead of inferring it.
    """
    install(curve)
    out = {}
    for level in levels:
        marginal = [level] * MAX_DEPTH
        price = (marginal, prefix_costs(marginal))
        row = {}
        for prompt in RANKED_PROMPTS:
            shares, depths = [], []
            for seed in seeds:
                entry = cache[(seed, prompt)]
                run = simulate(None, entry["factory"](entry["p_target"]),
                               windows, price=price)
                shares.append(run["depth_counts"][0] / run["rounds"])
                depths.append(run["mean_depth"])
            row[prompt] = {"non_drafting_share": statistics.fmean(shares),
                           "mean_depth": statistics.fmean(depths)}
        out["%.6f" % level] = row
    return out


def probe_walker(rate: float, seed: int):
    """Force one draft on a share `rate` of otherwise non-drafting rounds.

    `recordAcceptOutcome(acceptedCount: 0, drafts: [])` updates nothing:
    the accept loop is empty, `acceptedCount < drafts.count` is false, and the
    optimism branch needs a non-empty draft list. So a round that drafts
    nothing records nothing, and once `positionAcceptEMA[0]` falls under the
    depth-0 threshold the session cannot draft again for the rest of the
    request. `positionAcceptEMA` is a session-lifetime `private var`, so the
    state is absorbing within one request.

    This walker is a DIAGNOSTIC, not a proposal. It exists to separate a
    threshold cut, which only moves the absorbing boundary, from an escape
    mechanism, which destroys the absorbing state.
    """
    rng = random.Random(seed)

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        depth = walk(ema, margin, offer, adjust, ctx, force, price)
        if depth == 0 and offer >= 1 and rng.random() < rate:
            return 1
        return depth
    return chooser


def probe_sweep(cache, seeds, curve, windows, receipt, rates) -> dict:
    """Non-drafting share and median against the forced-draft rate."""
    install(curve)
    base_cache = {}
    out = {}
    for rate in rates:
        rows, per_seed = {}, []
        merged = {}
        for seed in seeds:
            ratios = {}
            for prompt in RANKED_PROMPTS:
                entry = cache[(seed, prompt)]
                slot = (seed, prompt)
                if slot in base_cache:
                    base = base_cache[slot]
                else:
                    base = simulate(None, entry["factory"](entry["p_target"]),
                                    windows)
                    base_cache[slot] = base
                run = simulate(None, entry["factory"](entry["p_target"]),
                               windows,
                               walker=probe_walker(rate, seed * 1000 + 7))
                ratios[prompt] = {
                    "ratio": run["us_per_token"] / base["us_per_token"]}
                merged.setdefault(prompt, []).append(
                    (run["depth_counts"][0] / run["rounds"],
                     run["mean_depth"], ratios[prompt]["ratio"]))
            per_seed.append(median_pct(receipt, ratios))
        for prompt, values in merged.items():
            rows[prompt] = {
                "non_drafting_share": statistics.fmean(v[0] for v in values),
                "mean_depth": statistics.fmean(v[1] for v in values),
                "ratio": statistics.fmean(v[2] for v in values)}
        out["%.4f" % rate] = {"per_prompt": rows,
                              "median_pct": statistics.fmean(per_seed)}
    return out


def attribute(r1: dict, form: str) -> dict:
    """Which of S or P carries each of the three measured signatures."""
    arms = r1["forms"][form]
    ship = arms["A_ship"]["per_prompt"]
    out = {}
    for name in ("S_subsidy", "P_spike", "pb6"):
        rows = arms[name]["per_prompt"]
        out[name] = {p: {
            "d_depth": rows[p]["mean_depth"] - ship[p]["mean_depth"],
            "d_non_drafting": (rows[p]["non_drafting_share"]
                               - ship[p]["non_drafting_share"]),
            "ratio": rows[p]["ratio"],
        } for p in ship}
    verdict = {
        "plutarch_unlock_S": out["S_subsidy"]["plutarch"]["d_non_drafting"],
        "plutarch_unlock_P": out["P_spike"]["plutarch"]["d_non_drafting"],
        "essays_deepening_S": out["S_subsidy"]["essays"]["d_depth"],
        "essays_deepening_P": out["P_spike"]["essays"]["d_depth"],
        "beagle_loss_S": out["S_subsidy"]["beagle"]["ratio"],
        "beagle_loss_P": out["P_spike"]["beagle"]["ratio"],
    }
    return {"per_arm": out, "verdict": verdict}


# ------------------------------------------------------------------ R2 tables

def r2_cells(cells_path, receipt, order_ship) -> dict:
    """Rank vectors for the E140 2x2, from the published per-prompt ratios."""
    blob = json.loads(cells_path.read_text())["cells"]
    out = {}
    for key, entry in blob.items():
        ratios = {p: {"ratio": v}
                  for p, v in entry["per_prompt_ratio"].items()}
        row = rank_row(receipt, ratios, order_ship)
        row.pop("raws")
        row["curve_lopo"] = entry["curve_lopo_mean"]
        out[key] = row
    return out


def r2_tier_grid(cache, seeds, curve, receipt, order_ship, windows) -> dict:
    """The E134 tier grid, with a rank vector at every tier."""
    base_cache = {}
    out = {}
    for tier in TIER_GRID:
        per_seed, merged, pairs = [], {}, set()
        for seed in seeds:
            rows = tier_ratios(cache, seed, curve, tier, windows,
                               base_cache=base_cache, curve_key="tier")
            per_seed.append(median_pct(receipt, rows))
            pairs.add(tuple(rank_vector(raw_ratios(receipt, rows))[3:5]))
            for prompt, row in rows.items():
                merged.setdefault(prompt, []).append(row["ratio"])
        ratios = {p: {"ratio": statistics.fmean(v)} for p, v in merged.items()}
        row = rank_row(receipt, ratios, order_ship)
        row.pop("raws")
        row["median_pct"] = statistics.fmean(per_seed)
        row["pair_churn"] = len(pairs)
        out["%.4f" % tier] = row
    return out


def _cell_rows(cache, seed, curve, price, walker, windows, base_cache,
               curve_key, only=None):
    """One arm on one curve. `base_cache` shares the unpriced denominator.

    Five arms at one perturbation share one denominator per prompt and seed,
    so the memo removes four fifths of the base work. The key must name the
    exact curve, because a leave-one-prompt-out fold is a different curve.
    """
    install(curve)
    out = {}
    for prompt in ([only] if only else RANKED_PROMPTS):
        entry = cache[(seed, prompt)]
        slot = (curve_key, seed, prompt)
        if slot in base_cache:
            base = base_cache[slot]
        else:
            base = simulate(None, entry["factory"](entry["p_target"]), windows)
            base_cache[slot] = base
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       price=price, walker=walker)
        out[prompt] = {"ratio": run["us_per_token"] / base["us_per_token"],
                       "mean_depth": run["mean_depth"]}
    return out


def r2_perturbation(cache, seeds, curves, masses, receipt, order_ship,
                    windows, forms, tier_form) -> dict:
    """The E140 cliff-perturbation table, with a rank vector in every cell."""
    step, round_us = cliff_step(curves[tier_form])
    alphonse = ALPHONSE_ROUND_SHARE * round_us / step
    fractions = sorted([0.0, 0.05, 0.10, alphonse, 0.25, 0.375, 0.50])
    base_cache = {}
    out = {}
    for form in forms:
        base_lopo = lopo_curves(masses, form) if form in FORMS else None
        for fraction in fractions:
            curve = perturb(curves[form], fraction)
            lopo = ({p: perturb(c, fraction) for p, c in base_lopo.items()}
                    if base_lopo is not None else None)
            tag = "%s|%.6f" % (form, fraction)
            for cell in PERTURB_ARMS:
                price = price_for(CELLS[cell]["price"], curve)
                walker = (lookahead_walker()
                          if CELLS[cell]["walk"] == "argmax" else None)
                merged, per_seed, pairs = {}, [], set()
                for seed in seeds:
                    rows = _cell_rows(cache, seed, curve, price, walker,
                                      windows, base_cache, tag)
                    per_seed.append(median_pct(receipt, rows))
                    pairs.add(tuple(rank_vector(
                        raw_ratios(receipt, rows))[3:5]))
                    for prompt, row in rows.items():
                        merged.setdefault(prompt, []).append(row["ratio"])
                ratios = {p: {"ratio": statistics.fmean(v)}
                          for p, v in merged.items()}
                row = rank_row(receipt, ratios, order_ship)
                row.pop("raws")
                row["in_sample"] = statistics.fmean(per_seed)
                row["pair_churn"] = len(pairs)
                if lopo is not None:
                    held, folded_merged = [], {}
                    for seed in seeds:
                        folded = {}
                        for prompt in RANKED_PROMPTS:
                            fold_curve = lopo[prompt]
                            fold_price = price_for(CELLS[cell]["price"],
                                                   fold_curve)
                            one = _cell_rows(cache, seed, fold_curve,
                                             fold_price, walker, windows,
                                             base_cache,
                                             "%s|lopo|%s" % (tag, prompt),
                                             only=prompt)
                            folded[prompt] = one[prompt]
                            folded_merged.setdefault(prompt, []).append(
                                one[prompt]["ratio"])
                        held.append(median_pct(receipt, folded))
                    row["curve_lopo"] = statistics.fmean(held)
                    lopo_ratios = {p: {"ratio": statistics.fmean(v)}
                                   for p, v in folded_merged.items()}
                    lopo_row = rank_row(receipt, lopo_ratios, order_ship)
                    row["rank_vector_lopo"] = lopo_row["rank_vector"]
                    row["reordered_lopo"] = lopo_row["reordered"]
                    row["pair_changed_lopo"] = lopo_row["pair_changed"]
                out["%s|%s|%.6f" % (form, cell, fraction)] = row
    return {"fractions": fractions, "step_us": step, "cells": out}


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--tier-form", default="per_drafting_round")
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--cells", type=pathlib.Path,
                    default=HERE / "e140-artifacts/cells.json")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/r1r2.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 F7 R1 and R2  zero GPU")

    # ------------------------------------------------------------- R1 part 1
    print("\n## R1a  the threshold table, rebuilt from source")
    ship = (list(PRICE_MARGINAL), list(PRICE_CUMULATIVE))
    pb6 = pb6_price(SHIPPED_TIER, SHIPPED_CLIFF)
    t_ship, t_pb6 = thresholds(ship), thresholds(pb6)
    print("within  = 8 * 0.18 / (7 + 1.45) = %.10f" % pb6[0][0])
    print("spike   = within * 1.45         = %.10f" % pb6[0][SHIPPED_CLIFF])
    print("total held: sum(ship) %.10f  sum(pb6) %.10f  gap %.3e"
          % (sum(ship[0]), sum(pb6[0]), abs(sum(ship[0]) - sum(pb6[0]))))
    print("%5s %12s %12s %10s %10s" % ("depth", "ship", "pb6", "pb6/ship",
                                       "subsidy %"))
    worst = 0.0
    table = []
    for depth, adv_ship, adv_pb6, adv_ratio in ADVISOR_THRESHOLDS:
        ratio = t_pb6[depth] / t_ship[depth]
        worst = max(worst, abs(t_ship[depth] - adv_ship),
                    abs(t_pb6[depth] - adv_pb6),
                    abs(ratio - adv_ratio))
        table.append({"depth": depth, "ship": t_ship[depth],
                      "pb6": t_pb6[depth], "ratio": ratio,
                      "advisor_ship": adv_ship, "advisor_pb6": adv_pb6,
                      "advisor_ratio": adv_ratio})
        print("%5d %12.6f %12.6f %10.4f %10.2f"
              % (depth, t_ship[depth], t_pb6[depth], ratio,
                 (1.0 - ratio) * 100.0))
    print("largest disagreement with F7 section 3: %.3e (F7 prints 6 or 4 "
          "decimals)" % worst)
    subsidies = [(1.0 - table[d]["ratio"]) * 100.0
                 for d in range(MAX_DEPTH) if d != SHIPPED_CLIFF]
    print("subsidy range off the boundary: %.2f%% to %.2f%%"
          % (min(subsidies), max(subsidies)))
    print("the subsidy is NOT flat and NOT monotone: it shrinks 5.33 -> 3.52 "
          "below the spike, then jumps to its maximum just above it, because "
          "`cumulative[0]` is 1.0 and is not scaled with the marginals.")

    # ------------------------------------------------------------- the replay
    control = load_board_receipt(args.board, CONTROL_RECEIPT)
    treatment = load_board_receipt(args.board, TREATMENT_RECEIPT)
    receipt = load_board_receipt(args.board, args.receipt)
    curves, _ = load_curves()
    masses = load_masses()
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)

    print("\n## R1b  the two mechanisms, replayed against the real A/B")
    print("control %s score %.8f ; treatment %s score %.8f ; measured "
          "%+.4f %%" % (CONTROL_RECEIPT, control["score"], TREATMENT_RECEIPT,
                        treatment["score"],
                        (treatment["score"] / control["score"] - 1.0) * 100.0))
    r1 = run_r1(cache, seeds, curves, masses, control, treatment,
                args.windows)
    print("control rank vector   %s" % " ".join(r1["control_rank_vector"]))
    print("treatment rank vector %s" % " ".join(r1["treatment_rank_vector"]))
    print("Rule 121 binding gap around the median pair: %.2f %%"
          % r1["control_binding_gap_pct"])

    for form in CURVE_FORMS:
        print("\n### form %s" % form)
        print("%-10s %7s %7s %7s %7s %7s | %7s %7s %7s %7s"
              % ("prompt", "shipD", "S D", "P D", "pb6 D", "measD",
                 "S rat", "P rat", "pb6rat", "measrat"))
        arms = r1["forms"][form]
        for prompt in RANKED_PROMPTS:
            print("%-10s %7.4f %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f "
                  "%7.4f"
                  % (prompt,
                     arms["A_ship"]["per_prompt"][prompt]["mean_depth"],
                     arms["S_subsidy"]["per_prompt"][prompt]["mean_depth"],
                     arms["P_spike"]["per_prompt"][prompt]["mean_depth"],
                     arms["pb6"]["per_prompt"][prompt]["mean_depth"],
                     r1["measured_depth"][prompt],
                     arms["S_subsidy"]["per_prompt"][prompt]["ratio"],
                     arms["P_spike"]["per_prompt"][prompt]["ratio"],
                     arms["pb6"]["per_prompt"][prompt]["ratio"],
                     r1["measured_ratio"][prompt]))
        for name in R1_ARMS:
            entry = arms[name]
            print("  %-12s median %+8.4f  lopo %8s  reordered %-5s  pair %s"
                  % (name, entry["median_pct"],
                     "%+.4f" % entry["median_pct_lopo"]
                     if "median_pct_lopo" in entry else "n/a",
                     entry["rank"]["reordered"],
                     "/".join(entry["rank"]["median_pair"])))
        print("  measured median %+8.4f" % r1["measured_median_pct"])

    print("\n## R1c  attribution, on form %s" % args.tier_form)
    attr = attribute(r1, args.tier_form)
    v = attr["verdict"]
    print("plutarch non-drafting share, change vs ship:  S %+0.4f   P %+0.4f"
          % (v["plutarch_unlock_S"], v["plutarch_unlock_P"]))
    print("essays mean depth, change vs ship:            S %+0.4f   P %+0.4f"
          % (v["essays_deepening_S"], v["essays_deepening_P"]))
    print("beagle candidate-time ratio (1 = no change):  S %8.4f  P %8.4f"
          % (v["beagle_loss_S"], v["beagle_loss_P"]))

    print("\n## R1d  the depth-0 unlock curve, form %s" % args.tier_form)
    levels = sorted({0.18, subsidy_marginal()[0]}
                    | {0.10 + 0.005 * i for i in range(17)})
    unlock = unlock_curve(cache, seeds, curves[args.tier_form], args.windows,
                          levels)
    print("depth-0 threshold is exactly the uniform marginal level, because "
          "`cumulative[0]` is 1.0 and `expected` is 0.0 on the first step.")
    print("%10s " % "threshold" + " ".join("%8s" % p for p in RANKED_PROMPTS))
    for key in sorted(unlock, key=float):
        print("%10s " % key + " ".join(
            "%8.4f" % unlock[key][p]["non_drafting_share"]
            for p in RANKED_PROMPTS))
    ship_level, sub_level = "%.6f" % 0.18, "%.6f" % subsidy_marginal()[0]
    print("\nnon-drafting share, ship 0.180000 -> subsidy %s" % sub_level)
    for prompt in RANKED_PROMPTS:
        before = unlock[ship_level][prompt]["non_drafting_share"]
        after = unlock[sub_level][prompt]["non_drafting_share"]
        print("  %-10s %.4f -> %.4f  (%+.4f, %.2f %% of the non-drafting "
              "mass)" % (prompt, before, after, after - before,
                         100.0 * (before - after) / before if before else 0.0))

    print("\n## R1e  the absorbing non-drafting state, form %s"
          % args.tier_form)
    print("a round that drafts nothing records nothing, so `positionAccept"
          "EMA[0]` freezes below the depth-0 threshold and the session cannot "
          "draft again inside the request. A threshold cut moves the "
          "absorbing boundary; it does not remove the state.")
    rates = [0.0, 0.02, 0.05, 0.15, 0.30]
    probe = probe_sweep(cache, seeds, curves[args.tier_form], args.windows,
                        control, rates)
    print("%8s %10s " % ("rate", "median %")
          + " ".join("%8s" % p for p in RANKED_PROMPTS))
    for key in sorted(probe, key=float):
        print("%8s %+10.4f " % (key, probe[key]["median_pct"])
              + " ".join("%8.4f"
                         % probe[key]["per_prompt"][p]["non_drafting_share"]
                         for p in RANKED_PROMPTS))
    print("plutarch mean depth against the forced-draft rate:")
    for key in sorted(probe, key=float):
        print("  rate %s  depth %.4f  ratio %.4f"
              % (key, probe[key]["per_prompt"]["plutarch"]["mean_depth"],
                 probe[key]["per_prompt"]["plutarch"]["ratio"]))
    print("  measured at %s: depth %.4f, non-drafting 0, ratio %.4f"
          % (TREATMENT_RECEIPT, r1["measured_depth"]["plutarch"],
             r1["measured_ratio"]["plutarch"]))

    # ------------------------------------------------------------------- R2
    order_ship = rank_vector(ship_raws(receipt))
    print("\n## R2  rank vectors, against replay receipt %s" % args.receipt)
    print("ship order  %s" % " ".join(order_ship))
    print("median pair %s" % "/".join(order_ship[3:5]))
    print("Rule 121 binding gap: %.2f %%" % binding_gap(ship_raws(receipt)))

    cells_rank = r2_cells(args.cells, receipt, order_ship)
    print("\n### the E140 2x2")
    print("%-34s %10s %10s %-18s" % ("cell", "curve-lopo", "reordered",
                                     "median pair"))
    for key in sorted(cells_rank):
        row = cells_rank[key]
        print("%-34s %10s %10s %-18s"
              % (key,
                 "%+.4f" % row["curve_lopo"] if row["curve_lopo"] is not None
                 else "n/a",
                 "YES" if row["reordered"] else "no",
                 "/".join(row["median_pair"])
                 + (" *" if row["pair_changed"] else "")))

    print("\n### the E134 tier grid, form %s" % args.tier_form)
    tier_rank = r2_tier_grid(cache, seeds, curves[args.tier_form], receipt,
                             order_ship, args.windows)
    print("%8s %10s %10s %-20s %6s" % ("tier", "median %", "reordered",
                                       "median pair", "churn"))
    for key in sorted(tier_rank, key=float):
        row = tier_rank[key]
        print("%8s %+10.4f %10s %-20s %6d"
              % (key, row["median_pct"], "YES" if row["reordered"] else "no",
                 "/".join(row["median_pair"])
                 + (" *" if row["pair_changed"] else ""), row["pair_churn"]))

    pert = None
    if not args.skip_perturb:
        print("\n### the E140 cliff perturbation")
        pert = r2_perturbation(cache, seeds, curves, masses, receipt,
                               order_ship, args.windows, list(CURVE_FORMS),
                               args.tier_form)
        print("%-38s %10s %10s %9s %-20s %6s"
              % ("form|cell|cut", "in-sample", "curve-lopo", "reordered",
                 "median pair", "churn"))
        for key in sorted(pert["cells"]):
            row = pert["cells"][key]
            print("%-38s %+10.4f %10s %9s %-20s %6d"
                  % (key, row["in_sample"],
                     "%+.4f" % row["curve_lopo"] if "curve_lopo" in row
                     else "n/a",
                     "YES" if row["reordered"] else "no",
                     "/".join(row["median_pair"])
                     + (" *" if row["pair_changed"] else ""),
                     row["pair_churn"]))

    flagged = sum(1 for r in cells_rank.values() if r["reordered"])
    flagged_t = sum(1 for r in tier_rank.values() if r["reordered"])
    print("\n## flags")
    print("2x2 cells reordered      %d of %d" % (flagged, len(cells_rank)))
    print("tier grid points         %d of %d" % (flagged_t, len(tier_rank)))
    if pert:
        fp = sum(1 for r in pert["cells"].values() if r["reordered"])
        pc = sum(1 for r in pert["cells"].values() if r["pair_changed"])
        print("perturbation cells       %d of %d reordered, %d change the pair"
              % (fp, len(pert["cells"]), pc))

    payload = {
        "harness": "local instrument",
        "gpu_used": False,
        "windows": args.windows,
        "seeds": seeds,
        "receipt": args.receipt,
        "control_receipt": CONTROL_RECEIPT,
        "treatment_receipt": TREATMENT_RECEIPT,
        "threshold_table": table,
        "threshold_worst_error": worst,
        "r1": r1,
        "attribution": attr,
        "unlock_curve": unlock,
        "probe_sweep": probe,
        "r2_cells": cells_rank,
        "r2_tier_grid": tier_rank,
        "r2_perturbation": pert,
        "ship_rank_vector": order_ship,
        "binding_gap_pct": binding_gap(ship_raws(receipt)),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
