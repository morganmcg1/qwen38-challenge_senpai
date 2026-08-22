#!/usr/bin/env python3
"""E140 F3 arm (a) and F4 items 3, 4 and 8: the post-tight curve.

`harness=local instrument`. Zero GPU.

F3 asked for a uniform per-round shift of `-2257 us` as a gate, and for a
logarithmic shift normalised to the same mean. F4 retracted the normalisation
and replaced it with the measured law from receipt `572b2cc4` and the rival
pair `02742bf0 -> ed608e64`:

    round saving at verify width M  =  COEF * ln(M / cols(M))

`cols(M)` is the number of launched columns at width M under the tight grid.
Reading the law back out of F4's own table confirms the form exactly: the cut
into width 2 is `COEF * ln(2) = 898.8 us` against F4's `898.9`, and the cut
into width 8 with `cols(8) = 2` is `-(COEF*ln(4) - COEF*ln(7)) = +725.8 us`
against F4's `-725.7`, which is the step GROWING.

Two column tables are in play. `onePass67` keeps one column up to width 7;
the shipped table splits widths 6 and 7 into two columns. Thorfinn ships the
second one, so that is the curve the next ranked receipt will walk.

Arms:

    A_ship        flat 0.18                       greedy   the anchor
    C_flatlook    flat 0.18                       argmax   the null gate
    D_curvelook   measured post-tight curve       argmax   the hypothesis
    E_pb6         flat + one boundary at width 6  greedy   the merged arm
    F_pb6look     the same, argmax                argmax   the composition
    G_pb68        flat + boundaries at 6 and 8    greedy   F4 item 8
    H_pb68look    the same, argmax                argmax   the composition

`G_pb68`'s width-8 tier is fitted in sample on each post-tight curve, which
favours the fitted rival over the parameter-free argmax. That is the direction
that makes a win for the argmax hard to argue away.

Usage:
  python3 e140_posttight.py --json e140-artifacts/posttight.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e128_price import MAX_DEPTH, RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct, simulate  # noqa: E402
from e140_cells import (  # noqa: E402
    greedy_walker, install, install_pb68, load_masses, lookahead_walker,
    lopo_curves, run_cell, summarise, transfer_cache,
)
from e140_lookahead import (  # noqa: E402
    CURVE_FORMS, SHIPPED_CLIFF, SHIPPED_CLIFF8, SHIPPED_TIER, curve_price,
    load_curves, pb68_price, walk_argmax,
)

ARMS = ("A_ship", "C_flatlook", "D_curvelook", "E_pb6", "F_pb6look",
        "G_pb68", "H_pb68look")

# F4's regression on `623e77af -> 572b2cc4`, replicated at -1329.0 on the
# independent rival pair `02742bf0 -> ed608e64`.
LAUNCH_COEF = 1296.8
# F3's flat model of the same saving, kept as a gate rather than a candidate.
UNIFORM_SHIFT_US = -2257.0

COLUMN_TABLES = {
    "onePass67": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 3},
    "shipped": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3},
}

TIER8_GRID = (1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 2.0, 2.25,
              2.5)


def uniform_shift(curve, us: float = UNIFORM_SHIFT_US) -> dict:
    """Move every round cost by the same absolute amount, changing no step."""
    out = dict(curve)
    out["uniform"] = curve.get("uniform", 0.0) + us
    out["name"] = "%s|uniform%+.0fus" % (curve.get("name", "?"), us)
    return out


def log_launch_shift(curve, table: str, coef: float = LAUNCH_COEF) -> dict:
    """Subtract `coef * ln(M / cols(M))` from the round cost at each width."""
    cols = COLUMN_TABLES[table]
    out = dict(curve)
    per_width = {int(w): v for w, v in (curve.get("per_width") or {}).items()}
    for rows in range(1, MAX_DEPTH + 2):
        columns = cols.get(rows, cols[max(cols)])
        saving = coef * math.log(rows / columns) if rows >= columns else 0.0
        per_width[rows] = per_width.get(rows, 0.0) - saving
    out["per_width"] = per_width
    out["name"] = "%s|log-%s" % (curve.get("name", "?"), table)
    return out


VARIANTS = {
    "base": lambda c: dict(c),
    "uniform": uniform_shift,
    "log_onePass67": lambda c: log_launch_shift(c, "onePass67"),
    "log_shipped": lambda c: log_launch_shift(c, "shipped"),
}


def step_table(curve) -> list[float]:
    """`step into width M` for M in 2..9, under `curve`."""
    saved = e128_price.CURVE
    try:
        install(curve)
        us = [e128_price.ranked_round_us(m) for m in range(1, MAX_DEPTH + 2)]
    finally:
        e128_price.CURVE = saved
    return [us[i + 1] - us[i] for i in range(len(us) - 1)]


def cap_probe(cache, seeds, curve, windows):
    """Does the argmax want a depth the shipped cap forbids, and where?

    The scored arm always uses the shipped `segmentedVerifyDepthCap`. This
    walker returns the capped choice and only records what an uncapped argmax
    would have chosen, so the diagnostic cannot change a single scored round.
    """
    counter = {"rounds": 0, "wants_past_cap": 0, "capped_sum": 0,
               "want_sum": 0, "want_hist": [0] * (MAX_DEPTH + 1)}

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        best = walk_argmax(ema, margin, offer, price=price)
        want = walk_argmax(ema, margin, offer, price=price,
                           cap_limit=MAX_DEPTH)
        counter["rounds"] += 1
        counter["capped_sum"] += best
        counter["want_sum"] += want
        counter["want_hist"][want] += 1
        if want > best:
            counter["wants_past_cap"] += 1
        return best

    install(curve)
    price = curve_price(curve)
    for seed in seeds:
        for prompt in RANKED_PROMPTS:
            entry = cache[(seed, prompt)]
            simulate(None, entry["factory"](entry["p_target"]), windows,
                     price=price, walker=chooser)
    rounds = counter["rounds"] or 1
    return {
        "rounds": counter["rounds"],
        "wants_past_cap_share": counter["wants_past_cap"] / rounds,
        "mean_capped_depth": counter["capped_sum"] / rounds,
        "mean_uncapped_depth": counter["want_sum"] / rounds,
        "uncapped_depth_share": [v / rounds for v in counter["want_hist"]],
    }


def arm_row(cache, seeds, curve, lopo, receipt, windows, cell):
    in_sample, held_out, depths = [], [], []
    depth_counts = {}
    for seed in seeds:
        ratios = {}
        for prompt in RANKED_PROMPTS:
            row = run_cell(cache, seed, prompt, cell, curve, windows)
            ratios[prompt] = row
            depths.append(RANKED_PROMPTS[prompt]["weight"] * row["mean_depth"])
            for depth, count in enumerate(row["depth_counts"]):
                depth_counts[depth] = depth_counts.get(depth, 0) + count
        in_sample.append(median_pct(receipt, ratios))
        if lopo is not None:
            folded = {p: run_cell(cache, seed, p, cell, lopo[p], windows)
                      for p in RANKED_PROMPTS}
            held_out.append(median_pct(receipt, folded))
    total = sum(depth_counts.values()) or 1
    return {
        "in_sample": summarise(in_sample),
        "curve_lopo": summarise(held_out) if held_out else None,
        "mean_depth": sum(depths) / len(seeds),
        "depth_share": [depth_counts.get(d, 0) / total
                        for d in range(MAX_DEPTH + 1)],
    }


def fit_tier8(cache, seeds, curve, receipt, windows, walker_kind="greedy"):
    """The width-8 tier that maximises the in-sample median on `curve`."""
    install(curve)
    by_tier = {}
    for tier8 in TIER8_GRID:
        price = pb68_price(SHIPPED_TIER, tier8, SHIPPED_CLIFF, SHIPPED_CLIFF8)
        values = []
        for seed in seeds:
            ratios = {}
            for prompt in RANKED_PROMPTS:
                entry = cache[(seed, prompt)]
                base = simulate(None, entry["factory"](entry["p_target"]),
                                windows)
                walker = (lookahead_walker() if walker_kind == "argmax"
                          else greedy_walker())
                run = simulate(None, entry["factory"](entry["p_target"]),
                               windows, price=price, walker=walker)
                ratios[prompt] = {
                    "ratio": run["us_per_token"] / base["us_per_token"]}
            values.append(median_pct(receipt, ratios))
        by_tier[tier8] = statistics.fmean(values)
    best = max(by_tier, key=by_tier.get)
    return best, by_tier


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
    ap.add_argument("--forms", default=",".join(CURVE_FORMS))
    ap.add_argument("--tier-form", default="per_drafting_round")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/posttight.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 F3 arm (a) and F4 items 3, 4, 8  "
          "zero GPU")
    curves, _ = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)
    forms = [f for f in args.forms.split(",") if f]

    print("\n## the measured launch law, on form %s" % args.tier_form)
    print("  coefficient %.1f us, F4 regression on 572b2cc4, rival replicate "
          "-1329.0" % LAUNCH_COEF)
    print("  %-14s %s" % ("step into", " ".join(
        "%9d" % m for m in range(2, MAX_DEPTH + 2))))
    for name in ("base", "uniform", "log_onePass67", "log_shipped"):
        steps = step_table(VARIANTS[name](curves[args.tier_form]))
        print("  %-14s %s" % (name, " ".join("%9.1f" % v for v in steps)))
    base_steps = step_table(curves[args.tier_form])
    for name in ("log_onePass67", "log_shipped"):
        steps = step_table(VARIANTS[name](curves[args.tier_form]))
        print("  %-14s boundary-4 ratio step(6)/step(5) %.4f -> %.4f"
              % (name, base_steps[4] / base_steps[3],
                 steps[4] / steps[3]))

    # F4 item 8: the fitted rival gets its scalar re-fitted on every curve it
    # is judged on. A frozen tier would be a straw man.
    print("\n## F4 item 8: the width-8 tier fitted on each curve (form %s)"
          % args.tier_form)
    tier8_fits = {}
    for name in VARIANTS:
        curve = VARIANTS[name](curves[args.tier_form])
        best, by_tier = fit_tier8(cache, seeds, curve, receipt, args.windows)
        tier8_fits[name] = {"best_tier8": best, "by_tier": by_tier,
                            "at_one": by_tier[1.0]}
        print("  %-14s best tier8 %-6.2f -> %+7.4f   tier8 1.0 (= pb6) "
              "-> %+7.4f" % (name, best, by_tier[best], by_tier[1.0]))

    results = {}
    for form in forms:
        base_lopo = lopo_curves(masses, form) if form in FORMS else None
        print("\n## form %s" % form)
        print("  %-14s %s" % ("curve", " ".join("%-15s" % a for a in ARMS)))
        for name, make in VARIANTS.items():
            curve = make(curves[form])
            lopo = ({p: make(c) for p, c in base_lopo.items()}
                    if base_lopo is not None else None)
            install_pb68(SHIPPED_TIER,
                         tier8_fits[name if name in tier8_fits else "base"][
                             "best_tier8"])
            row = {}
            for cell in ARMS:
                row[cell] = arm_row(cache, seeds, curve, lopo, receipt,
                                    args.windows, cell)
            results[(form, name)] = row
            print("  %-14s %s" % (name, " ".join(
                "%+7.4f/%s" % (
                    row[a]["in_sample"][0],
                    "%+6.4f" % row[a]["curve_lopo"][0]
                    if row[a]["curve_lopo"] else "  n/a ")
                for a in ARMS)))

    print("\n## F4 item 4: does the argmax want a depth the cap forbids? "
          "(form %s)" % args.tier_form)
    probes = {}
    for name, make in VARIANTS.items():
        probe = cap_probe(cache, seeds, make(curves[args.tier_form]),
                          args.windows)
        probes[name] = probe
        print("  %-14s rounds %7d  wants past cap %.4f  mean depth capped "
              "%.4f uncapped %.4f"
              % (name, probe["rounds"], probe["wants_past_cap_share"],
                 probe["mean_capped_depth"], probe["mean_uncapped_depth"]))
        print("                 uncapped depth share %s" % " ".join(
            "%.3f" % v for v in probe["uncapped_depth_share"]))

    print("\n## the F3 gate: a uniform per-round shift changes no step")
    for form in forms:
        base_row = results[(form, "base")]
        shift_row = results[(form, "uniform")]
        for cell in ("E_pb6", "D_curvelook", "G_pb68"):
            before = (base_row[cell]["curve_lopo"]
                      or base_row[cell]["in_sample"])[0]
            after = (shift_row[cell]["curve_lopo"]
                     or shift_row[cell]["in_sample"])[0]
            print("  %-20s %-14s %+7.4f -> %+7.4f  (%+7.4f pp)"
                  % (form, cell, before, after, after - before))

    print("\n## F4 item 3: cell D and the fitted rivals on the post-tight "
          "curve")
    for form in forms:
        for name in ("base", "log_onePass67", "log_shipped"):
            row = results[(form, name)]
            def value(cell):
                return (row[cell]["curve_lopo"]
                        or row[cell]["in_sample"])[0]
            print("  %-20s %-14s D %+7.4f  E %+7.4f  F %+7.4f  G %+7.4f  "
                  "H %+7.4f   D-E %+7.4f  D-G %+7.4f"
                  % (form, name, value("D_curvelook"), value("E_pb6"),
                     value("F_pb6look"), value("G_pb68"),
                     value("H_pb68look"),
                     value("D_curvelook") - value("E_pb6"),
                     value("D_curvelook") - value("G_pb68")))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument", "gpu_used": False,
        "windows": args.windows, "seeds": seeds, "receipt": receipt["id"],
        "launch_coef": LAUNCH_COEF, "uniform_shift_us": UNIFORM_SHIFT_US,
        "column_tables": {k: {str(w): c for w, c in v.items()}
                          for k, v in COLUMN_TABLES.items()},
        "steps": {name: step_table(VARIANTS[name](curves[args.tier_form]))
                  for name in VARIANTS},
        "tier8_fits": {k: {"best_tier8": v["best_tier8"],
                           "at_one": v["at_one"],
                           "by_tier": {"%.2f" % t: s
                                       for t, s in v["by_tier"].items()}}
                       for k, v in tier8_fits.items()},
        "cap_probe": probes,
        "arms": {"%s|%s|%s" % (form, name, cell): row[cell]
                 for (form, name), row in results.items()
                 for cell in row},
    }, indent=2, default=list) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
