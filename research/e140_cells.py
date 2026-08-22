#!/usr/bin/env python3
"""E140 item 1, 2, 3 and 4: the 2x2 of price against walk, plus `pb6`.

`harness=local instrument`. Zero GPU.

    cell            price               walk      expectation
    A  ship         flat 0.18           greedy    0.0000, the anchor
    B  rankedprice  measured curve      greedy    about -2.85, calibration
    C  flatlook     flat 0.18           argmax    MUST be 0.0000, the gate
    D  curvelook    measured curve      argmax    the hypothesis
    E  pb6          flat + one barrier  greedy    +2.4683, the bar to beat

Two held-out disciplines are reported side by side, because the cells do not
all carry the same kind of free parameter.

`curve_lopo` is the discipline that matters for cells B and D. The measured
curve is inverted from the same eight ranked prompts the replayed median is
computed over, so an in-sample curve lets a cell read its own answer. Each
prompt is therefore scored under a curve refitted WITHOUT that prompt, and
both the policy and the cost model for that prompt use the refitted curve.

`tier_lopo` is the discipline E134 used for cell E. There the free parameter
is the tier factor, chosen from a grid without the scored prompt. Cells C and
D have no free parameter at all, so their grid holds one point and the
selection is degenerate by construction; that is reported, not hidden.

The transfer fit is shared across curve forms. `fit_transfer` bisects on
`mean_depth`, which the shipped flat price decides without reading any round
cost, so the fitted `delta` cannot depend on the curve. `--check-transfer`
proves that numerically instead of by reading.

Usage:
  python3 e140_cells.py --json e140-artifacts/cells.json
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
from e128_price import (  # noqa: E402
    MAX_DEPTH, RANKED_PROMPTS, load_board_receipt, pooled_positions,
    shift_p_vector,
)
from e134_item2_refit import (  # noqa: E402
    FORMS, MEASURED, fit_one, measured_curve,
)
from e134_rung2 import (  # noqa: E402
    VectorSampler, build_legs, fit_transfer, median_pct, simulate, walk,
)
from e140_lookahead import (  # noqa: E402
    CURVE_FORMS, SHIPPED_CLIFF, SHIPPED_TIER, curve_price, flat_price,
    load_curves, pb6_price, walk_argmax,
)
from e134_rung3 import boundary_price  # noqa: E402

MASSES_PATH = HERE / "e134-artifacts/item2-measured-curve.json"
TIER_GRID = (1.0, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70,
             2.0301, 4.2689)

# The advisor's published reference points, for calibration only.
REFERENCE = {"B_rankedprice": -2.8508, "E_pb6": 2.4683, "oracle": 9.1617}


class Tally:
    """Per-round record of what the lookahead rule did to the greedy rule."""

    def __init__(self):
        self.rounds = 0
        self.disagreements = 0
        self.deeper = 0
        self.shallower = 0
        self.deeper_capability = 0
        self.shallower_capability = 0
        self.deeper_extra_accepted = 0
        self.same_capability = 0
        self.same_rounds = 0
        self.pairs = {}
        self.cap_bound = 0
        self.deeper_hits = 0
        # Round-start estimator state, which is what decides whether a
        # boundary is crossable at all. A position the policy never reaches
        # is never updated, so it stays frozen at `EMA_PRIOR`.
        self.ema_sum = [0.0] * MAX_DEPTH
        self.reach_sum = [0.0] * (MAX_DEPTH + 1)
        # First-order price of the changed rounds only, with the trajectory
        # held fixed. `greedy` is what the shipped walk would have paid on
        # exactly those rounds; `best` is what lookahead paid instead.
        self.deeper_us_greedy = 0.0
        self.deeper_us_best = 0.0
        self.deeper_tokens_greedy = 0
        self.deeper_tokens_best = 0
        self.same_us = 0.0
        self.same_tokens = 0
        self.round_us = None

    def observe_state(self, ema, reach):
        for index in range(MAX_DEPTH):
            self.ema_sum[index] += ema[index]
        for index, value in enumerate(reach):
            self.reach_sum[index] += value

    def observe(self, greedy, best, capability, cap):
        self.rounds += 1
        if best >= cap:
            self.cap_bound += 1
        if greedy == best:
            self.same_rounds += 1
            self.same_capability += capability
            if self.round_us is not None:
                self.same_us += self.round_us(best + 1)
                self.same_tokens += 1 + min(capability, best)
            return
        if self.round_us is not None:
            self.deeper_us_greedy += self.round_us(greedy + 1)
            self.deeper_us_best += self.round_us(best + 1)
            self.deeper_tokens_greedy += 1 + min(capability, greedy)
            self.deeper_tokens_best += 1 + min(capability, best)
        self.disagreements += 1
        key = "%d->%d" % (greedy, best)
        self.pairs[key] = self.pairs.get(key, 0) + 1
        if best > greedy:
            self.deeper += 1
            self.deeper_capability += capability
            self.deeper_extra_accepted += (min(capability, best)
                                           - min(capability, greedy))
            if capability > greedy:
                self.deeper_hits += 1
        else:
            self.shallower += 1
            self.shallower_capability += capability

    def summary(self):
        def mean(total, count):
            return total / count if count else float("nan")
        return {
            "rounds": self.rounds,
            "disagreements": self.disagreements,
            "deeper_round_count": self.deeper,
            "shallower_round_count": self.shallower,
            "unchanged_round_count": self.same_rounds,
            "mean_capability_deeper": mean(self.deeper_capability,
                                           self.deeper),
            "mean_capability_shallower": mean(self.shallower_capability,
                                              self.shallower),
            "mean_capability_unchanged": mean(self.same_capability,
                                              self.same_rounds),
            "extra_accepted_per_deeper_round": mean(
                self.deeper_extra_accepted, self.deeper),
            "p_first_declined_draft_accepted_deeper": mean(self.deeper_hits,
                                                           self.deeper),
            "cap_bound_share": mean(self.cap_bound, self.rounds),
            "transitions": dict(sorted(self.pairs.items())),
            "mean_round_start_ema": [v / self.rounds if self.rounds else
                                     float("nan") for v in self.ema_sum],
            "mean_reach": [v / self.rounds if self.rounds else float("nan")
                           for v in self.reach_sum],
            "changed_us_per_token_greedy": mean(self.deeper_us_greedy,
                                                self.deeper_tokens_greedy),
            "changed_us_per_token_lookahead": mean(self.deeper_us_best,
                                                   self.deeper_tokens_best),
            "unchanged_us_per_token": mean(self.same_us, self.same_tokens),
        }


def reach_vector(ema, margin, offer):
    """`reach_d` at every depth, on the shipped inputs, for the diagnostic."""
    out, running = [], 1.0
    have_margin = not math.isnan(margin)
    for depth in range(MAX_DEPTH):
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        running *= p
        out.append(running)
    return out


def lookahead_walker(tally=None):
    """`simulate`'s depth rule, replaced by the argmax over feasible depths."""
    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        if adjust is not None or force is not None:
            raise SystemExit("E140 prices no `adjust` or `force` arm")
        best = walk_argmax(ema, margin, offer, price=price)
        if tally is not None:
            greedy = walk(ema, margin, offer, price=price)
            cap = min(min(offer, MAX_DEPTH), 7)
            tally.observe(greedy, best, ctx["capability"], cap)
            tally.observe_state(ema, reach_vector(ema, margin, offer))
        return best
    return chooser


def greedy_walker(tally=None):
    """The shipped walk, with the same tally hook, so the two are comparable."""
    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        depth = walk(ema, margin, offer, adjust, ctx, force, price)
        if tally is not None:
            cap = min(min(offer, MAX_DEPTH), 7)
            tally.observe(depth, depth, ctx["capability"], cap)
            tally.observe_state(ema, reach_vector(ema, margin, offer))
        return depth
    return chooser


# `F_pb6look` is not part of the assigned 2x2. It is added because cell D
# turns out to fire on no round at all, and the reason is a feedback loop
# between the price and the estimator rather than the greedy assumption. The
# estimator only updates a position the policy actually reaches, so a price
# that stops the walk at depth 4 freezes positions 5 to 7 at their prior and
# destroys the evidence the argmax would need to cross the cliff. `pb6` is the
# one shipped price that keeps those positions alive, so it is the only price
# on which a lookahead rule can fire at all.
CELLS = {
    "A_ship": {"price": "flat", "walk": "greedy"},
    "B_rankedprice": {"price": "curve", "walk": "greedy"},
    "C_flatlook": {"price": "flat", "walk": "argmax"},
    "D_curvelook": {"price": "curve", "walk": "argmax"},
    "E_pb6": {"price": "pb6", "walk": "greedy"},
    "F_pb6look": {"price": "pb6", "walk": "argmax"},
}


def price_for(kind, curve):
    if kind == "flat":
        return flat_price()
    if kind == "pb6":
        return pb6_price(SHIPPED_TIER, SHIPPED_CLIFF)
    return curve_price(curve)


def install(curve):
    e128_price.CURVE = curve


def transfer_cache(legs, windows, fit_windows, seeds) -> dict:
    """One `(delta, p_target)` per prompt and seed, shared by every curve."""
    cache = {}
    for seed in seeds:
        for prompt, spec in RANKED_PROMPTS.items():
            fixture = spec["fixture"]
            chosen = [legs[name] for name in fixture if name in legs]
            if len(chosen) != len(fixture):
                raise SystemExit("missing fixture legs for %s" % prompt)
            p_fixture = pooled_positions(chosen)
            rounds = [r for leg in chosen for r in leg["rounds_detail"]]

            def factory(p_target, rounds=rounds, p_fixture=p_fixture,
                        seed=seed):
                return VectorSampler(rounds, p_fixture, p_target, seed=seed)

            delta, _, _ = fit_transfer(factory, p_fixture, spec["depth"],
                                       fit_windows)
            cache[(seed, prompt)] = {
                "factory": factory, "delta": delta,
                "p_target": shift_p_vector(p_fixture, delta)}
    return cache


def run_cell(cache, seed, prompt, cell, curve, windows, tally=None):
    """One prompt's candidate-time ratio against the shipped arm."""
    entry = cache[(seed, prompt)]
    install(curve)
    base = simulate(None, entry["factory"](entry["p_target"]), windows)
    spec = CELLS[cell]
    price = price_for(spec["price"], curve)
    walker = (lookahead_walker(tally) if spec["walk"] == "argmax"
              else greedy_walker(tally))
    run = simulate(None, entry["factory"](entry["p_target"]), windows,
                   price=price, walker=walker)
    return {"ratio": run["us_per_token"] / base["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"],
            "depth_counts": run["depth_counts"],
            "base_depth_counts": base["depth_counts"],
            "base_us_per_token": base["us_per_token"],
            "us_per_token": run["us_per_token"]}


def tier_ratios(cache, seed, curve, tier, windows):
    install(curve)
    price = boundary_price(tier, SHIPPED_CLIFF)[:2]
    out = {}
    for prompt in RANKED_PROMPTS:
        entry = cache[(seed, prompt)]
        base = simulate(None, entry["factory"](entry["p_target"]), windows)
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       price=price)
        out[prompt] = {"ratio": run["us_per_token"] / base["us_per_token"],
                       "mean_depth": run["mean_depth"],
                       "accept_rate": run["accept_rate"]}
    return out


def load_masses(path: pathlib.Path = MASSES_PATH) -> dict:
    """E134 item 2's replayed width masses, with integer width keys restored.

    JSON has no integer keys, so a round trip turns `by_width[6]` into
    `by_width["6"]` and every design column silently reads zero. That makes
    the normal equations singular rather than wrong, which is the only reason
    this was caught by an exception instead of by a plausible number.
    """
    blob = json.loads(path.read_text())["width_masses"]
    return {prompt: dict(entry,
                         by_width={int(w): v
                                   for w, v in entry["by_width"].items()})
            for prompt, entry in blob.items()}


def lopo_curves(masses, form) -> dict:
    """One refitted curve per held-out prompt, plus the in-sample curve."""
    prompts = [p for p in MEASURED if p in masses]
    out = {}
    for held in prompts:
        kept = [p for p in prompts if p != held]
        fit = fit_one(masses, kept, [MEASURED[p]["delta_us"] for p in kept],
                      form)
        out[held] = measured_curve(fit, form)
    return out


def summarise(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


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
    ap.add_argument("--skip-tier-lopo", action="store_true")
    ap.add_argument("--check-transfer", action="store_true")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/cells.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 cells  zero GPU")
    curves, best_form = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    print("## attachment gate")
    print("legs %d ; attached %d ; accept mismatches %d ; margin mismatches "
          "%d ; unmatched %d" % (gate["legs"], gate["attached"],
                                 gate["accept_mismatch"],
                                 gate["margin_mismatch"], gate["unmatched"]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")

    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)

    if args.check_transfer:
        print("\n## transfer-fit curve independence")
        install(curves["pre_arm"])
        first = {p: cache[(seeds[0], p)]["delta"] for p in RANKED_PROMPTS}
        again = transfer_cache(legs, args.windows, args.fit_windows,
                               [seeds[0]])
        install(curves["per_drafting_round"])
        third = transfer_cache(legs, args.windows, args.fit_windows,
                               [seeds[0]])
        worst = max(abs(again[(seeds[0], p)]["delta"]
                        - third[(seeds[0], p)]["delta"])
                    for p in RANKED_PROMPTS)
        print("largest delta difference across two installed curves  %.3e"
              % worst)
        print("cached deltas: %s" % " ".join(
            "%s=%.4f" % (p, v) for p, v in first.items()))
        if worst > 1e-12:
            raise SystemExit("the transfer fit reads the curve; cache is void")

    results = {}
    tallies = {}
    for form in CURVE_FORMS:
        curve = curves[form]
        lopo = lopo_curves(masses, form) if form in FORMS else None
        for cell in CELLS:
            in_sample, held_out = [], []
            per_prompt = {}
            depths = []
            tally = Tally()
            tally.round_us = e128_price.ranked_round_us
            for seed in seeds:
                ratios = {}
                for prompt in RANKED_PROMPTS:
                    row = run_cell(cache, seed, prompt, cell, curve,
                                   args.windows, tally)
                    ratios[prompt] = row
                    per_prompt.setdefault(prompt, []).append(row["ratio"])
                    depths.append(RANKED_PROMPTS[prompt]["weight"]
                                  * row["mean_depth"])
                in_sample.append(median_pct(receipt, ratios))
                if lopo is not None:
                    folded = {}
                    for prompt in RANKED_PROMPTS:
                        folded[prompt] = run_cell(cache, seed, prompt, cell,
                                                  lopo[prompt], args.windows)
                    held_out.append(median_pct(receipt, folded))
            key = (form, cell)
            results[key] = {
                "mean_depth": sum(depths) / len(seeds),
                "in_sample": summarise(in_sample),
                "in_sample_values": in_sample,
                "curve_lopo": summarise(held_out) if held_out else None,
                "curve_lopo_values": held_out,
                "per_prompt_ratio": {p: statistics.fmean(v)
                                     for p, v in per_prompt.items()},
            }
            tallies[key] = tally.summary()
            print("  %-20s %-14s in-sample %+8.4f  curve-lopo %s"
                  % (form, cell, results[key]["in_sample"][0],
                     "%+8.4f" % results[key]["curve_lopo"][0]
                     if held_out else "    n/a"))

    tier_lopo = {}
    if not args.skip_tier_lopo:
        for form in CURVE_FORMS:
            curve = curves[form]
            by_seed = {seed: {tier: tier_ratios(cache, seed, curve, tier,
                                                args.windows)
                              for tier in TIER_GRID}
                       for seed in seeds}
            in_sample = max(
                statistics.fmean([median_pct(receipt, by_seed[s][tier])
                                  for s in seeds])
                for tier in TIER_GRID)
            per_seed, chosen = [], {}
            for seed in seeds:
                ratios = {}
                for prompt in RANKED_PROMPTS:
                    best, best_tier = None, None
                    for tier in TIER_GRID:
                        others = {p: r for p, r in by_seed[seed][tier].items()
                                  if p != prompt}
                        value = median_pct(receipt, others)
                        if best is None or value > best:
                            best, best_tier = value, tier
                    chosen.setdefault(prompt, []).append(best_tier)
                    ratios[prompt] = by_seed[seed][best_tier][prompt]
                per_seed.append(median_pct(receipt, ratios))
            mean, sd = summarise(per_seed)
            fixed = summarise([median_pct(receipt, by_seed[s][SHIPPED_TIER])
                               for s in seeds])
            tier_lopo[form] = {"in_sample": in_sample, "held_out": mean,
                               "held_out_sd": sd, "per_seed": per_seed,
                               "chosen": chosen,
                               "fixed_1.45": fixed}
            print("  tier LOPO %-20s in-sample %+8.4f  held-out %+8.4f "
                  "(sd %.4f)  fixed 1.45 %+8.4f"
                  % (form, in_sample, mean, sd, fixed[0]))

    print("\n## the 2x2 plus pb6, replayed ranked median percent")
    print("%-20s %-14s %10s %8s %12s %8s %10s %10s" % (
        "curve form", "cell", "in-sample", "sd", "curve-lopo", "sd",
        "disagree", "mean depth"))
    for form in CURVE_FORMS:
        for cell in CELLS:
            entry = results[(form, cell)]
            lopo = entry["curve_lopo"]
            print("%-20s %-14s %+10.4f %8.4f %12s %8s %10d %10.4f" % (
                form, cell, entry["in_sample"][0], entry["in_sample"][1],
                "%+.4f" % lopo[0] if lopo else "n/a",
                "%.4f" % lopo[1] if lopo else "",
                tallies[(form, cell)]["disagreements"],
                entry["mean_depth"]))

    print("\n## cell C gate inside the replayer")
    print("%-20s %10s %10s %10s" % (
        "curve form", "rounds", "disagree", "median %"))
    gate_total = 0
    for form in CURVE_FORMS:
        entry = tallies[(form, "C_flatlook")]
        gate_total += entry["disagreements"]
        print("%-20s %10d %10d %+10.4f" % (
            form, entry["rounds"], entry["disagreements"],
            results[(form, "C_flatlook")]["in_sample"][0]))

    print("\n## item 2, what lookahead does to the greedy schedule")
    print("   `shallow` is 0 in every row and that is a theorem, not a "
          "measurement. The greedy walk stops at the first depth `g` whose "
          "next step lowers `E/C`, so `E/C` is strictly increasing on "
          "`0..g` and no depth below `g` can be the argmax. Lookahead can "
          "only ever go DEEPER, so the assignment's suppression audit has an "
          "empty population by construction.")
    print("%-20s %-14s %8s %8s %10s %10s %10s %8s" % (
        "curve form", "cell", "deeper", "shallow", "cap(deep)", "cap(same)",
        "extra tok", "P(hit)"))
    for form in CURVE_FORMS:
        for cell in ("D_curvelook", "F_pb6look"):
            entry = tallies[(form, cell)]
            print("%-20s %-14s %8d %8d %10.4f %10.4f %10.4f %8.4f" % (
                form, cell, entry["deeper_round_count"],
                entry["shallower_round_count"],
                entry["mean_capability_deeper"],
                entry["mean_capability_unchanged"],
                entry["extra_accepted_per_deeper_round"],
                entry["p_first_declined_draft_accepted_deeper"]))
            if entry["transitions"]:
                print("%-35s transitions %s" % ("", entry["transitions"]))
                print("%-35s us/token on the changed rounds: greedy %.1f -> "
                      "lookahead %.1f (%+.2f %%), unchanged rounds %.1f"
                      % ("", entry["changed_us_per_token_greedy"],
                         entry["changed_us_per_token_lookahead"],
                         100.0 * (entry["changed_us_per_token_lookahead"]
                                  / entry["changed_us_per_token_greedy"]
                                  - 1.0),
                         entry["unchanged_us_per_token"]))

    print("\n## estimator state, mean round-start EMA by position (%s)"
          % best_form)
    print("   A position the policy never reaches is never updated. The "
          "prior is 0.85 * 0.98^i, so a frozen entry reads near it.")
    print("%-14s %s" % ("cell", " ".join("%7s" % ("p%d" % i)
                                         for i in range(MAX_DEPTH))))
    for cell in CELLS:
        entry = tallies[(best_form, cell)]
        print("%-14s %s" % (cell, " ".join(
            "%7.4f" % v for v in entry["mean_round_start_ema"])))
    print("%-14s %s" % ("EMA_PRIOR", " ".join(
        "%7.4f" % (0.85 * 0.98 ** i) for i in range(MAX_DEPTH))))
    print("\n   mean reach by depth, the quantity the walk compares")
    for cell in CELLS:
        entry = tallies[(best_form, cell)]
        print("%-14s %s" % (cell, " ".join(
            "%7.4f" % v for v in entry["mean_reach"][:MAX_DEPTH])))

    print("\n## item 4, cap interaction. `segmentedVerifyDepthCap` is 7 and "
          "the walk cap is min(offer, 8, 7)")
    print("%-20s %-14s %12s" % ("curve form", "cell", "share at cap"))
    for form in CURVE_FORMS:
        for cell in CELLS:
            print("%-20s %-14s %12.4f" % (
                form, cell, tallies[(form, cell)]["cap_bound_share"]))

    print("\n## item 3, curve-form robustness of cell D")
    values = [results[(form, "D_curvelook")]["in_sample"][0]
              for form in CURVE_FORMS]
    lopo_values = [results[(form, "D_curvelook")]["curve_lopo"][0]
                   for form in CURVE_FORMS
                   if results[(form, "D_curvelook")]["curve_lopo"]]
    print("in-sample spread across four forms  %.4f pp (min %+0.4f max %+0.4f)"
          % (max(values) - min(values), min(values), max(values)))
    if lopo_values:
        print("curve-lopo spread across three fitted forms  %.4f pp"
              % (max(lopo_values) - min(lopo_values)))

    print("\n## calibration against the advisor's published reference points")
    for cell, want in REFERENCE.items():
        if cell not in CELLS:
            continue
        got = results[(best_form, cell)]["in_sample"][0]
        print("%-16s published %+8.4f   here %+8.4f   gap %+8.4f"
              % (cell, want, got, got - want))

    headline_form = best_form
    headline = results[(headline_form, "D_curvelook")]
    bar = results[(headline_form, "E_pb6")]
    print("\n## verdict, on the E134 best form %s" % headline_form)
    print("e140_replayed_ranked_median_pct   %+.4f  (curve-lopo %s)"
          % (headline["in_sample"][0],
             "%+.4f" % headline["curve_lopo"][0]
             if headline["curve_lopo"] else "n/a"))
    print("cell E pb6 on the same discipline  %+.4f  (curve-lopo %s)"
          % (bar["in_sample"][0],
             "%+.4f" % bar["curve_lopo"][0] if bar["curve_lopo"] else "n/a"))
    print("advance bar %+.4f ; close bar %+.4f" % (REFERENCE["E_pb6"], 1.50))

    payload = {
        "harness": "local instrument", "gpu_used": False,
        "receipt": receipt["id"], "receipt_score": receipt["score"],
        "windows": args.windows, "fit_windows": args.fit_windows,
        "seeds": seeds, "attachment_gate": gate,
        "best_form": best_form, "tier_grid": list(TIER_GRID),
        "cells": {"%s|%s" % (form, cell): {
            "in_sample_mean": results[(form, cell)]["in_sample"][0],
            "in_sample_sd": results[(form, cell)]["in_sample"][1],
            "in_sample_values": results[(form, cell)]["in_sample_values"],
            "curve_lopo_mean": (results[(form, cell)]["curve_lopo"][0]
                                if results[(form, cell)]["curve_lopo"]
                                else None),
            "curve_lopo_sd": (results[(form, cell)]["curve_lopo"][1]
                              if results[(form, cell)]["curve_lopo"]
                              else None),
            "curve_lopo_values": results[(form, cell)]["curve_lopo_values"],
            "per_prompt_ratio": results[(form, cell)]["per_prompt_ratio"],
            "f83_weighted_mean_depth": results[(form, cell)]["mean_depth"],
            "tally": tallies[(form, cell)],
        } for form in CURVE_FORMS for cell in CELLS},
        "tier_lopo": tier_lopo,
        "reference": REFERENCE,
        "e140_cellC_depth_disagreements_replayer": gate_total,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
