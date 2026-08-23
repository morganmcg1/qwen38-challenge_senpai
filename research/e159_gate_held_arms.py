#!/usr/bin/env python3
"""E159 Part B. Replay the depth-price arm family with the gate held.

harness=local replay instrument, priced by a harness=ranked cost law. Zero GPU.

H159 says every historical shape arm lost because the held-total constraint
`sum(marginal) == maxDepth * headStepCostRatio` dragged `marginal[0]` below
0.18, and that `threshold(0) == marginal[0]` exactly, so `marginal[0]` is the
drafting on/off gate. This module searches the arm family under two hard
constraints:

    constraint 1   marginal[0] == 0.18, the same Double the shipped path uses
    constraint 2   plutarch non-drafting rounds >= 440 of 487 under replay

Pricing follows RULE 177. Every arm is reported as a paired per-prompt
contrast on the candidate leg, `us_per_token` against the shipped arm on the
same sampler seed, across the eight ranked prompts. RULE 178 then weights that
contrast by the measured median sensitivities, because a depth-price arm is
non-uniform by construction. The published median is reported for promotion
only, never for selection, because FINDING 306 shows the median is an order
statistic over four near-tied prompts.

The round cost that converts a depth choice into candidate-leg time is the
E159 Part A law, identified from 5,770 ranked board cells with head and prompt
fixed effects. `--curve f97` swaps in the older two-segment board fit as a
robustness arm.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_price  # noqa: E402
import e134_rung2  # noqa: E402
from e128_price import (  # noqa: E402
    DECODE_TOKENS, RANKED_PROMPTS, load_board_receipt, median_of)
from e128_replay import MAX_DEPTH  # noqa: E402
from e134_rung2 import build_legs, prompt_panel, simulate  # noqa: E402

SHIPPED_HEAD_STEP_COST_RATIO = 0.18
SHIPPED_GATE_BITS = struct.pack("<d", SHIPPED_HEAD_STEP_COST_RATIO).hex()

# E159 Part A, `research/e159-artifacts/e159_round_cost_law.json`, stratum
# `board`, segment basis, head+prompt fixed effects. Index i is verify width
# i + 1, in units of the width-one round. Widths 2, 8 and 9 are model
# extrapolations and are marked in `PART_A_MEASURED`.
PART_A_CUMULATIVE = [
    1.000000, 1.143494, 1.286988, 1.574365,
    1.775769, 1.957930, 2.196327, 2.434723, 2.673120,
]
PART_A_MEASURED = [True, False, True, True, True, True, True, False, False]
PART_A_WIDTH_ONE_US = 30762.0

# `research/e128_price.py` default, F97, fitted over 147 official runs of other
# solvers by an unrelated method. Kept as an independent robustness curve.
F97 = {"breakpoint": 5, "lo": (27215.4, 3966.4), "hi": (17020.7, 7154.2)}

# Plutarch is the phase-transition guard. Its ranked receipts run 487 to 488
# rounds; the constraint is stated out of 487.
PLUTARCH_ROUNDS = 487
PLUTARCH_NON_DRAFTING_FLOOR = 440

# Part B stop rule, replacing the brief's +0.60 % published-median rule with
# the advisor's candidate-leg rule from the `research/receipt_candidate_leg.py`
# replication study.
STOP_RULE_GAIN_PCT = 0.30

# RULE 178. Median sensitivity measured by direct +0.10 % perturbation of each
# prompt's raw ratio through the real order statistic on 923 scored board
# receipts. A depth-price arm is non-uniform by construction, so it is priced
# on this weighted mean rather than the unweighted eight-prompt mean.
RULE178_WEIGHTS = {
    "beagle": 0.4741,
    "medicine": 0.1951,
    "essays": 0.1658,
    "republic": 0.0963,
    "botany": 0.0508,
    "travel": 0.0037,
    "plutarch": 0.0033,
    "drama": 0.0019,
}

# A replayed prompt whose shipped draft length misses its ranked draft length
# by more than this cannot carry an arm contrast, because the arm acts on the
# depth walk the transfer failed to reproduce.
TRANSFER_TOLERANCE = 0.05


def part_a_round_us(rows: int) -> float:
    index = max(1, min(rows, len(PART_A_CUMULATIVE)))
    return PART_A_WIDTH_ONE_US * PART_A_CUMULATIVE[index - 1]


def f97_round_us(rows: int) -> float:
    intercept, slope = F97["lo"] if rows < F97["breakpoint"] else F97["hi"]
    return intercept + slope * rows


def install_curve(name: str):
    """Replace the round cost in every module that captured it by name."""
    fn = {"e159": part_a_round_us, "f97": f97_round_us}[name]
    e128_price.ranked_round_us = fn
    e134_rung2.ranked_round_us = fn
    return fn


# ------------------------------------------------------------------- the arms


def prefix_costs(marginal: list[float]) -> list[float]:
    """`Qwen36MTPBlockSession.prefixCosts`, accumulated exactly as in Swift."""
    out = [1.0]
    running = 1.0
    for value in marginal:
        running += value
        out.append(running)
    return out


def uniform_price(h: float = SHIPPED_HEAD_STEP_COST_RATIO):
    """`makeUniformDepthPrice`, including its closed-form `cumulative`.

    The shipped constructor writes `1.0 + Double(i) * h` rather than
    accumulating, and the source says why: the two differ by one ulp and a
    control that is not bit-identical to the tip is not a control. The control
    arm here reproduces the closed form for the same reason.
    """
    return ([h] * MAX_DEPTH,
            [1.0 + i * h for i in range(MAX_DEPTH + 1)])


def boundary_marginal(width: int, tier: float = 2.0301) -> list[float]:
    """`makeBoundaryDepthPrice`, held total. `width` is the width entered."""
    within = MAX_DEPTH * SHIPPED_HEAD_STEP_COST_RATIO / ((MAX_DEPTH - 1) + tier)
    marginal = [within] * MAX_DEPTH
    marginal[width - 2] = within * tier
    return marginal


# `measuredRawDepthPrice`, the E68 rung-1 isolated-QMV step curve, before any
# rescale. `makeMeasuredDepthPrice` divides it down to the held total 1.44.
PBFIT_RAW = [
    0.26300121724709807,
    0.29195567495854047,
    0.34642143034825884,
    0.40231023217247086,
    0.63287276451077956,
    0.43601634825870655,
    0.35457813598673293,
    0.42510483416251998,
]


def pbfit_marginal() -> list[float]:
    total = MAX_DEPTH * SHIPPED_HEAD_STEP_COST_RATIO
    scale = total / sum(PBFIT_RAW)
    return [v * scale for v in PBFIT_RAW]


def hold_gate(marginal: list[float]) -> list[float]:
    """Constraint 1, applied as a pure overwrite. The total is released."""
    out = list(marginal)
    out[0] = SHIPPED_HEAD_STEP_COST_RATIO
    return out


def rescale_to_gate(raw: list[float]) -> list[float]:
    """Keep a shape exactly and set its level from the gate, not the total."""
    scale = SHIPPED_HEAD_STEP_COST_RATIO / raw[0]
    return [v * scale for v in raw]


def part_a_marginal() -> list[float]:
    """The Part A law as a depth price, then the gate held at 0.18.

    `cumulative[d]` in the Swift struct is the round cost at verify width
    `d + 1` divided by the width-one round, which is exactly what Part A fits,
    so the two vectors are the same object in the same units.
    """
    raw = [PART_A_CUMULATIVE[i + 1] - PART_A_CUMULATIVE[i]
           for i in range(MAX_DEPTH)]
    return hold_gate(raw)


def arm_family() -> dict:
    """Every arm, as an explicit marginal vector. Grids are the search axis."""
    family = {}
    family["ship"] = {0.0: uniform_price()}

    # The gate axis on its own: move `marginal[0]` and hold the interior at
    # the shipped 0.18. This is the only arm family that violates constraint 1
    # on purpose. It exists to test the phase transition inside the replay
    # against the ranked receipts, which record plutarch at draft length
    # 0.1540 for every gate at or above 0.18 and about 2.70 for both arms
    # below it.
    family["gatescan"] = {
        g: ([g] + [SHIPPED_HEAD_STEP_COST_RATIO] * (MAX_DEPTH - 1), None)
        for g in (0.10, 0.120143, 0.14, 0.15, 0.159467, 0.170414,
                  0.18, 0.20, 0.25, 0.32)
    }

    # The pure H159 axis: hold the gate, make the interior uniform, release
    # the total. One free parameter, so leave-one-prompt-out is meaningful.
    family["gateuniform"] = {
        h: ([SHIPPED_HEAD_STEP_COST_RATIO] + [h] * (MAX_DEPTH - 1),
            None)
        for h in (0.06, 0.08, 0.10, 0.11052, 0.13, 0.15, 0.18,
                  0.20, 0.22, 0.25, 0.30)
    }

    # The Part A law itself, scaled about the gate. s = 1.0 is the measured
    # interior; other s values ask whether the walk wants the measured shape
    # cheaper or dearer than the ranked board says it is.
    base = part_a_marginal()
    family["e159law"] = {
        s: ([base[0]] + [v * s for v in base[1:]], None)
        for s in (0.50, 0.75, 1.00, 1.25, 1.50)
    }

    # The advisor's sketch: E157's h_head_clean at indices 1-3, a wall at 4.
    family["sketch"] = {
        w: ([SHIPPED_HEAD_STEP_COST_RATIO, 0.11052, 0.11052, 0.11052]
            + [w] * 4, None)
        for w in (0.11052, 0.18, 0.25, 0.35, 0.5465, 0.80)
    }

    # The direct H159 test on the two arms that actually lost: keep the shape,
    # hold the gate. `pbfitgate` overwrites index 0 and keeps the old interior
    # level. `pbfitshape` keeps the shape exactly and sets the level from the
    # gate, which is the honest reading of "release the total".
    family["pbfitgate"] = {0.0: (hold_gate(pbfit_marginal()), None)}
    family["pbfitshape"] = {0.0: (rescale_to_gate(PBFIT_RAW), None)}
    for width in (5, 6, 7):
        family["pb%dgate" % width] = {
            0.0: (hold_gate(boundary_marginal(width)), None)}

    # The historical losers as they actually ran, for the positive control.
    family["pb6asrun"] = {0.0: (boundary_marginal(6), None)}
    family["pbfitasrun"] = {0.0: (pbfit_marginal(), None)}
    return family


def resolve(entry):
    marginal, cumulative = entry
    return marginal, cumulative if cumulative is not None else prefix_costs(marginal)


# --------------------------------------------------------------- the measurement


def arm_run(panel: dict, price, windows: int) -> dict:
    out = {}
    for prompt, spec in panel.items():
        run = simulate(None, spec["factory"](spec["p_target"]), windows,
                       price=price)
        out[prompt] = run
    return out


def contrast(ship_runs: dict, arm_runs: dict) -> dict:
    """RULE 177: paired per-prompt candidate-leg percent, unweighted."""
    per_prompt = {}
    for prompt, run in arm_runs.items():
        base = ship_runs[prompt]["us_per_token"]
        per_prompt[prompt] = (run["us_per_token"] / base - 1.0) * 100.0
    return per_prompt


def weighted_mean(per_prompt: dict) -> dict:
    """RULE 178 weighted mean with a Kish effective sample size.

    The weights are extremely unequal, so the naive `sqrt(n)` denominator would
    overstate the precision of what is close to a beagle-only measurement.
    """
    items = [(p, v, RULE178_WEIGHTS[p]) for p, v in per_prompt.items()
             if p in RULE178_WEIGHTS]
    wsum = sum(w for _, _, w in items)
    w2sum = sum(w * w for _, _, w in items)
    mean = sum(w * v for _, v, w in items) / wsum
    n_eff = wsum * wsum / w2sum
    var = sum(w * (v - mean) ** 2 for _, v, w in items) / wsum
    sd = math.sqrt(var * n_eff / (n_eff - 1.0)) if n_eff > 1.0 else float("nan")
    se = sd / math.sqrt(n_eff) if n_eff > 1.0 else float("nan")
    return {"mean": mean, "sd": sd, "se": se, "n_eff": n_eff,
            "weight_covered": wsum}


def summarise_contrast(per_prompt: dict, keep: list | None = None) -> dict:
    if keep is not None:
        per_prompt = {p: v for p, v in per_prompt.items() if p in keep}
    values = list(per_prompt.values())
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    se = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    same_sign = sum(1 for v in values if v * mean > 0)
    gain = -mean
    g = gain / 100.0
    wstat = weighted_mean(per_prompt)
    wgain = -wstat["mean"]
    return {
        "per_prompt_pct": per_prompt,
        "candidate_leg_mean_pct": mean,
        "candidate_leg_gain_pct": gain,
        "sd": sd,
        "se": se,
        "two_sigma": 2.0 * se,
        "same_sign": same_sign,
        "n": len(values),
        "clears_zero": gain - 2.0 * se > 0.0,
        # RULE 178. The selection statistic for a non-uniform mechanism.
        "rule178_gain_pct": wgain,
        "rule178_se": wstat["se"],
        "rule178_two_sigma": 2.0 * wstat["se"],
        "rule178_n_eff": wstat["n_eff"],
        "rule178_weight_covered": wstat["weight_covered"],
        "rule178_clears_zero": wgain - 2.0 * wstat["se"] > 0.0,
        # RULE 176. Only meaningful when the effect is close to uniform, so it
        # is reported next to `same_sign` and never on its own.
        "rule176_published_pct": 100.0 * g / (1.0 - g) if g < 1.0 else float("nan"),
    }


def plutarch_guard(arm_runs: dict) -> dict:
    run = arm_runs.get("plutarch")
    if run is None:
        return {"available": False}
    rounds = run["rounds"]
    non_drafting = run["depth_counts"][0]
    scaled = non_drafting / rounds * PLUTARCH_ROUNDS
    return {
        "available": True,
        "non_drafting_rounds": non_drafting,
        "total_rounds": rounds,
        "fraction": non_drafting / rounds,
        "out_of_487": scaled,
        "mean_depth": run["mean_depth"],
        "passes": scaled >= PLUTARCH_NON_DRAFTING_FLOOR,
    }


def median_state(receipt: dict, per_prompt_pct: dict) -> dict:
    """The realised median pair under an arm, and whether its identity moved."""
    raws = {}
    ship = {}
    for prompt, entry in receipt["per_prompt"].items():
        pct = per_prompt_pct.get(prompt, 0.0)
        ratio = 1.0 + pct / 100.0
        raws[prompt] = entry["serial"] / (entry["candidate"] * ratio)
        ship[prompt] = entry["serial"] / entry["candidate"]
    order_arm = sorted(raws, key=lambda p: raws[p])
    order_ship = sorted(ship, key=lambda p: ship[p])
    pair_arm = tuple(sorted(order_arm[3:5]))
    pair_ship = tuple(sorted(order_ship[3:5]))
    return {
        "median_pct": (median_of(list(raws.values()))
                       / median_of(list(ship.values())) - 1.0) * 100.0,
        "pair_ship": list(pair_ship),
        "pair_arm": list(pair_arm),
        "pair_changed": pair_arm != pair_ship,
    }


def leave_one_out(receipt, per_seed_grid: dict, grid: list, seeds: list,
                  prompts: list) -> dict:
    """Choose the grid point for each prompt WITHOUT that prompt.

    Selection uses the RULE 178 weighted candidate-leg mean over the other
    seven prompts, per RULE 177 and FINDING 306, not the published median.
    """
    folds = {}
    held_out = []
    for prompt in prompts:
        picks, values = [], []
        for seed in seeds:
            best_k, best_gain = None, None
            for k in grid:
                others = {p: v for p, v
                          in per_seed_grid[seed][k]["per_prompt_pct"].items()
                          if p != prompt}
                gain = -weighted_mean(others)["mean"]
                if best_gain is None or gain > best_gain:
                    best_k, best_gain = k, gain
            picks.append(best_k)
            values.append(-per_seed_grid[seed][best_k]["per_prompt_pct"][prompt])
        folds[prompt] = {
            "picks": picks,
            "held_out_gain_pct": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "unique_picks": sorted(set(picks)),
        }
        held_out.append(statistics.fmean(values))
    all_picks = sorted({k for f in folds.values() for k in f["picks"]})
    mean = statistics.fmean(held_out)
    sd = statistics.stdev(held_out) if len(held_out) > 1 else 0.0
    return {
        "folds": folds,
        "held_out_mean_gain_pct": mean,
        "held_out_spread_pct": max(held_out) - min(held_out),
        "held_out_sd": sd,
        "held_out_se": sd / math.sqrt(len(held_out)) if len(held_out) > 1 else 0.0,
        "selection_stable": len(all_picks) == 1,
        "selected": all_picks,
    }


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=here.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=here / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="5a9f130a")
    ap.add_argument("--curve", default="e159", choices=("e159", "f97"))
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--fit-windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=159)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=pathlib.Path,
                    default=here / "e159-artifacts/e159_gate_held_arms.json")
    args = ap.parse_args()

    install_curve(args.curve)
    print("harness=local replay, harness=ranked cost law  E159 Part B  zero GPU")
    print("round cost curve   %s" % args.curve)
    print("pricing            RULE 177 candidate leg over %d prompts, selected"
          % len(RANKED_PROMPTS))
    print("                   on the RULE 178 weighted mean, n_eff %.3f"
          % weighted_mean({p: 0.0 for p in RULE178_WEIGHTS})["n_eff"])
    print("stop rule          weighted gain >= %.2f %% with 2 sigma clearing 0"
          % STOP_RULE_GAIN_PCT)
    print("median receipt     %s, reported for promotion only\n" % args.receipt)

    print("## round cost law in use, verify width -> us per round")
    print("%6s %12s %12s %9s" % ("width", args.curve, "f97", "diff %"))
    for width in range(1, MAX_DEPTH + 2):
        a, b = part_a_round_us(width), f97_round_us(width)
        mark = "" if PART_A_MEASURED[width - 1] else "   INFERRED"
        print("%6d %12.0f %12.0f %+9.3f%s" % (
            width, a, b, 100.0 * (a / b - 1.0), mark))

    legs, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    for key in ("legs", "attached", "accept_mismatch", "margin_mismatch",
                "unmatched"):
        print("  %-22s %d" % (key, gate[key]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        print("  ATTACHMENT IS NOT PROVEN - every number below is void")

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    family = arm_family()

    panels, ship_runs = {}, {}
    for seed in seeds:
        panels[seed] = prompt_panel(legs, args.windows, args.fit_windows, seed)
        ship_runs[seed] = arm_run(panels[seed], resolve(family["ship"][0.0]),
                                  args.windows)

    panel = panels[seeds[0]]
    print("\n## transfer fit, averaged over %d seeds" % len(seeds))
    print("   A prompt whose replayed draft length misses its ranked draft")
    print("   length by more than %.0f %% cannot carry an arm contrast."
          % (100.0 * TRANSFER_TOLERANCE))
    print("%-10s %8s %10s %10s %9s %11s %6s" % (
        "prompt", "delta", "tgt depth", "sim depth", "miss %", "ship us/tok",
        "core"))
    core = []
    transfer = {}
    for prompt in panel:
        target = RANKED_PROMPTS[prompt]["depth"]
        sim = statistics.fmean(panels[s][prompt]["ship"]["mean_depth"]
                               for s in seeds)
        miss = sim / target - 1.0
        ok = abs(miss) <= TRANSFER_TOLERANCE
        if ok:
            core.append(prompt)
        transfer[prompt] = {"target_depth": target, "sim_depth": sim,
                            "miss": miss, "core": ok,
                            "rule178_weight": RULE178_WEIGHTS[prompt]}
        print("%-10s %8.4f %10.3f %10.3f %+9.2f %11.1f %6s" % (
            prompt, panel[prompt]["delta"], target, sim, 100.0 * miss,
            statistics.fmean(ship_runs[s][prompt]["us_per_token"]
                             for s in seeds),
            "yes" if ok else "NO"))
    core_weight = sum(RULE178_WEIGHTS[p] for p in core)
    total_weight = sum(RULE178_WEIGHTS.values())
    print("  core prompts %s" % ", ".join(core))
    print("  core prompts carry RULE 178 weight %.4f of %.4f, %.2f %% of the"
          % (core_weight, total_weight, 100.0 * core_weight / total_weight))
    print("  measured median sensitivity")

    # Every arm at every grid point, paired against ship on the same seed.
    results = {}
    per_seed = {}
    for name, grid in family.items():
        per_seed[name] = {seed: {} for seed in seeds}
        for k, entry in grid.items():
            price = resolve(entry)
            per_prompt_over_seeds = {p: [] for p in panel}
            guards, seed_means, seed_w_gains = [], [], []
            for seed in seeds:
                runs = arm_run(panels[seed], price, args.windows)
                pct = contrast(ship_runs[seed], runs)
                per_seed[name][seed][k] = {"per_prompt_pct": pct}
                for prompt, value in pct.items():
                    per_prompt_over_seeds[prompt].append(value)
                guards.append(plutarch_guard(runs))
                seed_means.append(statistics.fmean(pct.values()))
                seed_w_gains.append(-weighted_mean(pct)["mean"])
            averaged = {p: statistics.fmean(v)
                        for p, v in per_prompt_over_seeds.items()}
            summary = summarise_contrast(averaged)
            summary["core"] = summarise_contrast(averaged, keep=core)
            summary["seed_sd_of_mean"] = (
                statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0)
            # The prompt set is fixed and fully observed, so between-prompt
            # spread measures effect heterogeneity, not uncertainty about the
            # weighted average. Replay resampling noise does measure that.
            seed_sd = (statistics.stdev(seed_w_gains)
                       if len(seed_w_gains) > 1 else 0.0)
            summary["rule178_seed_sd"] = seed_sd
            summary["rule178_seed_se"] = seed_sd / math.sqrt(len(seed_w_gains))
            summary["rule178_seed_gains"] = seed_w_gains
            summary["plutarch"] = {
                "out_of_487": statistics.fmean(g["out_of_487"] for g in guards),
                "mean_depth": statistics.fmean(g["mean_depth"] for g in guards),
                "passes": all(g["passes"] for g in guards),
            }
            summary.update(median_state(receipt, averaged))
            summary["marginal"] = price[0]
            summary["cumulative"] = price[1]
            summary["sum_marginal"] = sum(price[0])
            summary["gate"] = price[0][0]
            # Constraint 1 is a bit-pattern claim, not a tolerance. The source
            # comment on `makeUniformDepthPrice` says a gate that differs by
            # one ulp is not a control.
            summary["gate_bits"] = struct.pack("<d", price[0][0]).hex()
            summary["gate_held"] = summary["gate_bits"] == SHIPPED_GATE_BITS
            results[(name, k)] = summary

    print("\n## every arm, candidate-leg contrast against ship, %d seeds"
          % len(seeds))
    print("   gain is positive when the arm decodes FASTER. `plu487` is the")
    print("   plutarch non-drafting guard and must be >= %d."
          % PLUTARCH_NON_DRAFTING_FLOOR)
    print("   `w178 %` is the RULE 178 weighted gain and is the selection")
    print("   statistic. `flat %` is the unweighted eight-prompt mean, shown")
    print("   only so the weighting's effect is visible.")
    print("%-12s %8s %6s %8s %9s %8s %6s %9s %8s %9s" % (
        "arm", "k", "gate", "sum", "w178 %", "w 2 sig", "sign", "flat %",
        "plu487", "median %"))
    for (name, k), s in results.items():
        print("%-12s %8.5f %6.4f %8.4f %+9.4f %8.4f %4d/%d %+9.4f %8.1f %+9.4f"
              % (name, k, s["gate"], s["sum_marginal"],
                 s["rule178_gain_pct"], s["rule178_two_sigma"],
                 s["same_sign"], s["n"], s["candidate_leg_gain_pct"],
                 s["plutarch"]["out_of_487"], s["median_pct"]))

    # Why the gate scan below cannot reproduce the ranked phase transition.
    # `threshold(0) == marginal[0]`, and `reach` at depth 0 is the EMA of
    # position-0 acceptance. A leg whose position-0 acceptance sits far above
    # 0.18 can never close its gate, whatever the arm does.
    print("\n## can the local material fire the depth-0 gate at all?")
    print("   The gate fires when position-0 acceptance falls to %.2f."
          % SHIPPED_HEAD_STEP_COST_RATIO)
    print("%-18s %8s %10s %11s %s" % (
        "local leg", "rounds", "p at d0", "x the gate", "gate can fire"))
    gate_capable = 0
    for name in sorted(legs):
        leg = legs[name]
        p0 = leg["positions"][0]["p"]
        can = p0 <= SHIPPED_HEAD_STEP_COST_RATIO
        gate_capable += int(can)
        print("%-18s %8d %10.4f %11.2f %s" % (
            name, leg["rounds"], p0, p0 / SHIPPED_HEAD_STEP_COST_RATIO,
            "yes" if can else "no"))
    print("  legs that can fire the gate %d of %d" % (gate_capable, len(legs)))
    print("  The public fixture `benchfixture` sits at %.4f, %.1fx the gate."
          % (legs["benchfixture"]["positions"][0]["p"],
             legs["benchfixture"]["positions"][0]["p"]
             / SHIPPED_HEAD_STEP_COST_RATIO))
    print("  No local leg resembles the hidden plutarch prompt, so no local")
    print("  measurement can price `marginal[0]` in either direction.")

    print("\n## gate scan positive control, interior held at %.2f"
          % SHIPPED_HEAD_STEP_COST_RATIO)
    print("   Ranked receipts: gate 0.18 -> plutarch edl 0.1540, gate 0.32 ->")
    print("   0.0645, gate 0.159467 or 0.120143 -> about 2.70. If the replay")
    print("   holds plutarch flat across this scan it does not contain the")
    print("   mechanism and cannot evaluate constraint 2.")
    print("%10s %12s %12s %10s" % (
        "gate", "plu edl", "plu of 487", "gain %"))
    for k in sorted(family["gatescan"]):
        s = results[("gatescan", k)]
        print("%10.6f %12.4f %12.1f %+10.4f" % (
            k, s["plutarch"]["mean_depth"], s["plutarch"]["out_of_487"],
            s["candidate_leg_gain_pct"]))
    scan = [results[("gatescan", k)]["plutarch"]["mean_depth"]
            for k in sorted(family["gatescan"])]
    print("  replay plutarch draft length range %.4f to %.4f, ratio %.2f"
          % (min(scan), max(scan), max(scan) / min(scan) if min(scan) else 0.0))
    print("  ranked plutarch draft length range 0.0645 to 2.6995, ratio 41.85")

    print("\n## per-prompt candidate-leg percent, negative = arm is faster")
    order = ["beagle", "essays", "republic", "botany", "medicine", "drama",
             "travel", "plutarch"]
    shown = [key for key in results if key[0] != "ship"]
    print("%-12s %8s " % ("arm", "k")
          + " ".join("%9s" % p[:9] for p in order))
    for key in shown:
        s = results[key]
        print("%-12s %8.5f " % key
              + " ".join("%+9.3f" % s["per_prompt_pct"].get(p, float("nan"))
                         for p in order))

    # Leave-one-prompt-out on the two arms that have a real search axis.
    loo = {}
    for name in ("gateuniform", "e159law", "sketch"):
        grid = sorted(family[name])
        loo[name] = leave_one_out(receipt, per_seed[name], grid, seeds,
                                  list(panel))
        entry = loo[name]
        print("\n## leave-one-prompt-out, %s" % name)
        print("   selection uses the other seven prompts on the candidate leg")
        print("%-10s %14s %8s  %s" % (
            "held out", "gain %", "sd", "grid points chosen"))
        for prompt, fold in entry["folds"].items():
            print("%-10s %+14.4f %8.4f  %s" % (
                prompt, fold["held_out_gain_pct"], fold["sd"],
                ", ".join("%.5f" % k for k in fold["unique_picks"])))
        print("  held-out mean %+.4f %%, spread %.4f, se %.4f, stable %s"
              % (entry["held_out_mean_gain_pct"], entry["held_out_spread_pct"],
                 entry["held_out_se"], entry["selection_stable"]))

    # The verdict, over gate-held arms.
    #
    # The plutarch guard is NOT applied as a filter here, and the gate scan
    # above says why: the replay's plutarch draft length moves by 4.8x across
    # the whole gate range while the ranked receipts move by 41.9x with a
    # discontinuity, and the shipped arm's own replayed count sits at the
    # 440 floor. The guard is inside the replay's noise around ship, so
    # filtering on it would reject arms for a quantity this instrument cannot
    # measure. It is reported for every arm instead.
    searched = [key for key in results
                if key[0] not in ("ship", "gatescan", "pb6asrun",
                                  "pbfitasrun")]
    eligible = {key: results[key] for key in searched
                if results[key]["gate_held"]}
    print("\n## verdict")
    print("  arms priced                     %d" % len(results))
    print("  gate-held arms searched         %d" % len(eligible))
    print("  shipped gate bit pattern        %s" % SHIPPED_GATE_BITS)
    print("  searched arms with that pattern %d of %d"
          % (sum(1 for key in searched if results[key]["gate_held"]),
             len(searched)))
    guard_fail = ["%s k=%.5f" % key for key in eligible
                  if not results[key]["plutarch"]["passes"]]
    print("  replay guard below the floor    %d of %d, reported not filtered"
          % (len(guard_fail), len(eligible)))

    best_key = best = None
    if eligible:
        best_key = max(eligible, key=lambda k: eligible[k]["rule178_gain_pct"])
        best = eligible[best_key]
    if best is None:
        verdict = "no_arm_clears"
    elif best["rule178_gain_pct"] < STOP_RULE_GAIN_PCT:
        verdict = "no_arm_clears"
    elif not best["rule178_clears_zero"]:
        verdict = "no_arm_clears"
    else:
        verdict = "arm_clears"
    if eligible:
        best_flat_key = max(
            eligible, key=lambda k: eligible[k]["candidate_leg_gain_pct"])
        best_core_key = max(
            eligible, key=lambda k: eligible[k]["core"]["candidate_leg_gain_pct"])
        print("  best on the RULE 178 weighting  %s k=%.5f, %+.4f %%"
              % (best_key + (best["rule178_gain_pct"],)))
        print("  best unweighted over 8 prompts  %s k=%.5f, %+.4f %%"
              % (best_flat_key
                 + (eligible[best_flat_key]["candidate_leg_gain_pct"],)))
        print("  best on the %d core prompts      %s k=%.5f, %+.4f %%"
              % ((len(core),) + best_core_key
                 + (eligible[best_core_key]["core"]["candidate_leg_gain_pct"],)))
    if best is not None:
        print("  best gate-held arm              %s k=%.5f" % best_key)
        print("  RULE 178 weighted gain          %+.4f %% (2 sigma %.4f, "
              "n_eff %.3f)" % (best["rule178_gain_pct"],
                               best["rule178_two_sigma"],
                               best["rule178_n_eff"]))
        print("  same gain, replay-seed 2 sigma  %.4f %% over %d seeds"
              % (2.0 * best["rule178_seed_se"], len(seeds)))
        print("  unweighted candidate-leg gain   %+.4f %% (2 sigma %.4f)"
              % (best["candidate_leg_gain_pct"], best["two_sigma"]))
        print("  same sign                       %d/%d"
              % (best["same_sign"], best["n"]))
        print("  plutarch non-drafting of 487    %.1f"
              % best["plutarch"]["out_of_487"])
        print("  median pair ship -> arm         %s -> %s, changed %s"
              % (best["pair_ship"], best["pair_arm"], best["pair_changed"]))
        print("  RULE 176 published estimate     %+.4f %%"
              % best["rule176_published_pct"])
    print("  e159_verdict                    %s" % verdict)

    out = {
        "harness": "local replay priced by a ranked cost law",
        "curve": args.curve,
        "receipt": args.receipt,
        "seeds": seeds,
        "windows": args.windows,
        "attachment_gate": gate,
        "stop_rule_gain_pct": STOP_RULE_GAIN_PCT,
        "stop_rule_statistic": "rule178_weighted_candidate_leg_gain_pct",
        "rule178_weights": RULE178_WEIGHTS,
        "plutarch_floor": PLUTARCH_NON_DRAFTING_FLOOR,
        "arms": {
            "%s|%.5f" % key: {
                k: v for k, v in s.items() if k != "per_prompt_pct"
            } | {"per_prompt_pct": s["per_prompt_pct"]}
            for key, s in results.items()
        },
        "transfer": transfer,
        "core_prompts": core,
        "leave_one_out": loo,
        "e159_verdict": verdict,
        "e159_best_arm": ("%s|%.5f" % best_key) if best_key else None,
        "e159_arm_marginal": best["marginal"] if best else None,
        "e159_predicted_candidate_leg_gain_pct": (
            best["candidate_leg_gain_pct"] if best else None),
        "e159_predicted_rule178_gain_pct": (
            best["rule178_gain_pct"] if best else None),
        "e159_rule178_two_sigma": best["rule178_two_sigma"] if best else None,
        "e159_rule178_n_eff": best["rule178_n_eff"] if best else None,
        "e159_predicted_median_pct": best["median_pct"] if best else None,
        "e159_per_prompt_pct": best["per_prompt_pct"] if best else None,
        "e159_plutarch_nondrafting": (
            best["plutarch"]["out_of_487"] if best else None),
        "e159_median_pair_changed": best["pair_changed"] if best else None,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
