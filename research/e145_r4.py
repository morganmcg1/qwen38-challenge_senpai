#!/usr/bin/env python3
"""E145 R4: the `(h, tier)` plane, searched against the LIVE curve.

`harness=local instrument`. Zero GPU.

WHY A PLANE AND NOT A LINE. `makeBoundaryDepthPrice` has two scalars, not one:

    total  = count * h
    within = total / (count - 1 + tier)
    marginal[cliff] = within * tier

The campaign has only ever moved `tier`, holding `h` at the shipped `0.18`.
FINDING 210 shows why that never worked: `tier` sits in the DENOMINATOR of
`within`, so raising the cliff automatically CHEAPENS every shallow boundary.
One scalar therefore cannot raise the cliff without also buying extra shallow
depth, and the two effects fight. `h` is the axis that sets the overall level,
so the two together can raise the cliff and hold the shallow price fixed.

WHAT DECIDES A CELL. The published score is the median of eight raw ratios,
which is the mean of the two middle values. On the current board `beagle`
holds the lower middle slot and the upper middle slot is occupied by `essays`
in 89.7 percent of receipts, `republic` in 6.9 percent and `medicine` in 3.4
percent. CAMPAIGN RULE 126 says a shaped gain is priced as an expectation over
that serial lottery, and CAMPAIGN RULE 129 says a NON-UNIFORM mechanism is
scored at its worst of the candidate upper-slot prompts, never at their mean.
So the reported objective is

    objective = (gain[beagle] + min gain over the four upper-slot prompts) / 2

and the SPREAD across those four is reported beside every cell. A cell with a
high objective and a wide spread is a cell whose value depends on which prompt
the lottery draws, and Rule 129 already priced that at its worst.

`median_pct`, the replayed median that lets the sort order move, is reported
in the same table as a cross check. The two answer different questions: the
objective asks what the mechanism is worth under the worst draw, and
`median_pct` asks what this one receipt's replay says.

WHICH CURVE. The cost model is the R2 live curve by default, because R3 showed
the replayed curve disagrees with it by 13.8 percent at width 7 and inverts the
sign of the 6 to 7 step. `--cost replayed` reproduces the old plane for
comparison. Both uses of the curve are ratios, so the local-to-ranked level
factor cancels; R3 gates that numerically.

Usage:
  python3 research/e145_r4.py --json research/e145-artifacts/r4.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import (  # noqa: E402
    RANKED_PROMPTS, load_board_receipt,
)
from e128_replay import PRICE_CUMULATIVE, PRICE_MARGINAL  # noqa: E402
from e134_rung2 import build_legs, median_pct, simulate  # noqa: E402
from e140_cells import install, transfer_cache  # noqa: E402
from e140_lookahead import SHIPPED_CLIFF, load_curves  # noqa: E402
from e145_r3 import (  # noqa: E402
    MEASURED_LABEL, REPLAYED_LABEL, installable, measured_points,
    serial_round_us, width1_candidates,
)
from e145_curve import LEGS_JSON  # noqa: E402

SHIPPED_H = 0.18
SHIPPED_TIER = 1.45

# FINDING 4 re-centred the grid after Advisor Error 156. The shipped cell sits
# inside it, so the control is a grid point rather than a separate calculation.
H_GRID = (0.15, 0.16, 0.17, SHIPPED_H, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28,
          0.30)
# The objective is a step function of the price, because the price only moves
# the score by changing an integer depth. A first coarse pass put the whole
# ridge on the single point `tier = 1.45` and dropped 2.9 percentage points by
# `tier = 2.0`, which a coarse grid cannot tell from a knife edge. The tier
# axis is therefore dense between 1.0 and 2.0 and coarse above it.
TIER_GRID = (1.0, 1.1, 1.2, 1.3, 1.4, SHIPPED_TIER, 1.5, 1.6, 1.7, 1.8, 1.9,
             2.0, 2.5, 3.0, 4.0, 5.0)

# The lower middle slot, and the four prompts the serial lottery can put in
# the upper middle slot. `drama`, `travel` and `plutarch` carry zero F83
# weight and never reach either middle slot on the current board.
LOWER_SLOT = "beagle"
UPPER_SLOT = ("essays", "republic", "medicine", "botany")


def plane_price(h: float, tier: float, cliff: int = SHIPPED_CLIFF):
    """`makeBoundaryDepthPrice` with its hard-coded 0.18 opened up to `h`.

    `h = 0.18, tier = 1.0` is `makeUniformDepthPrice` exactly, and
    `h = 0.18, tier = 1.45` is the shipped `pb6` arm exactly. Both identities
    are gated in `main`.
    """
    count = len(PRICE_MARGINAL)
    within = count * h / (count - 1 + tier)
    marginal = [within] * count
    marginal[cliff] = within * tier
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)]


def two_boundary_price(h: float, tier6: float, tier7: float,
                       cliff6: int = SHIPPED_CLIFF,
                       cliff7: int = SHIPPED_CLIFF + 1):
    """`pb67`: the live curve's SECOND expensive boundary, priced.

    R3 measured two adjacent large steps, into verify width 6 and into verify
    width 7, where the replayed curve had one spike and then a nearly free
    step. `pb68` was fitted on the replayed curve and selected `tier8 = 1.0`,
    that is, it collapsed to `pb6`, because there was nothing at 7 to 8 to
    price. On the live curve the second boundary is at 6 to 7, so this is the
    rival the measurement asks for. `tier7 = 1` reproduces `plane_price`
    exactly, which `main` gates.
    """
    count = len(PRICE_MARGINAL)
    within = count * h / (count - 2 + tier6 + tier7)
    marginal = [within] * count
    marginal[cliff6] = within * tier6
    marginal[cliff7] = within * tier7
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)]


def cell_ratios(cache, seed, curve, price, windows, base_cache) -> dict:
    """One candidate-time ratio per ranked prompt, against the shipped arm."""
    install(curve)
    out = {}
    for prompt in RANKED_PROMPTS:
        entry = cache[(seed, prompt)]
        slot = (id(curve), seed, prompt)
        if slot not in base_cache:
            base_cache[slot] = simulate(
                None, entry["factory"](entry["p_target"]), windows)
        base = base_cache[slot]
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       price=price)
        out[prompt] = {"ratio": run["us_per_token"] / base["us_per_token"],
                       "mean_depth": run["mean_depth"],
                       "accept_rate": run["accept_rate"]}
    return out


def slot_objective(ratios: dict) -> dict:
    """Rule 126 and Rule 129, applied to one cell's per-prompt ratios."""
    gain = {p: 100.0 * (1.0 / ratios[p]["ratio"] - 1.0) for p in ratios}
    upper = {p: gain[p] for p in UPPER_SLOT if p in gain}
    worst_prompt = min(upper, key=upper.get)
    values = list(upper.values())
    return {
        "gain_lower_slot": gain[LOWER_SLOT],
        "gain_upper_slot_worst": upper[worst_prompt],
        "worst_upper_slot_prompt": worst_prompt,
        "gain_upper_slot_best": max(values),
        "gain_upper_slot_mean": statistics.fmean(values),
        "upper_slot_spread": max(values) - min(values),
        "objective": 0.5 * (gain[LOWER_SLOT] + upper[worst_prompt]),
        "objective_at_mean_upper": 0.5 * (gain[LOWER_SLOT]
                                          + statistics.fmean(values)),
        "gain_per_prompt": gain,
    }


def search(cache, seeds, receipt, windows, curve) -> dict:
    base_cache: dict = {}
    out = {}
    for h in H_GRID:
        for tier in TIER_GRID:
            price = plane_price(h, tier)
            objectives, medians, spreads, worst = [], [], [], []
            per_prompt: dict[str, list[float]] = {}
            depths = []
            for seed in seeds:
                ratios = cell_ratios(cache, seed, curve, price, windows,
                                     base_cache)
                slots = slot_objective(ratios)
                objectives.append(slots["objective"])
                spreads.append(slots["upper_slot_spread"])
                worst.append(slots["worst_upper_slot_prompt"])
                medians.append(median_pct(receipt, ratios))
                for prompt, row in ratios.items():
                    per_prompt.setdefault(prompt, []).append(row["ratio"])
                    depths.append(RANKED_PROMPTS[prompt]["weight"]
                                  * row["mean_depth"])
            out["%.4f|%.4f" % (h, tier)] = {
                "h": h, "tier": tier,
                "within_price": price[0][0],
                "cliff_price": price[0][SHIPPED_CLIFF],
                "objective_mean": statistics.fmean(objectives),
                "objective_sd": (statistics.stdev(objectives)
                                 if len(objectives) > 1 else 0.0),
                "upper_slot_spread_mean": statistics.fmean(spreads),
                "worst_upper_slot_prompt": statistics.mode(worst),
                "median_pct_mean": statistics.fmean(medians),
                "f83_weighted_mean_depth": sum(depths) / len(seeds),
                "per_prompt_ratio": {p: statistics.fmean(v)
                                     for p, v in per_prompt.items()},
            }
    return out


def search_pb67(cache, seeds, receipt, windows, curve, h: float) -> dict:
    """The second boundary, at the flat level the single-boundary plane chose."""
    base_cache: dict = {}
    out = {}
    for tier6 in TIER_GRID:
        for tier7 in TIER_GRID:
            price = two_boundary_price(h, tier6, tier7)
            objectives, spreads, medians = [], [], []
            for seed in seeds:
                ratios = cell_ratios(cache, seed, curve, price, windows,
                                     base_cache)
                slots = slot_objective(ratios)
                objectives.append(slots["objective"])
                spreads.append(slots["upper_slot_spread"])
                medians.append(median_pct(receipt, ratios))
            out["%.4f|%.4f" % (tier6, tier7)] = {
                "tier6": tier6, "tier7": tier7, "h": h,
                "cliff6_price": price[0][SHIPPED_CLIFF],
                "cliff7_price": price[0][SHIPPED_CLIFF + 1],
                "objective_mean": statistics.fmean(objectives),
                "upper_slot_spread_mean": statistics.fmean(spreads),
                "median_pct_mean": statistics.fmean(medians),
            }
    return out


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
    ap.add_argument("--anchor", default="measured",
                    choices=("measured", "extrapolated", "serial"))
    ap.add_argument("--cost", default=MEASURED_LABEL,
                    choices=(MEASURED_LABEL, REPLAYED_LABEL))
    ap.add_argument("--skip-pb67", action="store_true")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e145-artifacts/r4.json")
    args = ap.parse_args()

    print("harness=local instrument  E145 R4 (h, tier) plane  zero GPU")

    shipped_uniform = plane_price(SHIPPED_H, 1.0)
    if max(abs(a - b) for a, b in zip(shipped_uniform[0], PRICE_MARGINAL)) > 1e-12:
        raise SystemExit("plane_price(0.18, 1.0) is not the shipped uniform"
                         " table; the parameterisation is wrong")
    print("  gate: plane_price(%.2f, 1.0) reproduces makeUniformDepthPrice"
          % SHIPPED_H)

    legs = json.loads(LEGS_JSON.read_text())["legs"]
    points, anchor = measured_points(legs)
    candidates = width1_candidates(points, anchor, serial_round_us(legs))
    if args.anchor not in candidates:
        print("e145_r4: the %s width-1 anchor is not available; have %s"
              % (args.anchor, sorted(candidates)))
        return 1
    points[1] = candidates[args.anchor]

    replayed_curves, best_form = load_curves()
    curve = (installable(points, best_form) if args.cost == MEASURED_LABEL
             else replayed_curves[best_form])
    print("  cost model %s (%s), width-1 anchor %s at %.1f us"
          % (args.cost, curve["name"], args.anchor, points[1]))

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs_e128, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    print("  attachment gate: %d legs, %d attached, 0 mismatches"
          % (gate["legs"], gate["attached"]))

    cache = transfer_cache(legs_e128, args.windows, args.fit_windows, seeds)
    grid = search(cache, seeds, receipt, args.windows, curve)

    shipped_key = "%.4f|%.4f" % (SHIPPED_H, SHIPPED_TIER)
    uniform_key = "%.4f|%.4f" % (SHIPPED_H, 1.0)
    shipped = grid[shipped_key]

    print("\n## objective, percent. Rows are h, columns are tier.")
    print("  %6s %s" % ("h", " ".join("%7.2f" % t for t in TIER_GRID)))
    for h in H_GRID:
        row = [grid["%.4f|%.4f" % (h, t)]["objective_mean"] for t in TIER_GRID]
        mark = " <- shipped h" if h == SHIPPED_H else ""
        print("  %6.2f %s%s" % (h, " ".join("%+7.3f" % v for v in row), mark))

    print("\n## spread across the four upper-slot prompts, percentage points")
    print("  %6s %s" % ("h", " ".join("%7.2f" % t for t in TIER_GRID)))
    for h in H_GRID:
        row = [grid["%.4f|%.4f" % (h, t)]["upper_slot_spread_mean"]
               for t in TIER_GRID]
        print("  %6.2f %s" % (h, " ".join("%7.3f" % v for v in row)))

    print("\n## median_pct, the replayed median that lets the sort order move")
    print("  %6s %s" % ("h", " ".join("%7.2f" % t for t in TIER_GRID)))
    for h in H_GRID:
        row = [grid["%.4f|%.4f" % (h, t)]["median_pct_mean"]
               for t in TIER_GRID]
        print("  %6.2f %s" % (h, " ".join("%+7.3f" % v for v in row)))

    best_key = max(grid, key=lambda k: grid[k]["objective_mean"])
    best = grid[best_key]
    print("\n## controls and the best cell")
    for label, key in (("uniform  h=0.18 tier=1.00", uniform_key),
                       ("shipped  h=0.18 tier=1.45", shipped_key),
                       ("best     h=%.2f tier=%.2f"
                        % (best["h"], best["tier"]), best_key)):
        row = grid[key]
        print("  %-28s objective %+7.3f   spread %6.3f   worst slot %-9s"
              "   median_pct %+7.3f   mean depth %.3f"
              % (label, row["objective_mean"], row["upper_slot_spread_mean"],
                 row["worst_upper_slot_prompt"], row["median_pct_mean"],
                 row["f83_weighted_mean_depth"]))
    print("  best cell against the shipped cell: %+.3f percentage points"
          % (best["objective_mean"] - shipped["objective_mean"]))

    ridge = sorted(grid.values(), key=lambda r: -r["objective_mean"])[:12]
    print("\n## the ridge: the twelve best cells, by the two prices they set")
    print("  %6s %6s %9s %9s %9s" % ("h", "tier", "shallow", "cliff",
                                     "objective"))
    for row in ridge:
        print("  %6.2f %6.2f %9.4f %9.4f %+9.3f"
              % (row["h"], row["tier"], row["within_price"],
                 row["cliff_price"], row["objective_mean"]))
    cliffs = [r["cliff_price"] for r in ridge]
    withins = [r["within_price"] for r in ridge]
    ridge_stats = {
        "cliff_price_spread_pct": 100.0 * (max(cliffs) / min(cliffs) - 1.0),
        "shallow_price_spread_pct": 100.0 * (max(withins) / min(withins) - 1.0),
        "objective_spread_pp": (ridge[0]["objective_mean"]
                                - ridge[-1]["objective_mean"]),
    }
    print("  across those twelve cells the cliff price varies by %.2f %% and"
          " the shallow price by %.2f %%, for %.3f percentage points of"
          " objective"
          % (ridge_stats["cliff_price_spread_pct"],
             ridge_stats["shallow_price_spread_pct"],
             ridge_stats["objective_spread_pp"]))

    pb67 = {}
    pb67_best = None
    if not args.skip_pb67:
        gate67 = two_boundary_price(SHIPPED_H, SHIPPED_TIER, 1.0)
        if max(abs(a - b) for a, b in
               zip(gate67[0], plane_price(SHIPPED_H, SHIPPED_TIER)[0])) > 1e-12:
            raise SystemExit("two_boundary_price(h, tier, 1.0) is not"
                             " plane_price(h, tier); pb67 is not nested")
        pb67 = search_pb67(cache, seeds, receipt, args.windows, curve,
                           best["h"])
        pb67_best = max(pb67, key=lambda k: pb67[k]["objective_mean"])
        nested = pb67["%.4f|%.4f" % (best["tier"], 1.0)]
        row = pb67[pb67_best]
        print("\n## pb67, the live curve's second boundary, at h = %.2f"
              % best["h"])
        print("  %6s %s" % ("tier6", " ".join("%7.2f" % t for t in TIER_GRID)))
        for tier6 in TIER_GRID:
            values = [pb67["%.4f|%.4f" % (tier6, t7)]["objective_mean"]
                      for t7 in TIER_GRID]
            print("  %6.2f %s" % (tier6,
                                  " ".join("%+7.3f" % v for v in values)))
        print("  nested control tier7 = 1.00 at tier6 = %.2f: %+.3f, which"
              " must equal the plane cell %+.3f"
              % (best["tier"], nested["objective_mean"],
                 best["objective_mean"]))
        print("  best pb67 cell tier6 = %.2f tier7 = %.2f: objective %+.3f,"
              " spread %.3f, %+.3f pp against the best single boundary"
              % (row["tier6"], row["tier7"], row["objective_mean"],
                 row["upper_slot_spread_mean"],
                 row["objective_mean"] - best["objective_mean"]))

    blob = {
        "harness": "local instrument",
        "gpu_used": False,
        "cost_model": args.cost,
        "curve_name": curve["name"],
        "width1_anchor_used": args.anchor,
        "width1_us": points[1],
        "receipt": args.receipt,
        "seeds": seeds,
        "windows": args.windows,
        "h_grid": list(H_GRID),
        "tier_grid": list(TIER_GRID),
        "lower_slot": LOWER_SLOT,
        "upper_slot_candidates": list(UPPER_SLOT),
        "grid": grid,
        "shipped_cell": shipped_key,
        "uniform_cell": uniform_key,
        "best_cell": best_key,
        "best_minus_shipped_pp": (best["objective_mean"]
                                  - shipped["objective_mean"]),
        "ridge": ridge,
        "ridge_stats": ridge_stats,
        "pb67_grid": pb67,
        "pb67_best_cell": pb67_best,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
