#!/usr/bin/env python3
"""E145 R7: prune the action set, then let the argmax use what is left.

`harness=local instrument`. Zero GPU.

WHY THIS RUNS. H140 measured `+0.0000 pp` from replacing the shipped first-order
marginal walk with a full argmax, while E140's oracle beat the same EMA rule by
1.39 pp. A rule that is already optimal cannot be 1.39 pp from the ceiling, so
one of the two statements is about the wrong object. R2 supplies the missing
piece: the live cost curve is not convex. It has a +30.6 percent wall at width
5 to 6 and then a nearly free 7 to 8 step, and every published depth-policy
result was priced through a rebuilt curve that has neither feature.

R7-0. Before any policy is simulated, ask which widths a rational rule could
ever choose on the measured curve. The answer is exact rather than empirical,
and it is much simpler than the literature assumes.

R7-1. Give the argmax only the surviving widths and re-price it, then repeat
with oracle acceptance. If the argmax with oracle acceptance beats the argmax
with the EMA, the acceptance estimate is the defect. If oracle acceptance does
not help either, the cost table is wrong as well.

WHAT IS AND IS NOT CLAIMED. The curve is LOCAL, measured on this M4 Pro. Both
of its uses are scale free -- the price table normalises by width 1 and the
cost model enters as a ratio against an unpriced arm on the same curve -- so
only the SHAPE transfers, and it is not proven that the M5 shape is the same.
Every percentage here is a replayed estimate, never a measured speedup.

Usage:
  python3 research/e145_r7.py --json research/e145-artifacts/r7.json
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
from e128_replay import SEGMENTED_VERIFY_DEPTH_CAP  # noqa: E402
from e134_rung2 import (  # noqa: E402
    build_legs, median_pct, oracle_depth, simulate,
)
from e140_cells import (  # noqa: E402
    flat_price, greedy_walker, install, price_for, transfer_cache,
)
from e140_lookahead import curve_price, load_curves, walk_argmax  # noqa: E402
from e145_curve import BASIS, LEGS_JSON  # noqa: E402
from e145_r3 import (  # noqa: E402
    installable, measured_points, serial_round_us, width1_candidates,
)

# The widest round the scored session can ask for: the shipped
# `segmentedVerifyDepthCap` is 7 drafts, so width is at most 8.
MAX_WIDTH = min(MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP) + 1


# ------------------------------------------------------------------ R7-0

def admissibility(cost: dict[int, float]) -> dict:
    """Which widths can ever be the strict argmax, on this cost curve?

    The argmax maximises `E(M) / C(M)`, with `E(M) = 1 + sum_{j<M-1} reach_j`
    the expected emitted tokens and `reach_j` the product of the first `j + 1`
    per-step acceptance probabilities. Nothing about the acceptance model is
    assumed except the one fact that holds for every such model: `reach` is
    nonincreasing, because each term multiplies the previous one by a
    probability.

    That single fact makes the test exact. For `M' > M`,

        E(M') / E(M) = 1 + (reach_{M-1} + ... + reach_{M'-2}) / E(M)

    is maximised over all monotone nonnegative reach vectors at the all-accept
    vertex `reach = 1`, where `E(M) = M`. So

        max E(M') / E(M) = M' / M

    and `value(M') > value(M)` is possible if and only if
    `M' / M > C(M') / C(M)`, that is `C(M') / M' < C(M) / M`.

    Therefore a width is admissible exactly when its cost PER TOKEN is a new
    running minimum. The bound is attained, not conservative: the all-accept
    vertex is a legal acceptance model, so no admissible width is wrongly
    admitted and no inadmissible width is wrongly excluded.

    The comparison is strict and `walk_argmax` keeps the shallower depth on a
    tie, so a width that only equals an earlier width is inadmissible too.
    """
    widths = sorted(w for w in cost if w <= MAX_WIDTH)
    per_token = {w: cost[w] / w for w in widths}
    admissible, best = [], None
    rows = []
    for width in widths:
        value = per_token[width]
        is_new = best is None or value < best
        blocker, margin_pct = None, None
        if not is_new:
            blocker = min(
                (w for w in widths if w < width and per_token[w] <= value),
                key=lambda w: per_token[w],
            )
            margin_pct = 100.0 * (value / per_token[blocker] - 1.0)
        if is_new:
            admissible.append(width)
            best = value
        rows.append(
            {
                "width": width,
                "cost_us": cost[width],
                "cost_per_token_us": value,
                "admissible": is_new,
                "blocked_by": blocker,
                "misses_by_pct": margin_pct,
            }
        )
    return {
        "rows": rows,
        "admissible_widths": admissible,
        "admissible_depths": [w - 1 for w in admissible],
        "cheapest_width": min(per_token, key=per_token.get),
    }


def domination_pairs(cost: dict[int, float]) -> list[dict]:
    """Every ordered pair, with the acceptance ratio the wider width needs."""
    widths = sorted(w for w in cost if w <= MAX_WIDTH)
    out = []
    for i, low in enumerate(widths):
        for high in widths[i + 1:]:
            ratio = cost[high] / cost[low]
            out.append(
                {
                    "narrow": low,
                    "wide": high,
                    "cost_ratio": ratio,
                    "max_token_ratio": high / low,
                    "wide_can_win": high / low > ratio,
                    "headroom_pct": 100.0 * (high / low / ratio - 1.0),
                }
            )
    return out


def per_leg_costs(legs: list[dict]) -> dict[int, list[float]]:
    """Each R2 width's individual timed legs, for the uncertainty replay."""
    out: dict[int, list[float]] = {}
    for leg in legs:
        if not leg["slot"].startswith("r2-") or leg["pin"] in (None, "none"):
            continue
        out.setdefault(int(leg["pin"]) + 1, []).append(
            leg["round_us_from_blocks"]
        )
    return out


def admissibility_uncertainty(cost: dict[int, float],
                              per_leg: dict[int, list[float]],
                              draws: int, seed: int) -> dict:
    """How often is each width admissible, given R2's own leg-to-leg spread?

    R2 timed two legs per width inside one palindrome. Their half-range is the
    only per-width uncertainty this experiment measured, so it is used as a
    one-sigma scale rather than invented. Width 1 came from the R2b session and
    has no pair here, so it carries the mean relative spread of the others.
    """
    rng = random.Random(seed)
    scale = {}
    spreads = []
    for width, values in per_leg.items():
        if len(values) < 2:
            continue
        sigma = 0.5 * (max(values) - min(values))
        scale[width] = sigma
        spreads.append(sigma / statistics.fmean(values))
    fallback = statistics.fmean(spreads) if spreads else 0.0
    for width in cost:
        scale.setdefault(width, fallback * cost[width])

    hits = {w: 0 for w in cost if w <= MAX_WIDTH}
    sets: dict[str, int] = {}
    for _ in range(draws):
        drawn = {
            w: max(1.0, rng.gauss(c, scale.get(w, 0.0)))
            for w, c in cost.items()
        }
        result = admissibility(drawn)
        for width in result["admissible_widths"]:
            hits[width] += 1
        key = ",".join(str(w) for w in result["admissible_widths"])
        sets[key] = sets.get(key, 0) + 1
    return {
        "draws": draws,
        "one_sigma_us": scale,
        "fallback_relative_sigma": fallback,
        "admissible_frequency": {w: hits[w] / draws for w in hits},
        "set_frequency": {k: v / draws for k, v in
                          sorted(sets.items(), key=lambda kv: -kv[1])},
    }


# ------------------------------------------------------------------ R7-1

def restricted_argmax_walker(allowed: set[int], tally: dict | None = None):
    """`walk_argmax` over an allowed depth set, changing nothing else.

    The per-step probability, the EMA entry, the depth-0 and depth-1 top-two
    margin clamps and their divisors are the shipped ones, exactly as in
    `e140_lookahead.walk_argmax`. The only difference is that a depth outside
    `allowed` is never a candidate for the maximum. Depth 0 is always allowed:
    not drafting is always available.
    """
    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        if adjust is not None or force is not None:
            raise SystemExit("R7 prices no `adjust` or `force` arm")
        full = walk_argmax(ema, margin, offer, price=price)
        if full in allowed:
            best = full
        else:
            best = _argmax_over(ema, margin, offer, price, allowed)
        if tally is not None:
            tally["full"][full] = tally["full"].get(full, 0) + 1
            tally["used"][best] = tally["used"].get(best, 0) + 1
            tally["pruned"] += int(best != full)
        return best
    return chooser


def _argmax_over(ema, margin, offer, price, allowed: set[int]) -> int:
    """`walk_argmax`'s objective, evaluated only at the allowed depths."""
    import math

    _, cumulative = price or e128_price.ranked_price_table()
    cap = min(min(offer, MAX_DEPTH), SEGMENTED_VERIFY_DEPTH_CAP)
    if cap <= 0:
        return 0
    best, best_value = 0, 1.0 / cumulative[0]
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        reach *= p
        expected += reach
        depth += 1
        if depth not in allowed:
            continue
        value = (1.0 + expected) / cumulative[depth]
        if value > best_value:
            best_value, best = value, depth
    return best


def restricted_oracle_walker(allowed: set[int] | None):
    """`oracle_depth` over an allowed depth set. Not implementable.

    With the capability known, `E(M) = 1 + min(capability, M - 1)`, so this is
    the same argmax as `restricted_argmax_walker` with a perfect acceptance
    model rather than the EMA. It is the ceiling for any depth rule on this
    cost curve, and the difference between it and the EMA arm is exactly the
    price of the estimate.
    """
    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        capability = ctx["capability"]
        if allowed is None:
            return oracle_depth(offer, capability)
        cap_depth = min(offer, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
        best, best_cost = 0, e128_price.ranked_round_us(1)
        for depth in range(1, cap_depth + 1):
            if depth not in allowed:
                continue
            cost = (e128_price.ranked_round_us(depth + 1)
                    / (1.0 + min(capability, depth)))
            if cost < best_cost:
                best, best_cost = depth, cost
        return best
    return chooser


def run_arm(cache, seed, prompt, cost_curve, price, walker, windows) -> dict:
    """One prompt's candidate-time ratio against the shipped arm."""
    entry = cache[(seed, prompt)]
    install(cost_curve)
    base = simulate(None, entry["factory"](entry["p_target"]), windows)
    run = simulate(None, entry["factory"](entry["p_target"]), windows,
                   price=price, walker=walker)
    rounds = run["rounds"]
    return {
        "ratio": run["us_per_token"] / base["us_per_token"],
        "mean_depth": run["mean_depth"],
        "accept_rate": run["accept_rate"],
        "depth_counts": run["depth_counts"],
        "rounds": rounds,
        "us_per_token": run["us_per_token"],
        "base_us_per_token": base["us_per_token"],
    }


def score_arms(cache, seeds, receipt, windows, arms, admissible) -> dict:
    admitted = set(admissible)
    out = {}
    for name, spec in arms.items():
        values, per_prompt = [], {}
        hist = [0] * (MAX_DEPTH + 2)
        rounds = 0
        depths = []
        for seed in seeds:
            ratios = {}
            for prompt in RANKED_PROMPTS:
                row = run_arm(cache, seed, prompt, spec["cost"],
                              spec["price"], spec["walker"](), windows)
                ratios[prompt] = row
                per_prompt.setdefault(prompt, []).append(row["ratio"])
                for index, count in enumerate(row["depth_counts"]):
                    hist[index] += count
                rounds += row["rounds"]
                depths.append(RANKED_PROMPTS[prompt]["weight"]
                              * row["mean_depth"])
            values.append(median_pct(receipt, ratios))
        out[name] = {
            "median_pct_mean": statistics.fmean(values),
            "median_pct_sd": (statistics.stdev(values)
                              if len(values) > 1 else 0.0),
            "median_pct_values": values,
            "weighted_mean_depth": sum(depths) / len(seeds),
            "rounds": rounds,
            "width_histogram": {str(i + 1): hist[i] / rounds
                                for i in range(MAX_WIDTH)},
            "frac_rounds_selecting_8": hist[7] / rounds,
            "frac_rounds_inadmissible": sum(
                hist[w - 1] for w in range(1, MAX_WIDTH + 1)
                if w not in admitted) / rounds,
            "per_prompt_ratio": {p: statistics.fmean(v)
                                 for p, v in per_prompt.items()},
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
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--skip-replay", action="store_true",
                    help="R7-0 only; the arithmetic needs no replayer")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e145-artifacts/r7.json")
    args = ap.parse_args()

    print("harness=local instrument  E145 R7  zero GPU")

    legs = json.loads(LEGS_JSON.read_text())["legs"]
    points, anchor = measured_points(legs)
    serial = serial_round_us(legs)
    candidates = width1_candidates(points, anchor, serial)
    points[1] = candidates[args.anchor]
    per_leg = per_leg_costs(legs)

    replayed_curves, best_form = load_curves()
    replayed = replayed_curves[best_form]
    measured = installable(points, best_form)
    saved_curve = e128_price.CURVE
    install(replayed)
    replayed_points = {w: e128_price.ranked_round_us(w)
                       for w in range(1, MAX_WIDTH + 1)}
    e128_price.CURVE = saved_curve

    print("\n## R7-0  which widths can EVER be the strict argmax?")
    print("  basis %s ; width-1 anchor %s = %.1f us"
          % (BASIS, args.anchor, points[1]))
    meas = admissibility(points)
    print("  %-6s %12s %14s %-11s %s"
          % ("width", "cost us", "us/token", "admissible", "note"))
    for row in meas["rows"]:
        note = ""
        if not row["admissible"]:
            note = ("blocked by width %d, misses by %+.4f %%"
                    % (row["blocked_by"], row["misses_by_pct"]))
        print("  %-6d %12.1f %14.1f %-11s %s"
              % (row["width"], row["cost_us"], row["cost_per_token_us"],
                 row["admissible"], note))
    print("  admissible widths %s  (depths %s)"
          % (meas["admissible_widths"], meas["admissible_depths"]))
    print("  cheapest width per token: %d" % meas["cheapest_width"])

    pairs = domination_pairs(points)
    seven = [p for p in pairs if p["wide"] == 7 and not p["wide_can_win"]]
    print("\n  width 7 is beaten by width %s under EVERY acceptance model"
          % ", ".join(str(p["narrow"]) for p in seven))
    for pair in seven:
        print("    vs width %d: cost ratio %.6f, best possible token ratio"
              " %.6f, short by %.4f %%"
              % (pair["narrow"], pair["cost_ratio"], pair["max_token_ratio"],
                 -pair["headroom_pct"]))
    m7_ever = int(7 in meas["admissible_widths"])

    uncertainty = admissibility_uncertainty(points, per_leg, args.draws,
                                            args.seed)
    print("\n## R7-0  is the pruning robust to R2's own leg-to-leg spread?")
    for width in sorted(uncertainty["admissible_frequency"]):
        print("  width %d admissible in %6.2f %% of %d draws (1 sigma %.1f us)"
              % (width,
                 100.0 * uncertainty["admissible_frequency"][width],
                 uncertainty["draws"],
                 uncertainty["one_sigma_us"].get(width, 0.0)))
    print("  most frequent admissible sets:")
    for key, freq in list(uncertainty["set_frequency"].items())[:5]:
        print("    {%s}  %6.2f %%" % (key, 100.0 * freq))

    out = {
        "harness": "local",
        "frame": "decode",
        "gpu_used": False,
        "rung": "R7",
        "basis": BASIS,
        "width1_anchor": args.anchor,
        "width1_us": points[1],
        "measured_cost_us": {str(w): points[w] for w in sorted(points)},
        "e145_r7_admissibility": meas,
        "e145_r7_admissible_set": meas["admissible_widths"],
        "e145_r7_admissible_depths": meas["admissible_depths"],
        "e145_r7_m7_ever_optimal": m7_ever,
        "e145_r7_domination_pairs": pairs,
        "e145_r7_admissibility_uncertainty": uncertainty,
    }

    if replayed_points:
        rep = admissibility(replayed_points)
        out["e145_r7_admissible_set_replayed"] = rep["admissible_widths"]
        print("\n  on the REPLAYED curve the admissible set is %s, so the"
              " pruning is a property of the measured shape, not of the test"
              % rep["admissible_widths"])

    if args.skip_replay:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("\nwrote %s" % args.json)
        return 0

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs_e128, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    print("  legs %d ; attached %d ; accept mismatches %d ; margin mismatches"
          " %d ; unmatched %d"
          % (gate["legs"], gate["attached"], gate["accept_mismatch"],
             gate["margin_mismatch"], gate["unmatched"]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    out["attachment_gate"] = gate

    cache = transfer_cache(legs_e128, args.windows, args.fit_windows, seeds)
    install(measured)
    measured_price = curve_price(measured)
    install(replayed)
    replayed_price = curve_price(replayed)
    flat = flat_price()
    allowed = set(meas["admissible_depths"])
    wide = allowed | {7}
    every = set(range(MAX_DEPTH + 1))

    # `A_ship` is the shipped rule and the shipped flat price. `C_flatlook`
    # changes ONLY the rule, so the difference between it and `argmax_full`
    # isolates the price table from the decision rule. The two replayed arms
    # are the control that decides whether a difference belongs to the curve or
    # to this module.
    arms = {
        "A_ship": {"cost": measured, "price": flat,
                   "walker": lambda: greedy_walker()},
        "C_flatlook": {"cost": measured, "price": flat,
                       "walker": lambda: restricted_argmax_walker(every)},
        "argmax_full": {"cost": measured, "price": measured_price,
                        "walker": lambda: restricted_argmax_walker(every)},
        "argmax_admissible": {"cost": measured, "price": measured_price,
                              "walker": lambda: restricted_argmax_walker(
                                  set(allowed))},
        "argmax_admissible_plus8": {"cost": measured, "price": measured_price,
                                    "walker": lambda: restricted_argmax_walker(
                                        set(wide))},
        "oracle_full": {"cost": measured, "price": measured_price,
                        "walker": lambda: restricted_oracle_walker(None)},
        "oracle_admissible": {"cost": measured, "price": measured_price,
                              "walker": lambda: restricted_oracle_walker(
                                  set(allowed))},
        "argmax_full_on_replayed": {"cost": replayed, "price": replayed_price,
                                    "walker": lambda: restricted_argmax_walker(
                                        every)},
        "oracle_full_on_replayed": {"cost": replayed, "price": replayed_price,
                                    "walker": lambda: restricted_oracle_walker(
                                        None)},
    }
    scored = score_arms(cache, seeds, receipt, args.windows, arms,
                        meas["admissible_widths"])
    out["e145_r7_arms"] = scored
    out["e145_r7_argmax_replayed_pct"] = (
        scored["argmax_admissible"]["median_pct_mean"]
    )
    out["e145_r7_frac_rounds_selecting_8"] = (
        scored["argmax_full"]["frac_rounds_selecting_8"]
    )

    out["e145_r7_frac_rounds_inadmissible_shipped"] = (
        scored["A_ship"]["frac_rounds_inadmissible"]
    )
    out["e145_r7_argmax_equals_greedy_on_flat_price"] = (
        scored["C_flatlook"]["width_histogram"]
        == scored["A_ship"]["width_histogram"]
    )

    print("\n## R7-1  replayed median gain against the shipped arm, percent")
    print("  %-26s %10s %8s %8s %8s %10s"
          % ("arm", "median %", "sd", "depth", "w8 %", "inadmis %"))
    for name, row in scored.items():
        print("  %-26s %+10.4f %8.4f %8.4f %8.4f %10.4f"
              % (name, row["median_pct_mean"], row["median_pct_sd"],
                 row["weighted_mean_depth"],
                 100.0 * row["frac_rounds_selecting_8"],
                 100.0 * row["frac_rounds_inadmissible"]))

    print("\n## R7-1  realised width histogram, percent of rounds")
    header = "  %-26s" % "arm"
    header += "".join("%7d" % w for w in range(1, MAX_WIDTH + 1))
    print(header)
    for name, row in scored.items():
        line = "  %-26s" % name
        line += "".join("%7.2f" % (100.0 * row["width_histogram"][str(w)])
                        for w in range(1, MAX_WIDTH + 1))
        print(line)

    ema_gain = scored["argmax_admissible"]["median_pct_mean"]
    oracle_gain = scored["oracle_admissible"]["median_pct_mean"]
    out["e145_r7_oracle_minus_argmax_pp"] = oracle_gain - ema_gain
    print("\n## R7-1  verdict")
    print("  argmax over the admissible set, EMA acceptance   %+.4f %%"
          % ema_gain)
    print("  the same argmax with oracle acceptance           %+.4f %%"
          % oracle_gain)
    print("  the estimate costs %.4f pp" % (oracle_gain - ema_gain))
    if ema_gain >= 0.50:
        verdict = "advance: above the +0.50 % promotion threshold"
    elif ema_gain <= 0.25:
        verdict = "drop: below the +0.25 % stop rule"
    else:
        verdict = "hold: between the +0.25 % and +0.50 % rules"
    out["e145_r7_verdict"] = verdict
    print("  %s" % verdict)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
