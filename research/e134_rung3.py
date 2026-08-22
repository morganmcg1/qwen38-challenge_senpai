#!/usr/bin/env python3
"""E134 rung 3: where does the oracle gap live, boundary by boundary?

Rung 2 found no implementable arm above zero held out. Before closing the
axis, this rung asks the prior question the assignment sets: is the
`+8.5248` percent oracle gap concentrated at one boundary, or is it spread
across every boundary? A gap that is spread evenly cannot be captured by any
single-boundary rule, however good that rule's discrimination is.

The decomposition gives the shipped walk perfect knowledge at exactly one
boundary and leaves every other boundary shipped. Two forms of perfect
knowledge are priced:

  greedy    continue past depth d if and only if draft d+1 will be accepted;
  costaware continue past depth d if and only if the full cost-aware oracle
            would also have gone past d, so it can decline a draft that WILL
            be accepted when the next width costs more than it returns.

The difference between the two is the value of knowing the cost cliff, as
opposed to knowing the acceptance.

Run from `research/`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e128_price  # noqa: E402
from e128_price import (  # noqa: E402
    DECODE_TOKENS, RANKED_PROMPTS, load_board_receipt, ranked_round_us,
)
from e128_replay import (  # noqa: E402
    MAX_DEPTH, PRICE_CUMULATIVE, PRICE_MARGINAL, SEGMENTED_VERIFY_DEPTH_CAP,
)
from e134_rung2 import (  # noqa: E402
    build_legs, median_pct, oracle_depth, prompt_panel, simulate,
)

CAP = min(MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)

# `slopeonly_b6`, the E128 headline fit of OUR OWN measured round cost. The
# board curve that `e128_price` ships breaks at M >= 5; our own fit breaks at
# M >= 6 with a larger jump. Which curve is in force decides whether declining
# an acceptable draft can ever be cheaper, so both are priced here.
OUR_CURVE = {"breakpoint": 6,
             "lo": (27725.39691958033, 3446.0718068476417),
             "hi": (27725.396919580293, 5323.531364694667)}

# Our own curve after thorfinn's `{6:6, 7:7}` one-pass table promotes. Widths
# 6 and 7 stop reading the projection weights twice, so they fall onto our
# fitted one-pass line and the pass cliff moves from M=6 to M=8. Both segments
# are unchanged; only the breakpoint moves, which is exactly the parameter-free
# prediction FINDING 156 asked for. Pricing the tier family on this curve says
# whether a boundary-4 constant survives that table change.
ONEPASS67_CURVE = {"breakpoint": 8,
                   "lo": (27725.39691958033, 3446.0718068476417),
                   "hi": (27725.396919580293, 5323.531364694667)}
CURVES = {"ours": OUR_CURVE, "onepass67": ONEPASS67_CURVE}
CLIFF_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)
FLAT_LEVELS = (0.18, 0.19, 0.20, 0.22, 0.24, 0.27, 0.30, 0.35, 0.45, 0.60)
TIER_FACTORS = (1.0, 1.10, 1.20, 1.33, 1.50, 1.75, 2.0301, 2.50, 3.00, 4.2689)
FAMILIES = {"cliff": CLIFF_WEIGHTS, "flat": FLAT_LEVELS, "tier": TIER_FACTORS}


def greedy_at(boundaries):
    """Perfect acceptance knowledge at the named boundaries only."""
    wanted = frozenset(boundaries)

    def force(depth, ctx):
        if depth not in wanted:
            return None
        return ctx["capability"] > depth
    return force


def costaware_at(boundaries):
    """Perfect cost-aware knowledge at the named boundaries only."""
    wanted = frozenset(boundaries)

    def force(depth, ctx):
        if depth not in wanted:
            return None
        return oracle_depth(ctx["offer"], ctx["capability"]) > depth
    return force


def force_ratios(panel, force, windows):
    out = {}
    for prompt, entry in panel.items():
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       force=force)
        out[prompt] = {
            "ratio": run["us_per_token"] / entry["ship"]["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"]}
    return out


def oracle_ratios(panel, windows):
    out = {}
    for prompt, entry in panel.items():
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       oracle=True)
        out[prompt] = {
            "ratio": run["us_per_token"] / entry["ship"]["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"]}
    return out


def cliff_price(weight: float, cliff: int = 4) -> tuple:
    """The shipped flat price with the measured cliff placed at one boundary.

    `makeUniformDepthPrice` charges a flat `0.18` at every boundary. Our own
    fitted round cost charges a much larger step at the boundary where the
    projection weights are read a second time. This arm keeps the flat price
    everywhere else and raises only the cliff boundary, so it is the smallest
    implementable expression of the rung-3 decomposition. `weight = 0` is the
    shipped table exactly and `weight = 1` is the full measured step.

    Unlike the E128 `rankedprice` arm, which replaced every entry at once and
    lost 2.8508 percent, this changes one entry.
    """
    step = ranked_round_us(cliff + 2) - ranked_round_us(cliff + 1)
    flat = ranked_round_us(2) - ranked_round_us(1)
    ratio = step / flat if flat else 1.0
    marginal = list(PRICE_MARGINAL)
    marginal[cliff] = PRICE_MARGINAL[cliff] * (1.0 + weight * (ratio - 1.0))
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)], ratio


def boundary_price(weight: float, cliff: int = 4) -> tuple:
    """`makeBoundaryDepthPrice`, the shipped one-boundary form.

    `weight` is the tier factor. Unlike `cliff_price` this HOLDS THE TOTAL at
    `maxDepth * headStepCostRatio`, which is the convention every existing arm
    in `Qwen36MTPBlockSession` follows, so the shallow entries fall as the one
    priced entry rises. `weight = 1` is the shipped uniform table exactly.
    """
    count = len(PRICE_MARGINAL)
    total = count * 0.18
    within = total / (count - 1 + weight)
    marginal = [within] * count
    marginal[cliff] = within * weight
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)], weight


# `measuredRawDepthPrice` from `Qwen36MTPBlockSession.swift`. E68 rung 3
# measured this shape end to end on this host at -3.500 percent candidate MTP
# seconds per token, and E75 measured it at +0.33 percent on the crown kernel
# table. It is the only price shape in the tree with a real GPU result, so it
# is the external anchor for whether this replayer prices shapes correctly.
MEASURED_RAW = [0.26300121724709807, 0.29195567495854047,
                0.34642143034825884, 0.40231023217247086,
                0.63287276451077956, 0.43601634825870655,
                0.35457813598673293, 0.42510483416251998]


def measured_price() -> tuple:
    """`makeMeasuredDepthPrice`, rescaled to the shipped total."""
    total = len(PRICE_MARGINAL) * 0.18
    scale = total / sum(MEASURED_RAW)
    marginal = [value * scale for value in MEASURED_RAW]
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)]


def flat_price(level: float, cliff: int = 4) -> tuple:
    """A uniform price at `level` instead of the shipped `0.18`.

    This is the control that decides what the cliff arm is really doing. If a
    flat rise pays as well as a targeted rise, then the finding is only that
    the shipped price is too low, and nothing about the cliff.
    """
    marginal = [level] * len(PRICE_MARGINAL)
    cumulative = [1.0]
    for value in marginal:
        cumulative.append(cumulative[-1] + value)
    return marginal, cumulative[:len(PRICE_CUMULATIVE)], level / 0.18


def price_ratios(panel, price, windows):
    out = {}
    for prompt, entry in panel.items():
        run = simulate(None, entry["factory"](entry["p_target"]), windows,
                       price=price)
        out[prompt] = {
            "ratio": run["us_per_token"] / entry["ship"]["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"]}
    return out


def greedy_is_cost_optimal() -> tuple[int, list]:
    """Can declining a draft that WILL be accepted ever be cheaper?

    Under the price curve currently in force, compare greedy perfect knowledge
    with the full cost-aware oracle over every reachable `(offer, capability)`.
    If they never differ, the whole oracle gap is false-positive elimination
    and no cost-only rule can pay.
    """
    differ = []
    for offer in range(1, MAX_DEPTH + 1):
        for capability in range(0, MAX_DEPTH + 2):
            greedy = min(capability, min(offer, MAX_DEPTH),
                         SEGMENTED_VERIFY_DEPTH_CAP)
            aware = oracle_depth(offer, capability)
            if greedy != aware:
                differ.append({"offer": offer, "capability": capability,
                               "greedy": greedy, "costaware": aware})
    return len(differ), differ


def boundary_census(panel, windows):
    """How many rounds is each boundary even in play for, under shipped?

    A boundary that is asked in few rounds, or that is nearly always answered
    the same way, cannot carry much of the oracle gap whatever a classifier
    scores on it.
    """
    rows = {}
    for prompt, entry in panel.items():
        run = simulate(None, entry["factory"](entry["p_target"]), windows)
        rounds = run["rounds"]
        asked = run["boundary_asked"]
        needed = run["boundary_needed"]
        per_boundary = []
        for d in range(CAP):
            per_boundary.append({
                "depth": d,
                "share_asked": asked[d] / rounds if rounds else 0.0,
                "p_needed_given_asked": (needed[d] / asked[d]
                                         if asked[d] else float("nan")),
            })
        rows[prompt] = {"rounds": rounds, "boundaries": per_boundary}
    return rows


def summarise(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=here.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=here / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--curve", choices=("board",) + tuple(CURVES),
                    default="board")
    ap.add_argument("--cliff", type=int, default=4)
    ap.add_argument("--only-cliff", action="store_true")
    ap.add_argument("--family", choices=tuple(FAMILIES), default="cliff")
    ap.add_argument("--grid", default="",
                    help="comma separated grid that replaces the family "
                         "default, used to refine around a peak")
    ap.add_argument("--json", type=pathlib.Path,
                    default=here / "e134-artifacts/rung3-boundaries.json")
    args = ap.parse_args()

    if args.curve in CURVES:
        e128_price.CURVE = CURVES[args.curve]
    print("## price curve in force: %s" % args.curve)
    print("round us by rows  %s" % " ".join(
        "%.1f" % ranked_round_us(m) for m in range(1, MAX_DEPTH + 2)))
    n_differ, differ = greedy_is_cost_optimal()
    print("cases where the cost-aware oracle declines an acceptable draft: %d"
          % n_differ)
    for case in differ[:8]:
        print("   offer %d capability %d  greedy %d  cost-aware %d" % (
            case["offer"], case["capability"], case["greedy"],
            case["costaware"]))

    legs, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    print("legs %d ; rounds attached %d ; accept mismatches %d ; "
          "margin mismatches %d ; unmatched %d" % (
              gate["legs"], gate["attached"], gate["accept_mismatch"],
              gate["margin_mismatch"], gate["unmatched"]))

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]

    plans = []
    if not args.only_cliff:
        plans.append(("full_oracle", None, None))
        for d in range(CAP):
            plans.append(("greedy@%d" % d, "greedy", [d]))
        for d in range(CAP):
            plans.append(("greedy_from@%d" % d, "greedy", list(range(d, CAP))))
        for d in range(CAP):
            plans.append(("greedy_upto@%d" % d, "greedy",
                          list(range(0, d + 1))))
        if n_differ:
            for d in range(CAP):
                plans.append(("costaware@%d" % d, "costaware", [d]))
    make_price = {"cliff": cliff_price, "flat": flat_price,
                  "tier": boundary_price}[args.family]
    grid = FAMILIES[args.family]
    if args.grid:
        grid = tuple(float(part) for part in args.grid.split(","))
    tag = args.family + "price@%.4f"
    for weight in grid:
        plans.append((tag % weight, "price", weight))
    fixed = {"pb5": boundary_price(2.0301, 3)[:2],
             "pb6": boundary_price(2.0301, 4)[:2],
             "pb7": boundary_price(2.0301, 5)[:2],
             "pbfit": measured_price(),
             "rankedprice": e128_price.ranked_price_table()}
    for name in fixed:
        plans.append((name, "table", name))

    _, _, cliff_ratio = make_price(grid[-1], args.cliff)
    print("\n## the %s price family, cliff boundary %d"
          % (args.family, args.cliff))
    print("the top grid point is %.4f times a shallow step" % cliff_ratio)

    per_seed = {name: [] for name, _, _ in plans}
    cliff_by_seed = {seed: {} for seed in seeds}
    census = None
    panel = None
    for seed in seeds:
        panel = prompt_panel(legs, args.windows, args.fit_windows, seed)
        if census is None:
            census = boundary_census(panel, args.windows)
        for name, kind, boundaries in plans:
            if kind is None:
                ratios = oracle_ratios(panel, args.windows)
            elif kind == "greedy":
                ratios = force_ratios(panel, greedy_at(boundaries),
                                      args.windows)
            elif kind == "costaware":
                ratios = force_ratios(panel, costaware_at(boundaries),
                                      args.windows)
            elif kind == "table":
                ratios = price_ratios(panel, fixed[boundaries], args.windows)
            else:
                marginal, cumulative, _ = make_price(boundaries, args.cliff)
                ratios = price_ratios(panel, (marginal, cumulative),
                                      args.windows)
                cliff_by_seed[seed][boundaries] = ratios
            per_seed[name].append(median_pct(receipt, ratios))

    print("\n## the shipped population at each boundary (seed %d)" % seeds[0])
    print("%-10s %6s %8s %8s %8s %8s %8s %8s %8s" % (
        "prompt", "rounds", "d0", "d1", "d2", "d3", "d4", "d5", "d6"))
    for prompt in RANKED_PROMPTS:
        entry = census.get(prompt)
        if entry is None:
            continue
        shares = " ".join("%8.3f" % b["share_asked"]
                          for b in entry["boundaries"])
        print("%-10s %6d %s" % (prompt, entry["rounds"], shares))
    print("   share of rounds where the walk asks that boundary")
    print("%-10s %6s %8s %8s %8s %8s %8s %8s %8s" % (
        "prompt", "", "d0", "d1", "d2", "d3", "d4", "d5", "d6"))
    for prompt in RANKED_PROMPTS:
        entry = census.get(prompt)
        if entry is None:
            continue
        shares = " ".join("%8.3f" % b["p_needed_given_asked"]
                          for b in entry["boundaries"])
        print("%-10s %6s %s" % (prompt, "", shares))
    print("   P(the right answer is yes | the boundary is asked). A value")
    print("   near 1 leaves a discriminator almost nothing to reject.")

    print("\n## replayed ranked median percent, perfect knowledge at named "
          "boundaries")
    print("%-22s %10s %8s" % ("arm", "median %", "sd"))
    for name, _, _ in plans:
        mean, sd = summarise(per_seed[name])
        print("%-22s %10.4f %8.4f" % (name, mean, sd))

    increments = {}
    if not args.only_cliff:
        single = [(d, summarise(per_seed["greedy@%d" % d])[0])
                  for d in range(CAP)]
        total = summarise(per_seed["full_oracle"])[0]
        positive = sum(value for _, value in single if value > 0)
        print("\n## concentration")
        print("full cost-aware oracle            %+8.4f" % total)
        print("sum of positive single boundaries %+8.4f" % positive)
        best_d, best_v = max(single, key=lambda item: item[1])
        print("largest single boundary           %+8.4f at d=%d"
              % (best_v, best_d))
        if positive > 1e-9:
            print("largest share of the positive sum %8.3f"
                  % (best_v / positive))
        print("depth-4 boundary alone            %+8.4f"
              % dict(single).get(4, float("nan")))

        print("\n## marginal value of one boundary, later boundaries already "
              "perfect")
        print("%-8s %10s %10s" % ("boundary", "increment", "share"))
        for d in range(CAP):
            here_value = summarise(per_seed["greedy_from@%d" % d])[0]
            rest = (summarise(per_seed["greedy_from@%d" % (d + 1)])[0]
                    if d + 1 < CAP else 0.0)
            increments[d] = here_value - rest
        span = sum(v for v in increments.values() if v > 0)
        for d in range(CAP):
            print("%-8d %+10.4f %10.3f" % (
                d, increments[d],
                increments[d] / span if span else float("nan")))
        print("sum of increments %+.4f against the full oracle %+.4f"
              % (sum(increments.values()), total))

    print("\n## %s price, leave-one-prompt-out, which is the headline"
          % args.family)
    print("   For each ranked prompt the weight is chosen WITHOUT that")
    print("   prompt, then applied to it, at every seed.")
    in_sample = max(summarise(per_seed[tag % w])[0] for w in grid)
    lofo, chosen = [], {}
    for seed in seeds:
        ratios = {}
        for prompt in panel:
            best, best_w = None, None
            for weight in grid:
                others = {p: r for p, r in cliff_by_seed[seed][weight].items()
                          if p != prompt}
                value = median_pct(receipt, others)
                if best is None or value > best:
                    best, best_w = value, weight
            chosen.setdefault(prompt, []).append(best_w)
            ratios[prompt] = cliff_by_seed[seed][best_w][prompt]
        lofo.append(median_pct(receipt, ratios))
    lofo_mean, lofo_sd = summarise(lofo)
    print("in-sample %+.4f ; held-out %+.4f ; sd %.4f ; gap %+.4f"
          % (in_sample, lofo_mean, lofo_sd, lofo_mean - in_sample))
    for prompt, picks in chosen.items():
        print("   %-10s weights %s" % (
            prompt, " ".join("%.4f" % w for w in picks)))
    print("\n%-10s %10s %10s %10s" % (
        "weight", "median %", "mean depth", "accept"))
    for weight in grid:
        mean, sd = summarise(per_seed[tag % weight])
        sample = cliff_by_seed[seeds[0]][weight]
        depth = sum(RANKED_PROMPTS[p]["weight"] * v["mean_depth"]
                    for p, v in sample.items())
        accept = sum(RANKED_PROMPTS[p]["weight"] * v["accept_rate"]
                     for p, v in sample.items())
        print("%-10.4f %+10.4f %10.3f %10.3f" % (weight, mean, depth, accept))

    print("\n## the price arms that already exist in the Swift tree")
    print("   `pbfit` is the external anchor: E68 rung 3 measured it end to")
    print("   end on this host at -3.500 percent candidate MTP seconds per")
    print("   token, and E75 measured it at +0.33 percent on the crown table.")
    print("%-8s %12s %8s   %s" % ("arm", "median %", "sd", "marginal"))
    for name in fixed:
        mean, sd = summarise(per_seed[name])
        shape = " ".join("%.3f" % v for v in fixed[name][0])
        print("%-8s %+12.4f %8.4f   %s" % (name, mean, sd, shape))
    print("%-8s %+12.4f %8.4f   %s" % (
        "ship", 0.0, 0.0, " ".join("%.3f" % v for v in PRICE_MARGINAL)))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "gate": gate,
        "receipt": args.receipt,
        "windows": args.windows,
        "fit_windows": args.fit_windows,
        "seeds": seeds,
        "decode_tokens": DECODE_TOKENS,
        "cap": CAP,
        "curve": args.curve,
        "cliff": args.cliff,
        "family": args.family,
        "cliff_ratio": cliff_ratio,
        "cliff_lofo": {"in_sample": in_sample, "held_out": lofo_mean,
                       "sd": lofo_sd, "per_seed": lofo,
                       "chosen": chosen},
        "round_us": [ranked_round_us(m) for m in range(1, MAX_DEPTH + 2)],
        "costaware_differs": differ,
        "increments": increments,
        "census": census,
        "arms": {name: {"per_seed": values,
                        "mean": summarise(values)[0],
                        "sd": summarise(values)[1]}
                 for name, values in per_seed.items()},
    }, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
