#!/usr/bin/env python3
"""E209 stage 0: replay the pre-registered depth controllers on the E168 corpus.

    python3 research/e209_desk.py [--draws N] [--json OUT]

The controllers are defined in `research/e209_controllers.py`, which is
committed before this replay runs. Everything here is harness=local desk
replay at cap 7 on the E168 `p7` pinned-depth-7 corpus, using the E207 port
whose positive control reproduced 1955 recorded rounds with 0 mismatches.

THE LEG RUNNER IS NEW, AND IT MATTERS. E207 priced a policy as a RATE over the
corpus round pool, `sum(cost) / sum(tokens)`. That is correct for a stationary
policy but it MIS-PRICES a settle window: a fixed 32-round window is a
different fraction of a 170-round leg than of a 250-round leg, and a controller
that locks a shallow level needs more rounds to emit the same 512 tokens. This
runner therefore replays a WHOLE LEG: it draws rounds from the exchangeable
pool until exactly 512 tokens are committed, and it narrows the last round to
the remaining token budget exactly as the trusted parent does. The settle tax
is then priced automatically, with no hand correction.

EXCHANGEABILITY AND WRAPAROUND. E203 stage 0a measured the accept-length
sequence to be exchangeable within a prompt (pooled rho1 = +0.0014, null sigma
0.0239), so a round is drawn as an exchangeable (margin, accept) pair. A leg
can need more rounds than the pool holds, so the order is four independent
permutations concatenated; a shallow-locked leg therefore re-uses rounds, which
is a resampling of the same exchangeable pool and not new information.

PAIRING. Every arm in a draw sees the SAME round order. The reported statistic
is the per-draw paired delta against the shipped rule, so the order effect --
which is the whole of the live rule's variance -- cancels in the mean and is
visible in the spread.

CENSORING IS PAID, NOT ASSUMED AWAY. A round played at depth d updates the EMA
from `accepted = min(k, d)` only, exactly as `recordAcceptOutcome` does. A
controller that holds a shallow level never observes the deeper positions and
its EMA degrades there. That is a real cost of the mechanism and the replay
charges it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e207_desk as D  # noqa: E402
import e209_controllers as C  # noqa: E402

PROMPTS = ["benchfixture", "dramatic", "english", "medicine", "narrative",
           "natural_history", "philosophy", "technical", "travel"]
PROSE = [p for p in PROMPTS if p != "benchfixture"]
CAP = 7
LEG_TOKENS = 512

# FINDING 540 reference values, prose-population median, harness=local.
FINDING_540_BEST_FIXED_PCT = 2.589
FINDING_540_VARIANCE_ONLY_PCT = 2.460

# Pre-registered decision thresholds from the assignment.
PROMOTE_MEDIAN_PCT = 1.0
PROMOTE_MIN_PER_PROMPT_PCT = -0.25
CLOSE_MEDIAN_PCT = 0.5


class ConstDepth:
    """Hindsight reference arm: one fixed depth for the whole leg."""

    kind = "reference"

    def __init__(self, depth):
        self.depth = depth
        self.name = "fixed-d%d" % depth

    def choose(self, _ema, _margin, cap, _round_index):
        return min(self.depth, cap)


def run_leg(pairs, factory, order, cap=CAP, tokens=LEG_TOKENS):
    """Replay one whole 512-token leg under one controller.

    The last round is narrowed to the remaining token budget, so the leg
    commits exactly `tokens` tokens and `decode_ms_per_token` is the total
    round cost divided by that fixed window.
    """
    controller = factory()
    ema = list(D.PRIOR)
    emitted = 0
    cost = 0.0
    depths = []
    accepted_total = 0
    index = 0
    while emitted < tokens:
        pair = pairs[order[index % len(order)]]
        index += 1
        round_cap = min(cap, tokens - emitted - 1)
        depth = controller.choose(ema, pair["m"], round_cap, len(depths))
        depth = max(0, min(depth, round_cap))
        accepted = min(pair["k"], depth)
        cost += D.round_ms(depth)
        emitted += 1 + accepted
        accepted_total += accepted
        depths.append(depth)
        D.update_ema(ema, accepted, depth)
    switches = sum(1 for a, b in zip(depths, depths[1:]) if a != b)
    return {"rounds": len(depths),
            "decode_ms_per_token": cost / tokens,
            "leg_ms_per_token": (D.SEED_MS + cost) / tokens,
            "ms_per_round": cost / len(depths),
            "mean_depth": sum(depths) / len(depths),
            "mean_accepted": accepted_total / len(depths),
            "switches": switches,
            "final_depth": depths[-1]}


def draw_order(pool_size, seed):
    """Four independent permutations concatenated: enough for any depth-0 leg."""
    rng = random.Random(seed)
    order = []
    for _ in range(4):
        block = list(range(pool_size))
        rng.shuffle(block)
        order.extend(block)
    return order


def evaluate(pairs, draws, seed_base):
    """Paired replay of every arm over the same draws."""
    orders = [draw_order(len(pairs), seed_base + n) for n in range(draws)]
    results = {}
    baseline = [run_leg(pairs, C.Shipped, order) for order in orders]
    results["shipped"] = summarize("shipped", "baseline", baseline, baseline)
    for factory in C.registry(CAP):
        legs = [run_leg(pairs, factory, order) for order in orders]
        name = factory().name
        results[name] = summarize(name, factory().kind, legs, baseline)
    for depth in range(CAP + 1):
        factory = (lambda d=depth: ConstDepth(d))
        legs = [run_leg(pairs, factory, order) for order in orders]
        results["fixed-d%d" % depth] = summarize(
            "fixed-d%d" % depth, "reference", legs, baseline)
    return results, baseline


def summarize(name, kind, legs, baseline):
    paired = [100.0 * (b["decode_ms_per_token"] - a["decode_ms_per_token"])
              / b["decode_ms_per_token"] for a, b in zip(legs, baseline)]
    paired_leg = [100.0 * (b["leg_ms_per_token"] - a["leg_ms_per_token"])
                  / b["leg_ms_per_token"] for a, b in zip(legs, baseline)]
    return {
        "name": name, "kind": kind,
        "pct_faster": statistics.fmean(paired),
        "pct_faster_sd": statistics.pstdev(paired) if len(paired) > 1 else 0.0,
        "pct_faster_leg_endpoint": statistics.fmean(paired_leg),
        "decode_ms_per_token": statistics.fmean(
            leg["decode_ms_per_token"] for leg in legs),
        "ms_per_round": statistics.fmean(leg["ms_per_round"] for leg in legs),
        "rounds": statistics.fmean(leg["rounds"] for leg in legs),
        "mean_depth": statistics.fmean(leg["mean_depth"] for leg in legs),
        "mean_accepted": statistics.fmean(leg["mean_accepted"] for leg in legs),
        "switches_per_leg": statistics.fmean(leg["switches"] for leg in legs),
        "final_depth_mode": statistics.mode(
            [leg["final_depth"] for leg in legs]),
    }


def ms_per_round_equivalent(row, shipped):
    """Convert the ms/token gain to ms/round at the shipped tokens-per-round.

    Same conversion as E207, so the number is comparable with the 0.2 ms/round
    composition bar and with FINDING 540.
    """
    gain = shipped["decode_ms_per_token"] - row["decode_ms_per_token"]
    return gain * (LEG_TOKENS / shipped["rounds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=209)
    parser.add_argument("--json",
                        default="research/e209-artifacts/stage0-controllers.json")
    args = parser.parse_args()

    per_prompt = {}
    for prompt in PROMPTS:
        pairs = D.corpus(prompt)
        if not pairs:
            continue
        results, _ = evaluate(pairs, args.draws, args.seed)
        shipped = results["shipped"]
        for row in results.values():
            row["ms_per_round_equivalent"] = ms_per_round_equivalent(
                row, shipped)
        best_fixed = min(
            (results["fixed-d%d" % d] for d in range(CAP + 1)),
            key=lambda row: row["decode_ms_per_token"])
        level = int(shipped["mean_depth"] + 0.5)
        per_prompt[prompt] = {
            "pool_rounds": len(pairs),
            "arms": results,
            "best_fixed_depth": int(best_fixed["name"].split("d")[-1]),
            "best_fixed_pct": best_fixed["pct_faster"],
            "variance_only_depth": level,
            "variance_only_pct": results["fixed-d%d" % level]["pct_faster"],
        }

    names = [name for name in per_prompt["dramatic"]["arms"]
             if per_prompt["dramatic"]["arms"][name]["kind"]
             not in ("baseline", "reference")]

    summary = {"harness": "local", "cap": CAP, "draws": args.draws,
               "leg_tokens": LEG_TOKENS,
               "corpus": "E168 p7 pinned depth 7, whole-leg replay",
               "finding540_best_fixed_pct": FINDING_540_BEST_FIXED_PCT,
               "finding540_variance_only_pct": FINDING_540_VARIANCE_ONLY_PCT}

    # The hindsight ceiling re-measured under THIS leg runner, so "fraction
    # captured" is a ratio inside one instrument rather than across two.
    ceiling = statistics.median(
        [per_prompt[p]["best_fixed_pct"] for p in PROSE])
    variance_only = statistics.median(
        [per_prompt[p]["variance_only_pct"] for p in PROSE])
    summary["ceiling_prose_median_pct"] = ceiling
    summary["variance_only_prose_median_pct"] = variance_only

    table = {}
    for name in names:
        prose = [per_prompt[p]["arms"][name]["pct_faster"] for p in PROSE]
        table[name] = {
            "prose_median_pct": statistics.median(prose),
            "prose_min_pct": min(prose),
            "prose_max_pct": max(prose),
            "benchfixture_pct":
                per_prompt["benchfixture"]["arms"][name]["pct_faster"],
            "prose_median_ms_per_round": statistics.median(
                [per_prompt[p]["arms"][name]["ms_per_round_equivalent"]
                 for p in PROSE]),
            "switches_per_leg": statistics.fmean(
                [per_prompt[p]["arms"][name]["switches_per_leg"]
                 for p in PROSE]),
            "shipped_switches_per_leg": statistics.fmean(
                [per_prompt[p]["arms"]["shipped"]["switches_per_leg"]
                 for p in PROSE]),
            "fraction_of_ceiling": statistics.median(prose) / ceiling,
            "promotion_eligible": name not in C.PROMOTION_INELIGIBLE,
            "per_prompt_pct": {p: per_prompt[p]["arms"][name]["pct_faster"]
                               for p in PROMPTS if p in per_prompt},
            "per_prompt_sd": {p: per_prompt[p]["arms"][name]["pct_faster_sd"]
                              for p in PROMPTS if p in per_prompt},
        }
    summary["controllers"] = table

    eligible = {n: r for n, r in table.items() if r["promotion_eligible"]}
    best = max(eligible, key=lambda n: eligible[n]["prose_median_pct"])
    summary["best_controller"] = best
    summary["best_prose_median_pct"] = eligible[best]["prose_median_pct"]
    summary["best_prose_min_pct"] = eligible[best]["prose_min_pct"]

    # Leave-one-prompt-out: choose the configuration on 7 prose prompts, score
    # it on the 8th. 15 configurations against 8 prompts overfit easily, so the
    # honest statistic is the held-out one.
    lopo = {}
    for held in PROSE:
        rest = [p for p in PROSE if p != held]
        pick = max(eligible, key=lambda n: statistics.median(
            [per_prompt[p]["arms"][n]["pct_faster"] for p in rest]))
        lopo[held] = {"picked": pick,
                      "held_out_pct": per_prompt[held]["arms"][pick]
                      ["pct_faster"]}
    summary["lopo"] = lopo
    summary["lopo_median_pct"] = statistics.median(
        [row["held_out_pct"] for row in lopo.values()])
    summary["lopo_min_pct"] = min(row["held_out_pct"] for row in lopo.values())

    median = summary["best_prose_median_pct"]
    minimum = summary["best_prose_min_pct"]
    if median >= PROMOTE_MEDIAN_PCT and minimum >= PROMOTE_MIN_PER_PROMPT_PCT:
        verdict = "STAGE 1: promote to the candidate surface behind an arm"
    elif median < CLOSE_MEDIAN_PCT:
        verdict = "CLOSE: the family has a measured PRACTICAL ceiling below " \
                  "+0.5%; decisive negative"
    elif median >= PROMOTE_MEDIAN_PCT:
        verdict = "REPORT AND STOP: median clears +1.0%% but the minimum " \
                  "per-prompt delta %.3f%% is below the -0.25%% robustness " \
                  "floor" % minimum
    else:
        verdict = "REPORT AND STOP: +0.5%% to +1.0%% band; the advisor decides"
    summary["verdict"] = verdict

    lines = render(summary, per_prompt, names)
    print(lines)

    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"summary": summary, "per_prompt": per_prompt}, indent=1,
        sort_keys=True) + "\n")
    out.with_suffix(".txt").write_text(lines + "\n")
    print("\nwrote %s" % out)


def render(summary, per_prompt, names):
    lines = []
    lines.append("E209 stage 0: online depth controllers (harness=local, "
                 "cap %d, %d draws)" % (CAP, summary["draws"]))
    lines.append("corpus E168 p7; whole-leg replay to exactly %d tokens; "
                 "paired round orders" % LEG_TOKENS)
    lines.append("")
    lines.append("HINDSIGHT CEILING re-measured under this leg runner "
                 "(prose median):")
    lines.append("  best fixed depth per prompt : %+.3f%%   "
                 "(FINDING 540 rate-based: %+.3f%%)"
                 % (summary["ceiling_prose_median_pct"],
                    FINDING_540_BEST_FIXED_PCT))
    lines.append("  variance-only (shipped level): %+.3f%%   "
                 "(FINDING 540 rate-based: %+.3f%%)"
                 % (summary["variance_only_prose_median_pct"],
                    FINDING_540_VARIANCE_ONLY_PCT))
    lines.append("")
    lines.append("%-14s %9s %9s %9s %9s %8s %8s %7s"
                 % ("controller", "prose med", "prose min", "prose max",
                    "bench", "ms/rd", "switch", "frac"))
    for name in names:
        row = summary["controllers"][name]
        flag = "" if row["promotion_eligible"] else "  DIAG"
        lines.append("%-14s %+9.3f %+9.3f %+9.3f %+9.3f %+8.3f %8.1f %7.2f%s"
                     % (name, row["prose_median_pct"], row["prose_min_pct"],
                        row["prose_max_pct"], row["benchfixture_pct"],
                        row["prose_median_ms_per_round"],
                        row["switches_per_leg"], row["fraction_of_ceiling"],
                        flag))
    shipped_switches = summary["controllers"][names[0]][
        "shipped_switches_per_leg"]
    lines.append("")
    lines.append("shipped rule depth switches per leg (prose mean): %.1f"
                 % shipped_switches)
    lines.append("")
    lines.append("BEST PROMOTION-ELIGIBLE: %s  prose median %+.3f%%  "
                 "prose min %+.3f%%"
                 % (summary["best_controller"],
                    summary["best_prose_median_pct"],
                    summary["best_prose_min_pct"]))
    lines.append("  fraction of the hindsight ceiling captured: %.2f"
                 % summary["controllers"][summary["best_controller"]]
                 ["fraction_of_ceiling"])
    lines.append("")
    lines.append("PER-PROMPT DELTAS for %s" % summary["best_controller"])
    best_row = summary["controllers"][summary["best_controller"]]
    for prompt in PROMPTS:
        if prompt not in best_row["per_prompt_pct"]:
            continue
        tag = " (benchfixture, reported separately)" \
            if prompt == "benchfixture" else ""
        lines.append("  %-16s %+7.3f%%  (sd over draws %.3f)%s"
                     % (prompt, best_row["per_prompt_pct"][prompt],
                        best_row["per_prompt_sd"][prompt], tag))
    lines.append("")
    lines.append("LEAVE-ONE-PROMPT-OUT (configuration chosen on 7 prose "
                 "prompts, scored on the 8th)")
    for held, row in summary["lopo"].items():
        lines.append("  hold out %-16s pick %-14s -> %+7.3f%%"
                     % (held, row["picked"], row["held_out_pct"]))
    lines.append("  held-out median %+.3f%%   held-out min %+.3f%%"
                 % (summary["lopo_median_pct"], summary["lopo_min_pct"]))
    lines.append("")
    lines.append("PRE-REGISTERED DECISION: %s" % summary["verdict"])
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
