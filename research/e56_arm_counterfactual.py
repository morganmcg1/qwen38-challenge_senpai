#!/usr/bin/env python3
"""E56 r2: price each weight-stream crossing separately, at rank and locally.

  python3 research/e56_arm_counterfactual.py [--windows N] [--seed N]

WHAT THE REVISION ASKS. Session 2 measured one arm that priced BOTH live QMV
crossings and won 2.35 % locally. The advisor's counterfactual then showed that
two thirds of that win came from narrowing verify width 9, a width the two
score-setting prompts almost never run, while the part that acts where the
ranked prompts live moved beagle the wrong way. This script decomposes the
price into the four arms the session measures, and prices each one twice: once
against a truth curve for THIS host class and once against a truth curve
transferred to the ranked M5.

THE ARMS. Each arm names the crossings its price is allowed to see. The three
redistribution arms keep the mean price pinned to `headStepCostRatio`, so none
of them is a disguised `h` retune; the two `h` arms change the mean on purpose,
because the advisor reports that 0.18 was mismeasured and the directly measured
value is 0.224:

  base     the shipped scalar price, flat at h = 0.18
  s45      prices only the 4 -> 5 crossing (weight streams 1 -> 2)
  s89      prices only the 8 -> 9 crossing, which E55 deleted from the live
           dispatch table; the arm holds that older geometry fixed so the
           erosion can be priced against an unchanged treatment
  h224     flat price at h = 0.224
  s45h224  the 4 -> 5 crossing priced at h = 0.224

The tables are rebuilt by `e56_walk_probe.cost_table` from the live Swift
constants, so this script and the binaries under test cannot disagree.

THE TRUTH CURVES, which is where session 2 went wrong. A price is what the
schedule believes; a truth curve is what the machine charges. Session 2 read
its ranked conclusion off an M4 Pro truth curve without the transfer factor.
Here the ranked curves come from the identified two-parameter transfer in
`research/e56_g_correction.py`:

    round_M5(d) = c1 * (1 + g * cum_M4(d) / C0_M4),  g in [0.7388, 0.7778]

and the local curve comes from this host's own pinned-width sweep, whose
crossing is 2.03x a within-tier row against the 1.40x the M4 Pro ladder
implies. That disagreement is not noise to be averaged away: it decides whether
the 4 -> 5 step is payable at all, so both are carried through to the end.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e53_width_mixture as E53
import e56_g_correction as G
import e56_stream_aware_schedule as E56
import e56_walk_probe as PROBE
import qmv_score_leverage as Q

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "e56-arm-counterfactual.json"

ARMS = ("base", "s45", "s89", "h224", "s45h224")

# Mean draft length published for the two score-setting prompts, from the
# receipt row the campaign measures against (E53).
RANKED_TARGET_MEAN = {"beagle": 485 / 107, "medicine": 472 / 99}

# This host's pinned-verify-width sweep (E56 session 2b, eight legs, palindrome
# order, every leg exact and thermally gated). Round seconds at verify widths
# 3, 4, 5 and 6.
LOCAL_ROUND_SECONDS = {3: 0.12912, 4: 0.15878, 5: 0.21549, 6: 0.24170}

# Measured, not assumed: the sequential per-draft acceptance of the base pair
# in E56 session 3 (research/e56_analyze.py --session s3, `accept p`). An
# earlier 0.99 was a guess and was too high, which made the local fixture look
# like a place where depth never has to be rationed.
LOCAL_FIXTURE_ACCEPT = 0.9625


def arm_prices(source: str) -> dict:
    """One `Price` per arm, rebuilt from the live Swift constants."""
    prices = {}
    for arm in ARMS:
        priced, head = PROBE.ARMS[arm]
        marginal, _ = PROBE.cost_table(source, priced, head)
        parts = []
        if priced:
            parts.append(f"prices crossings at verify width {sorted(priced)}")
        if head is not None:
            parts.append(f"head step cost ratio {head}")
        prices[arm] = E56.Price(arm, marginal,
                                ", ".join(parts) or "shipped scalar price")
    return prices


def ranked_truth(g: float) -> list[float]:
    """Ranked M5 round cost in microseconds, indexed by draft depth."""
    return [1000.0 * G.C1_M5 * (1.0 + g * G.CUM_M4[d] / G.C0_M4)
            for d in range(E53.MAX_DEPTH + 1)]


def reshaped_truth(truth: list[float], crossing_ratio: float,
                   ipg: dict) -> list[float]:
    """Same curve, but with the crossing steps at `crossing_ratio` x a row.

    The depth-8 round cost is preserved exactly, so this changes only the SHAPE
    of the ladder. It asks the question this host's sweep raises: if the ranked
    machine charges its weight-stream crossings the way this host does, does
    the conclusion move?
    """
    marginals = [truth[d + 1] - truth[d] for d in range(E53.MAX_DEPTH)]
    crosses = [PROBE.streams(d + 2, ipg) > PROBE.streams(d + 1, ipg)
               for d in range(E53.MAX_DEPTH)]
    total = sum(marginals)
    weight = sum(crossing_ratio if cross else 1.0 for cross in crosses)
    unit = total / weight
    out = [truth[0]]
    for cross in crosses:
        out.append(out[-1] + unit * (crossing_ratio if cross else 1.0))
    return out


def local_truth(ipg: dict) -> tuple[list[float], dict]:
    """This host's round cost in microseconds, from the pinned-width sweep.

    The sweep pins verify widths 3 to 6, so it measures the two within-tier
    steps that surround the 4 -> 5 crossing and the crossing itself. Depths 0
    and 1 are below the sweep and are filled with the M4 Pro ladder's shape
    scaled to this host's within-tier row, which is stated rather than hidden:
    no arm's decision turns on them, because every arm's first payable step is
    depth 0 at any acceptance the fixture reaches.
    """
    seconds = [LOCAL_ROUND_SECONDS[w] for w in (3, 4, 5, 6)]
    steps = [b - a for a, b in zip(seconds, seconds[1:])]   # 3->4, 4->5, 5->6
    within = (steps[0] + steps[2]) / 2.0
    crossing = steps[1]
    e1_within = statistics.fmean(
        [(E56.E1_C_US[d + 1] - E56.E1_C_US[d]) / 1e6 for d in (2, 4, 5, 6)])
    scale = within / e1_within
    shallow = [scale * (E56.E1_C_US[d + 1] - E56.E1_C_US[d]) / 1e6
               for d in (0, 1)]
    marginals = []
    for depth in range(E53.MAX_DEPTH):
        if depth < 2:
            marginals.append(shallow[depth])
        elif PROBE.streams(depth + 2, ipg) > PROBE.streams(depth + 1, ipg):
            marginals.append(crossing)
        else:
            marginals.append(within)
    depth0 = seconds[0] - marginals[0] - marginals[1]
    curve = [depth0]
    for step in marginals:
        curve.append(curve[-1] + step)
    provenance = {
        "measured_round_seconds": LOCAL_ROUND_SECONDS,
        "within_tier_step_seconds": within,
        "crossing_step_seconds": crossing,
        "crossing_over_within": crossing / within,
        "depth0_round_seconds": depth0,
        "shallow_steps_from_e1_scaled_by": scale,
    }
    return [1e6 * value for value in curve], provenance


def local_fixture_fit() -> dict:
    """A burst fit that is not bursty: the public fixture's flat acceptance."""
    return {"share_easy": 1.0, "persistence": 1.0,
            "q_easy": LOCAL_FIXTURE_ACCEPT, "q_hard": LOCAL_FIXTURE_ACCEPT,
            "mean_draft_len": None, "accept_ratio": LOCAL_FIXTURE_ACCEPT}


def replay_arms(fit: dict, prices: dict, truth: list[float], windows: int,
                seed: int) -> dict:
    """Every arm on one prompt and one truth curve, paired against `base`."""
    runs = {arm: E56.replay(fit, prices[arm], truth, windows, seed)
            for arm in ARMS}
    out = {}
    for arm in ARMS:
        run = runs[arm]
        delta = E56.paired_delta_pct(runs["base"], run)
        out[arm] = {
            "us_per_token": run["us_per_token"],
            "us_per_token_sem": run["us_per_token_sem"],
            "mean_draft_len": run["mean_draft_len"],
            "accept_ratio": run["accept_ratio"],
            "rounds": run["rounds"],
            "mixture": run["mixture"],
            "stops_at_width_5": run["mixture"].get(5, 0.0),
            "decode_pct": delta["decode_pct"],
            "decode_pct_sem": delta["decode_pct_sem"],
            "leg_pct": delta["leg_pct"],
            # A leg that takes less time is a speedup, and the score reads the
            # speedup. The advisor lost a sign here once by hand.
            "leg_gain_pct": -delta["leg_pct"],
        }
    return out


def decision_diff(fit: dict, prices: dict, truth: list[float], windows: int,
                  seed: int) -> dict:
    """Where each arm's walk disagrees with the shipped walk, and by how much.

    This is the mechanism question behind the beagle / medicine asymmetry: the
    same rule can cut depth in rounds where the extra rows were still paying.
    """
    import random

    out = {}
    for arm in ARMS:
        if arm == "base":
            continue
        shallower = deeper = same = 0
        depth_loss = 0
        for index in range(windows):
            rng_base = random.Random(seed + index)
            rng_arm = random.Random(seed + index)
            model_base = E53.BurstAcceptance(
                fit["share_easy"], fit["persistence"], fit["q_easy"],
                fit["q_hard"])
            model_arm = E53.BurstAcceptance(
                fit["share_easy"], fit["persistence"], fit["q_easy"],
                fit["q_hard"])
            sched_base = E56.PricedSchedule(prices["base"])
            sched_arm = E56.PricedSchedule(prices[arm])
            emitted_base = emitted_arm = 0
            # Drive both walks from the same acceptance draws for as long as
            # they stay in step. Once they diverge the comparison is no longer
            # round-for-round, so stop counting that window.
            while emitted_base < E53.TOTAL_TOKENS and emitted_arm < E53.TOTAL_TOKENS:
                probs_base, margin_base = model_base.step(rng_base)
                probs_arm, margin_arm = model_arm.step(rng_arm)
                depth_b = sched_base.depth(E53.OFFERED_DEPTH, margin_base)
                depth_a = sched_arm.depth(E53.OFFERED_DEPTH, margin_arm)
                if depth_a < depth_b:
                    shallower += 1
                    depth_loss += depth_b - depth_a
                elif depth_a > depth_b:
                    deeper += 1
                else:
                    same += 1
                draws = [rng_base.random() for _ in range(E53.MAX_DEPTH)]
                for _ in range(E53.MAX_DEPTH):
                    rng_arm.random()
                accepted_b = 0
                while accepted_b < depth_b and draws[accepted_b] < probs_base[accepted_b]:
                    accepted_b += 1
                accepted_a = 0
                while accepted_a < depth_a and draws[accepted_a] < probs_arm[accepted_a]:
                    accepted_a += 1
                sched_base.record(accepted_b, depth_b)
                sched_arm.record(accepted_a, depth_a)
                emitted_base += 1 + accepted_b
                emitted_arm += 1 + accepted_a
        rounds = shallower + deeper + same
        out[arm] = {
            "rounds_compared": rounds,
            "shallower_share": shallower / rounds if rounds else 0.0,
            "deeper_share": deeper / rounds if rounds else 0.0,
            "unchanged_share": same / rounds if rounds else 0.0,
            "mean_depth_given_shallower": (depth_loss / shallower
                                           if shallower else 0.0),
        }
    return out


def crossing_acceptance(prices: dict, depth: int = 3) -> dict:
    """Smallest flat acceptance at which each arm still takes a given step."""
    out = {}
    for arm in ARMS:
        marginal, cumulative = prices[arm].marginals, prices[arm].cum
        crossing = None
        for step in range(10000, 0, -1):
            p = step / 10000.0
            if PROBE.walk(p, marginal, cumulative) > depth:
                crossing = p
            else:
                break
        out[arm] = crossing
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=2000)
    parser.add_argument("--control-windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    source = PROBE.read_source()
    ipg = PROBE.inputs_per_group(source)
    prices = arm_prices(source)
    local_curve, local_provenance = local_truth(ipg)
    crossing_ratio = local_provenance["crossing_over_within"]

    truths = {
        "e1_m4_pro": list(E56.E1_C_US),
        "ranked_m5_g_lo": ranked_truth(G.G_BAND[0]),
        "ranked_m5_g_hi": ranked_truth(G.G_BAND[1]),
        "ranked_m5_g_lo_local_shape": reshaped_truth(
            ranked_truth(G.G_BAND[0]), crossing_ratio, ipg),
        "ranked_m5_g_hi_local_shape": reshaped_truth(
            ranked_truth(G.G_BAND[1]), crossing_ratio, ipg),
        "this_host": local_curve,
    }

    fits = E56.load_burst_fits()
    prompt_fits = {
        prompt: E56.central_fit(fits[prompt], RANKED_TARGET_MEAN[prompt])
        for prompt in RANKED_TARGET_MEAN
    }
    prompt_fits["local_fixture"] = local_fixture_fit()

    report = {
        "windows": args.windows,
        "seed": args.seed,
        "ipg": {str(k): v for k, v in ipg.items()},
        "prices": {
            arm: {
                "marginals": prices[arm].marginals,
                "cumulative": prices[arm].cum,
                "mean_marginal": prices[arm].mean_marginal(),
                "closed_steps": PROBE.closed_steps(prices[arm].marginals,
                                                   prices[arm].cum),
                "note": prices[arm].note,
            } for arm in ARMS
        },
        "crossing_ratio_used_by_the_price": PROBE.stream_cost_ratio(source),
        "truth_curves_us": truths,
        "local_truth_provenance": local_provenance,
        "smallest_flat_acceptance_that_still_crosses": {
            "depth3_width4_to_5": crossing_acceptance(prices, depth=3),
            "depth7_width8_to_9": crossing_acceptance(prices, depth=7),
        },
        "counterfactual": {},
        "score": {},
        "decision_diff": {},
    }

    for prompt, fit in prompt_fits.items():
        report["counterfactual"][prompt] = {
            "fit": {k: v for k, v in fit.items()
                    if k in ("share_easy", "persistence", "q_easy", "q_hard",
                             "mean_draft_len", "accept_ratio")},
        }
        for name, truth in truths.items():
            report["counterfactual"][prompt][name] = replay_arms(
                fit, prices, truth, args.windows, args.seed)

    # The revision asks for each arm's gain twice. `ranked_mixture` reads each
    # score-setting prompt's own acceptance process. `local_fixture` asks what
    # the same arm would be worth if both prompts behaved like the public local
    # fixture, which is the weighting a local-only reading implicitly applies.
    # The gap between the two is the transfer risk, stated as a number.
    for name in truths:
        ranked_block, local_block = {}, {}
        for arm in ARMS:
            ranked_gains = {
                prompt: report["counterfactual"][prompt][name][arm]["leg_gain_pct"]
                for prompt in RANKED_TARGET_MEAN
            }
            local_gain = (report["counterfactual"]["local_fixture"][name][arm]
                          ["leg_gain_pct"])
            local_gains = {prompt: local_gain for prompt in RANKED_TARGET_MEAN}
            ranked_block[arm] = {
                "leg_gain_pct": ranked_gains,
                "score_pct": Q.score_pct_from_leg_gains(ranked_gains),
            }
            local_block[arm] = {
                "leg_gain_pct": local_gains,
                "score_pct": Q.score_pct_from_leg_gains(local_gains),
            }
        report["score"][name] = {
            "ranked_mixture": ranked_block,
            "local_fixture": local_block,
            "gap_pct": {
                arm: (ranked_block[arm]["score_pct"]
                      - local_block[arm]["score_pct"]) for arm in ARMS
            },
        }

    for prompt, fit in prompt_fits.items():
        report["decision_diff"][prompt] = decision_diff(
            fit, prices, truths["this_host"], args.control_windows, args.seed)

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print_report(report)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


def print_report(report: dict) -> None:
    print("E56 r2 -- one arm per weight-stream crossing\n")
    print(f"crossing ratio inside the price: "
          f"{report['crossing_ratio_used_by_the_price']:.4f}  (this host, "
          f"round level)")
    print(f"crossing ratio this host measures in its truth curve: "
          f"{report['local_truth_provenance']['crossing_over_within']:.4f}")
    e1 = report["truth_curves_us"]["e1_m4_pro"]
    e1_marginals = [e1[d + 1] - e1[d] for d in range(8)]
    print(f"crossing ratio the M4 Pro ladder implies: "
          f"{e1_marginals[3] / ((e1_marginals[2] + e1_marginals[4]) / 2):.4f}"
          "  <- the two disagree, and the disagreement is load bearing\n")

    print("PRICE PER ARM (fraction of a depth-0 round; every mean is h)")
    print(f"  {'arm':<7}{'marginals':<66}closed")
    for arm, price in report["prices"].items():
        marginals = " ".join(f"{value:.4f}" for value in price["marginals"])
        print(f"  {arm:<7}{marginals:<66}{price['closed_steps']}")

    print("\nSMALLEST FLAT ACCEPTANCE THAT STILL TAKES THE STEP")
    for step, values in report["smallest_flat_acceptance_that_still_crosses"].items():
        print(f"  {step:<22}" + "  ".join(
            f"{arm}={'never' if value is None else format(value, '.4f')}"
            for arm, value in values.items()))

    for prompt in report["counterfactual"]:
        print(f"\n=== {prompt} ===")
        fit = report["counterfactual"][prompt]["fit"]
        print(f"  fit q_easy={fit['q_easy']:.4f} q_hard={fit['q_hard']:.4f} "
              f"share_easy={fit['share_easy']:.4f} "
              f"persistence={fit['persistence']:.4f}")
        for truth in report["truth_curves_us"]:
            block = report["counterfactual"][prompt][truth]
            print(f"  truth {truth}")
            print(f"    {'arm':<7}{'us/token':>10}{'decode %':>10}"
                  f"{'draft len':>11}{'width 5 share':>15}")
            for arm in ARMS:
                row = block[arm]
                print(f"    {arm:<7}{row['us_per_token']:>10.1f}"
                      f"{row['decode_pct']:>10.4f}"
                      f"{row['mean_draft_len']:>11.4f}"
                      f"{row['stops_at_width_5']:>15.4f}")

    print("\nSCORE PER ARM, WEIGHTED TWO WAYS (positive is a better score)")
    print("  ranked = each prompt's own acceptance; local = both prompts")
    print("  weighted as though they behaved like the public local fixture")
    for truth, block in report["score"].items():
        print(f"  {truth}")
        print(f"    {'arm':<7}{'ranked score':>14}{'local score':>14}"
              f"{'gap':>10}   ranked leg gains")
        for arm in ARMS:
            gains = "  ".join(
                f"{name} {value:+.4f} %" for name, value
                in block["ranked_mixture"][arm]["leg_gain_pct"].items())
            print(f"    {arm:<7}"
                  f"{block['ranked_mixture'][arm]['score_pct']:>+14.4f}"
                  f"{block['local_fixture'][arm]['score_pct']:>+14.4f}"
                  f"{block['gap_pct'][arm]:>+10.4f}   {gains}")

    print("\nWHERE THE ARMS DISAGREE WITH THE SHIPPED WALK")
    for prompt, arms in report["decision_diff"].items():
        print(f"  {prompt}")
        for arm, row in arms.items():
            print(f"    {arm:<7}shallower {row['shallower_share']:.4f}  "
                  f"deeper {row['deeper_share']:.4f}  "
                  f"unchanged {row['unchanged_share']:.4f}  "
                  f"mean rows cut {row['mean_depth_given_shallower']:.2f}")


if __name__ == "__main__":
    main()
