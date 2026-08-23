#!/usr/bin/env python3
"""E150 R0.5b: bracket the linearised rule across both campaign cost curves.

`harness=local`. Zero GPU. Zero Swift. Rule 79 is not engaged: every number
is an offline replay price, not a timing contrast.

Why this module exists
----------------------
Advisor feedback F2 promoted Rule 151. The mechanism R0.5 selected does not
inherit width->= 6 mass from the shipped schedule, it MANUFACTURES its own, so
the Rule 143 mass discount does not apply to it. What it is exposed to is the
per-width COST curve, and the campaign owns two curves that disagree exactly
on the two steps this rule crosses:

    step      replayed    measured/k     ratio
    4->5        3446.1        5759.0     1.67x
    5->6       16241.3       13851.1     0.85x
    6->7        1626.1       12295.1     7.56x   <-- the rule crosses here
    7->8        7490.5        1876.2     0.25x   <-- and here

They agree within 1.7x everywhere else. So a single-curve price is not a
result, it is one corner of a bracket, and the honest deliverable is the whole
2x2 of "which curve chose the policy" against "which curve priced it".

The four cells
--------------
Let `L(c)` be the linearised rule with clamp off and mu* solved on curve `c`,
and let `S` be the shipped ratio rule with the clamp on, which is what deploys
today. Every median percent below is already relative to that curve's own base
run, so a gain is a difference of two percentages on ONE curve.

    gain_on_measured   = L(measured) priced on measured  -  S on measured
    gain_on_replayed   = L(replayed) priced on replayed  -  S on replayed
    transfer_to_*      = L(other curve) priced here      -  S here
    cross_regret       = transfer_to_*  -  gain_on_*      (never positive)

`e150_policy_gain_min_over_curves_pp` is the worst self-consistent cell, which
is the number the advisor asked for to decide the submission. The transfer
cells are reported beside it because they are the real deployment risk: E145
R7-1 already showed `argmax_full` chosen on the replayed curve scores -3.1234
when it is priced on the measured curve.

Decoupling decision from evaluation
-----------------------------------
`score_arm` installs the evaluation curve and hands the same price table to
the walker, so by default a policy always believes whatever curve is pricing
it. `believe` wraps a chooser so it ignores that table and keeps its own. The
policy then believes one curve while reality bills it on the other, which is
exactly the transfer question.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import DECODE_TOKENS, RANKED_PROMPTS, ranked_round_us  # noqa: E402,E501
from e128_replay import SEGMENTED_VERIFY_DEPTH_CAP  # noqa: E402
from e134_rung2 import simulate  # noqa: E402
from e140_cells import install  # noqa: E402
from e150_lib import (  # noqa: E402
    R7_SHIPPED_PCT, REPRO_TOLERANCE_PP, build_env, score_arm, write_artifact)
from e150_predict import fixed_state_walker  # noqa: E402
from e150_r05 import BISECTION_MAX_STEPS, BISECTION_TOLERANCE, STATES  # noqa: E402,E501

# The bisection bracket. R0.5 solved every measured-curve cell inside it.
MU_LO, MU_HI = 0.20, 0.90


def believe(inner, decide_price):
    """Freeze the price table a chooser reasons with.

    Without this the chooser is handed the evaluation curve's table by
    `simulate`, so a cross cell would silently become a self-consistent one.
    """
    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        return inner(ema, margin, offer, adjust, ctx, force, decide_price)
    return chooser


def curve_of(env, name: str):
    """`(curve, price)` for a campaign curve named `measured` or `replayed`."""
    if name == "measured":
        return env.measured, env.measured_price
    if name == "replayed":
        return env.replayed, env.replayed_price
    raise SystemExit("unknown curve %r" % name)


def achieved_cost_per_token(env, curve_name: str, state_name: str, rule: str,
                            clamp, mu: float) -> float:
    """Normalised cost per emitted token when the policy prices at `mu`."""
    curve, price = curve_of(env, curve_name)
    install(curve)
    unit = ranked_round_us(1)
    total_us = total_tokens = 0.0
    for seed in env.seeds:
        for prompt in RANKED_PROMPTS:
            entry = env.cache[(seed, prompt)]
            walker = believe(
                fixed_state_walker(STATES[state_name](seed, prompt, entry),
                                   rule, clamp, fixed_lam=mu,
                                   cap_limit=SEGMENTED_VERIFY_DEPTH_CAP),
                price)
            install(curve)
            row = simulate(None, entry["factory"](entry["p_target"]),
                           env.windows, price=price, walker=walker)
            total_us += row["us_per_token"] * DECODE_TOKENS
            total_tokens += DECODE_TOKENS
    return (total_us / total_tokens) / unit


def solve_mu(env, curve_name: str, state_name: str, rule: str, clamp) -> dict:
    """Bisect `achieved_cost_per_token(mu) - mu` on one curve."""
    trace = []

    def g(mu: float) -> float:
        achieved = achieved_cost_per_token(env, curve_name, state_name, rule,
                                           clamp, mu)
        trace.append({"mu": mu, "achieved": achieved, "residual": achieved - mu})
        return achieved - mu

    lo, hi = MU_LO, MU_HI
    g_lo, g_hi = g(lo), g(hi)
    if g_lo < 0.0 or g_hi > 0.0:
        raise SystemExit(
            "no sign change on [%g, %g] for curve %s: g(lo)=%+0.6f "
            "g(hi)=%+0.6f" % (lo, hi, curve_name, g_lo, g_hi))
    steps = 0
    while hi - lo > BISECTION_TOLERANCE and steps < BISECTION_MAX_STEPS:
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        steps += 1
    mu_star = 0.5 * (lo + hi)
    return {
        "curve": curve_name,
        "mu_star": mu_star,
        "lambda_star": 1.0 / mu_star,
        "serial_round_us": trace and ranked_round_us(1),
        "bisection_steps": steps,
        "converged": hi - lo <= BISECTION_TOLERANCE,
        "trace": trace,
    }


def price_cell(env, decide_curve: str, eval_curve: str, state_name: str,
               rule: str, clamp, mu) -> dict:
    """Price one policy, decided on one curve and billed on another."""
    _, decide_price = curve_of(env, decide_curve)
    eval_curve_obj, eval_price = curve_of(env, eval_curve)

    def make(seed, prompt, entry):
        lam = mu.get(prompt) if isinstance(mu, dict) else mu
        return believe(
            fixed_state_walker(STATES[state_name](seed, prompt, entry),
                               rule, clamp, fixed_lam=lam,
                               cap_limit=SEGMENTED_VERIFY_DEPTH_CAP),
            decide_price)
    return score_arm(env, eval_curve, eval_curve_obj, eval_price, make)


def row(name: str, cell: dict) -> None:
    print("  %-42s %+9.4f  sd %.4f  depth %.4f  inadm %.4f"
          % (name, cell["median_pct_mean"], cell["median_pct_sd"],
             cell["weighted_mean_depth"], cell["frac_rounds_inadmissible"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--json", type=str, default="curve_bracket.json")
    args = ap.parse_args()

    env = build_env(args.windows, args.seeds)
    print("E150 R0.5b - the linearised rule across both campaign cost curves")
    print("  harness=local  gpu_used=False  rule_79=not_engaged")
    print("  Rule 151: no mass discount. This mechanism makes its own "
          "width-8 mass.")
    print("  windows %d  seeds %d" % (args.windows, args.seeds))

    # ---------------------------------------------------------- the control
    # The measured/shipped/clamped cell is the published R7 number. If it
    # moves, the environment is wrong and no bracket below is readable.
    # `greedy` is the rule that ships: continue while marginal gain per
    # marginal cost beats the round's own average. `ratio` is the linearised
    # rule R0.5 selected, so using it here would compare the candidate with
    # itself and quietly report a gain of zero against the wrong reference.
    shipped = {}
    for curve in ("measured", "replayed"):
        shipped[curve] = price_cell(env, curve, curve, "shipped", "greedy",
                                    None, None)
    print("\n## the deployed policy on each curve")
    for curve in ("measured", "replayed"):
        row("shipped greedy, clamp on, %s" % curve, shipped[curve])
    control = abs(shipped["measured"]["median_pct_mean"] - R7_SHIPPED_PCT)
    control_ok = control < REPRO_TOLERANCE_PP
    print("  control |measured shipped - published R7| %.4f pp  pass %s"
          % (control, control_ok))

    # ------------------------------------------------------------- solve mu
    print("\n## mu* solved separately on each curve, clamp off")
    solved = {}
    for curve in ("measured", "replayed"):
        solved[curve] = solve_mu(env, curve, "shipped", "ratio", {})
        print("  %-10s mu* %.9f  lambda* %.4f  steps %d  converged %s"
              % (curve, solved[curve]["mu_star"], solved[curve]["lambda_star"],
                 solved[curve]["bisection_steps"],
                 solved[curve]["converged"]))

    # ------------------------------------------------------------ the 2 x 2
    print("\n## the 2x2: which curve chose the policy, which curve billed it")
    cells = {}
    for decide in ("measured", "replayed"):
        for evaluate in ("measured", "replayed"):
            key = "decide_%s_eval_%s" % (decide, evaluate)
            cells[key] = price_cell(env, decide, evaluate, "shipped", "ratio",
                                    {}, solved[decide]["mu_star"])
            row(key, cells[key])

    def gain(decide: str, evaluate: str) -> float:
        return (cells["decide_%s_eval_%s" % (decide, evaluate)]
                ["median_pct_mean"]
                - shipped[evaluate]["median_pct_mean"])

    gain_measured = gain("measured", "measured")
    gain_replayed = gain("replayed", "replayed")
    transfer_to_replayed = gain("measured", "replayed")
    transfer_to_measured = gain("replayed", "measured")
    regret_on_replayed = transfer_to_replayed - gain_replayed
    regret_on_measured = transfer_to_measured - gain_measured

    print("\n## gains against the deployed policy on the SAME curve")
    print("  %-46s %+9.4f pp" % ("chosen on measured, billed on measured",
                                 gain_measured))
    print("  %-46s %+9.4f pp" % ("chosen on replayed, billed on replayed",
                                 gain_replayed))
    print("\n## transfer: the policy believes one curve, reality bills the other")
    print("  %-46s %+9.4f pp" % ("chosen on measured, billed on replayed",
                                 transfer_to_replayed))
    print("  %-46s %+9.4f pp" % ("chosen on replayed, billed on measured",
                                 transfer_to_measured))
    print("  %-46s %+9.4f pp" % ("cross regret on replayed",
                                 regret_on_replayed))
    print("  %-46s %+9.4f pp" % ("cross regret on measured",
                                 regret_on_measured))
    if max(regret_on_replayed, regret_on_measured) > 0.0:
        # Not a bug. mu* is the fixed point of POOLED normalised cost per
        # token, while the score is a receipt-weighted MEDIAN over per-prompt
        # ratios. The fixed point does not maximise that median, so a policy
        # carrying the other curve's scalar can score higher on this one.
        print("  note: a positive regret is possible because mu* is a fixed "
              "point of pooled\n        cost per token, not an argmax of the "
              "receipt-weighted median.")

    worst_self = min(gain_measured, gain_replayed)
    worst_any = min(gain_measured, gain_replayed, transfer_to_replayed,
                    transfer_to_measured)

    out = {
        "harness": "local",
        "gpu_used": False,
        "rung": "E150 R0.5b curve bracket",
        "frame": "replay_median_pct_frame",
        "rule_151": ("no Rule 143 mass discount: the mechanism creates its "
                     "own width-8 mass rather than inheriting it"),
        "windows": args.windows,
        "seeds": env.seeds,
        "e150_curve_bracket_control_pp": control,
        "e150_curve_bracket_control_ok": control_ok,
        "e150_mu_star_by_curve": {c: solved[c]["mu_star"] for c in solved},
        "e150_mu_star_solution": solved,
        "e150_shipped_by_curve": {
            c: shipped[c]["median_pct_mean"] for c in shipped},
        "e150_curve_bracket_cells": {
            k: {"median_pct": v["median_pct_mean"],
                "median_pct_sd": v["median_pct_sd"],
                "weighted_mean_depth": v["weighted_mean_depth"],
                "frac_rounds_inadmissible": v["frac_rounds_inadmissible"],
                "width_histogram": v["width_histogram"]}
            for k, v in cells.items()},
        "e150_policy_gain_on_measured_pct": gain_measured,
        "e150_policy_gain_on_replayed_pct": gain_replayed,
        "e150_policy_transfer_to_replayed_pp": transfer_to_replayed,
        "e150_policy_transfer_to_measured_pp": transfer_to_measured,
        "e150_policy_cross_regret_on_replayed_pp": regret_on_replayed,
        "e150_policy_cross_regret_on_measured_pp": regret_on_measured,
        "e150_policy_cross_regret_pp": min(regret_on_replayed,
                                           regret_on_measured),
        "e150_policy_gain_min_over_curves_pp": worst_self,
        "e150_policy_gain_min_over_all_cells_pp": worst_any,
    }
    path = write_artifact(args.json, out)

    print("\n## R0.5b headline")
    print("  worst self-consistent cell   %+9.4f pp   <- the submission number"
          % worst_self)
    print("  worst cell of all four       %+9.4f pp" % worst_any)
    if not control_ok:
        print("  WARNING: the published R7 control missed, nothing is readable")
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
