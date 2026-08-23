#!/usr/bin/env python3
"""E150 R0.5: is the shipped depth rule the right optimality form?

`harness=local instrument`. Zero GPU. Zero Swift. Rule 79 binds: this module
publishes no timing contrast. Every number is an offline price against the
E145 R3 measured per-width round cost curve, through the E128 replayer.

The question, from advisor feedback F1 section 4
------------------------------------------------
The shipped session breaks out of its depth walk on

    reach > price.marginal[d] * (1 + expected) / price.cumulative[d]

which is a RATIO rule: continue while marginal gain per marginal cost beats
THIS ROUND's own average gain per average cost. Renewal-reward theory says
the long-run optimum is instead the LINEARISED rule

    choose d maximising   E[tokens | d]  -  lambda_star * C(d)

where `lambda_star` is one scalar, the optimal achievable long-run tokens per
microsecond, found as the fixed point that makes the policy's own achieved
rate equal the rate it was told to assume. The shipped rule substitutes the
current round's realised rate for that global constant. Nobody on this
campaign has tested the substitution.

Parameterisation
----------------
Two reciprocal forms of the same Dinkelbach step exist and mixing them is the
easy way to get a wrong number here, so both are carried explicitly:

    maximise  T - lambda * C     with  lambda = tokens per unit cost
    maximise  mu * T - C         with  mu     = cost per token = 1 / lambda

`walk_ratio` implements the second. The fixed point is the `mu` whose induced
policy achieves exactly `mu` cost per token, so the root solved for below is

    g(mu) = achieved_cost_per_token(mu) - mu = 0

`g` is continuous and decreasing, so bisection is safe, and the run asserts
the sign change rather than assuming it.

Costs are normalised by `ranked_round_us(1)`, the cost of a round that drafts
nothing, which is exactly the unit `PRICE_CUMULATIVE` already uses.

Usage:
  python3 e150_r05.py --json r05.json
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e150_lib  # noqa: E402
from e128_price import (  # noqa: E402
    DECODE_TOKENS, RANKED_PROMPTS, ranked_round_us)
from e128_replay import SEGMENTED_VERIFY_DEPTH_CAP  # noqa: E402
from e134_rung2 import simulate  # noqa: E402
from e140_cells import install  # noqa: E402
from e150_lib import (  # noqa: E402
    R7_CLAIRVOYANT_ARGMAX_PCT, R7_CLAMP_NONE_PCT, R7_ORACLE_DEPTH_PCT,
    R7_SHIPPED_PCT, REPRO_TOLERANCE_PP, build_env, score_arm, write_artifact,
)
from e150_predict import fixed_state_walker, hit_vector  # noqa: E402

# `mu_star` is near 0.4 and the simulator's own noise is around 1e-4, so
# resolving the root past 1e-4 buys precision the corpus cannot support.
BISECTION_TOLERANCE = 1e-4
BISECTION_MAX_STEPS = 40
# The sensitivity sweep. If the price is flat across this range then solving
# `mu_star` in sample cannot matter, whatever the fold structure.
MU_SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00)


def shipped_state(seed, prompt, entry):
    """The information the shipped tree actually holds: its own EMA."""
    def state_of(ema, margin, offer, ctx):
        return ema
    return state_of


def clairvoyant_state(seed, prompt, entry):
    """The round's realised capability, as the R7 oracle arms read it."""
    def state_of(ema, margin, offer, ctx):
        return hit_vector(ctx["capability"])
    return state_of


STATES = {"shipped": shipped_state, "clairvoyant": clairvoyant_state}


def achieved_cost_per_token(env, state_name: str, clamp, mu: float,
                            prompts=None,
                            cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP
                            ) -> float:
    """Pooled normalised cost per token of the linearised policy at `mu`.

    Pooled, not averaged per prompt: the fixed point wanted here is the one
    scalar a deployed policy would have to carry, and it cannot know which
    prompt it is serving. `prompts` restricts the corpus, which is how the
    leave-one-prompt-out folds are solved.
    """
    install(env.measured)
    unit = ranked_round_us(1)
    total_us = total_tokens = 0.0
    for seed in env.seeds:
        for prompt in (prompts or RANKED_PROMPTS):
            entry = env.cache[(seed, prompt)]
            walker = fixed_state_walker(
                STATES[state_name](seed, prompt, entry), "ratio", clamp,
                fixed_lam=mu, cap_limit=cap_limit)
            install(env.measured)
            row = simulate(None, entry["factory"](entry["p_target"]),
                           env.windows, price=env.measured_price,
                           walker=walker)
            total_us += row["us_per_token"] * DECODE_TOKENS
            total_tokens += DECODE_TOKENS
    return (total_us / total_tokens) / unit


def solve_mu_star(env, state_name: str, clamp, lo: float, hi: float,
                  verbose: bool = True, prompts=None,
                  cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP) -> dict:
    """Bisect `g(mu) = achieved_cost_per_token(mu) - mu` to its root."""
    trace = []

    def g(mu: float) -> float:
        achieved = achieved_cost_per_token(env, state_name, clamp, mu, prompts,
                                           cap_limit)
        trace.append({"mu": mu, "achieved_cost_per_token": achieved,
                      "residual": achieved - mu})
        if verbose:
            print("    mu %.6f -> achieved %.6f  residual %+0.6f"
                  % (mu, achieved, achieved - mu))
        return achieved - mu

    g_lo, g_hi = g(lo), g(hi)
    if g_lo < 0.0 or g_hi > 0.0:
        raise SystemExit(
            "no sign change bracketing the fixed point on [%g, %g]: "
            "g(lo)=%+0.6f g(hi)=%+0.6f. Widen the bracket rather than "
            "reporting an unconverged mu_star." % (lo, hi, g_lo, g_hi))
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
        "mu_star_cost_per_token_normalised": mu_star,
        "lambda_star_tokens_per_normalised_cost": 1.0 / mu_star,
        "lambda_star_tokens_per_us": 1.0 / (mu_star * ranked_round_us(1)),
        "serial_round_us": ranked_round_us(1),
        "bracket_low": lo, "bracket_high": hi,
        "bisection_steps": steps,
        "converged": hi - lo <= BISECTION_TOLERANCE,
        "trace": trace,
    }


def price(env, state_name: str, rule: str, clamp, fixed_lam=None,
          cap_limit: int = SEGMENTED_VERIFY_DEPTH_CAP) -> dict:
    """`fixed_lam` may be one scalar or a per-prompt map for the LOPO arm."""
    def make(seed, prompt, entry):
        lam = (fixed_lam.get(prompt) if isinstance(fixed_lam, dict)
               else fixed_lam)
        return fixed_state_walker(STATES[state_name](seed, prompt, entry),
                                  rule, clamp, fixed_lam=lam,
                                  cap_limit=cap_limit)
    return score_arm(env, "measured", env.measured, env.measured_price, make)


def summarise(row: dict, mu) -> dict:
    """The fields every R0.5 arm must publish, including the safety ones.

    `frac_rounds_inadmissible` and the width histogram are not decoration. A
    linearised rule drafts deeper than the shipped rule, so the first way this
    result could be an artefact is by spending its rounds at widths the
    measured curve never actually measured.
    """
    return {
        "median_pct": row["median_pct_mean"],
        "median_pct_sd": row["median_pct_sd"],
        "weighted_mean_depth": row["weighted_mean_depth"],
        "weighted_accept_rate": row["weighted_accept_rate"],
        "width1_share": row["width_histogram"]["1"],
        "width_histogram": row["width_histogram"],
        "frac_rounds_inadmissible": row["frac_rounds_inadmissible"],
        "mu_used": mu,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--solve-windows", type=int, default=60,
                    help="cheaper corpus for the bisection itself")
    ap.add_argument("--solve-seeds", type=int, default=2)
    ap.add_argument("--json", type=pathlib.Path,
                    default=e150_lib.ARTIFACTS / "r05.json")
    args = ap.parse_args()

    print("harness=local instrument  E150 R0.5  zero GPU  zero Swift")
    print("policy FORM only: the information state is held fixed in every "
          "comparison below.")
    env = build_env(windows=args.windows, seeds=args.seeds)
    print("  attachment gate %s" % env.gate)
    solver_env = build_env(windows=args.solve_windows, seeds=args.solve_seeds)

    out = dict(env.identity())
    out.update({
        "rung": "E150 R0.5",
        "question": "is the shipped ratio rule the right optimality form?",
        "solve_windows": args.solve_windows,
        "solve_seeds": args.solve_seeds,
    })

    print("\n## solve the renewal fixed point by bisection")
    # The fixed point moves with the clamp, because the clamp changes the
    # policy whose achieved rate the root has to match. Solving once and
    # reusing the value across clamp settings would price the wrong policy.
    solved = {}
    for state_name in ("shipped", "clairvoyant"):
        for clamp_name, clamp in (("clamped", None), ("noclamp", {})):
            key = "%s_%s" % (state_name, clamp_name)
            print("  %s" % key)
            solved[key] = solve_mu_star(solver_env, state_name, clamp,
                                        0.05, 3.0)
            print("    mu_star %.6f  lambda_star %.6f tokens per normalised "
                  "cost"
                  % (solved[key]["mu_star_cost_per_token_normalised"],
                     solved[key]["lambda_star_tokens_per_normalised_cost"]))
    out["e150_lambda_star"] = (
        solved["shipped_clamped"]["lambda_star_tokens_per_normalised_cost"])
    out["e150_lambda_star_solution"] = solved

    print("\n## the 2x2x2 table: information state x rule x clamp")
    table = {}
    for state_name in ("shipped", "clairvoyant"):
        for clamp_name, clamp in (("clamped", None), ("noclamp", {})):
            mu_star = solved["%s_%s" % (state_name, clamp_name)][
                "mu_star_cost_per_token_normalised"]
            for rule, lam in (("greedy", None), ("argmax", None),
                              ("linearised", mu_star)):
                key = "%s_%s_%s" % (state_name, rule, clamp_name)
                rule_name = "ratio" if rule == "linearised" else rule
                row = price(env, state_name, rule_name, clamp, lam)
                table[key] = summarise(row, lam)
                print("  %-34s %+9.4f  sd %.4f  depth %.4f  inadmissible "
                      "%.4f"
                      % (key, row["median_pct_mean"], row["median_pct_sd"],
                         row["weighted_mean_depth"],
                         row["frac_rounds_inadmissible"]))

    checks = {
        "shipped_greedy_clamped_reproduces_r7": abs(
            table["shipped_greedy_clamped"]["median_pct"] - R7_SHIPPED_PCT)
        <= REPRO_TOLERANCE_PP,
        "shipped_greedy_noclamp_reproduces_r7": abs(
            table["shipped_greedy_noclamp"]["median_pct"] - R7_CLAMP_NONE_PCT)
        <= REPRO_TOLERANCE_PP,
        "clairvoyant_argmax_clamped_reproduces_r7": abs(
            table["clairvoyant_argmax_clamped"]["median_pct"]
            - R7_CLAIRVOYANT_ARGMAX_PCT) <= REPRO_TOLERANCE_PP,
        "clairvoyant_argmax_noclamp_reproduces_r7": abs(
            table["clairvoyant_argmax_noclamp"]["median_pct"]
            - R7_ORACLE_DEPTH_PCT) <= REPRO_TOLERANCE_PP,
    }
    for name, ok in checks.items():
        print("  check %-46s %s" % (name, ok))

    print("\n## sensitivity: how much does the priced result depend on mu?")
    sweep = []
    for mu in MU_SWEEP:
        row = price(env, "shipped", "ratio", {}, mu)
        sweep.append({"mu": mu, **summarise(row, mu)})
        print("  mu %.2f  %+9.4f  depth %.4f  inadmissible %.4f"
              % (mu, row["median_pct_mean"], row["weighted_mean_depth"],
                 row["frac_rounds_inadmissible"]))

    print("\n## width cap: is the linearised gain only the width-8 cell?")
    # The measured curve makes width 8 nearly free at the margin,
    # C(8) - C(7) = 3162.2 us, the same increment the curve extrapolates to
    # width 9. The ratio rule excludes width 8 by 0.43 % of cost per token,
    # so both rules are decided by sub-1 % precision in one cell. Capping the
    # rule below that cell separates a real policy-form gain from a gain that
    # is only an opinion about C(8).
    cap_rows = []
    for cap in (3, 4, 5, 6, SEGMENTED_VERIFY_DEPTH_CAP):
        for clamp_name, clamp in (("noclamp", {}), ("clamped", None)):
            mu_capped = solve_mu_star(
                solver_env, "shipped", clamp, 0.05, 3.0, verbose=False,
                cap_limit=cap)["mu_star_cost_per_token_normalised"]
            row = price(env, "shipped", "ratio", clamp, mu_capped, cap)
            base = price(env, "shipped", "greedy", clamp, None, cap)
            cap_rows.append({
                "depth_cap": cap, "max_width": cap + 1, "clamp": clamp_name,
                **summarise(row, mu_capped),
                "shipped_rule_same_cap_pct": base["median_pct_mean"],
                "policy_form_gain_pp": (row["median_pct_mean"]
                                        - base["median_pct_mean"]),
            })
            print("  cap width %d %-8s linearised %+9.4f  shipped rule "
                  "%+9.4f  gain %+0.4f pp  w8 %.4f"
                  % (cap + 1, clamp_name, row["median_pct_mean"],
                     base["median_pct_mean"],
                     row["median_pct_mean"] - base["median_pct_mean"],
                     row["width_histogram"]["8"]))

    print("\n## leave one prompt out: solve mu on 7 prompts, price all 8")
    fold_mu = {}
    for held in RANKED_PROMPTS:
        others = [p for p in RANKED_PROMPTS if p != held]
        fold = solve_mu_star(solver_env, "shipped", {}, 0.05, 3.0,
                             verbose=False, prompts=others)
        fold_mu[held] = fold["mu_star_cost_per_token_normalised"]
        print("  hold out %-10s mu_star %.6f  (%d steps)"
              % (held, fold_mu[held], fold["bisection_steps"]))
    lopo_row = price(env, "shipped", "ratio", {}, fold_mu)
    lopo = summarise(lopo_row, None)
    lopo["fold_mu_star"] = fold_mu
    print("  LOPO linearised, no clamp   %+9.4f  depth %.4f  inadmissible "
          "%.4f" % (lopo_row["median_pct_mean"],
                    lopo_row["weighted_mean_depth"],
                    lopo_row["frac_rounds_inadmissible"]))

    capped = next(r for r in cap_rows if r["depth_cap"] == 4)
    out.update({
        "e150_r05_mu_sensitivity": sweep,
        "e150_r05_width_cap_sweep": cap_rows,
        "e150_policy_form_gain_pp_width5_capped": capped["policy_form_gain_pp"],
        "e150_gain_survives_width_cap": (
            capped["policy_form_gain_pp"] > 0.1218),
        "e150_r05_lopo_linearised_noclamp": lopo,
        "e150_r05_lopo_minus_pooled_pp": (
            lopo_row["median_pct_mean"]
            - table["shipped_linearised_noclamp"]["median_pct"]),
    })

    ship = table["shipped_greedy_clamped"]["median_pct"]
    lin = table["shipped_linearised_clamped"]["median_pct"]
    lin_free = table["shipped_linearised_noclamp"]["median_pct"]
    clair_ratio = table["clairvoyant_argmax_clamped"]["median_pct"]
    clair_lin = table["clairvoyant_linearised_clamped"]["median_pct"]
    clair_lin_free = table["clairvoyant_linearised_noclamp"]["median_pct"]
    floor = 0.1218
    best_deployable = max(lin, lin_free, lopo["median_pct"])
    inadmissible = max(row["frac_rounds_inadmissible"]
                       for row in table.values())

    out.update({
        "e150_r05_table": table,
        "e150_r05_control_checks": checks,
        "e150_r05_controls_all_reproduced": all(checks.values()),
        "e150_argmax_linearised_pct": lin,
        "e150_argmax_linearised_pct_noclamp": lin_free,
        "e150_argmax_linearised_pct_clairvoyant": clair_lin,
        "e150_argmax_linearised_pct_clairvoyant_noclamp": clair_lin_free,
        "e150_policy_form_gain_pp": lin - ship,
        "e150_policy_form_gain_pp_noclamp": lin_free - ship,
        "e150_policy_form_gain_pp_lopo": lopo["median_pct"] - ship,
        "e150_policy_form_gain_pp_clairvoyant": clair_lin - clair_ratio,
        "e150_policy_form_recovers_clamp_cost_pp": (
            clair_lin_free - R7_ORACLE_DEPTH_PCT),
        "e150_published_oracle_is_not_the_ceiling": (
            clair_lin_free > R7_ORACLE_DEPTH_PCT + floor),
        "e150_revised_headroom_axis_pp": clair_lin_free - R7_SHIPPED_PCT,
        "e150_max_frac_rounds_inadmissible": inadmissible,
        "e150_r05_verdict": (
            "policy form pays: the linearised rule beats the shipped ratio "
            "rule by more than the noise floor on shipped information"
            if best_deployable - ship > floor else
            "tie: the shipped ratio form is not the binding constraint"),
        "e150_noise_floor_pp": floor,
    })

    path = write_artifact(args.json.name, out)

    print("\n## R0.5 headline")
    print("  lambda_star            %.6f tokens per normalised cost"
          % out["e150_lambda_star"])
    print("  mu_star                %.6f normalised cost per token"
          % solved["shipped_clamped"]["mu_star_cost_per_token_normalised"])
    print("  shipped state, ratio rule (ships today) %+8.4f %%" % ship)
    print("  shipped state, linearised, clamp on     %+8.4f %%  (%+0.4f pp)"
          % (lin, lin - ship))
    print("  shipped state, linearised, no clamp     %+8.4f %%  (%+0.4f pp)"
          % (lin_free, lin_free - ship))
    print("  the same, mu solved leave one out       %+8.4f %%  (%+0.4f pp)"
          % (lopo["median_pct"], lopo["median_pct"] - ship))
    print("  F4 noise floor                           %8.4f pp" % floor)
    print("  clairvoyant, ratio rule, clamp on       %+8.4f %%" % clair_ratio)
    print("  clairvoyant, linearised, clamp on       %+8.4f %%" % clair_lin)
    print("  clairvoyant, linearised, no clamp       %+8.4f %%"
          % clair_lin_free)
    print("  versus the published oracle %+0.4f      %+8.4f pp"
          % (R7_ORACLE_DEPTH_PCT,
             out["e150_policy_form_recovers_clamp_cost_pp"]))
    print("  worst inadmissible width share           %8.4f" % inadmissible)
    print("  gain with depth capped to width 5       %+8.4f pp  (survives: "
          "%s)" % (out["e150_policy_form_gain_pp_width5_capped"],
                   out["e150_gain_survives_width_cap"]))
    print("  controls all reproduced                  %s"
          % out["e150_r05_controls_all_reproduced"])
    print("  verdict: %s" % out["e150_r05_verdict"])
    if not out["e150_r05_controls_all_reproduced"]:
        print("  WARNING: a positive control missed its published cell, so "
              "no number above is readable.")
    print("\nwrote %s" % path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
