#!/usr/bin/env python3
"""E145 R7-2 and R7-3: what the acceptance state is worth, and to whom.

`harness=local instrument`. Zero GPU.

R7-2 reconciles two campaign numbers that both travel under the word
"oracle" and disagree by 13.5 pp on the same replayed curve:

    E140 `Boracle_rankedprice`   -4.5296 %   (curve_lopo, -4.5149 in sample)
    E145 R7 `oracle_full_on_replayed`  +8.9390 %

They are not the same arm. The campaign vocabulary overloaded one word:

    name              what the policy is given          when
    estimator oracle  the TRUE per-position marginal    once, per prompt
                      acceptance vector `p_target`,
                      substituted for the EMA state
    decision oracle   the realised capability of THIS   every round
                      round, then the depth that
                      minimises cost per realised token

E140 measured the first (`e140_oracle_state.py:66-79` substitutes
`p_target` for `ema` and leaves the rule alone; its own summary keys are
`estimator_worth_pp` and `argmax_worth_with_perfect_estimator_pp`).
E128 and E145 R7 measure the second (`e128-results.md:823`, "the oracle
arm knows, for each round, whether each draft token will be accepted").
E140's own module records the decision oracle at
`e140_cells.py:70  REFERENCE["oracle"] = 9.1617`, so E140 never claimed
-4.53 was a ceiling.

This file runs both, plus the full factorial they are corners of, in one
process against one cache, so the reconciliation is a measurement and not
a reading of two artifacts.

    state         rule    | replayed curve | measured curve
    ema           greedy  | E140 B         | R7 shipped-rule control
    ema           argmax  | E140 D, R7     | R7 `argmax_full`
    truep         greedy  | E140 Boracle   | the distributional ceiling
    truep         argmax  | E140 Doracle   |   under the shipped rule
    clairvoyant   greedy  | threshold rule fed perfect per-round info
    clairvoyant   argmax  | argmax fed perfect per-round info
    oracle_depth  -       | E128 oracle    | R7 `oracle_full`

R7-3 then asks how accurate a per-round predictor must be to collect the
R7 headroom. Two families feed the argmax on the measured curve:

    shrink  p_hat[i] = lam * 1[i < capability] + (1 - lam) * ema[i]
    noise   p_hat[i] = clip(1[i < capability] + N(0, sigma), 0, 1)

`lam = 0` is the shipped estimator and `lam = 1` is perfect per-round
knowledge, so the ladder spans exactly the R7 gap by construction.

Usage:
  python3 e145_r7_state.py --json e145-artifacts/r7-state.json
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
from e134_rung2 import (  # noqa: E402
    build_legs, median_pct, oracle_depth, walk,
)
from e140_cells import install, transfer_cache  # noqa: E402
from e140_lookahead import curve_price, load_curves, walk_argmax  # noqa: E402
from e145_curve import LEGS_JSON  # noqa: E402
from e145_r3 import (  # noqa: E402
    installable, measured_points, serial_round_us, width1_candidates,
)
from e145_r7 import MAX_WIDTH, admissibility, run_arm  # noqa: E402

# The two published points this rung has to land on, and the band E128
# reported for the decision oracle across its receipt variants.
E140_ESTIMATOR_ORACLE_IN_SAMPLE = -4.5148987193991665
E140_ESTIMATOR_ORACLE_LOPO = -4.529642090803134
E140_ARGMAX_EMA_IN_SAMPLE = -3.123394074355753
E128_DECISION_ORACLE_BAND = (8.5248, 9.0800)
E140_PUBLISHED_ORACLE_REFERENCE = 9.1617
RECONCILE_TOLERANCE_PP = 0.05

LAM_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
NOISE_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CAPTURE_FLOOR = 0.10


def state_walker(rule: str, state: str, p_target, *, lam: float = 1.0,
                 sigma: float = 0.0, noise_seed: int = 0):
    """One chooser, parameterised by what acceptance state it may read.

    `rule` selects the shipped threshold walk or `walk_argmax`. Nothing
    else changes: the price, the caps and the depth-0 and depth-1 margin
    clamps are the shipped ones in both cases. `state` selects what is
    handed to that rule in place of the EMA.
    """
    vector = [p_target[min(i, len(p_target) - 1)] for i in range(MAX_DEPTH)]
    rng = random.Random(noise_seed)

    def apply(st, margin, offer, ctx, price):
        if rule == "argmax":
            return walk_argmax(st, margin, offer, price=price)
        return walk(st, margin, offer, None, ctx, None, price)

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        if adjust is not None or force is not None:
            raise SystemExit("R7-2 prices no `adjust` or `force` arm")
        if state == "oracle_depth":
            return oracle_depth(offer, ctx["capability"])
        if state == "ema":
            st = ema
        elif state == "truep":
            st = vector
        else:
            capability = ctx["capability"]
            hit = [1.0 if i < capability else 0.0 for i in range(MAX_DEPTH)]
            if state == "clairvoyant":
                st = hit
            elif state == "shrink":
                st = [lam * hit[i] + (1.0 - lam) * ema[i]
                      for i in range(MAX_DEPTH)]
            elif state == "noise":
                st = [min(1.0, max(0.0, hit[i] + rng.gauss(0.0, sigma)))
                      for i in range(MAX_DEPTH)]
            else:
                raise SystemExit("unknown acceptance state %r" % state)
        return apply(st, margin, offer, ctx, price)
    return chooser


def score_state_arms(cache, seeds, receipt, windows, arms, admitted) -> dict:
    """`e145_r7.score_arms`, but the walker may read the prompt's vector."""
    out = {}
    for name, spec in arms.items():
        values, per_prompt = [], {}
        hist = [0] * (MAX_DEPTH + 2)
        rounds, depths = 0, []
        for seed in seeds:
            ratios = {}
            for prompt in RANKED_PROMPTS:
                entry = cache[(seed, prompt)]
                walker = state_walker(
                    spec["rule"], spec["state"], entry["p_target"],
                    lam=spec.get("lam", 1.0), sigma=spec.get("sigma", 0.0),
                    noise_seed=seed)
                row = run_arm(cache, seed, prompt, spec["cost"],
                              spec["price"], walker, windows)
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
            "frac_rounds_inadmissible": sum(
                hist[w - 1] for w in range(1, MAX_WIDTH + 1)
                if w not in admitted) / rounds,
            "per_prompt_ratio": {p: statistics.fmean(v)
                                 for p, v in per_prompt.items()},
        }
    return out


def curves_and_prices(anchor: str):
    """The same two cost environments R7 priced, rebuilt here."""
    legs = json.loads(LEGS_JSON.read_text())["legs"]
    points, width1 = measured_points(legs)
    serial = serial_round_us(legs)
    points[1] = width1_candidates(points, width1, serial)[anchor]
    replayed_curves, best_form = load_curves()
    replayed = replayed_curves[best_form]
    measured = installable(points, best_form)
    install(measured)
    measured_price = curve_price(measured)
    install(replayed)
    replayed_price = curve_price(replayed)
    return points, measured, replayed, measured_price, replayed_price


def rung_2(cache, seeds, receipt, windows, env, admitted) -> dict:
    """The factorial the two published `oracle` numbers are corners of."""
    measured, replayed, measured_price, replayed_price = env
    arms = {}
    for curve_name, cost, price in (("replayed", replayed, replayed_price),
                                    ("measured", measured, measured_price)):
        for state in ("ema", "truep", "clairvoyant"):
            for rule in ("greedy", "argmax"):
                arms["%s_%s_%s" % (curve_name, state, rule)] = {
                    "cost": cost, "price": price,
                    "state": state, "rule": rule}
        arms["%s_oracle_depth" % curve_name] = {
            "cost": cost, "price": price,
            "state": "oracle_depth", "rule": "argmax"}
    scored = score_state_arms(cache, seeds, receipt, windows, arms, admitted)

    def value(name):
        return scored[name]["median_pct_mean"]

    e140_repro = value("replayed_truep_greedy")
    e140_argmax_repro = value("replayed_ema_argmax")
    e128_repro = value("replayed_oracle_depth")
    argmax_measured = value("measured_ema_argmax")
    oracle_measured = value("measured_oracle_depth")
    dist_measured = value("measured_truep_argmax")
    realised_gap = oracle_measured - argmax_measured
    dist_gap = dist_measured - argmax_measured
    low, high = E128_DECISION_ORACLE_BAND
    checks = {
        "e140_estimator_oracle_reproduced": abs(
            e140_repro - E140_ESTIMATOR_ORACLE_IN_SAMPLE)
        <= RECONCILE_TOLERANCE_PP,
        "e140_argmax_ema_reproduced": abs(
            e140_argmax_repro - E140_ARGMAX_EMA_IN_SAMPLE)
        <= RECONCILE_TOLERANCE_PP,
        "e128_decision_oracle_in_band": low - 0.5 <= e128_repro <= high + 0.5,
    }
    return {
        "arms": scored,
        "checks": checks,
        "e145_r7_oracle_reconciled": 1.0 if all(checks.values()) else 0.0,
        "e145_r7_e140_estimator_oracle_repro_pct": e140_repro,
        "e145_r7_e140_estimator_oracle_published_pct":
            E140_ESTIMATOR_ORACLE_IN_SAMPLE,
        "e145_r7_e140_estimator_oracle_lopo_pct": E140_ESTIMATOR_ORACLE_LOPO,
        "e145_r7_e128_decision_oracle_repro_pct": e128_repro,
        "e145_r7_e128_decision_oracle_band": list(E128_DECISION_ORACLE_BAND),
        "e145_r7_e140_published_oracle_reference_pct":
            E140_PUBLISHED_ORACLE_REFERENCE,
        "e145_r7_two_oracles_differ_pp": e128_repro - e140_repro,
        "e145_r7_distributional_ceiling_measured_pct": dist_measured,
        "e145_r7_distributional_headroom_pp": dist_gap,
        "e145_r7_realised_headroom_pp": realised_gap,
        "e145_r7_headroom_that_is_per_round_frac": (
            1.0 - dist_gap / realised_gap if realised_gap else 0.0),
        "e145_r7_margin_clamp_cost_with_perfect_info_pp": (
            oracle_measured - value("measured_clairvoyant_argmax")),
        "e145_r7_greedy_cannot_use_perfect_info_pp": (
            value("measured_clairvoyant_greedy") - oracle_measured),
    }


def rung_3(cache, seeds, receipt, windows, env, admitted, floor_pct) -> dict:
    """How accurate must a per-round predictor be to collect the gap?"""
    measured, _replayed, measured_price, _rp = env
    arms = {}
    for lam in LAM_GRID:
        arms["shrink_%.2f" % lam] = {
            "cost": measured, "price": measured_price,
            "state": "shrink", "rule": "argmax", "lam": lam}
    for sigma in NOISE_GRID:
        arms["noise_%.2f" % sigma] = {
            "cost": measured, "price": measured_price,
            "state": "noise", "rule": "argmax", "sigma": sigma}
    scored = score_state_arms(cache, seeds, receipt, windows, arms, admitted)
    base = scored["shrink_0.00"]["median_pct_mean"]
    top = scored["shrink_1.00"]["median_pct_mean"]
    span = top - base

    def captured(name):
        row = scored[name]
        return {
            "median_pct": row["median_pct_mean"],
            "median_pct_sd": row["median_pct_sd"],
            "captured_frac": ((row["median_pct_mean"] - base) / span
                              if span else 0.0),
            "weighted_mean_depth": row["weighted_mean_depth"],
            "frac_rounds_inadmissible": row["frac_rounds_inadmissible"],
        }

    lam_rows = [dict(lam=lam, **captured("shrink_%.2f" % lam))
                for lam in LAM_GRID]
    noise_rows = [dict(sigma=s, **captured("noise_%.2f" % s))
                  for s in NOISE_GRID]
    below = [r["sigma"] for r in noise_rows if r["captured_frac"] < floor_pct]
    half = [r["lam"] for r in lam_rows if r["captured_frac"] >= 0.50]
    return {
        "arms": scored,
        "e145_r7_ladder_base_pct": base,
        "e145_r7_ladder_top_pct": top,
        "e145_r7_ladder_span_pp": span,
        "e145_r7_captured_frac_vs_lam": lam_rows,
        "e145_r7_captured_frac_vs_noise": noise_rows,
        "e145_r7_noise_sigma_below_floor": (min(below) if below else None),
        "e145_r7_capture_floor": floor_pct,
        "e145_r7_lam_for_half_the_gap": (min(half) if half else None),
    }


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
    ap.add_argument("--skip-rung3", action="store_true")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e145-artifacts/r7-state.json")
    args = ap.parse_args()

    print("harness=local instrument  E145 R7-2 and R7-3  zero GPU")
    points, measured, replayed, measured_price, replayed_price = (
        curves_and_prices(args.anchor))
    env = (measured, replayed, measured_price, replayed_price)
    admitted = set(admissibility(points)["admissible_widths"])
    print("  admissible widths on the measured curve %s" % sorted(admitted))

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs_e128, gate = build_legs(args.accept, args.runs)
    print("  attachment gate: legs %d ; attached %d ; mismatches %d/%d/%d"
          % (gate["legs"], gate["attached"], gate["accept_mismatch"],
             gate["margin_mismatch"], gate["unmatched"]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs_e128, args.windows, args.fit_windows, seeds)

    out = {
        "harness": "local", "frame": "decode", "gpu_used": False,
        "rung": "R7-2,R7-3", "windows": args.windows, "seeds": seeds,
        "receipt": receipt["id"], "width1_anchor": args.anchor,
        "attachment_gate": gate,
    }

    two = rung_2(cache, seeds, receipt, args.windows, env, admitted)
    out.update({k: v for k, v in two.items() if k != "arms"})
    out["e145_r7_state_factorial"] = two["arms"]

    print("\n## R7-2  the acceptance-state factorial, replayed median percent")
    print("  %-30s %10s %8s %8s %10s"
          % ("arm", "median %", "sd", "depth", "inadmis %"))
    for name, row in two["arms"].items():
        print("  %-30s %+10.4f %8.4f %8.4f %10.4f"
              % (name, row["median_pct_mean"], row["median_pct_sd"],
                 row["weighted_mean_depth"],
                 100.0 * row["frac_rounds_inadmissible"]))

    print("\n## R7-2  reconciliation")
    print("  E140 `Boracle_rankedprice`, published in sample   %+8.4f"
          % E140_ESTIMATOR_ORACLE_IN_SAMPLE)
    print("  replayed + true marginal vector + greedy, here    %+8.4f"
          % two["e145_r7_e140_estimator_oracle_repro_pct"])
    print("  E128 decision oracle, published band              "
          "%+8.4f .. %+8.4f" % E128_DECISION_ORACLE_BAND)
    print("  replayed + realised capability + argmin, here     %+8.4f"
          % two["e145_r7_e128_decision_oracle_repro_pct"])
    print("  the two `oracle` names differ by                  %+8.4f pp"
          % two["e145_r7_two_oracles_differ_pp"])
    for name, ok in two["checks"].items():
        print("  check %-40s %s" % (name, ok))
    print("  e145_r7_oracle_reconciled = %.1f"
          % two["e145_r7_oracle_reconciled"])

    print("\n## R7-2  what each information set is worth on the MEASURED"
          " curve")
    print("  perfect marginal distribution buys                %+8.4f pp"
          % two["e145_r7_distributional_headroom_pp"])
    print("  perfect per-round outcome buys                    %+8.4f pp"
          % two["e145_r7_realised_headroom_pp"])
    print("  share of the headroom that is per-round only      %8.2f %%"
          % (100.0 * two["e145_r7_headroom_that_is_per_round_frac"]))
    print("  the depth-0/1 margin clamps cost, with perfect"
          " info            %+8.4f pp"
          % two["e145_r7_margin_clamp_cost_with_perfect_info_pp"])
    print("  the shipped threshold rule, given perfect info    %+8.4f pp"
          " vs the argmin"
          % two["e145_r7_greedy_cannot_use_perfect_info_pp"])

    if not args.skip_rung3:
        three = rung_3(cache, seeds, receipt, args.windows, env, admitted,
                       CAPTURE_FLOOR)
        out.update({k: v for k, v in three.items() if k != "arms"})
        out["e145_r7_predictor_ladder"] = three["arms"]
        print("\n## R7-3  how good must a per-round predictor be?")
        print("  ladder spans %+.4f -> %+.4f, that is %.4f pp"
              % (three["e145_r7_ladder_base_pct"],
                 three["e145_r7_ladder_top_pct"],
                 three["e145_r7_ladder_span_pp"]))
        print("  %-8s %10s %10s %8s %10s"
              % ("lambda", "median %", "captured", "depth", "inadmis %"))
        for row in three["e145_r7_captured_frac_vs_lam"]:
            print("  %-8.2f %+10.4f %9.2f %% %8.4f %10.4f"
                  % (row["lam"], row["median_pct"],
                     100.0 * row["captured_frac"],
                     row["weighted_mean_depth"],
                     100.0 * row["frac_rounds_inadmissible"]))
        print("  %-8s %10s %10s %8s %10s"
              % ("sigma", "median %", "captured", "depth", "inadmis %"))
        for row in three["e145_r7_captured_frac_vs_noise"]:
            print("  %-8.2f %+10.4f %9.2f %% %8.4f %10.4f"
                  % (row["sigma"], row["median_pct"],
                     100.0 * row["captured_frac"],
                     row["weighted_mean_depth"],
                     100.0 * row["frac_rounds_inadmissible"]))
        print("  smallest sigma with capture below %.0f %%: %s"
              % (100.0 * CAPTURE_FLOOR,
                 three["e145_r7_noise_sigma_below_floor"]))
        print("  smallest lambda that keeps half the gap: %s"
              % three["e145_r7_lam_for_half_the_gap"])

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
