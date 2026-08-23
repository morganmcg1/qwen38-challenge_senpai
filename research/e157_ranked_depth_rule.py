"""E157 R0: the head-clean ranked depth price, and what it says about depth.

Two things changed after FINDING 295 forced a head audit:

  1. The ranked `drama` width sweep is not head-homogeneous. Eleven head
     digests appear across twenty-one cells, and the single deepest cell owns
     a head of its own. Holding the head fixed removes the curvature I
     reported earlier, so the honest ranked price is a straight line fit
     inside one head over widths 1.05 to 3.24.
  2. FINDING 286's ranked row law reproduces the independently fitted local
     row law under one divisor. That is testable here: search for a divisor
     that maps both local parameters onto both published ranked parameters
     under ordinary rounding. If one exists, FINDING 286 carries no ranked
     information about depth, and RULE 166 recovers the divisor that built it.

Sign convention in words: `d_star` is the number of drafts a round should
propose. `d_star` above the live drafted mean means the live schedule drafts
too few rows, which is the opposite of this experiment's hypothesis.

    python3 research/e157_ranked_depth_rule.py
"""
from __future__ import annotations

import json
import os

ARTIFACTS = os.path.join(os.path.dirname(__file__), "e157-artifacts")
MAX_DRAFTS = 8

# Fitted here from 156 raw local rounds, harness=local, declared q2/q4 head.
LOCAL_A_US = 23421.071843078098
LOCAL_B_US = 15708.498145888343
# FINDING 286 as published: clean_round_us = 8,434 + 5,657 x rows.
F286_FIXED_US = 8434.0
F286_ROW_US = 5657.0
# Shipped cost model, Qwen36MTPBlockSession.swift.
SHIPPED_H = 0.18


def divisor_interval(local: float, published: float) -> tuple[float, float]:
    """Divisors that map `local` onto `published` under ordinary rounding."""
    return local / (published + 0.5), local / (published - 0.5)


def circularity_test() -> dict:
    low_a, high_a = divisor_interval(LOCAL_A_US, F286_FIXED_US)
    low_b, high_b = divisor_interval(LOCAL_B_US, F286_ROW_US)
    low, high = max(low_a, low_b), min(high_a, high_b)
    single = low <= high
    local_h = LOCAL_B_US / (LOCAL_A_US + LOCAL_B_US)
    f286_h = F286_ROW_US / (F286_FIXED_US + F286_ROW_US)
    return {
        "harness": "local versus published ranked law",
        "fixed_term_divisor_interval": [low_a, high_a],
        "row_term_divisor_interval": [low_b, high_b],
        "single_divisor_reproduces_both": single,
        "divisor_interval": [low, high] if single else None,
        "divisor_point": (low + high) / 2 if single else None,
        "local_depth_price_h": local_h,
        "finding_286_depth_price_h": f286_h,
        "depth_price_relative_difference": abs(f286_h - local_h) / local_h,
        # A scalar divisor cannot change a ratio of two scaled terms, so an
        # identical h is the signature of a rescaled law rather than a fit.
        "rule_166_interval_contains_divisor": (
            single and low <= 2.78 and high >= 2.72
        ),
    }


def expected_tokens(p: float, drafts: int) -> float:
    """Primary token plus the accepted prefix, geometric in conditional p."""
    if drafts <= 0:
        return 1.0
    if p >= 1.0:
        return 1.0 + drafts
    return 1.0 + p * (1.0 - p**drafts) / (1.0 - p)


def best_drafts(p: float, round_us) -> tuple[int, float]:
    best_d, best_rate = 0, expected_tokens(p, 0) / round_us(0)
    for drafts in range(1, MAX_DRAFTS + 1):
        rate = expected_tokens(p, drafts) / round_us(drafts)
        if rate > best_rate:
            best_d, best_rate = drafts, rate
    return best_d, best_rate


def linear_round(fixed_us: float, row_us: float):
    return lambda drafts: fixed_us + row_us * (drafts + 1.0)


def quadratic_round(beta0: float, beta1: float, beta2: float):
    def cost(drafts: float) -> float:
        rows = drafts + 1.0
        return beta0 + beta1 * rows + beta2 * rows * rows

    return cost


def main() -> None:
    with open(os.path.join(ARTIFACTS, "e157_head_confound.json")) as handle:
        confound = json.load(handle)
    with open(os.path.join(ARTIFACTS, "e157_ranked_row_law.json")) as handle:
        row_law = json.load(handle)

    strat = confound["ranked_sweep_head_stratification"]
    clean = strat["single_head_linear"]
    fixed_effect = strat["head_fixed_effect_quadratic"]

    clean_fixed = clean["fixed_us"]
    clean_row = clean["marginal_us_per_row"]
    clean_h = clean_row / (clean_fixed + clean_row)

    # The head fixed-effect model is the honest upper bound on curvature: its
    # point estimate is not significant, so carry the two-sigma edge too.
    curvature_upper = (
        fixed_effect["curvature_us_per_row_squared"]
        + 2.0 * fixed_effect["curvature_stderr"]
    )

    prices = {
        "head_clean_ranked_linear": linear_round(clean_fixed, clean_row),
        "head_clean_ranked_curvature_2sigma": quadratic_round(
            fixed_effect["intercept_us"],
            fixed_effect["linear_us_per_row"],
            curvature_upper,
        ),
        "shipped_flat_0p18": lambda d: 1.0 + SHIPPED_H * d,
        "finding_286": linear_round(F286_FIXED_US, F286_ROW_US),
        "local_m4_pro": linear_round(LOCAL_A_US, LOCAL_B_US),
    }

    per_prompt = row_law["receipts"][0]["per_prompt"]
    rows_out = []
    for entry in per_prompt:
        p = entry["conditional_p_fitted"]
        live = entry["edl_drafted_mean"]
        record = {
            "prompt": entry.get("prompt", entry.get("name")),
            "conditional_p": p,
            "live_drafted_mean": live,
            "clean_us_per_round": entry["clean_us_per_round"],
        }
        for name, cost in prices.items():
            d_star, rate = best_drafts(p, cost)
            live_rate = expected_tokens(p, round(live)) / cost(round(live))
            record[f"d_star_{name}"] = d_star
            record[f"gain_pct_{name}"] = 100.0 * (rate / live_rate - 1.0)
        record["depth_error_head_clean"] = (
            record["d_star_head_clean_ranked_linear"] - live
        )
        rows_out.append(record)

    signs = {
        "too_deep" if r["depth_error_head_clean"] < -0.5
        else "too_shallow" if r["depth_error_head_clean"] > 0.5
        else "about_right"
        for r in rows_out
    }
    verdict = {
        "harness": "ranked",
        "e157_head_clean_ranked_row_price_us": clean_row,
        "e157_head_clean_ranked_row_price_stderr_us": clean["marginal_stderr"],
        "e157_head_clean_ranked_depth_price_h": clean_h,
        "e157_shipped_depth_price_h": SHIPPED_H,
        "e157_shipped_price_overstates_depth_by": SHIPPED_H / clean_h - 1.0,
        "e157_identified_width_range": clean["width_range"],
        "e157_depth_price_extrapolated_above_width": clean["width_range"][1],
        "e157_depth_error_sign": (
            list(signs)[0] if len(signs) == 1 else sorted(signs)
        ),
        "e157_hypothesis_two_rows_too_deep": (
            all(r["depth_error_head_clean"] <= -1.5 for r in rows_out)
        ),
        "e157_curvature_significant_after_head_control": (
            abs(fixed_effect["curvature_sigma"]) >= 2.0
        ),
    }

    out = {
        "experiment": "e157-r0-head-clean-ranked-depth-rule",
        "circularity_test": circularity_test(),
        "head_clean_price": {
            "harness": "ranked",
            "fixed_us": clean_fixed,
            "row_us": clean_row,
            "row_stderr_us": clean["marginal_stderr"],
            "depth_price_h": clean_h,
            "r_squared": clean["r_squared"],
            "n_cells": clean["n"],
            "width_range": clean["width_range"],
            "head": strat["largest_head"],
            "curvature_point_us": fixed_effect["curvature_us_per_row_squared"],
            "curvature_stderr_us": fixed_effect["curvature_stderr"],
            "curvature_two_sigma_upper_us": curvature_upper,
        },
        "per_prompt": rows_out,
        "verdict": verdict,
    }

    circ = out["circularity_test"]
    print("--- FINDING 286 circularity test ---")
    print(f"single divisor reproduces both parameters : "
          f"{circ['single_divisor_reproduces_both']}")
    print(f"divisor interval  : {circ['divisor_interval']}")
    print(f"local h           : {circ['local_depth_price_h']:.5f}")
    print(f"FINDING 286 h     : {circ['finding_286_depth_price_h']:.5f}")
    print(f"relative gap in h : {circ['depth_price_relative_difference']:.2e}")
    print(f"RULE 166 interval contains it : "
          f"{circ['rule_166_interval_contains_divisor']}")

    print("\n--- head-clean ranked price, harness=ranked ---")
    print(f"row price   {clean_row:.0f} +- {clean['marginal_stderr']:.0f} us/row"
          f"  (n={clean['n']}, one head, R2={clean['r_squared']:.4f})")
    print(f"depth price h = {clean_h:.4f}   shipped = {SHIPPED_H}"
          f"   overstated by {100*(SHIPPED_H/clean_h-1):.0f} %")
    print(f"identified over widths {clean['width_range'][0]:.2f}"
          f" to {clean['width_range'][1]:.2f}")

    print("\n--- optimal drafts per prompt ---")
    header = (f"{'prompt':10s} {'p':>6s} {'live':>5s} {'clean':>6s} "
              f"{'2sig':>5s} {'ship':>5s} {'F286':>5s} {'local':>5s} {'err':>6s}")
    print(header)
    for r in rows_out:
        print(
            f"{str(r['prompt'])[:10]:10s} {r['conditional_p']:6.3f} "
            f"{r['live_drafted_mean']:5.2f} "
            f"{r['d_star_head_clean_ranked_linear']:6d} "
            f"{r['d_star_head_clean_ranked_curvature_2sigma']:5d} "
            f"{r['d_star_shipped_flat_0p18']:5d} "
            f"{r['d_star_finding_286']:5d} "
            f"{r['d_star_local_m4_pro']:5d} "
            f"{r['depth_error_head_clean']:6.2f}"
        )
    print(f"\nverdict: {json.dumps(verdict, indent=1)}")

    path = os.path.join(ARTIFACTS, "e157_ranked_depth_rule.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
