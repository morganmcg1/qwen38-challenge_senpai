"""E149 F3: does the width-cost cliff explain the per-prompt null sd?

harness=ranked for the null sds, harness=local-replay for the width masses and
the cost curve. Zero GPU. Nothing is measured here; this file only combines
published ranked receipts with two tables the campaign ledger already owns.

THE THREE ASKS

  e149_per_prompt_null_sd_vs_per_step_p
      the eight n=11 null sds against the F92 implied per-step p, rank
      correlation with an EXACT permutation null over all 8! orderings.

  e149_per_prompt_width_dispersion
      the sd of realised verify width per prompt, and the share of its pairwise
      dispersion that crosses the cost cliff.

  e149_cliff_straddle_index
      cost dispersion under the measured curve divided by cost dispersion under
      a linear curve of the same average slope, at the same width distribution.

TWO SOURCE DISCREPANCIES, BOTH FLAGGED AND NEITHER RESOLVED HERE

  1. F3 quotes cliff steps of 29,134.5 us at 5->6 and 25,861.5 us at 6->7. No
     table in this checkout carries either number. The only rebuilt width-cost
     curve in the ledger is at senpai/campaign-ledger.md:49487-49500, and its
     shape is different: ONE cliff at 5->6 and a COLLAPSED 6->7 step. This file
     uses the ledger curve and reports the disagreement.
  2. The replayed width masses at senpai/campaign-ledger.md:49473-49485 come
     from a 6-seed by 200-window local replay, not from the ranked legs. For
     botany the masses at 5..8 drafts alone imply a mean draft length of 6.84,
     above the ranked 6.148, so the replay sits at a different operating point.
     Every width statistic here is therefore labelled harness=local-replay and
     must not be read as a ranked width distribution.
"""

import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import e146_lib as L
import e148_lib as E
from e149_null_sd import PAYING_FOUR, pearson, spearman

OUT = os.path.join(HERE, "e149-f3-cliff.json")
NULL_SD_JSON = os.path.join(HERE, "e149-null-sd.json")

# senpai/campaign-ledger.md:49473-49485, "Replayed shipped width masses, 6
# seeds by 200 windows". m(k) is the mass at k DRAFT tokens, so the verify
# width is k + 1 and the top of the range, m(8), is width 9. prompt ->
# (replay rounds, drafting fraction, m5, m6, m7, m8).
REPLAY_WIDTH_MASS = {
    "plutarch": (586578, 0.0808, 0.0038, 0.0001, 0.0000, 0.0000),
    "drama": (248714, 0.5589, 0.0674, 0.1091, 0.0900, 0.0824),
    "travel": (246541, 0.9726, 0.1308, 0.0870, 0.0172, 0.0000),
    "beagle": (142498, 1.0000, 0.0920, 0.0739, 0.0589, 0.3388),
    "republic": (121413, 1.0000, 0.0454, 0.0432, 0.0835, 0.4807),
    "essays": (113061, 1.0000, 0.1995, 0.1558, 0.1408, 0.2958),
    "medicine": (109093, 1.0000, 0.1474, 0.0641, 0.0556, 0.4850),
    "botany": (86848, 1.0000, 0.0761, 0.0859, 0.0940, 0.6605),
}

# senpai/campaign-ledger.md:49489-49500. Verify-width cost in microseconds for
# widths 1..9. `per_round` is the tier the campaign ships.
CURVES = {
    "pre_arm": [31171.5, 34617.5, 38063.6, 41509.7, 44955.8, 59666.6, 64990.1,
                70313.6, 75637.2],
    "per_round": [31173.2, 34619.3, 38065.4, 41511.4, 44957.5, 61198.8,
                  62824.9, 70315.4, 75638.9],
    "per_draft": [31171.5, 34568.0, 38014.1, 41460.2, 44906.3, 61473.0,
                  62941.5, 70264.1, 75587.7],
    "proportional": [31151.7, 34595.6, 38039.5, 41483.4, 44927.3, 61310.0,
                     63010.3, 70269.1, 75589.2],
}
SHIPPED_CURVE = "per_round"

# F3's quoted steps, carried only so the disagreement is explicit.
F3_QUOTED_STEPS = {"5->6": 29134.5, "6->7": 25861.5, "below_5": 3446.0}

# The cliff sits between verify width 5 and verify width 6 on the ledger curve.
CLIFF_LOW_WIDTH = 5


def truncated_geometric_drafts(p, kmax):
    """Mass over 0..kmax drafts for an accept-until-first-failure chain."""
    mass = [(1.0 - p) * p ** k for k in range(kmax)]
    mass.append(p ** kmax)
    return mass


def width_distribution(prompt, allocation):
    """Verify-width distribution, width 1..9, from the ledger replay masses.

    The ledger publishes only m(5..8) and the drafting fraction. The remaining
    drafting mass sits at 0..4 drafts and its shape is not published, so it is
    supplied by `allocation` and the answer is reported under all three.
    """
    _, draft_frac, m5, m6, m7, m8 = REPLAY_WIDTH_MASS[prompt]
    top = {5: m5, 6: m6, 7: m7, 8: m8}
    remainder = draft_frac - sum(top.values())
    if remainder < -1e-9:
        raise ValueError("%s: published top masses exceed the drafting "
                         "fraction" % prompt)
    remainder = max(remainder, 0.0)

    low = [0.0] * 5
    if allocation == "geometric":
        shape = truncated_geometric_drafts(L.STEP_P[prompt], 4)
        total = sum(shape)
        for k in range(5):
            low[k] = remainder * shape[k] / total
    elif allocation == "all_at_zero_drafts":
        low[0] = remainder
    elif allocation == "all_at_four_drafts":
        low[4] = remainder
    else:
        raise ValueError(allocation)

    # A non-drafting round still verifies one row, so it sits at width 1.
    dist = [0.0] * 10  # index by verify width 1..9, slot 0 unused
    dist[1] += 1.0 - draft_frac
    for k in range(5):
        dist[k + 1] += low[k]
    for k, mass in top.items():
        dist[k + 1] += mass
    total = sum(dist)
    return [v / total for v in dist]


def moments(dist, values):
    m1 = sum(dist[w] * values[w - 1] for w in range(1, 10))
    m2 = sum(dist[w] * values[w - 1] ** 2 for w in range(1, 10))
    return m1, math.sqrt(max(m2 - m1 * m1, 0.0))


def cliff_analysis(prompt, allocation, curve):
    dist = width_distribution(prompt, allocation)
    widths = list(range(1, 10))
    mean_w, sd_w = moments(dist, widths)
    mean_c, sd_c = moments(dist, curve)

    # A linear curve with the same average slope over the measured range.
    slope = (curve[8] - curve[0]) / 8.0
    sd_linear = slope * sd_w
    index = sd_c / sd_linear if sd_linear > 0 else float("nan")

    below = sum(dist[w] for w in range(1, CLIFF_LOW_WIDTH + 1))
    above = 1.0 - below
    return {
        "allocation": allocation,
        "distribution_width_1_to_9": dist[1:],
        "mean_verify_width": mean_w,
        "sd_verify_width": sd_w,
        "mean_round_cost_us": mean_c,
        "sd_round_cost_us": sd_c,
        "linear_slope_us_per_width": slope,
        "sd_round_cost_linear_us": sd_linear,
        "cliff_straddle_index": index,
        "mass_at_or_below_width_%d" % CLIFF_LOW_WIDTH: below,
        "mass_above_cliff": above,
        "cliff_crossing_pair_fraction": 2.0 * below * above,
    }


def exact_permutation_null(xs, ys, stat):
    """Exact two-sided permutation null over every ordering of ys."""
    observed = stat(xs, ys)
    n = len(ys)
    total = 0
    at_least = 0
    absolutes = []
    for perm in itertools.permutations(range(n)):
        value = stat(xs, [ys[i] for i in perm])
        absolutes.append(abs(value))
        total += 1
        if abs(value) >= abs(observed) - 1e-12:
            at_least += 1
    absolutes.sort()
    return {
        "observed": observed,
        "permutations": total,
        "two_sided_p": at_least / float(total),
        "abs_null_p95": absolutes[int(0.95 * total)],
        "abs_null_p99": absolutes[int(0.99 * total)],
    }


def main():
    with open(NULL_SD_JSON) as fh:
        nulls = json.load(fh)
    tot = nulls["per_prompt_total"]
    prompts = list(L.PROMPT_ORDER)
    sds = [tot[p]["sd_pp"] for p in prompts]
    step_p = [L.STEP_P[p] for p in prompts]

    result = {"harness_null_sd": "ranked",
              "harness_width_and_curve": "local-replay",
              "curve_source": "senpai/campaign-ledger.md:49487-49500",
              "mass_source": "senpai/campaign-ledger.md:49473-49485",
              "shipped_curve": SHIPPED_CURVE}

    # ------------------------------------------------------------------
    # 0. reproduce the F92 per-step p table
    # ------------------------------------------------------------------
    f92 = {"beagle": 0.9341, "botany": 0.9598, "medicine": 0.9639,
           "essays": 0.9647, "republic": 0.9661}
    print("=== F92 per-step p, reproduced from the published receipts ===")
    print("  %-9s %10s %10s %10s" % ("prompt", "mine", "F3 table", "diff"))
    repro = {}
    for p in prompts:
        if p in f92:
            repro[p] = {"mine": L.STEP_P[p], "f3": f92[p],
                        "diff": L.STEP_P[p] - f92[p]}
            print("  %-9s %10.4f %10.4f %+10.4f"
                  % (p, L.STEP_P[p], f92[p], L.STEP_P[p] - f92[p]))
        else:
            print("  %-9s %10.4f %10s %10s" % (p, L.STEP_P[p], "-", "-"))
    result["f92_per_step_p_reproduction"] = repro
    result["per_step_p"] = {p: L.STEP_P[p] for p in prompts}

    # ------------------------------------------------------------------
    # 1. null sd against per-step p
    # ------------------------------------------------------------------
    print("\n=== e149_per_prompt_null_sd_vs_per_step_p ===")
    sp = exact_permutation_null(step_p, sds, spearman)
    pe = exact_permutation_null(step_p, sds, pearson)
    idx5 = [prompts.index(p) for p in E.WEIGHTED_FIVE]
    sp5 = exact_permutation_null([step_p[i] for i in idx5],
                                 [sds[i] for i in idx5], spearman)
    result["null_sd_vs_per_step_p"] = {"eight_prompt_spearman": sp,
                                       "eight_prompt_pearson": pe,
                                       "weighted_five_spearman": sp5}
    for name, cell in (("spearman, 8 prompts", sp), ("pearson, 8 prompts", pe),
                       ("spearman, weighted five", sp5)):
        print("  %-26s %+.4f   exact two-sided p %.4f over %d permutations, "
              "|null| p95 %.4f"
              % (name, cell["observed"], cell["two_sided_p"],
                 cell["permutations"], cell["abs_null_p95"]))

    # CONFOUND. F3's per-step p, F2's round count and the mean draft length are
    # not independent axes on these eight prompts. If they rank the prompts the
    # same way, no ranking test on eight points can separate them.
    rounds = [L.F219_ANCHOR[p][2] for p in prompts]
    dlen = [L.F219_ANCHOR[p][1] for p in prompts]
    confound = {
        "per_step_p_vs_round_count": spearman(step_p, rounds),
        "per_step_p_vs_mean_draft_length": spearman(step_p, dlen),
        "round_count_vs_mean_draft_length": spearman(rounds, dlen),
    }
    result["driver_confound"] = confound
    print("  confound between the candidate drivers, Spearman:")
    for name, value in confound.items():
        print("    %-34s %+.4f" % (name, value))

    # ------------------------------------------------------------------
    # 2 and 3. width dispersion and the cliff-straddle index
    # ------------------------------------------------------------------
    curve = CURVES[SHIPPED_CURVE]
    steps = [curve[i + 1] - curve[i] for i in range(8)]
    print("\n=== the ledger curve, and F3's quoted steps ===")
    print("  ledger %s steps, width 1->2 .. 8->9: %s"
          % (SHIPPED_CURVE, " ".join("%.1f" % s for s in steps)))
    print("  F3 quotes 5->6 %.1f and 6->7 %.1f; the ledger reads %.1f and "
          "%.1f. NEITHER F3 NUMBER APPEARS IN THIS CHECKOUT."
          % (F3_QUOTED_STEPS["5->6"], F3_QUOTED_STEPS["6->7"],
             steps[4], steps[5]))
    result["ledger_curve_steps_us"] = steps
    result["f3_quoted_steps_us"] = F3_QUOTED_STEPS
    result["f3_quoted_steps_found_in_checkout"] = False

    allocations = ["geometric", "all_at_zero_drafts", "all_at_four_drafts"]
    table = {}
    for allocation in allocations:
        print("\n--- allocation of the unpublished sub-5-draft mass: %s ---"
              % allocation)
        print("  %-9s %8s %8s %10s %10s %8s %9s %9s"
              % ("prompt", "meanW", "sdW", "meanC_us", "sdC_us", "index",
                 "P(W<=5)", "cross"))
        for p in prompts:
            cell = cliff_analysis(p, allocation, curve)
            table.setdefault(p, {})[allocation] = cell
            print("  %-9s %8.4f %8.4f %10.1f %10.1f %8.3f %9.4f %9.4f"
                  % (p, cell["mean_verify_width"], cell["sd_verify_width"],
                     cell["mean_round_cost_us"], cell["sd_round_cost_us"],
                     cell["cliff_straddle_index"],
                     cell["mass_at_or_below_width_%d" % CLIFF_LOW_WIDTH],
                     cell["cliff_crossing_pair_fraction"]))
    result["e149_per_prompt_width_dispersion"] = {
        p: {a: {"sd_verify_width": table[p][a]["sd_verify_width"],
                "mean_verify_width": table[p][a]["mean_verify_width"],
                "cliff_crossing_pair_fraction":
                    table[p][a]["cliff_crossing_pair_fraction"]}
            for a in allocations} for p in prompts}
    result["e149_cliff_straddle_index"] = {
        p: {a: table[p][a]["cliff_straddle_index"] for a in allocations}
        for p in prompts}
    result["cliff_detail"] = table

    # ------------------------------------------------------------------
    # 4. does the index track the null sd?
    # ------------------------------------------------------------------
    print("\n=== does the cliff-straddle index track the null sd? ===")
    verdict = {}
    for allocation in allocations:
        index = [table[p][allocation]["cliff_straddle_index"] for p in prompts]
        sdw = [table[p][allocation]["sd_verify_width"] for p in prompts]
        cost_sd = [table[p][allocation]["sd_round_cost_us"] for p in prompts]
        cross = [table[p][allocation]["cliff_crossing_pair_fraction"]
                 for p in prompts]
        cell = {
            "index_vs_null_sd": exact_permutation_null(index, sds, spearman),
            "cost_sd_vs_null_sd": exact_permutation_null(cost_sd, sds,
                                                         spearman),
            "width_sd_vs_null_sd": exact_permutation_null(sdw, sds, spearman),
            "cross_vs_null_sd": exact_permutation_null(cross, sds, spearman),
        }
        verdict[allocation] = cell
        print("  %-20s index rho %+.4f (p %.4f)  costsd rho %+.4f (p %.4f)  "
              "widthsd rho %+.4f (p %.4f)"
              % (allocation, cell["index_vs_null_sd"]["observed"],
                 cell["index_vs_null_sd"]["two_sided_p"],
                 cell["cost_sd_vs_null_sd"]["observed"],
                 cell["cost_sd_vs_null_sd"]["two_sided_p"],
                 cell["width_sd_vs_null_sd"]["observed"],
                 cell["width_sd_vs_null_sd"]["two_sided_p"]))
    result["account_tests"] = verdict

    # beagle against the other four paying prompts on the index itself
    print("\n=== beagle against the paying four, on each candidate driver ===")
    drivers = {}
    for allocation in allocations:
        row = {}
        for name in ("cliff_straddle_index", "sd_verify_width",
                     "sd_round_cost_us", "cliff_crossing_pair_fraction"):
            b = table["beagle"][allocation][name]
            others = [table[p][allocation][name] for p in PAYING_FOUR]
            row[name] = {"beagle": b, "paying_four_mean": E.mean(others),
                         "ratio": b / E.mean(others) if E.mean(others) else
                         float("nan"),
                         "beagle_is_extreme": b > max(others)}
            print("  %-20s %-30s beagle %10.4f  four-mean %10.4f  %.2fx  "
                  "extreme %s"
                  % (allocation, name, b, E.mean(others), row[name]["ratio"],
                     row[name]["beagle_is_extreme"]))
        drivers[allocation] = row
    result["beagle_against_paying_four"] = drivers

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
