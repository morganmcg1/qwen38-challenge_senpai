"""E149 rung 0a: adjudicate `165d4ba7` and republish the at-zero null block.

harness=ranked, zero GPU. Every input is a published per-prompt field of a
ranked Yukon receipt. Nothing local enters this file.

WHAT THIS DECIDES

`165d4ba7` (jonathan308, rejected, 3.69634719) declares `yukon reset 684821e`
and states that the only difference against the promoted archive is the marker
sentence in the free-text `note` field of `mtp-head.manifest.json`. That is the
same class as our own VERIFIED null pair `b8b8b860 -> 44559d02`, but it is
DECLARED, not verified: the archive is not published, the two rows carry
different submission commits, and the row belongs to another account. Its
candidate leg reads +0.1772 % weighted-five against the crown, which is worse
than any member of the operative n=10 at-zero block and above the 0.15 pp gate
on its own.

The corrector is applied exactly as E148 R-B reproduced it:

  * fit `k` in the DECODE frame, seed prefill removed (F227);
  * `steps = round(k / 879.0)`, F240/RULE 132 constant;
  * ambiguous when `|k/879 - steps| >= 0.35`;
  * price in the TOTAL-LEG frame on the weighted five (F237).

Both fit sets are reported: the registered six cohort prompts named in the
assignment, and the schedule-matched set that E148 made operative. A row enters
the block only if BOTH classify it at zero steps, because the block's truth is
"zero mechanism, zero state", and a disagreement between the two fits means the
state term is not resolved.

FINDINGS 240 AND 241 CAVEAT, STATED UP FRONT. Ledger 309 item 8 retires E146's
per-row corrections on NONZERO-step rows and keeps the classifier for refusal
only. This rung uses the classifier for exactly that job: to decide whether
`165d4ba7` sits at the zero lattice point, where the correction subtracts
nothing and the reported value is the RAW weighted-five candidate-leg read. No
retired per-row correction is used anywhere in this file.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import e146_lib as L
import e148_lib as E

OUT = os.path.join(HERE, "e149-rung0a.json")
GATE_PP = 0.15
CROWN = "684821ed"
NEW_ROW = "165d4ba7"

# The operative at-zero block exactly as ledger 309 item 9 published it, in the
# order the ledger prints its values. `parent` is the anchor E148 priced each
# row against. `verified` is True only where the campaign owns both rows and
# can inspect the diff itself; every other row rests on its author's word plus
# the three published decision fields (head provenance, schedule, accepted
# pair count), which are never timing fields.
AT_ZERO_N10 = [
    ("106573b9", "1760479a", False, "same-content resample of frontier e8f14c44"),
    ("64508884", "3ba6ee9d", False, "fresh-crown resample, zero-delta draw"),
    ("aff3b543", "1760479a", False, "E146 declared null"),
    ("b8e0f27c", "3ba6ee9d", False, "E146 declared null"),
    ("3a18ff21", CROWN, False, "E146 declared null"),
    ("b6cb0fea", "0b8602e1", False, "new-crown resample, zero-delta draw"),
    ("f7d59543", CROWN, False, "zero-delta resample of promoted main"),
    ("4debb1df", CROWN, False, "same-content resample of frontier main"),
    ("bed5081a", "48423d09", False, "same-content redraw of frontier c0dbec05"),
    ("44559d02", "b8b8b860", True,
     "OURS: only the mtp-head manifest note text differs, scored surface "
     "byte-identical"),
]
# The published n=10 values, carried so this run can prove it reproduces the
# operative block before it adds an eleventh member.
LEDGER_N10_VALUES = [-0.0516, 0.0027, 0.0210, 0.0516, 0.0147, 0.1065, 0.0011,
                     -0.1114, 0.0211, -0.0044]
LEDGER_N10_SD = 0.0577
LEDGER_N10_ABS_MAX = 0.1114
LEDGER_N10_CHI2_UPPER = 0.0949
LEDGER_N10_MDE = 0.1154

# The three draws of one declared-identical crown tree.
THREE_DRAWS = [CROWN, "f7d59543", NEW_ROW]

# FRAME RECONCILIATION. The brief prices `165d4ba7` at +0.1772 % "weighted
# five" and `f7d59543` at +0.0011 % "weighted five" in one table. Those two
# numbers come from two different estimators. This run reproduces both to the
# last published digit and reports every block statistic in all three frames,
# because the gate verdict is frame-dependent.
#
#   unweighted : the arithmetic mean of the five per-prompt percentages. This
#                is what `e148_lib.correct` computes and what every one of the
#                ten existing block members was priced with.
#   f83        : the F83 marginal median weights, applied as a SUM and not
#                renormalised, so the five weights total 0.9192 and the result
#                is systematically shrunk by 8 % as well as re-weighted.
#   median_pair: the row's own realised median pair, Rule 118's lead frame.
F83_WEIGHTS = {"beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
               "botany": 0.0124, "republic": 0.0100}


def frame_values(anchor, row):
    """One pair priced in the three frames, total leg."""
    per_prompt = E.pct_diff(anchor, row, "total", L.PROMPT_ORDER)
    pair = E.median_pair(anchor)
    return {
        "unweighted": E.mean([per_prompt[p] for p in E.WEIGHTED_FIVE]),
        "f83": sum(F83_WEIGHTS[p] * per_prompt[p] for p in F83_WEIGHTS),
        "median_pair": E.mean([per_prompt[p] for p in pair]),
        "median_pair_prompts": pair,
        "per_prompt": per_prompt,
    }


def gamma_p(a, x):
    """Regularized lower incomplete gamma P(a, x), series and continued fraction."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_cdf(x, df):
    return gamma_p(df / 2.0, x / 2.0)


def chi2_inv(p, df):
    """Inverse chi-square CDF by bisection. Enough digits for a 95 % bound."""
    low, high = 1e-9, 10.0 * df + 200.0
    for _ in range(300):
        mid = 0.5 * (low + high)
        if chi2_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def sigma_upper_95(sd, n):
    """One-sided 95 % upper confidence bound on sigma from a sample sd."""
    if n < 2:
        return float("nan")
    df = n - 1
    return math.sqrt(df * sd * sd / chi2_inv(0.05, df))


def block_stats(values):
    n = len(values)
    mean = E.mean(values)
    sd = E.sd(values) if n > 1 else float("nan")
    upper = sigma_upper_95(sd, n)
    return {
        "n": n,
        "values": values,
        "abs_max_pp": max(abs(v) for v in values),
        "sd_pp": sd,
        "rms_pp": math.sqrt(sum(v * v for v in values) / n),
        "mean_pp": mean,
        "sigma_chi2_upper95_pp": upper,
        "mde_2sigma_pp": 2.0 * sd,
        "mde_2sigma_upper95_pp": 2.0 * upper,
    }


def frame_block(table11):
    """Block statistics at n=10 and n=11 in each of the three price frames."""
    keys = {"unweighted": "frame_unweighted_pct", "f83": "frame_f83_pct",
            "median_pair": "frame_median_pair_pct"}
    out = {}
    for name, field in keys.items():
        values11 = [e[field] for e in table11]
        out[name] = {"n=10": block_stats(values11[:-1]),
                     "n=11": block_stats(values11)}
    return out


def classify(anchor, row):
    """Both fit frames for one pair, plus the decision fields behind the claim."""
    registered = E.correct(anchor, row, fit_prompts=E.COHORT_PROMPTS,
                           price_prompts=E.WEIGHTED_FIVE)
    matched = E.correct_auto(anchor, row, price_prompts=E.WEIGHTED_FIVE)
    matched_prompts = E.matched_prompts(anchor, row)
    heads = {anchor.head(p) for p in L.PROMPT_ORDER} | {row.head(p)
                                                        for p in L.PROMPT_ORDER}
    frames = frame_values(anchor, row)
    return {
        "frame_unweighted_pct": frames["unweighted"],
        "frame_f83_pct": frames["f83"],
        "frame_median_pair_pct": frames["median_pair"],
        "per_prompt_total_pct": frames["per_prompt"],
        "row": row.id8, "parent": anchor.id8,
        "solver": row.solver, "score": row.score, "status": row.status,
        "registered_k_us_per_dr": registered["k_us_per_drafting_round"],
        "registered_steps_exact": registered["steps_exact"],
        "registered_steps": registered["steps"],
        "registered_ambiguous": registered["ambiguous_step"],
        "registered_corrected_total_pct": registered["corrected_total_pct"],
        "registered_raw_total_pct": registered["raw_total_pct"],
        "matched_k_us_per_dr": matched["k_us_per_drafting_round"],
        "matched_steps_exact": matched["steps_exact"],
        "matched_steps": matched["steps"],
        "matched_ambiguous": matched["ambiguous_step"],
        "matched_corrected_total_pct": matched["corrected_total_pct"],
        "matched_raw_total_pct": matched["raw_total_pct"],
        "median_pair": matched["median_pair"],
        "median_pair_pct": matched["median_pair_pct"],
        "median_pair_corrected_pct": matched["median_pair_corrected_pct"],
        "prefill_pct": matched.get("prefill_pct"),
        "n_matched_prompts": len(matched_prompts),
        "matched_prompts": matched_prompts,
        "same_head_provenance_all_eight": len(heads) == 1,
        "head_provenance": row.head("beagle"),
        "fit_r2": matched["fit_r2"],
        "residual_sd_pp": matched["residual_sd_pp"],
    }


def three_draw_dispersion(rows_by_id):
    """One-way dispersion of three draws of one declared-identical tree.

    The three receipts publish the same tree, so every difference between them
    is session noise. For each weighted-five prompt the three candidate legs are
    expressed as a percentage of their own three-draw mean, then averaged over
    the five prompts. The spread of those three numbers is the honest
    single-receipt noise figure for a crown-class candidate.
    """
    draws = [rows_by_id[i] for i in THREE_DRAWS]
    per_prompt = {}
    for prompt in L.PROMPT_ORDER:
        legs = [L.leg(d, prompt, "total") for d in draws]
        centre = E.mean(legs)
        per_prompt[prompt] = {
            "seconds_per_token": legs,
            "mean": centre,
            "pct_of_mean": [100.0 * (v / centre - 1.0) for v in legs],
            "sd_pp": 100.0 * E.sd(legs) / centre,
            "range_pp": 100.0 * (max(legs) - min(legs)) / centre,
        }
    w5 = [E.mean([per_prompt[p]["pct_of_mean"][i] for p in E.WEIGHTED_FIVE])
          for i in range(3)]
    f83 = [sum(F83_WEIGHTS[p] * per_prompt[p]["pct_of_mean"][i]
               for p in F83_WEIGHTS) for i in range(3)]
    eight = [E.mean([per_prompt[p]["pct_of_mean"][i] for p in L.PROMPT_ORDER])
             for i in range(3)]
    scores = [d.score for d in draws]
    medians = [d.published_median() for d in draws]
    return {
        "draws": THREE_DRAWS,
        "solvers": [d.solver for d in draws],
        "per_prompt": per_prompt,
        "weighted_five_pct_of_mean": w5,
        "weighted_five_mean_pp": E.mean(w5),
        "weighted_five_sd_pp": E.sd(w5),
        "weighted_five_range_pp": max(w5) - min(w5),
        "f83_pct_of_mean": f83,
        "f83_sd_pp": E.sd(f83),
        "f83_range_pp": max(f83) - min(f83),
        "eight_prompt_pct_of_mean": eight,
        "eight_prompt_sd_pp": E.sd(eight),
        "eight_prompt_range_pp": max(eight) - min(eight),
        "published_scores": scores,
        "published_score_mean": E.mean(scores),
        "published_score_sd": E.sd(scores),
        "published_score_range": max(scores) - min(scores),
        "recomputed_medians": medians,
        "per_prompt_sd_pp_weighted_five": {
            p: per_prompt[p]["sd_pp"] for p in E.WEIGHTED_FIVE},
    }


def main():
    path, rows = E.load_rows()
    by_id = {r.id8: r for r in rows}
    print("board %s, %d scored rows" % (path, len(rows)))
    print("harness=ranked. Fit frame DECODE (prefill removed). "
          "Price frame TOTAL LEG on the weighted five %s."
          % ", ".join(E.WEIGHTED_FIVE))
    print("step constant %.1f us/drafting round, ambiguity threshold %.2f steps"
          % (E.STEP_US, E.AMBIGUOUS_STEP_FRACTION))

    missing = [i for i in [CROWN, NEW_ROW] + [r for r, _, _, _ in AT_ZERO_N10]
               if i not in by_id]
    if missing:
        raise SystemExit("rows absent from the board: %s" % ", ".join(missing))

    # ------------------------------------------------------------------
    # 1. classify the eleventh candidate
    # ------------------------------------------------------------------
    print("\n=== 1. classify %s against the crown %s ===" % (NEW_ROW, CROWN))
    new = classify(by_id[CROWN], by_id[NEW_ROW])
    print("  solver %s, status %s, published %.8f"
          % (new["solver"], new["status"], new["score"]))
    print("  registered six-prompt fit : k %+8.2f us/dr, steps_exact %+.4f -> "
          "%d step(s), ambiguous %s"
          % (new["registered_k_us_per_dr"], new["registered_steps_exact"],
             new["registered_steps"], new["registered_ambiguous"]))
    print("  schedule-matched fit (%d p): k %+8.2f us/dr, steps_exact %+.4f -> "
          "%d step(s), ambiguous %s"
          % (new["n_matched_prompts"], new["matched_k_us_per_dr"],
             new["matched_steps_exact"], new["matched_steps"],
             new["matched_ambiguous"]))
    print("  raw weighted-five %+.4f pp, corrected %+.4f pp (the correction "
          "subtracts nothing at zero steps)"
          % (new["matched_raw_total_pct"], new["matched_corrected_total_pct"]))
    print("  median pair %s %+.4f pp, prefill %+.4f pp"
          % ("+".join(new["median_pair"]), new["median_pair_pct"],
             new["prefill_pct"] if new["prefill_pct"] is not None
             else float("nan")))
    print("  decision fields: identical head provenance on all eight %s (%s), "
          "schedule identical on %d of 8 prompts"
          % (new["same_head_provenance_all_eight"], new["head_provenance"],
             new["n_matched_prompts"]))

    at_zero = (new["registered_steps"] == 0 and new["matched_steps"] == 0
               and not new["registered_ambiguous"]
               and not new["matched_ambiguous"])
    print("  classifies at zero steps in BOTH fit frames: %s" % at_zero)

    print("\n--- 1b. the brief's three quoted pairs, in every frame ---")
    print("%-22s %11s %11s %11s %9s"
          % ("pair", "unweighted", "F83", "medpair", "8p sd"))
    recon = {}
    for anchor_id, row_id in ((CROWN, "f7d59543"), (CROWN, NEW_ROW),
                              ("f7d59543", NEW_ROW)):
        frames = frame_values(by_id[anchor_id], by_id[row_id])
        values = list(frames["per_prompt"].values())
        recon["%s->%s" % (anchor_id, row_id)] = {
            "unweighted_pct": frames["unweighted"],
            "f83_pct": frames["f83"],
            "median_pair_pct": frames["median_pair"],
            "median_pair_prompts": frames["median_pair_prompts"],
            "eight_prompt_sd_pp": E.sd(values),
            "per_prompt_pct": frames["per_prompt"],
        }
        print("%-22s %+11.4f %+11.4f %+11.4f %9.4f"
              % ("%s -> %s" % (anchor_id, row_id), frames["unweighted"],
                 frames["f83"], frames["median_pair"], E.sd(values)))
    print("  the brief quotes +0.0011 for the first pair and +0.1772 and")
    print("  +0.1870 for the other two. The first is the unweighted column,")
    print("  the other two are the F83 column. Reading one table in two")
    print("  frames inflates the apparent gap between the draws.")

    # ------------------------------------------------------------------
    # 2. reproduce the operative n=10 block, then extend it
    # ------------------------------------------------------------------
    print("\n=== 2. the operative at-zero block, reproduced ===")
    print("%-9s %-9s %9s %6s %5s %10s %10s %9s  %s"
          % ("row", "parent", "k_us/dr", "st_ex", "steps", "corr_w5%",
             "medpair%", "provenance", "why"))
    table, values = [], []
    for row_id, parent_id, verified, why in AT_ZERO_N10:
        entry = classify(by_id[parent_id], by_id[row_id])
        entry["verified"] = verified
        entry["why"] = why
        table.append(entry)
        values.append(entry["matched_corrected_total_pct"])
        print("%-9s %-9s %+9.2f %+6.3f %5d %+10.4f %+10.4f %9s  %s"
              % (entry["row"], entry["parent"], entry["matched_k_us_per_dr"],
                 entry["matched_steps_exact"], entry["matched_steps"],
                 entry["matched_corrected_total_pct"],
                 entry["median_pair_corrected_pct"],
                 "VERIFIED" if verified else "declared", why[:34]))

    n10 = block_stats(values)
    drift = [abs(a - b) for a, b in zip(sorted(values),
                                        sorted(LEDGER_N10_VALUES))]
    print("\n  reproduction check against ledger 309 item 9: largest per-row "
          "difference %.4f pp" % max(drift))
    print("  n=10  abs_max %.4f (ledger %.4f)  sd %.4f (ledger %.4f)  "
          "chi2_95_upper %.4f (ledger %.4f)  2sigma MDE %.4f (ledger %.4f)"
          % (n10["abs_max_pp"], LEDGER_N10_ABS_MAX, n10["sd_pp"],
             LEDGER_N10_SD, n10["sigma_chi2_upper95_pp"], LEDGER_N10_CHI2_UPPER,
             n10["mde_2sigma_pp"], LEDGER_N10_MDE))

    n11 = None
    payload_frames = None
    verdict = "not-admitted"
    if at_zero:
        new["verified"] = False
        new["why"] = "declared yukon reset of the crown, manifest note only"
        table11 = table + [new]
        values11 = values + [new["matched_corrected_total_pct"]]
        n11 = block_stats(values11)
        verdict = "pass" if n11["abs_max_pp"] <= GATE_PP else "fail"
        print("\n=== 3. the at-zero block at n = 11 ===")
        print("%-9s %-9s %10s %11s %10s  %s"
              % ("row", "parent", "corr_w5%", "provenance", "medpair%", "why"))
        for entry in sorted(table11,
                            key=lambda e: e["matched_corrected_total_pct"]):
            print("%-9s %-9s %+10.4f %11s %+10.4f  %s"
                  % (entry["row"], entry["parent"],
                     entry["matched_corrected_total_pct"],
                     "VERIFIED" if entry["verified"] else "declared",
                     entry["median_pair_corrected_pct"], entry["why"][:44]))
        print("\n  values      %s"
              % ", ".join("%+.4f" % v for v in values11))
        print("  n           %d" % n11["n"])
        print("  abs max     %.4f pp" % n11["abs_max_pp"])
        print("  sd          %.4f pp" % n11["sd_pp"])
        print("  rms         %.4f pp" % n11["rms_pp"])
        print("  mean        %+.4f pp" % n11["mean_pp"])
        print("  sigma 95 %% chi-square upper  %.4f pp"
              % n11["sigma_chi2_upper95_pp"])
        print("  2 sigma MDE %.4f pp   (at the 95 %% sigma bound %.4f pp)"
              % (n11["mde_2sigma_pp"], n11["mde_2sigma_upper95_pp"]))
        print("\n  e149_at_zero_block_n11_gate_verdict = %s   (gate %.2f pp)"
              % (verdict, GATE_PP))
        print("  smallest gate the n=11 block still passes: %.4f pp"
              % n11["abs_max_pp"])
        print("  n=10 -> n=11 deltas: abs_max %+.4f, sd %+.4f, MDE %+.4f pp"
              % (n11["abs_max_pp"] - n10["abs_max_pp"],
                 n11["sd_pp"] - n10["sd_pp"],
                 n11["mde_2sigma_pp"] - n10["mde_2sigma_pp"]))

        frames = frame_block(table11)
        print("\n=== 3b. the same block in all three price frames ===")
        print("  the brief prices `165d4ba7` at +0.1772 and `f7d59543` at")
        print("  +0.0011 in one table. Those are two different estimators, so")
        print("  the block is reported in each of them.")
        print("%-9s %11s %11s %11s  %s"
              % ("row", "unweighted", "F83", "medpair", "provenance"))
        for entry in table11:
            print("%-9s %+11.4f %+11.4f %+11.4f  %s"
                  % (entry["row"], entry["frame_unweighted_pct"],
                     entry["frame_f83_pct"], entry["frame_median_pair_pct"],
                     "VERIFIED" if entry["verified"] else "declared"))
        print("%-24s %9s %9s %9s %9s %9s"
              % ("frame / block", "abs_max", "sd", "rms", "MDE_2s", "gate"))
        for key in ("unweighted", "f83", "median_pair"):
            for label in ("n=10", "n=11"):
                stats = frames[key][label]
                print("%-24s %9.4f %9.4f %9.4f %9.4f %9s"
                      % ("%s %s" % (key, label), stats["abs_max_pp"],
                         stats["sd_pp"], stats["rms_pp"], stats["mde_2sigma_pp"],
                         "pass" if stats["abs_max_pp"] <= GATE_PP else "FAIL"))
        payload_frames = frames
    else:
        print("\n=== 3. NOT admitted: the row does not classify at zero steps ===")
        print("  the block stays at n = 10 and every published statistic stands")

    # ------------------------------------------------------------------
    # 4. three draws of one tree
    # ------------------------------------------------------------------
    print("\n=== 4. within-tree dispersion, three draws of one declared tree ===")
    disp = three_draw_dispersion(by_id)
    print("  draws %s (%s)" % (", ".join(disp["draws"]),
                               ", ".join(disp["solvers"])))
    print("  each draw's weighted-five candidate leg as % of the three-draw mean:")
    for name, value in zip(disp["draws"], disp["weighted_five_pct_of_mean"]):
        print("    %-9s %+.4f %%" % (name, value))
    print("  weighted-five  mean %+.4f, sd %.4f, range %.4f pp"
          % (disp["weighted_five_mean_pp"], disp["weighted_five_sd_pp"],
             disp["weighted_five_range_pp"]))
    print("  eight-prompt   sd %.4f, range %.4f pp"
          % (disp["eight_prompt_sd_pp"], disp["eight_prompt_range_pp"]))
    print("  per-prompt sd over the three draws:")
    for prompt in L.PROMPT_ORDER:
        print("    %-9s %.4f pp" % (prompt, disp["per_prompt"][prompt]["sd_pp"]))
    print("  published scores %s" % ", ".join("%.8f" % s
                                              for s in disp["published_scores"]))
    print("  published score mean %.8f, sd %.8f, range %.8f"
          % (disp["published_score_mean"], disp["published_score_sd"],
             disp["published_score_range"]))

    payload = {
        "harness": "ranked",
        "board_path": path,
        "board_rows": len(rows),
        "gate_pp": GATE_PP,
        "step_us": E.STEP_US,
        "price_prompts": E.WEIGHTED_FIVE,
        "e149_165d4ba7_fitted_k_us_per_dr": new["registered_k_us_per_dr"],
        "e149_165d4ba7_steps_exact": new["registered_steps_exact"],
        "e149_165d4ba7_fitted_k_us_per_dr_matched_fit": new["matched_k_us_per_dr"],
        "e149_165d4ba7_steps_exact_matched_fit": new["matched_steps_exact"],
        "e149_165d4ba7_classifies_at_zero": at_zero,
        "e149_165d4ba7_raw_weighted_five_pct": new["matched_raw_total_pct"],
        "e149_165d4ba7_median_pair_pct": new["median_pair_pct"],
        "e149_165d4ba7_provenance": "declared",
        "e149_at_zero_block_n11_gate_verdict": verdict,
        "e149_at_zero_block_n11_smallest_passing_gate_pp":
            n11["abs_max_pp"] if n11 else None,
        "e149_at_zero_block_n11_gate_verdict_by_frame": {
            key: ("pass" if payload_frames[key]["n=11"]["abs_max_pp"] <= GATE_PP
                  else "fail") for key in payload_frames
        } if payload_frames else None,
        "e149_at_zero_block_n11_smallest_passing_gate_by_frame": {
            key: payload_frames[key]["n=11"]["abs_max_pp"]
            for key in payload_frames
        } if payload_frames else None,
        "e149_165d4ba7_f83_frame_pct": new["frame_f83_pct"],
        "frame_reconciliation": recon,
        "e149_at_zero_block_frames": payload_frames,
        "at_zero_block_n10": n10,
        "at_zero_block_n11": n11,
        "at_zero_block_rows": table + ([new] if at_zero else []),
        "new_row": new,
        "three_draw_dispersion": disp,
        "ledger309_n10_reference": {
            "values": LEDGER_N10_VALUES, "sd_pp": LEDGER_N10_SD,
            "abs_max_pp": LEDGER_N10_ABS_MAX,
            "sigma_chi2_upper95_pp": LEDGER_N10_CHI2_UPPER,
            "mde_2sigma_pp": LEDGER_N10_MDE,
            "largest_per_row_reproduction_difference_pp": max(drift),
        },
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
