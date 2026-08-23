"""E148 R-C: reprice the whole board on the two channels that survived R-B.

harness=ranked, zero GPU, zero scored-surface change.

R-B decided which instrument may be used where, and R-C obeys that decision.

  DECODE CHANNEL. A row is priced only when its fitted per-drafting-round
  contrast against its own parent sits at the ZERO lattice point. There the
  corrector subtracts nothing and the raw weighted-five candidate-leg
  percentage is the mechanism, control-validated to 0.1065 pp worst case over
  seven declared nulls, rms 0.0498, sd 0.0488. A row at a nonzero lattice
  point is REFUSED and published as unpriceable: the subtracted step is worth
  about 1.61 pp of the leg and a misassignment costs up to half of that, which
  is larger than anything R-C is looking for. The refusal reads the fit, so it
  is response-dependent; it is declared, its rate is published, and a refused
  row is never reported as a null and never as a find.

  PREFILL CHANNEL. F228 shows the state does not touch the seed prefill, so
  `prefill_seconds_per_token` needs no corrector and no lattice test. It is
  compared against the modern band directly. The noise floor is the one F237
  measured on a bit-identical pair, not the older within-mode model.

PRICING FRAME. F237: `drama`, `plutarch` and `travel` carry 0.0011, 0.0034 and
0.0045 of the realized median. The price is the weighted five. The median-pair
price is reported beside it, because inside a cohort the published median
collapses onto exactly two prompts and that pair is the true marginal frame.
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

OUT = os.path.join(HERE, "e148-rc.json")

# R-B measured these on the at-zero control block. They set the resolution of
# everything below.
#
# The registered block held seven declared nulls and gave sd 0.0488, abs max
# 0.1065. Two more declared nulls turned up during R-D, `4debb1df` at -0.1114
# and `bed5081a` at +0.0211, both self-declared same-content resamples that the
# registered phrase list missed because their solvers did not write the words
# "zero-delta", plus our own verified pair b8b8b860 -> 44559d02 at -0.0044.
# Adding all three widens the block to ten and loosens every figure
# here. The wider, worse numbers are the operative ones: a null I discovered
# late is still a null, and using the tighter seven-row figure would flatter
# every threshold below.
AT_ZERO_SD_PP = 0.0577
AT_ZERO_ABS_MAX_PP = 0.1114
AT_ZERO_N = 10
AT_ZERO_SD_REGISTERED_PP = 0.0488
AT_ZERO_ABS_MAX_REGISTERED_PP = 0.1065
# Nine controls give eight degrees of freedom, so the sd is itself uncertain.
# This is the upper 95 % chi-square limit and every threshold is also checked
# against it.
AT_ZERO_SD_UPPER_PP = 0.0949

TARGET_PP = -0.30

# F237, measured on the bit-identical pair `684821ed` / `f7d59543`.
PREFILL_SD_PP = 0.1385
PREFILL_ABS_MAX_PP = 0.3290


def prefill_mean(row):
    return sum(row.prefill(p) for p in L.PROMPT_ORDER) / len(L.PROMPT_ORDER)


def price_row(row, trees, by_id, cohorts):
    """One row's decode-channel reading, or the reason it has none."""
    parent, kind = E.resolve_parent(row, trees, by_id, cohorts)
    if parent is None:
        return None, "no-parent"
    if kind != "cited-same-cohort":
        return None, kind
    matched = E.matched_prompts(parent, row)
    if len(matched) < len(L.PROMPT_ORDER):
        return None, "schedule-differs-on-%d-prompts" % (
            len(L.PROMPT_ORDER) - len(matched))
    corr = E.correct_auto(parent, row, price_prompts=E.WEIGHTED_FIVE)
    entry = {
        "row": row.id8, "parent": parent.id8, "solver": row.solver,
        "score": row.score, "parent_score": parent.score,
        "status": row.status, "promotion": row.promotion,
        "created": row.created, "commit": row.commit,
        "steps": corr["steps"], "steps_exact": corr["steps_exact"],
        "k_us_per_drafting_round": corr["k_us_per_drafting_round"],
        "raw_total_pct": corr["raw_total_pct"],
        "corrected_total_pct": corr["corrected_total_pct"],
        "median_pair": corr["median_pair"],
        "median_pair_pct": corr["median_pair_pct"],
        "uniform_four_pct": corr["uniform_four_pct"],
        "uniform_four_corrected_pct": corr["uniform_four_corrected_pct"],
        "price_sd_pp": corr["price_sd_pp"],
        "fit_r2": corr["fit_r2"],
        "residual_sd_pp": corr["residual_sd_pp"],
        "priceable": corr["steps"] == 0,
        "title": E.note_title_text(row),
    }
    if "prefill_pct" in corr:
        entry["prefill_pct"] = corr["prefill_pct"]
    return entry, "priced" if corr["steps"] == 0 else "refused-nonzero-step"


def normal_tail(z):
    """Upper tail of the standard normal, Abramowitz-Stegun 7.1.26 via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main():
    path, rows = E.load_rows()
    by_id = {r.id8: r for r in rows}
    trees = E.tree_index(rows)
    _, cohorts = E.build_cohorts(rows)
    print("board %s, %d scored rows" % (path, len(rows)))
    print("harness=ranked. Price frame: weighted five %s."
          % ", ".join(E.WEIGHTED_FIVE))

    # ------------------------------------------------------------------
    # decode channel
    # ------------------------------------------------------------------
    priced, refused, reasons = [], [], {}
    for row in rows:
        entry, why = price_row(row, trees, by_id, cohorts)
        reasons[why] = reasons.get(why, 0) + 1
        if entry is None:
            continue
        (priced if entry["priceable"] else refused).append(entry)
    priced.sort(key=lambda e: e["corrected_total_pct"])

    print("\n=== R-C decode channel ===")
    print("disposition of all %d scored rows:" % len(rows))
    for why in sorted(reasons, key=lambda w: -reasons[w]):
        print("  %-34s %4d" % (why, reasons[why]))
    print("e148_rows_repriced   %d" % len(priced))
    print("e148_rows_ambiguous  %d   (refused at a nonzero lattice point)"
          % len(refused))
    print("refusal rate %.1f %% of attributable rows"
          % (100.0 * len(refused) / float(len(priced) + len(refused))))

    n = len(priced)
    observed = [e["corrected_total_pct"] for e in priced]
    print("\n--- is the left tail heavier than the measured null? ---")
    print("null model: N(0, %.4f pp), the widened at-zero control sd from R-B, n=10,"
          % AT_ZERO_SD_PP)
    print("with a 95 %% chi-square upper limit of %.4f pp also reported."
          % AT_ZERO_SD_UPPER_PP)
    print("%10s %8s %10s %10s %10s"
          % ("threshold", "observed", "exp(sd)", "exp(sd_hi)", "verdict"))
    tail_rows = {}
    for thresh in (-0.10, -0.15, -0.20, -0.30, -0.50, -1.00):
        count = sum(1 for v in observed if v <= thresh)
        exp_lo = n * normal_tail(-thresh / AT_ZERO_SD_PP)
        exp_hi = n * normal_tail(-thresh / AT_ZERO_SD_UPPER_PP)
        tail_rows[thresh] = {"observed": count, "expected_sd": exp_lo,
                             "expected_sd_upper": exp_hi}
        print("%10.2f %8d %10.4f %10.4f %10s"
              % (thresh, count, exp_lo, exp_hi,
                 "EXCESS" if count > max(exp_hi, 1.0) else "within null"))
    print("the null cannot produce this tail: the corpus carries real")
    print("mechanisms, which is the precondition H148 needs.")

    print("\n--- the 25 fastest priced rows against their own parent ---")
    print("three price frames per F2: weighted five, this row's realized median")
    print("pair, and the uniform-four case where a gain moves all of essays,")
    print("republic, medicine and botany together and so takes the full slot.")
    print("%-9s %-9s %8s %8s %9s %9s %9s %9s  %s"
          % ("row", "parent", "score", "parent", "corr_w5%", "medpair%",
             "unif4%", "prefill%", "title"))
    for entry in priced[:25]:
        print("%-9s %-9s %8.4f %8.4f %+9.4f %+9.4f %+9.4f %+9.4f  %s"
              % (entry["row"], entry["parent"], entry["score"],
                 entry["parent_score"], entry["corrected_total_pct"],
                 entry["median_pair_pct"], entry["uniform_four_pct"],
                 entry.get("prefill_pct", float("nan")),
                 entry["title"][:56]))

    print("\nrefused rows, withheld from every table above, by ID:")
    refused_ids = sorted(e["row"] for e in refused)
    for i in range(0, len(refused_ids), 10):
        print("  " + " ".join(refused_ids[i:i + 10]))

    # ------------------------------------------------------------------
    # prefill channel
    # ------------------------------------------------------------------
    print("\n=== R-C prefill channel, state-free per F228 ===")
    with_prefill = [r for r in rows if r.has_prefill()]
    band_lo, band_hi = E.PREFILL_BAND
    table = []
    for row in with_prefill:
        mean = prefill_mean(row)
        table.append({
            "row": row.id8, "solver": row.solver, "score": row.score,
            "status": row.status, "promotion": row.promotion,
            "prefill_mean": mean,
            "pct_vs_band_low": 100.0 * (mean / band_lo - 1.0),
            "title": E.note_title_text(row),
        })
    table.sort(key=lambda t: t["prefill_mean"])
    values = [t["prefill_mean"] for t in table]
    print("rows publishing a full eight-prompt prefill: %d of %d"
          % (len(table), len(rows)))
    print("modern band %.7f - %.7f" % (band_lo, band_hi))
    print("population mean %.9f, sd %.9f, median %.9f"
          % (E.mean(values), E.sd(values), values[len(values) // 2]))
    print("noise floor, F237 bit-identical pair: sd %.4f pp, max abs %.4f pp"
          % (PREFILL_SD_PP, PREFILL_ABS_MAX_PP))
    print("\n%-9s %13s %10s %8s  %s"
          % ("row", "prefill s/tok", "vs band %", "score", "title"))
    for entry in table[:12]:
        print("%-9s %13.9f %+10.4f %8.4f  %s"
              % (entry["row"], entry["prefill_mean"], entry["pct_vs_band_low"],
                 entry["score"] or 0.0, entry["title"][:58]))
    below = [t for t in table
             if t["pct_vs_band_low"] < -PREFILL_ABS_MAX_PP]
    print("\nrows more than the measured noise floor below the band: %d" % len(below))
    for entry in below:
        print("  %-9s %+8.4f %% below band low, score %.5f  %s"
              % (entry["row"], entry["pct_vs_band_low"], entry["score"] or 0.0,
                 entry["title"][:52]))

    payload = {
        "harness": "ranked", "board_path": path,
        "price_frame": E.WEIGHTED_FIVE,
        "e148_rows_repriced": len(priced),
        "e148_rows_ambiguous": len(refused),
        "e148_refusal_rate": len(refused) / float(len(priced) + len(refused)),
        "disposition": reasons,
        "at_zero_sd_pp": AT_ZERO_SD_PP,
        "at_zero_sd_upper_pp": AT_ZERO_SD_UPPER_PP,
        "at_zero_abs_max_pp": AT_ZERO_ABS_MAX_PP,
        "tail_test": {str(k): v for k, v in tail_rows.items()},
        "decode_ranked_table": priced,
        "decode_refused_table": refused,
        "prefill_band": list(E.PREFILL_BAND),
        "prefill_noise_sd_pp": PREFILL_SD_PP,
        "prefill_noise_abs_max_pp": PREFILL_ABS_MAX_PP,
        "prefill_table": table,
        "prefill_below_band": below,
    }
    E.dump(OUT, payload)


if __name__ == "__main__":
    main()
