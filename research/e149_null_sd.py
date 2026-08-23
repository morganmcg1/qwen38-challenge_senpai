"""E149 F2 section 4: the per-prompt candidate-leg null sd over the n=11 block.

harness=ranked, zero GPU. Every input is a published per-prompt field of a
ranked Yukon receipt.

WHAT THIS DECIDES

Rung 0a measured, over three draws of one declared-identical crown tree, a
per-prompt candidate-leg sd of beagle 0.1850 pp against essays 0.0301 pp. Three
draws carry two degrees of freedom, so that ratio is not decisive. The eleven
at-zero contrasts carry ten degrees of freedom per prompt, so this file
re-estimates the same quantity on the larger block and asks what accounts for
the spread.

FRAME, RULE 144. Every number here is named. The lead price frame of the block
is `unweighted mean of the weighted five`, but a PER-PROMPT sd is not a price
and has no weighting: it is the dispersion of one prompt's candidate-leg
percentage across the eleven contrasts. Both leg frames are reported, TOTAL
(the block's price frame, F237) and DECODE (the fit frame, F227).

CONTRAST VERSUS RECEIPT. Each block member is a DIFFERENCE of two receipts. If
the two draws are independent, `sd_contrast = sqrt(2) * sd_receipt`, so the
single-receipt figure comparable with rung 0a's three-draw number is the
contrast sd divided by sqrt(2). Both are printed. Four of the eleven contrasts
share the crown as their anchor, so the eleven are not fully independent; that
caveat is carried in the output and is not corrected for.
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
from e149_rung0a import AT_ZERO_N10, CROWN, NEW_ROW, THREE_DRAWS, sigma_upper_95

OUT = os.path.join(HERE, "e149-null-sd.json")

PAYING_FOUR = ["essays", "republic", "medicine", "botany"]

# F239 per-prompt state exposure at 879 us per DRAFTING round, as the advisor
# published it in F2 section 4. Carried verbatim so the correlation test runs
# against the advisor's own table and not a re-derivation.
F239_STATE_EXPOSURE_PCT = {
    "beagle": 2.0070, "medicine": 1.8261, "essays": 1.8438, "botany": 1.6551,
    "republic": 1.8889, "plutarch": 0.2278, "drama": 2.6450, "travel": 2.5670,
}

# Rung 0a's three-draw per-prompt sd, for the cross-check column.
THREE_DRAW_SD_PP = {
    "beagle": 0.1850, "botany": 0.0292, "drama": 0.2674, "essays": 0.0301,
    "medicine": 0.0396, "plutarch": 0.3552, "republic": 0.0312, "travel": 0.2007,
}


# ----------------------------------------------------------------------
# distributions
# ----------------------------------------------------------------------

def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log1p(-x) + a * math.log(x)) \
        * _betacf(b, a, 1.0 - x) / b


def f_cdf(f, d1, d2):
    if f <= 0.0:
        return 0.0
    return betai(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))


def f_inv(p, d1, d2):
    low, high = 1e-9, 1e6
    for _ in range(300):
        mid = math.sqrt(low * high)
        if f_cdf(mid, d1, d2) < p:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high)


def variance_ratio_ci(var1, df1, var2, df2, level=0.95):
    """Two-sided CI on sigma1/sigma2 from two independent sample variances."""
    alpha = 1.0 - level
    ratio = var1 / var2
    lo = ratio / f_inv(1.0 - alpha / 2.0, df1, df2)
    hi = ratio / f_inv(alpha / 2.0, df1, df2)
    return math.sqrt(lo), math.sqrt(hi)


def pearson(xs, ys):
    n = len(xs)
    mx, my = E.mean(xs), E.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        for pos, i in enumerate(order):
            r[i] = float(pos + 1)
        return r
    return pearson(ranks(xs), ranks(ys))


# ----------------------------------------------------------------------
# the block
# ----------------------------------------------------------------------

def block_members():
    """The eleven at-zero contrasts, in the order rung 0a published them."""
    members = [(row, parent, verified) for row, parent, verified, _ in AT_ZERO_N10]
    members.append((NEW_ROW, CROWN, False))
    return members


def per_prompt_matrix(by_id, field):
    """prompt -> the eleven per-prompt candidate-leg percentages."""
    out = {p: [] for p in L.PROMPT_ORDER}
    labels = []
    for row, parent, _ in block_members():
        diffs = L.pct_diff(by_id[parent], by_id[row], field)
        labels.append("%s<-%s" % (row, parent))
        for p in L.PROMPT_ORDER:
            out[p].append(diffs[p])
    return labels, out


def prompt_stats(values):
    n = len(values)
    df = n - 1
    m = E.mean(values)
    s = E.sd(values)
    # A null block has a known true mean of zero, so the dispersion about zero
    # is also reported. It uses n degrees of freedom and is the figure that
    # matters if a reader wants the total error of a single contrast.
    rms0 = math.sqrt(sum(v * v for v in values) / n)
    return {
        "n": n, "df": df, "mean_pp": m, "sd_pp": s,
        "sd_upper95_pp": sigma_upper_95(s, n),
        "rms_about_zero_pp": rms0,
        "abs_max_pp": max(abs(v) for v in values),
        "receipt_sd_pp": s / math.sqrt(2.0),
        "values_pp": values,
    }


def rounds_per_second(prompt):
    """TOTAL rounds per second of candidate leg, from the F219 anchor table."""
    cand, _, rounds = L.F219_ANCHOR[prompt]
    return rounds / (L.DECODE_TOKENS * cand)


def main():
    path, rows = E.load_rows()
    by_id = {r.id8: r for r in rows}
    print("board %s, %d scored rows" % (path, len(rows)))
    print("harness=ranked. RULE 144 frames named on every table.")

    members = block_members()
    print("\nblock: %d at-zero contrasts" % len(members))
    anchors = {}
    for row, parent, _ in members:
        anchors[parent] = anchors.get(parent, 0) + 1
    print("anchors: %s" % ", ".join("%s x%d" % (k, v)
                                    for k, v in sorted(anchors.items())))

    result = {"board": path, "n_rows": len(rows),
              "members": ["%s<-%s" % (r, p) for r, p, _ in members],
              "anchor_multiplicity": anchors}

    for field in ("total", "decode"):
        labels, matrix = per_prompt_matrix(by_id, field)
        stats = {p: prompt_stats(matrix[p]) for p in L.PROMPT_ORDER}
        result["labels"] = labels
        result["per_prompt_%s" % field] = stats

        print("\n=== per-prompt null sd, leg frame %s, n=11, df=10 ==="
              % field.upper())
        print("  %-9s %9s %9s %9s %9s %9s %9s"
              % ("prompt", "mean", "sd", "sd_u95", "rms0", "absmax", "1-receipt"))
        for p in L.PROMPT_ORDER:
            s = stats[p]
            print("  %-9s %+9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
                  % (p, s["mean_pp"], s["sd_pp"], s["sd_upper95_pp"],
                     s["rms_about_zero_pp"], s["abs_max_pp"],
                     s["receipt_sd_pp"]))

        # beagle against the pooled other four paying prompts
        vb = stats["beagle"]["sd_pp"] ** 2
        pooled_var = E.mean([stats[p]["sd_pp"] ** 2 for p in PAYING_FOUR])
        ratio = math.sqrt(vb / pooled_var)
        lo, hi = variance_ratio_ci(vb, 10, pooled_var, 40)
        result["beagle_to_paying_four_%s" % field] = {
            "ratio": ratio, "ci95": [lo, hi],
            "beagle_sd_pp": stats["beagle"]["sd_pp"],
            "pooled_four_sd_pp": math.sqrt(pooled_var),
            "pooled_prompts": PAYING_FOUR, "df_numerator": 10,
            "df_denominator": 40,
        }
        print("  beagle %.4f / pooled(%s) %.4f  =  %.3fx  "
              "[95%% CI %.3f, %.3f]  df 10 vs 40"
              % (stats["beagle"]["sd_pp"], "+".join(PAYING_FOUR),
                 math.sqrt(pooled_var), ratio, lo, hi))

        # ---------------- accounts ----------------
        prompts = list(L.PROMPT_ORDER)
        sds = [stats[p]["sd_pp"] for p in prompts]
        exposure = [F239_STATE_EXPOSURE_PCT[p] for p in prompts]
        rps = [rounds_per_second(p) for p in prompts]
        dlen = [L.F219_ANCHOR[p][1] for p in prompts]
        rounds = [L.F219_ANCHOR[p][2] for p in prompts]

        accounts = {
            "f239_drafting_state_exposure_pct": exposure,
            "total_rounds_per_second_of_leg": rps,
            "mean_draft_length": dlen,
            "total_round_count": rounds,
        }
        acc_out = {}
        print("  accounts for the spread (8 prompts):")
        for name, xs in accounts.items():
            acc_out[name] = {"pearson_r": pearson(xs, sds),
                             "spearman_rho": spearman(xs, sds),
                             "values": dict(zip(prompts, xs))}
            print("    %-34s pearson %+.4f  spearman %+.4f"
                  % (name, acc_out[name]["pearson_r"],
                     acc_out[name]["spearman_rho"]))
        # the same two correlations restricted to the five paying prompts
        idx5 = [prompts.index(p) for p in E.WEIGHTED_FIVE]
        for name, xs in accounts.items():
            acc_out[name]["pearson_r_weighted_five"] = pearson(
                [xs[i] for i in idx5], [sds[i] for i in idx5])
            acc_out[name]["spearman_rho_weighted_five"] = spearman(
                [xs[i] for i in idx5], [sds[i] for i in idx5])
        result["accounts_%s" % field] = acc_out

        # implied per-round jitter. If the null noise is a per-round timing
        # jitter of k microseconds, then sd_p (per cent) = k * 1e-6 * rps_p *
        # 100, so k = sd_p / (1e-4 * rps_p) and k must be constant across
        # prompts for the account to hold.
        jitter = {p: stats[p]["sd_pp"] / (1e-4 * rounds_per_second(p))
                  for p in prompts}
        jl = [jitter[p] for p in prompts]
        result["implied_per_round_jitter_us_%s" % field] = {
            "per_prompt": jitter, "mean": E.mean(jl), "sd": E.sd(jl),
            "min": min(jl), "max": max(jl),
            "spread_ratio": max(jl) / min(jl),
        }
        print("  implied per-round jitter (us/round), constant if the account "
              "holds:")
        print("    " + "  ".join("%s %.2f" % (p, jitter[p]) for p in prompts))
        print("    mean %.2f  sd %.2f  spread %.2fx"
              % (E.mean(jl), E.sd(jl), max(jl) / min(jl)))

        # signal-to-noise corollary: a per-round mechanism's signal scales with
        # the same exposure, so its detectability per prompt is exposure / sd.
        snr = {p: (1e-4 * rounds_per_second(p)) / stats[p]["sd_pp"]
               for p in prompts}
        result["per_round_mechanism_snr_per_us_%s" % field] = snr
        print("  detectable per-round saving at 2 sigma on ONE receipt "
              "(us/round, lower is better):")
        print("    " + "  ".join("%s %.1f" % (p, 2.0 / snr[p])
                                 for p in prompts))

        # jackknife: is any single contrast carrying a prompt's dispersion?
        jack = {}
        for p in prompts:
            vals = stats[p]["values_pp"]
            worst_i, worst_sd = None, None
            for i in range(len(vals)):
                drop = vals[:i] + vals[i + 1:]
                s = E.sd(drop)
                if worst_sd is None or s < worst_sd:
                    worst_sd, worst_i = s, i
            jack[p] = {"sd_drop1_pp": worst_sd,
                       "dropped": labels[worst_i],
                       "dropped_value_pp": vals[worst_i],
                       "shrink_ratio": worst_sd / stats[p]["sd_pp"]}
        result["jackknife_%s" % field] = jack
        print("  jackknife, sd after dropping the single most influential "
              "contrast:")
        for p in prompts:
            j = jack[p]
            print("    %-9s %.4f -> %.4f (%.2fx) dropping %s at %+.4f"
                  % (p, stats[p]["sd_pp"], j["sd_drop1_pp"],
                     j["shrink_ratio"], j["dropped"], j["dropped_value_pp"]))

        # where the median-pair uncertainty lives. The published median is the
        # mean of two per-prompt ratios, so if the two legs were independent
        # its sd would be sqrt(var_lower + var_upper) / 2.
        var_b = stats["beagle"]["sd_pp"] ** 2
        var_o = pooled_var
        share = var_b / (var_b + var_o)
        implied_pair_sd = math.sqrt(var_b + var_o) / 2.0
        result["median_pair_variance_decomposition_%s" % field] = {
            "beagle_variance_share": share,
            "implied_median_pair_sd_pp_if_independent": implied_pair_sd,
            "implied_median_pair_2sigma_mde_pp": 2.0 * implied_pair_sd,
        }
        print("  median-pair decomposition: beagle carries %.1f %% of the "
              "variance; independent-prompt sd %.4f pp, 2 sigma %.4f pp"
              % (100.0 * share, implied_pair_sd, 2.0 * implied_pair_sd))

    # cross-check against rung 0a's three draws
    tot = result["per_prompt_total"]
    cross = {}
    print("\n=== cross-check: n=11 single-receipt sd against rung 0a's three "
          "draws (TOTAL frame) ===")
    print("  %-9s %12s %12s %8s" % ("prompt", "n=11/sqrt2", "3-draw", "ratio"))
    for p in L.PROMPT_ORDER:
        a = tot[p]["receipt_sd_pp"]
        b = THREE_DRAW_SD_PP[p]
        cross[p] = {"n11_receipt_sd_pp": a, "three_draw_sd_pp": b,
                    "ratio": a / b}
        print("  %-9s %12.4f %12.4f %8.2f" % (p, a, b, a / b))
    result["three_draw_cross_check"] = cross

    # round-count anomaly, stated as the advisor asked
    rc = {p: {"total_rounds": L.F219_ANCHOR[p][2],
              "rounds_per_second_of_leg": rounds_per_second(p),
              "sd_pp_total_frame": tot[p]["sd_pp"]} for p in L.PROMPT_ORDER}
    result["round_count_table"] = rc

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
