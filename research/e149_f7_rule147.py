#!/usr/bin/env python3
"""E149 F7 item 1. Test the Rule 147 per-prompt detectability table.

`harness=ranked`. Zero GPU.

F7 asks two things that are not the same question:

  (a) "check that against your n=11 block"
  (b) "independent confirmation of Rule 147 from a row you did not use to
      build it" -- the fresh receipt `0cf1637e` against the crown `684821ed`.

(a) is a REPRODUCTION check, not an independent test, if Rule 147 was itself
derived from this block. This module measures the provenance first: it
recomputes the per-prompt implied per-round jitter at n=10 and at n=11 and
reports the ratio against Rule 147 / 2. If the n=10 ratio is ~1.00 the table is
this block and (a) can only reproduce it.

(b) is the real test. `0cf1637e` is schedule-matched to the crown on all eight
prompts (identical `effective_mean_draft_len` digits and identical
`non_drafting_round_count`), so the pair admits a clean per-drafting-round
decomposition. The advisor's hypothesis is:

    the five paying prompts carry a real common effect, and the drama/travel
    excursion is mostly per-prompt noise

which under Rule 147 is a falsifiable statement: convert every per-prompt
percentage to microseconds per drafting round, fit ONE common per-round cost,
and ask whether the residuals are inside the per-prompt sigma the table
predicts. If drama and travel come back at |z| <= 2 while their raw
percentages are 2.5x the paying-five cluster, the table has explained the
excursion and it is confirmed. If they come back at |z| >> 2, the table
under-states those two prompts and the excursion is a real non-uniform effect.

Usage:
  python3 research/e149_f7_rule147.py --json research/e149-f7-rule147.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e146_lib as L  # noqa: E402
import e148_lib as E  # noqa: E402

CROWN = "684821ed"
NEW = "0cf1637e"

# RULE 147, exactly as F4 published it: the per-prompt saving in microseconds
# per drafting round that ONE ranked receipt can detect at 2 sigma.
RULE_147_2SIGMA_US = {
    "essays": 44.5, "medicine": 66.9, "republic": 70.9, "beagle": 119.5,
    "botany": 129.2, "drama": 123.8, "travel": 150.1, "plutarch": 211.9,
}

# The operative at-zero null block. `parent` is the anchor each row is priced
# against. The eleventh member `165d4ba7` is the one rung 0a added.
BLOCK = [
    ("106573b9", "1760479a"), ("64508884", "3ba6ee9d"),
    ("aff3b543", "1760479a"), ("b8e0f27c", "3ba6ee9d"),
    ("3a18ff21", CROWN), ("b6cb0fea", "0b8602e1"),
    ("f7d59543", CROWN), ("4debb1df", CROWN),
    ("bed5081a", "48423d09"), ("44559d02", "b8b8b860"),
    ("165d4ba7", CROWN),
]

PAYING_FIVE = ["beagle", "botany", "essays", "medicine", "republic"]
EXCURSION = ["drama", "travel"]


def sd(values, ddof=1):
    n = len(values)
    if n - ddof <= 0:
        return float("nan")
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - ddof))


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n))
                    * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else float("nan")


def basis_pct_per_us(anchor, prompt, field, unit="rule147"):
    """Percent of `anchor`'s leg moved by one microsecond per round.

    `rule147` reproduces the exact basis the table was built on
    (`e149_null_sd.rounds_per_second`): TOTAL rounds from the F219 anchor
    table over the F219 anchor's TOTAL-frame candidate leg. It is a fixed
    per-prompt constant, so it does not move with the row being priced.

    `drafting` is the alternative per-DRAFTING-round basis. It differs by a
    factor of one on the paying prompts and by 12x on plutarch, whose crown
    schedule leaves only about 11 drafting rounds in 487.
    """
    if unit == "rule147":
        _, _, rounds = L.F219_ANCHOR[prompt]
        cand = L.F219_ANCHOR[prompt][0]
        return rounds / (L.DECODE_TOKENS * cand) * 1e-6 * 100.0
    rounds = L.round_count(prompt, anchor.dlen(prompt))
    drafting = rounds - anchor.non_drafting(prompt)
    return drafting / (L.DECODE_TOKENS * L.leg(anchor, prompt, field)) * 1e-6 * 100.0


# ---------------------------------------------------------------- provenance

def block_jitter(by, field, members, unit="rule147"):
    """Per-prompt sd of the null block, in pp and in us per round."""
    out = {}
    for p in L.PROMPT_ORDER:
        vals, per_round = [], []
        for child, parent in members:
            a, c = by[parent], by[child]
            pct = 100.0 * (L.leg(c, p, field) / L.leg(a, p, field) - 1.0)
            vals.append(pct)
            per_round.append(pct / basis_pct_per_us(a, p, field, unit))
        out[p] = {
            "n": len(vals),
            "sd_pp": sd(vals),
            "sd_us_per_round": sd(per_round),
            "values_pp": vals,
        }
    return out


# ------------------------------------------------------------------ the test

def common_fit(units, sigmas, prompts):
    """Inverse-variance weighted common per-round cost and its residual zs."""
    wsum = sum(1.0 / sigmas[p] ** 2 for p in prompts)
    k = sum(units[p] / sigmas[p] ** 2 for p in prompts) / wsum
    se = math.sqrt(1.0 / wsum)
    z = {p: (units[p] - k) / sigmas[p] for p in prompts}
    chi2 = sum(z[p] ** 2 for p in prompts)
    return {
        "k_us_per_round": k,
        "se_us_per_round": se,
        "z": z,
        "chi2": chi2,
        "dof": len(prompts) - 1,
        "chi2_per_dof": chi2 / (len(prompts) - 1),
        "prompts": list(prompts),
    }


def chi2_sf(x, k):
    """Upper tail of chi-squared with k dof, via the regularized gamma."""
    if x <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:                      # series for P(a, x)
        term = 1.0 / a
        total = term
        n = 1
        while n < 10000:
            term *= xx / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
            n += 1
        p = total * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return 1.0 - p
    b, c = xx + 1.0 - a, 1e300            # continued fraction for Q(a, x)
    d, h = 1.0 / b, 1.0 / b
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def run(board):
    path, rows = L.load(board)
    by = {r.id8: r for r in rows}
    for rid in [CROWN, NEW] + [x for pair in BLOCK for x in pair]:
        if rid not in by:
            raise SystemExit(f"row {rid} is not on the scored board {path}")

    crown, new = by[CROWN], by[NEW]

    # Schedule match, stated rather than assumed.
    schedule = {
        p: {
            "dlen_crown": crown.dlen(p), "dlen_new": new.dlen(p),
            "dlen_identical": repr(crown.dlen(p)) == repr(new.dlen(p)),
            "non_drafting_crown": crown.non_drafting(p),
            "non_drafting_new": new.non_drafting(p),
            "non_drafting_identical":
                crown.non_drafting(p) == new.non_drafting(p),
            "head_crown": crown.head(p), "head_new": new.head(p),
            "head_identical": crown.head(p) == new.head(p),
        }
        for p in L.PROMPT_ORDER
    }
    schedule_matched = all(v["dlen_identical"] and v["non_drafting_identical"]
                           for v in schedule.values())

    out = {
        "harness": "ranked",
        "board": path,
        "scored_rows": len(rows),
        "crown": CROWN,
        "new_row": NEW,
        "new_row_score": new.score,
        "new_row_status": new.status,
        "crown_score": crown.score,
        "rule_147_2sigma_us": RULE_147_2SIGMA_US,
        "schedule_match": schedule,
        "schedule_matched_all_eight": schedule_matched,
    }

    # ---- provenance: is Rule 147 this block? -----------------------------
    prov = {}
    for field in ("decode", "total"):
        n10 = block_jitter(by, field, BLOCK[:10])
        n11 = block_jitter(by, field, BLOCK)
        prov[field] = {
            "n10": {p: n10[p]["sd_us_per_round"] for p in n10},
            "n11": {p: n11[p]["sd_us_per_round"] for p in n11},
            "n10_sd_pp": {p: n10[p]["sd_pp"] for p in n10},
            "n11_sd_pp": {p: n11[p]["sd_pp"] for p in n11},
            "ratio_n10_to_rule147_half": {
                p: n10[p]["sd_us_per_round"]
                / (RULE_147_2SIGMA_US[p] / 2.0) for p in n10},
            "ratio_n11_to_rule147_half": {
                p: n11[p]["sd_us_per_round"]
                / (RULE_147_2SIGMA_US[p] / 2.0) for p in n11},
        }
        r10 = list(prov[field]["ratio_n10_to_rule147_half"].values())
        r11 = list(prov[field]["ratio_n11_to_rule147_half"].values())
        prov[field]["ratio_n10_mean"] = sum(r10) / len(r10)
        prov[field]["ratio_n10_max_abs_dev"] = max(abs(v - 1.0) for v in r10)
        prov[field]["ratio_n11_mean"] = sum(r11) / len(r11)
        prov[field]["ratio_n11_max_abs_dev"] = max(abs(v - 1.0) for v in r11)
        ref = [RULE_147_2SIGMA_US[p] for p in L.PROMPT_ORDER]
        prov[field]["spearman_n10_vs_rule147"] = spearman(
            ref, [prov[field]["n10"][p] for p in L.PROMPT_ORDER])
        prov[field]["spearman_n11_vs_rule147"] = spearman(
            ref, [prov[field]["n11"][p] for p in L.PROMPT_ORDER])
    out["rule147_provenance"] = prov
    dec = prov["decode"]
    out["rule147_is_the_n10_block"] = bool(dec["ratio_n10_max_abs_dev"] < 0.10)
    out["rule147_is_the_n11_block"] = bool(dec["ratio_n11_max_abs_dev"] < 0.10)

    # ---- the independent test on 0cf1637e --------------------------------
    tests = {}
    for unit in ("rule147", "drafting"):
        for field in ("decode", "total"):
            pct = {p: 100.0 * (L.leg(new, p, field) / L.leg(crown, p, field) - 1.0)
                   for p in L.PROMPT_ORDER}
            basis = {p: basis_pct_per_us(crown, p, field, unit)
                     for p in L.PROMPT_ORDER}
            units = {p: pct[p] / basis[p] for p in L.PROMPT_ORDER}
            sigma = {p: RULE_147_2SIGMA_US[p] / 2.0 for p in L.PROMPT_ORDER}

            all8 = common_fit(units, sigma, L.PROMPT_ORDER)
            five = common_fit(units, sigma, PAYING_FIVE)
            z_excursion_vs_five = {
                p: (units[p] - five["k_us_per_round"]) / sigma[p]
                for p in EXCURSION}

            # Joint test of the excursion pair against the paying-five level.
            joint_chi2 = sum(v * v for v in z_excursion_vs_five.values())
            # Competing model: the effect is a uniform PERCENTAGE, not a
            # uniform per-round cost. Same data, same sigmas, expressed in pp.
            sigma_pp = {p: sigma[p] * basis[p] for p in L.PROMPT_ORDER}
            pct_five = common_fit(pct, sigma_pp, PAYING_FIVE)
            pct_all8 = common_fit(pct, sigma_pp, L.PROMPT_ORDER)

            tests[f"{unit}/{field}"] = {
                "unit": unit,
                "field": field,
                "pct": pct,
                "basis_pct_per_us": basis,
                "us_per_round": units,
                "sigma_us": sigma,
                "sigma_pp": sigma_pp,
                "common_fit_all8": all8,
                "common_fit_all8_p": chi2_sf(all8["chi2"], all8["dof"]),
                "common_fit_paying_five": five,
                "common_fit_paying_five_p": chi2_sf(five["chi2"], five["dof"]),
                "z_excursion_vs_paying_five_level": z_excursion_vs_five,
                "excursion_joint_chi2": joint_chi2,
                "excursion_joint_p": chi2_sf(joint_chi2, 2),
                "uniform_pct_model_paying_five": pct_five,
                "uniform_pct_model_paying_five_p":
                    chi2_sf(pct_five["chi2"], pct_five["dof"]),
                "uniform_pct_model_all8": pct_all8,
                "uniform_pct_model_all8_p":
                    chi2_sf(pct_all8["chi2"], pct_all8["dof"]),
                "mean8_pct": sum(pct.values()) / 8.0,
                "paying_five_mean_pct":
                    sum(pct[p] for p in PAYING_FIVE) / len(PAYING_FIVE),
            }
    out["independent_test_0cf1637e"] = tests

    # ---- verdict ---------------------------------------------------------
    d = tests["rule147/decode"]
    z5 = d["common_fit_paying_five"]["z"]
    zx = d["z_excursion_vs_paying_five_level"]
    five_inside = all(abs(v) <= 2.0 for v in z5.values())
    excursion_inside = all(abs(v) <= 2.0 for v in zx.values())
    plutarch_z = d["common_fit_all8"]["z"]["plutarch"]

    out["e149_rule147_paying_five_all_within_2sigma"] = bool(five_inside)
    out["e149_rule147_excursion_pair_within_2sigma"] = bool(excursion_inside)
    out["e149_rule147_max_abs_z_paying_five"] = max(abs(v) for v in z5.values())
    out["e149_rule147_max_abs_z_excursion"] = max(abs(v) for v in zx.values())
    out["e149_rule147_plutarch_z_all8"] = plutarch_z
    if five_inside and excursion_inside:
        verdict = "confirmed"
    elif five_inside:
        verdict = "confirmed_on_paying_five_understates_excursion_pair"
    else:
        verdict = "not_confirmed"
    out["e149_rule147_holds_on_0cf1637e"] = verdict
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=None)
    ap.add_argument("--json", default=str(HERE / "e149-f7-rule147.json"))
    args = ap.parse_args()

    out = run(args.board)
    pathlib.Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"board {out['board']}  scored rows {out['scored_rows']}")
    print(f"{NEW} score {out['new_row_score']} status {out['new_row_status']}")
    print(f"schedule matched on all eight: {out['schedule_matched_all_eight']}")
    print()
    print("PROVENANCE of Rule 147 (decode frame, Rule 147 total-round basis)")
    dec = out["rule147_provenance"]["decode"]
    hdr = "%-10s %10s %9s %7s %9s %7s" % (
        "prompt", "rule147/2", "n10 sd", "ratio", "n11 sd", "ratio")
    print(hdr)
    for p in L.PROMPT_ORDER:
        print("%-10s %10.2f %9.2f %7.3f %9.2f %7.3f" % (
            p, RULE_147_2SIGMA_US[p] / 2.0, dec["n10"][p],
            dec["ratio_n10_to_rule147_half"][p], dec["n11"][p],
            dec["ratio_n11_to_rule147_half"][p]))
    print("mean ratio n10 %.4f  max abs dev %.4f"
          % (dec["ratio_n10_mean"], dec["ratio_n10_max_abs_dev"]))
    print("mean ratio n11 %.4f  max abs dev %.4f"
          % (dec["ratio_n11_mean"], dec["ratio_n11_max_abs_dev"]))
    print("rule147_is_the_n10_block = %s" % out["rule147_is_the_n10_block"])
    print("rule147_is_the_n11_block = %s" % out["rule147_is_the_n11_block"])
    print("spearman rank vs rule147: n10 %.4f  n11 %.4f"
          % (dec["spearman_n10_vs_rule147"], dec["spearman_n11_vs_rule147"]))
    print()

    for key in ("rule147/decode", "rule147/total", "drafting/decode"):
        d = out["independent_test_0cf1637e"][key]
        print("INDEPENDENT TEST on %s   basis=%s" % (NEW, key))
        print("%-10s %8s %10s %8s %8s %8s"
              % ("prompt", "pct", "us/round", "sigma", "z(all8)", "z(five)"))
        for p in L.PROMPT_ORDER:
            z8 = d["common_fit_all8"]["z"][p]
            z5 = d["common_fit_paying_five"]["z"].get(p)
            if z5 is None:
                z5 = d["z_excursion_vs_paying_five_level"].get(p)
            z5s = "%8.2f" % z5 if z5 is not None else "       -"
            print("%-10s %8.4f %10.1f %8.1f %8.2f %s"
                  % (p, d["pct"][p], d["us_per_round"][p], d["sigma_us"][p],
                     z8, z5s))
        a8, f5 = d["common_fit_all8"], d["common_fit_paying_five"]
        print("  common all8  k = %8.1f +- %.1f  chi2/dof %8.2f  p %.4g"
              % (a8["k_us_per_round"], a8["se_us_per_round"],
                 a8["chi2_per_dof"], d["common_fit_all8_p"]))
        print("  common five  k = %8.1f +- %.1f  chi2/dof %8.2f  p %.4g"
              % (f5["k_us_per_round"], f5["se_us_per_round"],
                 f5["chi2_per_dof"], d["common_fit_paying_five_p"]))
        print("  excursion pair vs the five level: chi2 %.2f on 2 dof, p %.4g"
              % (d["excursion_joint_chi2"], d["excursion_joint_p"]))
        up = d["uniform_pct_model_paying_five"]
        print("  competing uniform-PERCENT model on the five: "
              "%.4f pp, chi2/dof %.2f, p %.4g"
              % (up["k_us_per_round"], up["chi2_per_dof"],
                 d["uniform_pct_model_paying_five_p"]))
        print()

    print("e149_rule147_holds_on_0cf1637e = %s"
          % out["e149_rule147_holds_on_0cf1637e"])
    print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

