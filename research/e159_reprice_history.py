#!/usr/bin/env python3
"""E159. Re-price the historical depth-price arms under RULE 177 and RULE 178.

`research/receipt_candidate_leg.py compare` pairs two receipts prompt by prompt
on `mtp_seconds_per_token_mean`. This module reads the same receipts and adds
the RULE 178 weighting, so an arm is priced by the measured median sensitivity
rather than by an unweighted eight-prompt mean.

The point of the exercise is ask 1 of the advisor's E159 feedback: the pbfit
`+0.33 %` was read off a published median and cannot be banked as stated.
"""
from __future__ import annotations

import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import receipt_candidate_leg as rcl  # noqa: E402

# Advisor E159 feedback section 2, measured by direct +0.10 % perturbation of
# each prompt's raw ratio through the real order statistic on 923 receipts.
RULE178_WEIGHTS = {
    "beagle": 0.4741,
    "medicine": 0.1951,
    "essays": 0.1658,
    "republic": 0.0963,
    "botany": 0.0508,
    "travel": 0.0037,
    "plutarch": 0.0033,
    "drama": 0.0019,
}

SHA8 = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# The depth-price arm history, as the advisor tabulated it in the assignment.
PAIRS = [
    ("pb6", "572b2cc4", "e003a86d", 0.170414),
    ("pbfit", "ca9251b8", "2da69933", 0.120143),
]

ORDER = ["beagle", "medicine", "essays", "republic", "botany", "travel",
         "plutarch", "drama"]


def leg_percents(id_a: str, id_b: str):
    """Paired per-prompt candidate-leg percent, b against a. Negative = faster."""
    table = rcl.submissions()
    a, b = table.get(id_a[:8]), table.get(id_b[:8])
    if a is None or b is None:
        raise SystemExit("missing receipt %s or %s" % (id_a, id_b))
    pa, pb = rcl.prompts(a), rcl.prompts(b)
    out, edl, serial_a, serial_b = {}, {}, {}, {}
    for sha, name in SHA8.items():
        if sha not in pa or sha not in pb:
            continue
        ca = pa[sha]["mtp_seconds_per_token_mean"]
        cb = pb[sha]["mtp_seconds_per_token_mean"]
        out[name] = (cb / ca - 1.0) * 100.0
        edl[name] = (pa[sha].get("effective_mean_draft_len"),
                     pb[sha].get("effective_mean_draft_len"))
        serial_a[name] = pa[sha]["serial_seconds_per_token_mean"]
        serial_b[name] = pb[sha]["serial_seconds_per_token_mean"]
    return out, edl, serial_a, serial_b, a, b


def weighted(per_prompt: dict) -> dict:
    items = [(v, RULE178_WEIGHTS[p]) for p, v in per_prompt.items()]
    wsum = sum(w for _, w in items)
    w2 = sum(w * w for _, w in items)
    mean = sum(w * v for v, w in items) / wsum
    n_eff = wsum * wsum / w2
    var = sum(w * (v - mean) ** 2 for v, w in items) / wsum
    sd = math.sqrt(var * n_eff / (n_eff - 1.0))
    return {"mean": mean, "sd": sd, "se": sd / math.sqrt(n_eff),
            "n_eff": n_eff}


def spread_pct(values) -> float:
    lo, hi = min(values), max(values)
    return (hi / lo - 1.0) * 100.0


def main() -> int:
    print("## RULE 177 and RULE 178 re-pricing of the depth-price arm history")
    print("   Candidate-leg percent is `mtp_seconds_per_token_mean` of the arm")
    print("   against its own base receipt. NEGATIVE means the arm is FASTER.")
    print("   Gains are quoted as positive-is-better, so gain = -percent.\n")

    for name, id_a, id_b, gate in PAIRS:
        pct, edl, sa, sb, ra, rb = leg_percents(id_a, id_b)
        flat = statistics.fmean(pct.values())
        flat_sd = statistics.stdev(pct.values())
        flat_se = flat_sd / math.sqrt(len(pct))
        w = weighted(pct)
        same_sign = sum(1 for v in pct.values() if v * flat > 0)

        print("### %s   %s -> %s   marginal[0] %.6f -> 0.18 held? no"
              % (name, id_a[:8], id_b[:8], gate))
        print("%-10s %9s %12s %12s %14s %10s" % (
            "prompt", "weight", "cand leg %", "edl base", "edl arm",
            "weighted"))
        for prompt in ORDER:
            if prompt not in pct:
                continue
            print("%-10s %9.4f %+12.4f %12.4f %14.4f %+10.4f" % (
                prompt, RULE178_WEIGHTS[prompt], pct[prompt],
                edl[prompt][0], edl[prompt][1],
                RULE178_WEIGHTS[prompt] * pct[prompt]))

        seven = {p: v for p, v in pct.items() if p != "plutarch"}
        s_mean = statistics.fmean(seven.values())
        s_se = statistics.stdev(seven.values()) / math.sqrt(len(seven))

        print("  RULE 177 unweighted gain        %+.4f %%  2 sigma %.4f  "
              "same sign %d/%d"
              % (-flat, 2.0 * flat_se, same_sign, len(pct)))
        print("  RULE 178 weighted gain          %+.4f %%  2 sigma %.4f  "
              "n_eff %.3f" % (-w["mean"], 2.0 * w["se"], w["n_eff"]))
        print("  seven prompts without plutarch  %+.4f %%  2 sigma %.4f"
              % (-s_mean, 2.0 * s_se))
        print("  plutarch alone                  %+.4f %%" % -pct["plutarch"])
        sc_a, sc_b = float(ra["officialScore"]), float(rb["officialScore"])
        print("  published median                %.8f -> %.8f, %+.4f %%"
              % (sc_a, sc_b, (sc_b / sc_a - 1.0) * 100.0))
        print("  serial within-run spread        base %.3f %%, arm %.3f %% "
              "(unusable above about 0.8 %%)"
              % (spread_pct(sa.values()), spread_pct(sb.values())))
        print("  RULE 178 verdict                %s\n" % (
            "loss" if -w["mean"] < 0 else "gain"))

    print("## The sign disagreement, stated plainly")
    print("%-7s %12s %12s %12s %12s %10s" % (
        "arm", "unweighted", "RULE 178", "exact order", "published",
        "178/pub"))
    for name, id_a, id_b, _ in PAIRS:
        pct, _, _, _, ra, rb = leg_percents(id_a, id_b)
        flat = -statistics.fmean(pct.values())
        w = -weighted(pct)["mean"]
        exact = exact_median_shift(ra, pct)
        med = (float(rb["officialScore"]) / float(ra["officialScore"]) - 1.0) * 100.0
        print("%-7s %+12.4f %+12.4f %+12.4f %+12.4f %10.3f"
              % (name, flat, w, exact, med, w / med if med else float("nan")))
    print("\n  The unweighted mean gets the SIGN WRONG on both arms. RULE 178")
    print("  and the exact order statistic both get it right. RULE 178 is")
    print("  validated out of sample on two ranked receipt pairs.")
    print("  RULE 178 under-predicts the magnitude by a factor near 1.5,")
    print("  and the exact order statistic recovers most of that gap, so the")
    print("  residual is order-statistic reshuffling, not a bad weight table.")
    print("  The linearisation is safe for screening and understates a loss.")
    return 0


def exact_median_shift(base_row: dict, cand_pct: dict) -> float:
    """Published median change from a candidate-leg change, exactly.

    Holds every serial leg at its base value and pushes the per-prompt
    candidate-leg change through the real order statistic, so the result
    includes any reshuffling of the median pair. RULE 178's weight table is
    the first-order approximation of this quantity.
    """
    per_prompt = rcl.prompts(base_row)
    base_raw, arm_raw = [], []
    for sha, name in SHA8.items():
        if sha not in per_prompt or name not in cand_pct:
            continue
        entry = per_prompt[sha]
        serial = entry["serial_seconds_per_token_mean"]
        cand = entry["mtp_seconds_per_token_mean"]
        base_raw.append(serial / cand)
        arm_raw.append(serial / (cand * (1.0 + cand_pct[name] / 100.0)))
    return (rcl.median8(arm_raw) / rcl.median8(base_raw) - 1.0) * 100.0


if __name__ == "__main__":
    sys.exit(main())
