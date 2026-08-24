#!/usr/bin/env python3
"""E159 Part B robustness. Is the best gate-held arm a schedule or a curve fit?

The E159 Part A round-cost law and the independent F97 board fit agree within
1.4 % at every verify width except width 4, where they differ by 12.4 %. If a
12.4 % disagreement at one width reorders the arm family, the ranking is a
property of the cost curve and not of the depth schedule.
"""
from __future__ import annotations

import json
import pathlib
import sys

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e159-artifacts"
ORDER = ["beagle", "essays", "republic", "botany", "medicine", "drama",
         "travel", "plutarch"]


def main() -> int:
    curves = {}
    for name in ("e159", "f97"):
        path = ARTIFACTS / ("e159_gate_held_arms_%scurve.json" % name)
        curves[name] = json.loads(path.read_text())

    shared = sorted(set(curves["e159"]["arms"]) & set(curves["f97"]["arms"]))
    held = [a for a in shared
            if curves["e159"]["arms"][a]["gate_held"]
            and not a.startswith(("ship|", "gatescan|"))]

    print("## RULE 178 weighted gain, same arms, two measured cost curves")
    print("   Positive is faster. `seed 2s` is the replay's own resampling")
    print("   uncertainty over 7 seeds, which is the only variance term that")
    print("   describes this fixed, fully observed eight-prompt set.")
    print("%-20s %10s %9s %10s %9s %10s" % (
        "arm", "e159 %", "seed 2s", "f97 %", "seed 2s", "swing"))
    rows = []
    for arm in held:
        a = curves["e159"]["arms"][arm]
        b = curves["f97"]["arms"][arm]
        swing = a["rule178_gain_pct"] - b["rule178_gain_pct"]
        rows.append((arm, a, b, swing))
    rows.sort(key=lambda r: -r[1]["rule178_gain_pct"])
    for arm, a, b, swing in rows:
        print("%-20s %+10.4f %9.4f %+10.4f %9.4f %+10.4f" % (
            arm, a["rule178_gain_pct"], 2.0 * a["rule178_seed_se"],
            b["rule178_gain_pct"], 2.0 * b["rule178_seed_se"], swing))

    best_e159 = max(rows, key=lambda r: r[1]["rule178_gain_pct"])
    best_f97 = max(rows, key=lambda r: r[2]["rule178_gain_pct"])
    print("\n  best under e159 curve   %s  %+.4f %% (f97 says %+.4f %%)"
          % (best_e159[0], best_e159[1]["rule178_gain_pct"],
             best_e159[2]["rule178_gain_pct"]))
    print("  best under f97 curve    %s  %+.4f %% (e159 says %+.4f %%)"
          % (best_f97[0], best_f97[2]["rule178_gain_pct"],
             best_f97[1]["rule178_gain_pct"]))
    print("  same arm chosen         %s" % (best_e159[0] == best_f97[0]))
    print("  curve swing on the e159 winner   %+.4f %%, against a replay-seed"
          % best_e159[3])
    print("  2 sigma of %.4f %%. The curve moves it %.1fx its own noise."
          % (2.0 * best_e159[1]["rule178_seed_se"],
             abs(best_e159[3]) / (2.0 * best_e159[1]["rule178_seed_se"])))

    print("\n## per-prompt candidate-leg percent for the e159 winner, %s"
          % best_e159[0])
    print("   Negative means the arm decodes that prompt faster.")
    print("%-10s %9s %10s %10s %12s" % (
        "prompt", "weight", "e159 %", "f97 %", "weighted e159"))
    a, b = best_e159[1], best_e159[2]
    weights = curves["e159"]["rule178_weights"]
    for prompt in ORDER:
        print("%-10s %9.4f %+10.3f %+10.3f %+12.4f" % (
            prompt, weights[prompt], a["per_prompt_pct"][prompt],
            b["per_prompt_pct"][prompt],
            weights[prompt] * a["per_prompt_pct"][prompt]))

    print("\n## RULE 178 linear weights against the exact order statistic")
    print("   `exact %` applies the same per-prompt candidate-leg changes to")
    print("   the base receipt's raw ratios and re-sorts, so it includes any")
    print("   reshuffling of the median pair. RULE 178's table is its")
    print("   first-order approximation and is only valid while the sort")
    print("   order holds.")
    print("%-20s %10s %10s %8s %9s %8s" % (
        "arm", "w178 %", "exact %", "agree", "max |eff|", "reshuf"))
    disagree = 0
    for arm, row, _b, _s in rows:
        w, ex = row["rule178_gain_pct"], row["median_pct"]
        agree = (w > 0) == (ex > 0)
        if not agree:
            disagree += 1
        biggest = max(abs(v) for v in row["per_prompt_pct"].values())
        print("%-20s %+10.4f %+10.4f %8s %9.3f %8s" % (
            arm, w, ex, "yes" if agree else "NO", biggest,
            "yes" if row["pair_changed"] else "no"))
    print("  sign disagreements %d of %d gate-held arms" % (disagree, len(rows)))
    positive_exact = [r for r in rows if r[1]["median_pct"] > 0]
    print("  gate-held arms with a POSITIVE exact prediction: %d"
          % len(positive_exact))
    print("  The near-tied pack that supplies the 5th value spans 0.844 %.")
    print("  Every arm here moves at least one prompt by more than that, so")
    print("  every arm reshuffles the sort and the linear weights do not hold.")

    print("\n## what the stop rule needs, given the weighting")
    n_eff = curves["e159"]["arms"][held[0]]["rule178_n_eff"]
    print("  RULE 178 Kish effective sample size            %.3f of 8" % n_eff)
    print("  Per-prompt spread that still clears +0.30 % at 2 sigma, if the")
    print("  between-prompt spread is used as the variance:  sd < %.4f %%"
          % (0.30 * (n_eff ** 0.5) / 2.0))
    print("  Observed per-prompt sd of the e159 winner:      sd = %.4f %%"
          % a["sd"])
    print("  A mechanism uniform enough to pass that test is by definition a")
    print("  uniform mechanism, which RULE 178 prices unweighted instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
