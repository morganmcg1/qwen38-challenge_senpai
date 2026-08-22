#!/usr/bin/env python3
"""F15 reconciliation: why the advisor's log law and my flat law disagree.

    YUKON_API_TOKEN=... python3 research/e135_f15_reconcile.py 623e77af 572b2cc4

F14 (mine) reported that a flat launch saving survives on the seven drafting
prompts and after a drafting-fraction correction. F15 (advisor) reported four
fits over the same receipt pair and selected

    C:  saving = a * ln(Mbar)      with NO intercept,  R^2 = 0.8861

over

    A:  saving = a                 (flat),             R^2 = 0.0000

and called the flat model dead. This script tests five specific objections to
that selection. Every number below is computed from the receipts, not quoted.

  1 CENTERING. A model without an intercept has no centered R^2, so the usual
    library default reports the UNCENTERED R^2, which is measured against zero
    rather than against the sample mean. A's 0.0000 is centered. Comparing the
    two selects the model with the more flattering denominator, not the model
    with the smaller error. The fair statements are (a) both R^2 on the same
    denominator and (b) an error criterion that has no denominator at all.

  2 ROUND RECOVERY. F14 and F15 report different mean savings from the same
    receipts, so they used different round counts R. R is not in the receipt;
    it is recovered from the exact rational `effective_mean_draft_len`. The
    verdict must not depend on which feasible multiple is chosen.

  3 MONOTONICITY. A no-intercept log law is increasing in Mbar and passes
    through the origin. If the observed saving instead FALLS as Mbar rises
    across the drafting prompts, no such law fits, whatever its R^2.

  4 CONFOUND. The single lowest-width prompt is also the prompt that declines
    to draft in almost every round. A non-drafting round dispatches no
    multi-row verification matvec, so its launch saving is zero by
    construction. ln(Mbar) and the drafting fraction are near-collinear at
    that point, and either one explains the same near-zero saving.

  5 CONSEQUENCE. Under the shipped `onepass67` route the tight grid removes
    M-1 columns at widths 3..7. Turning the log law into the per-column slope
    it implies gives a value that the measured per-drafting-round slope
    excludes by many standard errors.

PRESS, the leave-one-out sum of squared prediction errors, is the criterion
used for the model choice. It is invariant to centering, it charges every
model for its own parameters, and it is computable in closed form from the hat
matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, "research")

from e135_f14_regression import (  # noqa: E402
    board,
    invert,
    receipt,
    recover_rounds,
)

# Two-sided 5 % critical value of Student's t at 6 degrees of freedom, which is
# eight prompts fitted with an intercept and one slope.
T_CRITICAL_DOF6 = 2.447
# The shipped route dispatches one pass at widths 6 and 7 and a tiered switch
# elsewhere; the tight grid deletes M-1 empty columns at widths 3..7.
ONEPASS_WIDTHS = (3, 4, 5, 6, 7)


def fit(y: list[float], columns: list[list[float]], intercept: bool = True):
    """OLS with an optional intercept, plus centered and uncentered R^2 and PRESS."""
    n = len(y)
    design = [([1.0] if intercept else []) + [c[i] for c in columns]
              for i in range(n)]
    width = len(design[0])
    gram = [[sum(design[i][a] * design[i][b] for i in range(n))
             for b in range(width)] for a in range(width)]
    moment = [sum(design[i][a] * y[i] for i in range(n)) for a in range(width)]
    inverse = invert(gram)
    beta = [sum(inverse[a][b] * moment[b] for b in range(width))
            for a in range(width)]
    fitted = [sum(beta[a] * design[i][a] for a in range(width)) for i in range(n)]
    residual = [y[i] - fitted[i] for i in range(n)]
    sse = sum(r * r for r in residual)
    dof = max(1, n - width)
    variance = sse / dof
    se = [math.sqrt(max(0.0, variance * inverse[a][a])) for a in range(width)]

    mean = sum(y) / n
    sst_centered = sum((v - mean) ** 2 for v in y)
    sst_uncentered = sum(v * v for v in y)

    # PRESS uses the hat diagonal, h_ii = x_i' (X'X)^-1 x_i.
    press = 0.0
    for i in range(n):
        hat = sum(design[i][a] * inverse[a][b] * design[i][b]
                  for a in range(width) for b in range(width))
        press += (residual[i] / max(1e-12, 1.0 - hat)) ** 2

    return {
        "beta": beta,
        "se": se,
        "sigma": math.sqrt(variance),
        "sse": sse,
        "press": press,
        "r2_centered": 1.0 - sse / sst_centered if sst_centered else float("nan"),
        "r2_uncentered": 1.0 - sse / sst_uncentered,
        "params": width,
    }


def models(y: list[float], mbar: list[float]) -> dict:
    ln = [math.log(m) for m in mbar]
    return {
        "A  a": fit(y, []),
        "B  a + b*Mbar": fit(y, [mbar]),
        "C  a*ln(Mbar), no intercept": fit(y, [ln], intercept=False),
        "D  a + b*ln(Mbar)": fit(y, [ln]),
    }


def report(title: str, y: list[float], mbar: list[float]) -> dict:
    print("\n=== %s (n=%d) ===" % (title, len(y)))
    print("%-30s %10s %10s %9s %9s %9s %12s"
          % ("model", "coef(last)", "se", "sigma", "R2 cent", "R2 unc", "PRESS"))
    out = models(y, mbar)
    for name, f in out.items():
        print("%-30s %10.1f %10.1f %9.1f %9.4f %9.4f %12.3e"
              % (name, f["beta"][-1], f["se"][-1], f["sigma"],
                 f["r2_centered"], f["r2_uncentered"], f["press"]))
    best = min(out.items(), key=lambda kv: kv[1]["press"])
    print("lowest PRESS: %s" % best[0])
    d = out["D  a + b*ln(Mbar)"]
    t = d["beta"][1] / d["se"][1] if d["se"][1] else float("nan")
    print("nested test of ln(Mbar) inside an intercept model: b=%.1f se=%.1f "
          "t=%+.2f -> %s at 5 %%"
          % (d["beta"][1], d["se"][1], t,
             "SIGNIFICANT" if abs(t) > T_CRITICAL_DOF6 else "not significant"))
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in a))
    db = math.sqrt(sum((v - mb) ** 2 for v in b))
    return num / (da * db) if da and db else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    rows = board(os.environ["YUKON_API_TOKEN"])
    before, after = receipt(args.before, rows), receipt(args.after, rows)
    tokens = before["officialMetrics"]["decode_tokens"]
    new_by_hash = {p["prompt_sha256"]: p
                   for p in after["officialMetrics"]["per_prompt"]}

    table = []
    for old in before["officialMetrics"]["per_prompt"]:
        new = new_by_hash[old["prompt_sha256"]]
        mean_draft = new["effective_mean_draft_len"]
        rounds, feasible = recover_rounds(mean_draft, tokens)
        delta = (old["mtp_seconds_per_token_mean"]
                 - new["mtp_seconds_per_token_mean"])
        fraction = (rounds - new["non_drafting_round_count"]) / rounds
        table.append({
            "prompt": old["prompt_sha256"][:8],
            "mbar": 1.0 + mean_draft,
            "rounds": rounds,
            "feasible": feasible,
            "naive": tokens / (1.0 + mean_draft),
            "delta": delta,
            "saving": delta * (tokens / rounds) * 1e6,
            "fraction": fraction,
            "nondrafting": new["non_drafting_round_count"],
        })

    print("%-10s %8s %7s %8s %10s %10s %9s"
          % ("prompt", "Mbar", "R", "drafting", "us/round", "per-draft",
             "feasible"))
    for r in table:
        print("%-10s %8.3f %7d %8.3f %10.1f %10.1f %9s"
              % (r["prompt"], r["mbar"], r["rounds"], r["fraction"],
                 r["saving"], r["saving"] / r["fraction"],
                 ",".join(str(v) for v in r["feasible"][:3])))

    y = [r["saving"] for r in table]
    mbar = [r["mbar"] for r in table]

    # 1 CENTERING -----------------------------------------------------------
    print("\n################ 1 centering ################")
    all8 = report("all eight prompts, recovered R", y, mbar)
    a, c = all8["A  a"], all8["C  a*ln(Mbar), no intercept"]
    print("the advisor compared A's CENTERED R2 %.4f with C's UNCENTERED R2 %.4f"
          % (a["r2_centered"], c["r2_uncentered"]))
    print("on the same uncentered denominator: A %.4f vs C %.4f -> ln(Mbar) adds "
          "%.4f of uncentered variance"
          % (a["r2_uncentered"], c["r2_uncentered"],
             c["r2_uncentered"] - a["r2_uncentered"]))
    print("on error rather than R2: SSE A %.3e vs C %.3e; PRESS A %.3e vs C %.3e"
          % (a["sse"], c["sse"], a["press"], c["press"]))

    # 2 ROUND RECOVERY ------------------------------------------------------
    print("\n################ 2 round recovery ################")
    print("mean saving at recovered R: %.1f us/round" % (sum(y) / len(y)))
    naive_y = [r["delta"] * (tokens / r["naive"]) * 1e6 for r in table]
    print("mean saving at naive R:     %.1f us/round" % (sum(naive_y) / len(naive_y)))
    for label, alt_index in (("second feasible multiple", 1),
                             ("third feasible multiple", 2)):
        alt_y, ok = [], True
        for r in table:
            if len(r["feasible"]) <= alt_index:
                ok = False
                break
            alt_y.append(r["delta"] * (tokens / r["feasible"][alt_index]) * 1e6)
        if not ok:
            print("%s: unavailable for at least one prompt" % label)
            continue
        f = models(alt_y, mbar)
        print("%s: mean %.1f, PRESS A %.3e C %.3e D %.3e"
              % (label, sum(alt_y) / len(alt_y), f["A  a"]["press"],
                 f["C  a*ln(Mbar), no intercept"]["press"],
                 f["D  a + b*ln(Mbar)"]["press"]))
    report("all eight prompts, naive R", naive_y, mbar)

    # 3 MONOTONICITY --------------------------------------------------------
    print("\n################ 3 monotonicity ################")
    drafting = [r for r in table if r["nondrafting"] == 0]
    drafting.sort(key=lambda r: r["mbar"])
    print("drafting prompts sorted by width:")
    coefficient = c["beta"][0]
    for r in drafting:
        predicted = coefficient * math.log(r["mbar"])
        print("  %-10s Mbar %6.3f  observed %8.1f  C predicts %8.1f  error %+7.1f %%"
              % (r["prompt"], r["mbar"], r["saving"], predicted,
                 100.0 * (predicted - r["saving"]) / r["saving"]))
    lo, hi = drafting[0], drafting[-1]
    print("observed change across the drafting width range %.3f -> %.3f: %+.1f us"
          % (lo["mbar"], hi["mbar"], hi["saving"] - lo["saving"]))
    print("C requires a change of %+.1f us over the same range"
          % (coefficient * (math.log(hi["mbar"]) - math.log(lo["mbar"]))))
    print("rank correlation of saving with width over the drafting prompts: %+.3f"
          % pearson([float(i) for i in range(len(drafting))],
                    [r["saving"] for r in drafting]))
    report("drafting prompts only",
           [r["saving"] for r in drafting], [r["mbar"] for r in drafting])

    # 4 CONFOUND ------------------------------------------------------------
    print("\n################ 4 confound ################")
    ln_all = [math.log(r["mbar"]) for r in table]
    frac_all = [r["fraction"] for r in table]
    print("Pearson r(ln Mbar, drafting fraction), all eight: %+.4f"
          % pearson(ln_all, frac_all))
    keep = [i for i, r in enumerate(table) if r["nondrafting"] == 0]
    print("the same correlation over the %d drafting prompts is undefined "
          "(fraction is exactly 1.0 for all of them)" % len(keep))
    corrected = [r["saving"] / r["fraction"] for r in table]
    print("fitting the drafting fraction alone, all eight prompts:")
    frac_fit = fit(y, [frac_all])
    print("  saving = %.1f %+.1f * fraction, se %.1f, t %+.2f, PRESS %.3e"
          % (frac_fit["beta"][0], frac_fit["beta"][1], frac_fit["se"][1],
             frac_fit["beta"][1] / frac_fit["se"][1], frac_fit["press"]))
    report("per-drafting-round saving, all eight prompts", corrected, mbar)

    # 5 CONSEQUENCE ---------------------------------------------------------
    print("\n################ 5 consequence ################")
    per_round = report("per-drafting-round saving, drafting prompts only",
                       [r["saving"] / r["fraction"] for r in drafting],
                       [r["mbar"] for r in drafting])
    linear = per_round["B  a + b*Mbar"]
    print("measured per-drafting-round slope in Mbar: %+.1f +- %.1f us/token-column"
          % (linear["beta"][1], linear["se"][1]))
    print("columns the tight grid deletes at width M under onepass%s: M-1"
          % "".join(str(w) for w in (6, 7)))
    for width in ONEPASS_WIDTHS:
        implied = coefficient * math.log(width) / max(1, width - 1)
        sigma_away = ((implied - linear["beta"][1]) / linear["se"][1]
                      if linear["se"][1] else float("nan"))
        print("  width %d: C implies %8.1f us per deleted column, which the "
              "measured slope excludes at %5.1f sigma"
              % (width, implied, abs(sigma_away)))

    print("\nSUMMARY")
    print("  centering       : A uncentered %.4f vs C uncentered %.4f"
          % (a["r2_uncentered"], c["r2_uncentered"]))
    print("  lowest PRESS    : %s"
          % min(all8.items(), key=lambda kv: kv[1]["press"])[0])
    print("  monotonicity    : observed %+.1f us, C requires %+.1f us"
          % (hi["saving"] - lo["saving"],
             coefficient * (math.log(hi["mbar"]) - math.log(lo["mbar"]))))
    json.dump({"table": table}, open("research/out/e135-f15-reconcile.json", "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
