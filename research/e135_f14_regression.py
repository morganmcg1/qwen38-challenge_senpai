#!/usr/bin/env python3
"""F14: separate the flat, linear, and logarithmic models of the launch saving.

    YUKON_API_TOKEN=... python3 research/e135_f14_regression.py BEFORE AFTER

`BEFORE` and `AFTER` are submission id prefixes. The pair must differ by the
launch grid alone. Receipts are cached under `research/out/receipts/`.

WHY THREE MODELS DISAGREE
-------------------------
F189 measured a per-round saving of about 2,257 us on M5 but could not
separate its shape:

  flat         C_tight(M) = C_wide(M) - a
  linear       C_tight(M) = C_wide(M) - a - b*M
  logarithmic  C_tight(M) = C_wide(M) - a - b*ln(M)

A flat shift preserves every step in the cost curve, so the M=5-to-M=6 cliff
that `pb6` is fitted to survives intact. A logarithmic saving grows with width,
flattens the curve, and shrinks that cliff, which would make `pb6`'s 1.45 tier
factor wrong.

ROUND COUNT IS NOT IN THE RECEIPT
---------------------------------
The receipt reports seconds per TOKEN. Converting to seconds per ROUND needs
the round count `R`, and the receipt has no acceptance rate, so

    tokens = R + R * acceptance * mean_draft_len

is one equation in two unknowns. `R` is still recoverable, because
`effective_mean_draft_len` is an exact rational `proposed / R`, so the reduced
denominator of that rational divides `R`. Feasibility then pins the multiple:
acceptance must lie in (0, 1], which forces `R >= tokens / (1 + mean_draft_len)`,
and the observed acceptance band on this track narrows it further.

Both this recovery and the naive `R = tokens / (1 + mean_draft_len)` are
reported, because they differ by about 16 % and the verdict must not depend on
an unstated convention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import urllib.request
from fractions import Fraction

BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = pathlib.Path("research/out/receipts")
# Acceptance cannot exceed 1, and that is the only hard constraint. The
# smallest feasible round count is the maximum-acceptance solution; every
# larger multiple implies a lower rate. Prompt `c1ec5866` shows why a tighter
# band is wrong: it drafts in 38 of 487 rounds and accepts a third of what it
# proposes, so a 0.60 floor would have silently dropped it.
ACCEPTANCE_BAND = (0.0, 1.0)


def board(token: str) -> list[dict]:
    request = urllib.request.Request(
        "%s/benchmarks/%s/submissions?all=true" % (BASE, BENCHMARK_ID),
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())["submissions"]


def receipt(prefix: str, rows: list[dict]) -> dict:
    hits = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("%r matches %d submissions" % (prefix, len(hits)))
    row = hits[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ("%s.json" % row["id"])).write_text(json.dumps(row, indent=1))
    return row


def recover_rounds(mean_draft_len: float, tokens: int) -> tuple[int | None, list[int]]:
    """Round counts consistent with an exact `proposed / R` mean draft length."""
    denominator = Fraction(mean_draft_len).limit_denominator(4096).denominator
    floor = tokens / (1.0 + mean_draft_len)
    feasible = []
    rounds = denominator
    while rounds < tokens:
        if rounds >= floor:
            acceptance = (tokens - rounds) / (rounds * mean_draft_len)
            if ACCEPTANCE_BAND[0] < acceptance <= ACCEPTANCE_BAND[1]:
                feasible.append(rounds)
        rounds += denominator
    return (feasible[0] if feasible else None), feasible


def invert(matrix: list[list[float]]) -> list[list[float]]:
    width = len(matrix)
    work = [row[:] + [1.0 if i == j else 0.0 for j in range(width)]
            for i, row in enumerate(matrix)]
    for pivot in range(width):
        best = max(range(pivot, width), key=lambda r: abs(work[r][pivot]))
        work[pivot], work[best] = work[best], work[pivot]
        if abs(work[pivot][pivot]) < 1e-12:
            raise SystemExit("singular design; the regressors are collinear")
        scale = work[pivot][pivot]
        work[pivot] = [v / scale for v in work[pivot]]
        for row in range(width):
            if row == pivot:
                continue
            factor = work[row][pivot]
            work[row] = [v - factor * p for v, p in zip(work[row], work[pivot])]
    return [row[width:] for row in work]


def ols(y: list[float], columns: list[list[float]]):
    """Tiny OLS. `columns` excludes the intercept. Returns beta, se, sigma."""
    n = len(y)
    design = [[1.0] + [column[i] for column in columns] for i in range(n)]
    width = len(design[0])
    gram = [[sum(design[i][a] * design[i][b] for i in range(n))
             for b in range(width)] for a in range(width)]
    moment = [sum(design[i][a] * y[i] for i in range(n)) for a in range(width)]
    inverse = invert(gram)
    beta = [sum(inverse[a][b] * moment[b] for b in range(width))
            for a in range(width)]
    residuals = [y[i] - sum(beta[a] * design[i][a] for a in range(width))
                 for i in range(n)]
    dof = max(1, n - width)
    variance = sum(r * r for r in residuals) / dof
    se = [math.sqrt(max(0.0, variance * inverse[a][a])) for a in range(width)]
    return beta, se, math.sqrt(variance)


# Two-sided 5 % critical value of Student's t at 6 degrees of freedom, which is
# the 8-prompt receipt fitted with an intercept and one slope.
T_CRITICAL_DOF6 = 2.447


def analyse(label: str, savings: list[float], widths: list[float]) -> None:
    print("\n=== %s ===" % label)
    print("%-9s %-13s %-22s %-10s %s"
          % ("model", "intercept", "slope", "resid sd", "slope t"))
    beta, se, sigma = ols(savings, [])
    flat_sigma = sigma
    print("%-9s %-13.1f %-22s %-10.1f %s" % ("flat", beta[0], "-", sigma, "-"))

    verdicts = {}
    for name, regressor in (("linear", widths),
                            ("log", [math.log(w) for w in widths])):
        beta, se, sigma = ols(savings, [regressor])
        t = beta[1] / se[1] if se[1] else float("inf")
        verdicts[name] = (beta[1], se[1], t, sigma)
        print("%-9s %-13.1f %-22s %-10.1f %+.2f"
              % (name, beta[0], "%+.1f +/- %.1f" % (beta[1], se[1]), sigma, t))

    significant = {n: v for n, v in verdicts.items()
                   if abs(v[2]) >= T_CRITICAL_DOF6}
    print("flat residual sd %.1f us/round" % flat_sigma)
    if not significant:
        print("VERDICT flat survives: no slope reaches |t| = %.3f at 6 dof, so "
              "the cost curve keeps every step and pb6's 1.45 tier stands"
              % T_CRITICAL_DOF6)
        return
    best = max(significant, key=lambda n: abs(significant[n][2]))
    slope, error, t, _ = significant[best]
    print("VERDICT the %s model wins: slope %+.1f +/- %.1f us/round per unit, "
          "t=%+.2f. Hand b to edward. Do NOT change the tier here."
          % (best, slope, error, t))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    rows = board(os.environ["YUKON_API_TOKEN"])
    before, after = receipt(args.before, rows), receipt(args.after, rows)
    for row, name in ((before, "before"), (after, "after")):
        print("%-7s %s score=%s status=%s commit=%s"
              % (name, row["id"][:8], row.get("officialScore"),
                 row.get("status"), str(row.get("submissionCommitSha"))[:8]))

    tokens = before["officialMetrics"]["decode_tokens"]
    if tokens != after["officialMetrics"]["decode_tokens"]:
        raise SystemExit("the two receipts use different decode windows")

    by_hash = {p["prompt_sha256"]: p for p in after["officialMetrics"]["per_prompt"]}
    savings_recovered, savings_naive, widths = [], [], []

    print("\n%-10s %8s %8s %10s %10s %8s %8s"
          % ("prompt", "Mbar", "R(rec)", "R(naive)", "dt us/tok",
             "us/round", "naive"))
    for old in before["officialMetrics"]["per_prompt"]:
        new = by_hash.get(old["prompt_sha256"])
        if new is None:
            raise SystemExit("prompt %s is absent from the after receipt"
                             % old["prompt_sha256"][:8])
        mean_draft = new["effective_mean_draft_len"]
        if old["effective_mean_draft_len"] != mean_draft:
            print("NOTE prompt %s changed its schedule: %r -> %r"
                  % (old["prompt_sha256"][:8], old["effective_mean_draft_len"],
                     mean_draft))
        rounds, feasible = recover_rounds(mean_draft, tokens)
        if rounds is None:
            raise SystemExit("no feasible round count for prompt %s"
                             % old["prompt_sha256"][:8])
        naive = tokens / (1.0 + mean_draft)
        delta = old["mtp_seconds_per_token_mean"] - new["mtp_seconds_per_token_mean"]
        per_round = delta * (tokens / rounds) * 1e6
        per_round_naive = delta * (tokens / naive) * 1e6
        width = 1.0 + mean_draft
        savings_recovered.append(per_round)
        savings_naive.append(per_round_naive)
        widths.append(width)
        print("%-10s %8.3f %8d %10.1f %10.3f %8.1f %8.1f"
              % (old["prompt_sha256"][:8], width, rounds, naive,
                 delta * 1e6, per_round, per_round_naive))
        if len(feasible) > 1:
            print("           sensitivity: other feasible round counts %s"
                  % feasible[1:3])

    analyse("all prompts, recovered round counts", savings_recovered, widths)
    analyse("all prompts, naive round counts (sensitivity)",
            savings_naive, widths)

    # A prompt that declines to draft in most of its rounds spends most of its
    # time on the serial path, where the launch grid cannot act. It is a real
    # data point for the cost model, but it carries enormous leverage at the
    # low end of the width axis, so the fit is reported without it as well.
    keep = [i for i, p in enumerate(before["officialMetrics"]["per_prompt"])
            if by_hash[p["prompt_sha256"]]["non_drafting_round_count"] == 0]
    if len(keep) < len(widths):
        dropped = len(widths) - len(keep)
        analyse("drafting prompts only (%d dropped)" % dropped,
                [savings_recovered[i] for i in keep], [widths[i] for i in keep])

    # Dropping the prompt is blunt. The launch grid acts on the multi-row
    # verification matvec, and a non-drafting round never dispatches one, so
    # its expected saving is zero by construction. Dividing by the drafting
    # fraction puts every prompt on a per-DRAFTING-round axis and keeps all
    # eight. If the all-prompt slope was leverage from an idle prompt, it
    # disappears here.
    print("\n=== per-drafting-round saving, all prompts ===")
    print("%-10s %8s %10s %12s" % ("prompt", "Mbar", "drafting", "us/round"))
    corrected = []
    for i, old in enumerate(before["officialMetrics"]["per_prompt"]):
        new = by_hash[old["prompt_sha256"]]
        rounds, _ = recover_rounds(new["effective_mean_draft_len"], tokens)
        fraction = (rounds - new["non_drafting_round_count"]) / rounds
        corrected.append(savings_recovered[i] / fraction)
        print("%-10s %8.3f %10.3f %12.1f"
              % (old["prompt_sha256"][:8], widths[i], fraction, corrected[i]))
    analyse("per-drafting-round saving", corrected, widths)

    print("\nMbar range %.2fx over %d prompts"
          % (max(widths) / min(widths), len(widths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
