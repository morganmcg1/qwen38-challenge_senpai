#!/usr/bin/env python3
"""Bound the marginal cost of one extra launched threadgroup column.

The F14 regression finds the tight-grid saving is flat in mean verify width.
Under `onepass67` the tight grid removes `M - 1` columns at widths 3 to 7, so a
per-column saving model predicts a slope equal to the per-column cost. The
measured slope is far below that, and this quantifies by how much and states
what it implies for a table that launches one extra column at widths 6 and 7.
"""

import glob
import json
import sys
from fractions import Fraction

TOKENS = 512
# Slope of per-drafting-round saving on mean verify width, from
# `research/e135_f14_regression.py` on the 623e77af -> 572b2cc4 pair.
SLOPE_US = 82.5
SLOPE_SE_US = 70.5


def load(prefix):
    return json.load(open([p for p in glob.glob("research/out/receipts/*.json")
                           if prefix in p][0]))


def rounds_for(mean_draft_len):
    step = Fraction(mean_draft_len).limit_denominator(4096).denominator
    count = step
    while count < TOKENS:
        if count >= TOKENS / (1.0 + mean_draft_len):
            if 0 < (TOKENS - count) / (count * mean_draft_len) <= 1:
                return count
        count += step
    raise SystemExit("no feasible round count")


after = load(sys.argv[1])["officialMetrics"]
print("%-10s %8s %8s %12s %14s"
      % ("prompt", "Mbar", "rounds", "ms/round", "cols removed"))
round_times, removed_columns = [], []
for prompt in after["per_prompt"]:
    mean_draft = prompt["effective_mean_draft_len"]
    rounds = rounds_for(mean_draft)
    per_round = prompt["mtp_seconds_per_token_mean"] * TOKENS / rounds
    width = 1.0 + mean_draft
    removed = max(0.0, width - 1.0)
    round_times.append(per_round)
    removed_columns.append(removed)
    print("%-10s %8.3f %8d %12.3f %14.2f"
          % (prompt["prompt_sha256"][:8], width, rounds, per_round * 1e3,
             removed))

mean_round = sum(round_times) / len(round_times)
mean_removed = sum(removed_columns) / len(removed_columns)
mean_saving = 2210.0

print("\nmean round time      %.3f ms" % (mean_round * 1e3))
print("mean columns removed %.2f" % mean_removed)
print("flat saving          %.0f us/round" % mean_saving)
print("per-column cost implied by a proportional model: %.0f us/column"
      % (mean_saving / mean_removed))
implied = mean_saving / mean_removed
sigma = (implied - SLOPE_US) / SLOPE_SE_US
print("measured slope       %+.1f +/- %.1f us/column, which excludes the "
      "proportional model at %.1f sigma" % (SLOPE_US, SLOPE_SE_US, sigma))

upper = SLOPE_US + 2 * SLOPE_SE_US
print("\n2-sigma upper bound on one extra column: %.0f us" % upper)
print("as a share of one round: %.3f %%" % (100.0 * upper * 1e-6 / mean_round))
print("shipped adds one column at widths 6 and 7 only, so the worst case is "
      "that share of the rounds that run at those widths")
