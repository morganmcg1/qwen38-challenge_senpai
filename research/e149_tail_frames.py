"""E149 F2 section 3: the E148 R-C left-tail excess, re-run frame by frame.

harness=ranked, zero GPU. This does NOT re-run E148. It reuses E148 R-C's own
pricing loop unchanged, reproduces its published observed counts as a control,
and then does the two things F2 asked for:

  1. re-run the EXPECTED column at the n=11 lead-frame sigma (0.077360 pp)
     instead of the retired n=10 sigma (0.0577 pp);
  2. name the frame the OBSERVED counts were computed in.

RULE 144 PROBLEM, STATED UP FRONT. E148 R-C counted survivors on
`corrected_total_pct`, which is the UNWEIGHTED MEAN OF THE WEIGHTED FIVE. The
0.077360 pp sigma is the REALISED MEDIAN PAIR sd. Putting one against the other
is a cross-frame comparison, which Rule 144 forbids. So this file also counts
the same survivors on `median_pair_corrected_pct` and reports each frame
against its own sigma. The cross-frame column is printed only so the number F2
quoted can be located, and it is labelled as cross-frame.
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
from e148_rc import price_row, normal_tail

OUT = os.path.join(HERE, "e149-tail-frames.json")

THRESHOLDS = [-0.10, -0.15, -0.20, -0.30, -0.50, -1.00]

# The at-zero block sigma in each frame. n=10 is the retired figure E148 used;
# n=11 is what rung 0a republished after `165d4ba7` entered the block.
SIGMA = {
    "unweighted_weighted_five": {"n10": 0.057680, "n11": 0.062321},
    "median_pair": {"n10": 0.058338, "n11": 0.077360},
    "f83_marginal_weight_sum": {"n10": 0.054588, "n11": 0.072882},
}

# E148 R-C's published observed counts, carried as a reproduction control.
E148_OBSERVED = {-0.10: 88, -0.15: 52, -0.20: 35, -0.30: 14, -0.50: 10,
                 -1.00: 1}
E148_N = 375


def tail_column(values, n, sigma):
    return {str(t): {"observed": sum(1 for v in values if v <= t),
                     "expected": n * normal_tail(-t / sigma)}
            for t in THRESHOLDS}


def price_board(path):
    board, rows = E.load_rows(path)
    by_id = {r.id8: r for r in rows}
    trees = E.tree_index(rows)
    _, cohorts = E.build_cohorts(rows)
    priced = []
    for row in rows:
        entry, _ = price_row(row, trees, by_id, cohorts)
        if entry is not None and entry["priceable"]:
            priced.append(entry)
    return board, len(rows), priced


def main():
    path, scored, priced = price_board(None)
    n = len(priced)
    w5 = [e["corrected_total_pct"] for e in priced]
    mp = [e.get("median_pair_corrected_pct", e["median_pair_pct"])
          for e in priced]

    # Reproduction control. E148 R-C ran on the live board at 880 scored rows;
    # the board is now larger, so the counts cannot match exactly. The control
    # that is available is that the DEEP tail, where no new row landed, is
    # unchanged, and every shallow count moved by at most the number of new
    # priceable rows.
    counts = {t: sum(1 for v in w5 if v <= t) for t in THRESHOLDS}
    growth = n - E148_N
    drift = {t: counts[t] - E148_OBSERVED[t] for t in THRESHOLDS}
    control_ok = all(0 <= drift[t] <= growth for t in THRESHOLDS)
    print("live board %s, %d scored rows" % (path, scored))
    print("R-C priced survivors: %d, against %d published by E148 (+%d new "
          "priceable rows)" % (n, E148_N, growth))
    print("reproduction control, weighted-five observed counts against E148: "
          "%s, drift %s, monotone within board growth: %s"
          % ([counts[t] for t in THRESHOLDS], [drift[t] for t in THRESHOLDS],
             control_ok))

    result = {"harness": "ranked", "board": path, "n_scored": scored,
              "n_priced": n, "e148_published_n": E148_N,
              "e148_published_counts": {str(t): E148_OBSERVED[t]
                                        for t in THRESHOLDS},
              "observed_counts_now": {str(t): counts[t] for t in THRESHOLDS},
              "count_drift_since_e148": {str(t): drift[t] for t in THRESHOLDS},
              "reproduction_control_within_board_growth": control_ok,
              "sigma_table": SIGMA}

    print("\n=== within-frame: observed counted in the SAME frame as sigma ===")
    for frame, values in (("unweighted_weighted_five", w5),
                          ("median_pair", mp)):
        for tag in ("n10", "n11"):
            sigma = SIGMA[frame][tag]
            col = tail_column(values, n, sigma)
            result["within_frame_%s_%s" % (frame, tag)] = col
            print("  frame %-24s sigma %s = %.6f pp" % (frame, tag, sigma))
            print("    %10s %9s %12s %s"
                  % ("threshold", "observed", "expected", "verdict"))
            for t in THRESHOLDS:
                c = col[str(t)]
                print("    %10.2f %9d %12.4f %s"
                      % (t, c["observed"], c["expected"],
                         "EXCESS" if c["observed"] > max(c["expected"], 1.0)
                         else "within null"))

    print("\n=== cross-frame, F2's quoted column: observed in the "
          "weighted-five frame against the median-pair n=11 sigma ===")
    print("    RULE 144 says this pairing is not a valid comparison. It is "
          "printed only to locate F2's numbers.")
    cross = tail_column(w5, n, SIGMA["median_pair"]["n11"])
    result["cross_frame_w5_observed_vs_median_pair_sigma_n11"] = cross
    for t in THRESHOLDS:
        c = cross[str(t)]
        print("    %10.2f %9d %12.4f" % (t, c["observed"], c["expected"]))

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
