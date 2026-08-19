"""Attribute E41's local per-round width curve to the weight-stream count.

Answers the advisor's 07:51Z step-vs-quadratic question from data E41 already
collected. `streams(M) = ceil(M / IPG(M))` is read out of the `case M:` switch in
`quantized.h`, so the boundary is *given by source* rather than fitted, and the
regression has one fewer degree of freedom than a step model with a free
breakpoint.

The two IPG tables below are the shipped dispatch at the E41 assignment base and
at the current advisor tip. They differ only at M=5 and M=9, which is why the
1->2 stream boundary moves from 5->6 down to 4->5 on the rebased tree.

Run from the repository root; reads only committed artifacts.
"""

import json
import math

IPG_BY_BASE = {
    "04ad6bf": {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5},
    "efff400": {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3},
}
WIDTHS = list(range(3, 10))
METRICS = "research/e41-artifacts/e41-metrics.json"


def solve(rows, y):
    """Normal-equation least squares; tiny system, no numpy dependency."""
    k = len(rows[0])
    aug = [
        [sum(rows[i][p] * rows[i][q] for i in range(len(rows))) for q in range(k)]
        + [sum(rows[i][p] * y[i] for i in range(len(rows)))]
        for p in range(k)
    ]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(k):
            if r != col:
                f = aug[r][col] / aug[col][col]
                for q in range(col, k + 1):
                    aug[r][q] -= f * aug[col][q]
    return [aug[i][k] / aug[i][i] for i in range(k)]


def streams(m, ipg):
    return math.ceil(m / ipg[m])


def main():
    c_round = json.load(open(METRICS))["c_round_ms"]
    old, new = IPG_BY_BASE["04ad6bf"], IPG_BY_BASE["efff400"]

    print("M  IPGold sold  IPGnew snew  changed  T_base(04ad6bf)")
    for m in WIDTHS:
        print(
            "%d    %d     %d      %d      %d    %-4s  %8.3f"
            % (
                m, old[m], streams(m, old), new[m], streams(m, new),
                "YES" if old[m] != new[m] else "no", c_round[str(m)]["base"],
            )
        )

    for tag in ("base", "base_r1"):
        t = [c_round[str(m)][tag] for m in WIDTHS]
        d1 = [t[i + 1] - t[i] for i in range(len(t) - 1)]
        print()
        print("%s  T  %s" % (tag, " ".join("%8.3f" % x for x in t)))
        print("%s  d1 %s" % (" " * len(tag), " ".join("%8.3f" % x for x in d1)))
        # A quadratic in M forces non-decreasing first differences; this drop
        # is the model-free falsification of the quadratic family.
        print("%s  5->6 %.3f, 6->7 %.3f, DROP %.3f"
              % (" " * len(tag), d1[2], d1[3], d1[2] - d1[3]))

    t = [c_round[str(m)]["base"] for m in WIDTHS]
    s_old = [streams(m, old) for m in WIDTHS]
    rows = [[1.0, float(s_old[i]), float(WIDTHS[i])] for i in range(len(WIDTHS))]
    a, b, c = solve(rows, t)
    resid = [t[i] - (a + b * s_old[i] + c * WIDTHS[i]) for i in range(len(WIDTHS))]
    print()
    print("fit T = %.3f + %.3f*streams + %.3f*M   (M=3..9, 04ad6bf stream table)"
          % (a, b, c))
    print("residuals ms:", [round(r, 3) for r in resid],
          "max abs", round(max(abs(r) for r in resid), 3))

    print()
    print("PREDICTED efff400 curve, same coefficients, new stream table:")
    for m in WIDTHS:
        so, sn = streams(m, old), streams(m, new)
        print("  M=%d  s %d->%d   T %8.3f -> %8.3f   delta %+8.3f"
              % (m, so, sn, a + b * so + c * m, a + b * sn + c * m, b * (sn - so)))
    pred = {m: a + b * streams(m, new) + c * m for m in WIDTHS}
    print()
    print("predicted first differences on efff400:",
          {"%d->%d" % (m, m + 1): round(pred[m + 1] - pred[m], 2) for m in WIDTHS[:-1]})


if __name__ == "__main__":
    main()
