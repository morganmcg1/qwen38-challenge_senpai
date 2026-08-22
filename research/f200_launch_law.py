#!/usr/bin/env python3
"""FINDING 200 - the F14 width regression on OUR OWN tight-grid receipt.

Two ranked pairs now isolate exactly one mechanism, grid wide -> tight:

  rival  02742bf0 -> ed608e64   shipped plan, probe 0.15, shared entry point
  ours   623e77af -> 572b2cc4   onePass67 plan, probe 0.25, tiered entry points

Both pairs hold effective_mean_draft_len and non_drafting_round_count
digit-identical, so R is pinned and the round-time delta equals the
candidate seconds/token delta exactly.

The difference between the two pairs is the (plan x grid) interaction.
"""
import json
import math

BOARD = "/tmp/yukon-board/full.json"
R = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
     "republic": 93, "essays": 92, "medicine": 90, "botany": 81}
NAME = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
        "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
        "ea82dcb5": "republic", "3b10cb4d": "travel"}

# Table.onePass67 (ours) and Table.shipped (rival): (m, ipg)
IPG_OURS = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}
IPG_RIVAL = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}


def load():
    with open(BOARD) as fh:
        return json.load(fh)["submissions"]


def find(subs, pref):
    for s in subs:
        if s["id"].startswith(pref):
            return s
    raise SystemExit("not found " + pref)


def pp(row):
    out = {}
    for e in row["officialMetrics"]["per_prompt"]:
        nm = NAME.get(e["prompt_sha256"][:8])
        if nm:
            out[nm] = e
    return out


def ols(X, y):
    """X is a list of rows (list of floats), y a list.  Normal equations."""
    k = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)]
         for a in range(k)]
    b = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    # gaussian elimination
    M = [A[r][:] + [b[r]] for r in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for r in range(k):
            if r == c:
                continue
            f = M[r][c] / pv
            for j in range(c, k + 1):
                M[r][j] -= f * M[c][j]
    beta = [M[c][k] / M[c][c] for c in range(k)]
    fit = [sum(beta[a] * X[i][a] for a in range(k)) for i in range(len(X))]
    res = [y[i] - fit[i] for i in range(len(y))]
    ybar = sum(y) / len(y)
    sst = sum((v - ybar) ** 2 for v in y)
    sse = sum(v * v for v in res)
    dof = max(1, len(y) - k)
    return beta, (1.0 - sse / sst if sst else float("nan")), math.sqrt(sse / dof)


def pair(a8, b8, ipg, label):
    subs = load()
    A, B = find(subs, a8), find(subs, b8)
    pa, pb = pp(A), pp(B)
    rows = []
    for nm in sorted(pa, key=lambda n: pa[n]["effective_mean_draft_len"]):
        mbar = pa[nm]["effective_mean_draft_len"] + 1.0
        r = R[nm]
        ra = pa[nm]["mtp_seconds_per_token_mean"] * 512e6 / r
        rb = pb[nm]["mtp_seconds_per_token_mean"] * 512e6 / r
        rows.append((nm, mbar, ra, rb, rb - ra))
    print("=" * 78)
    print("%s   %s %.8f -> %s %.8f" % (label, a8, A["officialScore"], b8,
                                       B["officialScore"]))
    print("%-9s %6s %12s %12s %10s %9s %7s %7s" %
          ("prompt", "Mbar", "us/rnd A", "us/rnd B", "delta us", "delta %",
           "colWide", "colTgt"))
    for nm, mbar, ra, rb, d in rows:
        m = max(3, min(9, int(round(mbar))))
        cw = mbar
        ct = math.ceil(m / ipg[m])
        print("%-9s %6.3f %12.1f %12.1f %10.1f %9.4f %7.2f %7d" %
              (nm, mbar, ra, rb, d, 100.0 * d / ra, cw, ct))
    d = [r[4] for r in rows]
    mb = [r[1] for r in rows]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1))
    print("  saving mean %.1f us/round  sd %.1f  CV %.2f %%" %
          (-mean, sd, 100.0 * sd / abs(mean)))

    print("  --- three F189 models, dependent variable = delta us/round ---")
    b1, r2a, sa = ols([[1.0]] * n, d)
    print("  A flat            a=%9.1f                 R2 %7.4f  resid sd %7.1f"
          % (b1[0], r2a, sa))
    b2, r2b, sb = ols([[1.0, m] for m in mb], d)
    print("  B a + b*Mbar      a=%9.1f  b=%9.1f  R2 %7.4f  resid sd %7.1f"
          % (b2[0], b2[1], r2b, sb))
    b3, r2c, sc = ols([[math.log(m)] for m in mb], d)
    print("  C a*ln(Mbar)      a=%9.1f                 R2 %7.4f  resid sd %7.1f"
          % (b3[0], r2c, sc))
    b4, r2d, sd4 = ols([[1.0, math.log(m)] for m in mb], d)
    print("  D a + b*ln(Mbar)  a=%9.1f  b=%9.1f  R2 %7.4f  resid sd %7.1f"
          % (b4[0], b4[1], r2d, sd4))
    return rows


def main():
    o = pair("623e77af", "572b2cc4", IPG_OURS, "OURS  onePass67")
    r = pair("02742bf0", "ed608e64", IPG_RIVAL, "RIVAL shipped  ")

    print("=" * 78)
    print("THE (plan x grid) INTERACTION, ranked, per prompt")
    print("%-9s %10s %10s %10s   %s" %
          ("prompt", "ours d%", "rival d%", "inter pp", "colremoved o/r"))
    do = {x[0]: x for x in o}
    dr = {x[0]: x for x in r}
    so = sr = 0.0
    for nm in ["beagle", "essays", "medicine", "botany", "republic",
               "plutarch", "drama", "travel"]:
        a = 100.0 * do[nm][4] / do[nm][2]
        b = 100.0 * dr[nm][4] / dr[nm][2]
        m = max(3, min(9, int(round(do[nm][1]))))
        co = do[nm][1] - math.ceil(m / IPG_OURS[m])
        cr = dr[nm][1] - math.ceil(m / IPG_RIVAL[m])
        so += a
        sr += b
        print("%-9s %10.4f %10.4f %10.4f   %.2f / %.2f" %
              (nm, a, b, a - b, co, cr))
    print("mean      %10.4f %10.4f %10.4f" % (so / 8, sr / 8, (so - sr) / 8))
    print()
    print("F194 pre-registered one-pass cost under tight (medpair) +0.2649 %%")
    print("F138 pre-registered interaction band                    +1200..+3400 us isolated")


main()
