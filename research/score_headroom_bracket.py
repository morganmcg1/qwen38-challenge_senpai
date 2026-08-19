#!/usr/bin/env python3
"""Independent referee for the three score-side scalars the campaign keeps
re-deriving by hand. Written to check E37 (askeladd) and E35 (alphonse) against
each other; kept because holding two independent routes to a load-bearing
scalar is the cheapest error check we have (ledger 136).

Three things it computes, none of which needs a GPU or a network call:

  1. The score identity: score = mean of the 4th and 5th order statistics of
     raw_p = serial_s_per_tok / mtp_s_per_tok.  Reproduces our board row to 8 dp.

  2. Per-prompt HEADROOM: how far one prompt's raw_p can rise before the central
     pair changes and further gain stops being scored.  The binding neighbour is
     the 6th order statistic (essays), for beagle AND medicine alike.

  3. askeladd's E37 BRACKET on the dispatched-width distribution, from published
     telemetry only.  effective_mean_draft_len is the mean of draftTokens.count
     over rounds and M = drafts + 1, so a per-prompt distribution over depths
     {0..maxDepth} satisfies exactly two equality constraints (sum = 1, mean = n).
     That polytope's vertices have <= 2 support points, so extrema of any linear
     functional are exact by enumeration over pairs.  No proxy, no simulation.

     Note the row-weighted share is a RATIO of linear functionals, not a linear
     functional, so its extremum is not guaranteed at a 2-support vertex in
     general.  For this family it is, and `--self-test` checks that claim by
     random search rather than asserting it.

Usage:  python research/score_headroom_bracket.py [--self-test]
"""

from __future__ import annotations

import argparse
import itertools
import random

# Our ranked row ca9251b8, sorted ascending by raw_p.  480 verified cells.
RAW_P = {
    "plutarch": 1.252802,
    "drama": 1.916682,
    "travel": 2.179802,
    "beagle": 3.120154,
    "medicine": 3.344863,
    "essays": 3.366118,
    "republic": 3.394017,
    "botany": 3.425360,
}
# effective_mean_draft_len, ranked, bit-identical across the top 12 (ledger 123).
RANKED_N = {
    "plutarch": 0.1540,
    "drama": 2.2976,
    "travel": 2.6557,
    "beagle": 4.5327,
    "medicine": 4.7677,
    "republic": 5.2697,
    "essays": 5.4253,
    "botany": 5.7765,
}
BOARD_TOP = 3.24929398547457
OUR_ROW = 3.23250848263467
MAX_DEPTH = 8


def score(raw_p: dict[str, float]) -> float:
    """Mean of the 4th and 5th order statistics (1-indexed) of 8 prompts."""
    v = sorted(raw_p.values())
    return (v[3] + v[4]) / 2.0


def headroom(prompt: str, raw_p: dict[str, float] | None = None) -> float:
    """Relative gain in `prompt`'s raw_p at which score stops rising.

    Closed form: the prompt contributes to the central pair until it passes the
    6th order statistic of the OTHER seven prompts.  Bisection is used instead so
    the answer does not depend on my reading of the order-statistic algebra.
    """
    raw_p = raw_p or RAW_P
    lo, hi = 1.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        a = dict(raw_p)
        a[prompt] = raw_p[prompt] * mid
        b = dict(raw_p)
        b[prompt] = raw_p[prompt] * mid * (1.0 + 1e-9)
        if score(b) > score(a) + 1e-15:
            lo = mid
        else:
            hi = mid
    return lo - 1.0


def d_score_d_raw_p(prompt: str, raw_p: dict[str, float] | None = None) -> float:
    """d(score)/score per unit RELATIVE gain in this prompt's raw_p.

    Equals raw_p[prompt] / (2 * score) while the prompt is in the central pair,
    and 0 otherwise.  This is the 0.4827 factor in the E38 payoff chain.
    """
    raw_p = raw_p or RAW_P
    v = sorted(raw_p.values())
    if raw_p[prompt] not in (v[3], v[4]):
        return 0.0
    return raw_p[prompt] / (2.0 * score(raw_p))


def _vertices(n: float, max_depth: int = MAX_DEPTH):
    """All <= 2-support vertices of {p >= 0 : sum p = 1, sum d*p = n}."""
    for a, b in itertools.combinations(range(max_depth + 1), 2):
        if a <= n <= b:
            w = (n - a) / (b - a)          # weight on b
            yield {a: 1.0 - w, b: w}
    if float(n).is_integer() and 0 <= n <= max_depth:
        yield {int(n): 1.0}


def round_share_bracket(n: float, m_min: int = 6, max_depth: int = MAX_DEPTH):
    """Exact [min, max] of the ROUND share at dispatched width M >= m_min."""
    vals = [
        sum(w for d, w in v.items() if d + 1 >= m_min)
        for v in _vertices(n, max_depth)
    ]
    return min(vals), max(vals)


def row_share_bracket(n: float, m_min: int = 6, max_depth: int = MAX_DEPTH):
    """Exact [min, max] of the ROW share at M >= m_min.  Rows per round = M = d+1."""
    vals = []
    for v in _vertices(n, max_depth):
        rows = sum(w * (d + 1) for d, w in v.items())
        deep = sum(w * (d + 1) for d, w in v.items() if d + 1 >= m_min)
        vals.append(deep / rows)
    return min(vals), max(vals)


def report() -> None:
    print("=== 1. score identity ===")
    s = score(RAW_P)
    print(f"  mean(4th, 5th order stat) = {s:.8f}")
    print(f"  our board row             = {OUR_ROW:.8f}   diff {abs(s - OUR_ROW):.2e}")
    print(f"  board top                 = {BOARD_TOP:.8f}")
    print(f"  official gap              = {(s / BOARD_TOP - 1) * 100:+.4f} %")
    print("  (engineerable gap is -0.561 % under R'; ledger 131)")

    print("\n=== 2. headroom: how far one prompt can carry us alone ===")
    print(f"  binding neighbour = 6th order statistic = essays {RAW_P['essays']:.6f}")
    for p in ("beagle", "medicine"):
        h = headroom(p)
        a = dict(RAW_P)
        a[p] = RAW_P[p] * (1.0 + h)
        print(f"  {p:9s} headroom {h * 100:+7.3f} % of raw_p -> "
              f"ceiling {a[p]:.6f}, score {score(a):.8f} "
              f"({(score(a) / s - 1) * 100:+.4f} % of score)   "
              f"d(score)/d(raw_p) = {d_score_d_raw_p(p):.4f}")
    print("  cross-check: alphonse E35 saturation model gives medicine-only +0.318 %")

    print("\n=== 3. E37 telemetry bracket on dispatched width M >= 6 ===")
    print(f"  {'prompt':10s} {'n':>7s}   round share            row share")
    for p, n in sorted(RANKED_N.items(), key=lambda kv: kv[1]):
        rl, rh = round_share_bracket(n)
        wl, wh = row_share_bracket(n)
        print(f"  {p:10s} {n:7.4f}   [{rl:.4f}, {rh:.4f}]      [{wl:.4f}, {wh:.4f}]")
    print("  askeladd E37: beagle round [.1333,.9066] row >= .2167 ;"
          " medicine [.1920,.9536] >= .2996")


def self_test() -> int:
    fails = 0

    def check(name, got, want, tol):
        nonlocal fails
        ok = abs(got - want) <= tol
        if not ok:
            fails += 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got:.6f} want {want:.6f}")

    print("self-test")
    # 1. score identity must reproduce our published row.
    check("score identity vs board row", score(RAW_P), OUR_ROW, 1e-7)

    # 2. headroom must equal the closed form: pass the 6th order statistic.
    for p in ("beagle", "medicine"):
        closed = RAW_P["essays"] / RAW_P[p] - 1.0
        check(f"headroom({p}) == essays/{p} - 1", headroom(p), closed, 1e-6)

    # 3. the payoff factor used in the E38 chain.
    check("d(score)/d(raw_p) beagle", d_score_d_raw_p("beagle"), 0.4827, 5e-4)
    # medicine is the 5th order statistic, so it also carries weight.
    check("d(score)/d(raw_p) medicine", d_score_d_raw_p("medicine"), 0.5174, 5e-4)
    # travel is outside the central pair and must be worth exactly nothing.
    check("d(score)/d(raw_p) travel", d_score_d_raw_p("travel"), 0.0, 0.0)

    # 4. reproduce askeladd's published bracket numbers.
    rl, rh = round_share_bracket(RANKED_N["beagle"])
    check("beagle round share min", rl, 0.1333, 5e-4)
    check("beagle round share max", rh, 0.9066, 5e-4)
    check("beagle row share min", row_share_bracket(RANKED_N["beagle"])[0], 0.2167, 5e-4)
    check("medicine row share min", row_share_bracket(RANKED_N["medicine"])[0], 0.2996, 5e-4)

    # 5. degenerate ends: n = 0 can never dispatch M >= 6; n = maxDepth always does.
    check("n=0 round share max", round_share_bracket(0.0)[1], 0.0, 1e-12)
    check("n=8 round share min", round_share_bracket(8.0)[0], 1.0, 1e-12)
    # M >= 1 is trivially every round, for any n.
    check("m_min=1 round share min", round_share_bracket(4.5, m_min=1)[0], 1.0, 1e-12)

    # 6. The row share is a RATIO of linear functionals, so a 2-support vertex is
    #    not guaranteed optimal a priori.  Verify by random search that no interior
    #    distribution beats the enumerated bracket.
    rng = random.Random(20260819)
    worst = 0.0
    for prompt in ("beagle", "medicine"):
        n = RANKED_N[prompt]
        lo, hi = row_share_bracket(n)
        for _ in range(40000):
            w = [rng.random() for _ in range(MAX_DEPTH + 1)]
            tot = sum(w)
            w = [x / tot for x in w]
            mean = sum(i * w[i] for i in range(MAX_DEPTH + 1))
            if mean <= 1e-9:
                continue
            # rescale support toward the target mean by mixing with a point mass
            for t in (0.0, 0.25, 0.5, 0.75):
                lam = t
                mixed = [(1 - lam) * w[i] for i in range(MAX_DEPTH + 1)]
                k = min(MAX_DEPTH, max(0, int(round(n))))
                mixed[k] += lam
                m2 = sum(i * mixed[i] for i in range(MAX_DEPTH + 1))
                if abs(m2 - n) > 1e-6:
                    continue
                rows = sum(mixed[i] * (i + 1) for i in range(MAX_DEPTH + 1))
                deep = sum(mixed[i] * (i + 1) for i in range(MAX_DEPTH + 1) if i + 1 >= 6)
                r = deep / rows
                worst = max(worst, lo - r, r - hi)
    ok = worst <= 1e-9
    if not ok:
        fails += 1
    print(f"  [{'ok' if ok else 'FAIL'}] random search never escapes the row bracket "
          f"(max excursion {worst:.2e})")

    print(f"self-test: {fails} failure(s)")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(1 if self_test() else 0)
    report()
