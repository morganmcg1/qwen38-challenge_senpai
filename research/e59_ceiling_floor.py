#!/usr/bin/env python3
"""Can the shipped QMV register ceiling be LOWERED, not merely held?

Ledger 187(L) establishes that only an edit leaving the kernel maximum
unchanged can win at rank. That raises the dual question this tool answers:
is the shipped maximum of 104 a free parameter we could push down, which
would be a pure win with no cell-timing cost at all?

The answer is no, and the reason is a legality constraint rather than a
performance tradeoff. `quantized.h:1169` asserts `M % IPG != 1`, and
`IPG = ceil(M / ceil(M / 4))` is the shipped chooser. Enumerating every legal
configuration shows M=7 has no legal accumulator count below 4, so 104 is a
hard floor for the ceiling while M=7 remains dispatchable.

That makes thorfinn's r=2 row-block route the unique survivor for a different
and stronger reason than "it happens to fit": 104 cannot move, and 91 < 104.

Sources
  183(C)  affine ladders  r=4: 20 + 21*NA   r=2: 16 + 15*NA
  E44     competing r=2 anchor fit          r=2: 15 + 17*NA
  183(B)  one library, one pipeline, one allocation for all M = 1..9
  187(C)  census: <T,3,3>=83 <T,4,4>=104 <T,5,5>=125 (exact on single-group)
"""

import argparse
import math

# `_m` wrapper legality, quantized.h:1156-1175.
M_MIN, M_MAX = 3, 9
NA_MIN, NA_MAX = 2, 12

# Shipped dispatch table for out_vec_size >= 4096 (verified in source).
SHIPPED = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# 183(C) affine register ladders, max residual 0.25 reg over NA = 2..12.
LADDER_R4 = (20, 21)
# Two competing fits for r=2; 187(L) carries both as a band.
LADDER_R2_183C = (16, 15)
LADDER_R2_E44 = (15, 17)

# 187(C) census anchors, exact on single-group cells.
CENSUS_EXACT = {(3, 3): 83, (4, 4): 104, (5, 5): 125}


def regs(ladder, na):
    a, b = ladder
    return a + b * na


def shipped_ipg(m):
    """IPG = ceil(M / ceil(M / 4)), the shipped chooser."""
    return math.ceil(m / math.ceil(m / 4))


def legal(m, ipg):
    """quantized.h:1169 static_assert(M % IPG != 1), plus IPG <= M."""
    return 1 <= ipg <= m and m % ipg != 1


def legal_ipgs(m):
    return [ipg for ipg in range(1, m + 1) if legal(m, ipg)]


def streams(m, ipg):
    return math.ceil(m / ipg)


def report():
    print("=" * 74)
    print("1. The shipped chooser reproduces the shipped table")
    print("=" * 74)
    print(f"{'M':>3} {'shipped IPG':>12} {'ceil(M/ceil(M/4))':>18} {'match':>7}"
          f" {'streams':>8} {'r=4 regs':>9}")
    ceiling = 0
    for m in range(M_MIN, M_MAX + 1):
        pred = shipped_ipg(m)
        got = SHIPPED[m]
        r = regs(LADDER_R4, got)
        ceiling = max(ceiling, r)
        print(f"{m:>3} {got:>12} {pred:>18} {str(pred == got):>7}"
              f" {streams(m, got):>8} {r:>9}")
    print(f"\nshipped kernel maximum (r=4) = {ceiling}")
    assert ceiling == 104, ceiling

    print()
    print("=" * 74)
    print("2. Census anchors agree with the r=4 ladder on single-group cells")
    print("=" * 74)
    for (m, na), obs in sorted(CENSUS_EXACT.items()):
        pred = regs(LADDER_R4, na)
        print(f"  <T,{m},{na}>  census {obs:>4}   ladder {pred:>4}   "
              f"resid {obs - pred:+d}")
        assert obs == pred, (m, na, obs, pred)

    print()
    print("=" * 74)
    print("3. Every legal configuration, and the cheapest reachable per width")
    print("=" * 74)
    print(f"{'M':>3} {'legal IPG':>22} {'min legal NA':>13} {'regs at min':>12}"
          f" {'streams there':>14}")
    floors = {}
    for m in range(M_MIN, M_MAX + 1):
        opts = legal_ipgs(m)
        # IPG == NA for the _m wrapper; NA_MIN is the static_assert floor at
        # quantized.h:980 for the accumulator vector type.
        usable = [i for i in opts if i >= NA_MIN]
        best = min(usable)
        floors[m] = regs(LADDER_R4, best)
        print(f"{m:>3} {str(opts):>22} {best:>13} {floors[m]:>12}"
              f" {streams(m, best):>14}")

    hard_floor = max(floors.values())
    binding = [m for m, v in floors.items() if v == hard_floor]
    print(f"\nlowest achievable kernel maximum over all legal retablings"
          f" = {hard_floor}")
    print(f"binding widths (cannot go lower) = {binding}")

    print()
    print("=" * 74)
    print("4. Verdict")
    print("=" * 74)
    if hard_floor >= ceiling:
        print(f"  The ceiling CANNOT be lowered by retabling.")
        print(f"  Shipped maximum {ceiling} already equals the legal floor"
              f" {hard_floor},")
        print(f"  pinned by width(s) {binding}.")
        for m in binding:
            opts = [i for i in legal_ipgs(m) if i >= NA_MIN]
            print(f"    M={m}: legal accumulator counts {opts};"
                  f" none below {min(opts)}")
            for i in range(NA_MIN, min(opts)):
                print(f"      M={m} at NA={i} would be"
                      f" {m} %% {i} = {m % i} -> illegal")
    else:
        print(f"  A retabling to maximum {hard_floor} is legal. Investigate.")

    print()
    print("=" * 74)
    print("5. The r=2 row-block route against that hard floor")
    print("=" * 74)
    print(f"  shipped kernel maximum (immovable)      = {ceiling}")
    for name, lad in (("183(C) 16+15*NA", LADDER_R2_183C),
                      ("E44    15+17*NA", LADDER_R2_E44)):
        v = regs(lad, 5)
        verdict = "fits under ceiling" if v < ceiling else "RAISES ceiling"
        print(f"  <T,5,5> at r=2, {name}  = {v:>4}   {verdict}")
    print(f"  <T,5,5> at r=4, {LADDER_R4[0]}+{LADDER_R4[1]}*NA"
          f"  = {regs(LADDER_R4, 5):>4}   RAISES ceiling")
    print()
    print("  Both r=2 fits sit below the immovable floor, so the conclusion is")
    print("  robust to the intercept disagreement. Measure the census anyway;")
    print("  187(L) requires it rather than assuming the intercept.")

    print()
    print("=" * 74)
    print("6. What this adds to the E59 brief")
    print("=" * 74)
    print("  The r=2 route is not merely one option that happens to fit under")
    print("  104. It is the only route that can ever fit, because 104 is a")
    print("  legality floor and not a tuning choice. Any NA=5 cell at r=4")
    print("  costs +21 registers; no retabling can buy that back elsewhere.")


def self_test():
    checks = 0

    # 1. shipped chooser reproduces the shipped table exactly
    for m in range(M_MIN, M_MAX + 1):
        assert shipped_ipg(m) == SHIPPED[m], m
    checks += 1

    # 2. every shipped configuration is legal
    for m, ipg in SHIPPED.items():
        assert legal(m, ipg), (m, ipg)
    checks += 1

    # 3. census anchors match the r=4 ladder exactly on single-group cells
    for (m, na), obs in CENSUS_EXACT.items():
        assert regs(LADDER_R4, na) == obs, (m, na)
    checks += 1

    # 4. shipped maximum is 104
    assert max(regs(LADDER_R4, SHIPPED[m])
               for m in range(M_MIN, M_MAX + 1)) == 104
    checks += 1

    # 5. M=7 has no legal accumulator count below 4 -- the load-bearing claim
    assert [i for i in legal_ipgs(7) if i >= NA_MIN] == [4, 5, 7], legal_ipgs(7)
    assert 7 % 2 == 1 and 7 % 3 == 1 and 7 % 6 == 1
    checks += 1

    # 6. the legal floor equals the shipped maximum: no retabling helps
    floors = {m: regs(LADDER_R4, min(i for i in legal_ipgs(m) if i >= NA_MIN))
              for m in range(M_MIN, M_MAX + 1)}
    assert max(floors.values()) == 104, floors
    checks += 1

    # 7. both r=2 fits put <T,5,5> strictly below that floor
    assert regs(LADDER_R2_183C, 5) == 91
    assert regs(LADDER_R2_E44, 5) == 100
    assert regs(LADDER_R2_183C, 5) < 104 and regs(LADDER_R2_E44, 5) < 104
    checks += 1

    # 8. <T,5,5> at r=4 raises the ceiling, which is why E27 pays the dose
    assert regs(LADDER_R4, 5) == 125 > 104
    checks += 1

    # 9. the r=2 ladders disagree, so the intercept must be measured
    assert regs(LADDER_R2_183C, 5) != regs(LADDER_R2_E44, 5)
    checks += 1

    print(f"self-test OK: {checks} checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        report()
